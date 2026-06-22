#!/usr/bin/env python3
"""Exact full-catalog teacher-forced scoring for MARIUS (the decisive experiment).

This is the SPINE of the extension (see .context/extension-plan-2026-06-16.md). It
answers whether the catalog-reachability collapse is a SEARCH error (the beam
cannot reach an item the model would rank highly) or a MODEL-ranking error (the
model itself ranks the item low). No prior semantic-ID recommender paper runs this
as an empirical instrument; APAO shows it only qualitatively, and the doc-IR result
(2504.09935) is theoretical.

For a sample of test users it teacher-forces EVERY catalog item's L-tuple through
the depth decoder to obtain the exact joint log p(item | history) over the WHOLE
catalog, then compares the exact ranking against the model's own beam search on the
same users. It is read-only with respect to the authors' frozen code: it calls only
the public modules MARIUS.temporal_forward / mid_proj / depth_emb / depth_forward
and net.search, exactly as scripts/extensions/scoring.py and net.search do.

Two stages:
  run      (GPU, Snellius): score the catalog, dump per-user exact top-K + ranks +
           the beam top-K for the identical users. Writes exact_catalog_<cat>_seed<s>.npz
  analyze  (CPU, anywhere): read the npz, emit the search-vs-model decomposition table.

Token space (verified): token[l] = raw_code[l] + l*K_cb + n_special, K_cb = 256,
n_special = 2, L = 4. The catalog token table is built by inverting the committed
tuple_to_item.json under reports/extensions/topk/<category>/.

  # GPU (Snellius):
  python scripts/extensions/exact_catalog.py run \
      --category Sports_and_Outdoors --category-slug sports --seed 42 \
      --quant-method COSETTE_128d_256x4_<id>-col \
      --output-root "$SCRATCH/cosette_marius/outputs" \
      --max-users 1000 --chunk-size 1024 --eval-batch-size 32

  # CPU:
  python scripts/extensions/exact_catalog.py analyze --category Sports_and_Outdoors --seed 42

The 'analyze' stage and --selftest are pure numpy (no torch); 'run' imports torch
lazily so this file is importable/checkable on a torch-less machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

K_CB = 256        # codebook size per level (verified: codes span 0..255 in every level)
N_SPECIAL = 2     # len(SpecialTokens); also asserted from the model at runtime
DEFAULT_DUMP_ROOT = REPO / "reports" / "extensions" / "topk"
DEFAULT_OUT_ROOT = REPO / "reports" / "extensions" / "exact_catalog"
HALLUCINATION = -1


# ----------------------------------------------------------------------------
# Pure-numpy helpers (shared by run-stage post-processing, analyze, and selftest)
# ----------------------------------------------------------------------------

def build_catalog_token_table(t2i: dict, L: int, k_cb: int = K_CB, n_special: int = N_SPECIAL):
    """int64[n_catalog, L] token-space code table; row j = catalog item j.

    t2i maps "c0,c1,..,c{L-1}" (raw codes in [0, k_cb)) -> catalog index j.
    token[l] = raw[l] + l*k_cb + n_special, matching src/data/marius.py:Remap and
    the de-offset in compute_beyond_accuracy.load_model_recs.
    """
    n_catalog = len(t2i)
    table = np.full((n_catalog, L), -1, dtype=np.int64)
    level_off = np.arange(L, dtype=np.int64) * k_cb + n_special
    for key, j in t2i.items():
        raw = [int(x) for x in key.split(",")]
        if len(raw) != L:
            raise ValueError(f"code tuple {key!r} has length {len(raw)} != L={L}")
        table[int(j)] = np.asarray(raw, dtype=np.int64) + level_off
    if (table < 0).any():
        raise ValueError("catalog token table has gaps: tuple_to_item indices are not contiguous")
    return table


def tuple_to_item_idx(code_tokens_row, t2i: dict, L: int, k_cb: int = K_CB, n_special: int = N_SPECIAL):
    """De-offset a TOKEN-space L-tuple to a catalog index, or HALLUCINATION."""
    level_off = [l * k_cb + n_special for l in range(L)]
    raw = [int(c) - level_off[l] for l, c in enumerate(code_tokens_row)]
    if any(r < 0 or r >= k_cb for r in raw):
        return HALLUCINATION
    return t2i.get(",".join(str(r) for r in raw), HALLUCINATION)


def topk_and_target_rank(scores_row, target_idx, k):
    """Return (top-k indices by score desc, 1-based rank of target_idx).

    scores_row: float[n_catalog] exact log-probs. target_idx in [0,n_catalog) or -1.
    Rank uses strict-greater count + 1 (ties give the optimistic rank); with float
    log-probs exact ties are negligible.
    """
    n = scores_row.shape[0]
    kk = min(k, n)
    # argpartition for the top-kk, then sort just those.
    part = np.argpartition(-scores_row, kk - 1)[:kk]
    top = part[np.argsort(-scores_row[part])]
    if target_idx is None or target_idx < 0:
        rank = -1
    else:
        rank = int((scores_row > scores_row[target_idx]).sum()) + 1
    return top.astype(np.int64), rank


# ----------------------------------------------------------------------------
# GPU run stage (torch imported lazily)
# ----------------------------------------------------------------------------

def _score_full_catalog(net, batch_input, table_t, chunk_size, autocast_ctx):
    """float32[B, n_catalog] exact log p(item|history) for every catalog item.

    Mirrors scripts/extensions/scoring.score_marius_tuples EXACTLY (same per-step
    quantity net.search accumulates), but hoists temporal_forward/mid_proj out of
    the candidate loop and streams the catalog in chunks for memory.
    """
    import torch
    import torch.nn.functional as F

    B = batch_input.shape[0]
    N, L = table_t.shape
    with torch.no_grad(), autocast_ctx():
        temporal_tokens = net.temporal_forward(batch_input)          # B x T x D_temp
        mid = net.mid_proj(temporal_tokens)[:, -1, :]                # B x d  (== search slice)
        d = mid.shape[-1]
        out = torch.empty(B, N, dtype=torch.float32, device=mid.device)
        for s in range(0, N, chunk_size):
            cand = table_t[s:s + chunk_size]                         # Cc x L
            Cc = cand.shape[0]
            flat = cand[None, :, :].expand(B, Cc, L).reshape(B * Cc, L)   # B*Cc x L
            mid_rep = mid[:, None, None, :].expand(B, Cc, 1, d).reshape(B * Cc, 1, d)
            dec_embs = torch.cat([mid_rep, net.depth_emb(flat[:, :-1])], dim=1)  # B*Cc x L x d
            logits = net.depth_forward(dec_embs)                     # B*Cc x L x V
            log_probs = F.log_softmax(logits.float(), dim=-1)
            tok_lp = log_probs.gather(2, flat[:, :, None]).squeeze(-1)   # B*Cc x L
            out[:, s:s + Cc] = tok_lp.sum(dim=-1).view(B, Cc).float()
    return out


def run_stage(args) -> int:
    import contextlib

    import hydra
    import pytorch_lightning as L
    import ray
    import torch

    from scripts.benchmark_extensions import get_best_checkpoint, run_directory_for_seed
    from src.models import SpecialTokens
    from src.utils.tools import patch_fsspec

    pad = SpecialTokens.PAD.value
    n_special = len(SpecialTokens)
    assert n_special == N_SPECIAL, f"n_special {n_special} != {N_SPECIAL}"

    dump_dir = args.dump_root / args.category
    meta = json.loads((dump_dir / "meta.json").read_text())
    t2i = json.loads((dump_dir / "tuple_to_item.json").read_text())
    Lc = meta["L"]
    table = build_catalog_token_table(t2i, Lc)        # int64[n_catalog, L] (host)
    n_catalog = table.shape[0]
    assert n_catalog == meta["n_catalog"], (n_catalog, meta["n_catalog"])

    repo_root = REPO
    output_root = args.output_root or (repo_root / "outputs")
    results_dir = args.results_dir or (
        output_root / "results" if (output_root / "results").exists() else repo_root / "reports" / "results"
    )

    patch_fsspec()
    ray.init(ignore_reinit_error=True)

    run_directory = run_directory_for_seed(results_dir, method="marius", slug=args.category_slug, seed=args.seed)
    cfg, ckpt_path = get_best_checkpoint(output_root / "models", run_directory)
    cfg.data.ray_datasets.which = ["valid", "test"]
    if args.eval_batch_size is not None:
        cfg.data.datamodule.valid_batch_size = int(args.eval_batch_size)

    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")
    model = hydra.utils.instantiate(cfg["model"])
    model.full_hydra_config = cfg

    use_cuda = torch.cuda.is_available()
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_cuda \
        else contextlib.nullcontext

    state = {"n": 0}
    buf = {"exact_top": [], "exact_top_scores": [], "target_exact_rank": [],
           "target_codes": [], "beam_codes": [], "hist_len": [], "target_in_catalog": []}

    class ExactCollector(L.Callback):
        @torch.no_grad()
        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            if args.max_users is not None and state["n"] >= args.max_users:
                return
            net = pl_module.net
            # authoritative codebook-size check from the model's depth vocab.
            V = net.depth_emb.num_embeddings
            assert (V - n_special) % Lc == 0 and (V - n_special) // Lc == K_CB, \
                f"depth vocab {V} inconsistent with L={Lc}, K_cb={K_CB}, n_special={n_special}"

            inp = batch["input"]                                  # B x T x L (token space)
            table_t = torch.as_tensor(table, device=inp.device)
            scores = _score_full_catalog(net, inp, table_t, args.chunk_size, autocast_ctx)  # B x n_catalog
            with autocast_ctx():
                beam = pl_module.net.search(batch, n_results=args.beam_k)   # B x beam_k x L

            tgt = batch["target"]
            tgt = tgt[:, -1, :] if tgt.dim() == 3 else tgt        # B x L (token space)
            valid = (inp != pad).any(dim=2)
            hist_len = valid.sum(dim=1)

            scores_np = scores.detach().cpu().numpy()
            inp_np = inp.detach().cpu().numpy()
            tgt_np = tgt.detach().cpu().numpy()
            B = scores_np.shape[0]
            for b in range(B):
                if args.max_users is not None and state["n"] >= args.max_users:
                    break
                if args.filter_seen:
                    # Mirror search()'s filter_preds: drop the user's history items
                    # so exact-vs-beam is apples-to-apples (the beam masks seen items).
                    seen = {tuple_to_item_idx(inp_np[b, t], t2i, Lc) for t in range(inp_np.shape[1])}
                    seen.discard(HALLUCINATION)
                    if seen:
                        scores_np[b, list(seen)] = -np.inf
                target_idx = tuple_to_item_idx(tgt_np[b], t2i, Lc)
                top, rank = topk_and_target_rank(scores_np[b], target_idx, args.topk_store)
                buf["exact_top"].append(top.astype(np.int32))
                buf["exact_top_scores"].append(scores_np[b, top].astype(np.float32))
                buf["target_exact_rank"].append(rank)
                buf["target_in_catalog"].append(1 if target_idx >= 0 else 0)
                state["n"] += 1
            buf["target_codes"].append(tgt_np.astype(np.int64))
            buf["beam_codes"].append(beam.detach().cpu().numpy().astype(np.int64))
            buf["hist_len"].append(hist_len.detach().cpu().numpy().astype(np.int64))

    trainer = L.Trainer(accelerator="gpu" if use_cuda else "cpu", devices=1,
                        precision="bf16-mixed", logger=False, enable_progress_bar=False,
                        enable_model_summary=False, callbacks=[ExactCollector()])
    protocol = str(cfg.paths.protocol)
    ckpt_uri = ckpt_path if "://" in ckpt_path else f"{protocol}://{ckpt_path}"
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_uri)

    n = state["n"]
    # per-batch arrays (target_codes/beam/hist) may cover slightly more users than
    # the per-user lists when --max-users cut mid-batch; truncate to n for alignment.
    target_codes = np.concatenate(buf["target_codes"], axis=0)[:n]
    beam_codes = np.concatenate(buf["beam_codes"], axis=0)[:n]
    hist_len = np.concatenate(buf["hist_len"], axis=0)[:n]

    out_dir = args.out_dir or DEFAULT_OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_filtered" if args.filter_seen else ""
    npz = out_dir / f"exact_catalog_{args.category}_seed{args.seed}{suffix}.npz"
    np.savez_compressed(
        npz,
        exact_top=np.stack(buf["exact_top"]),                  # n x topk_store (catalog idx)
        exact_top_scores=np.stack(buf["exact_top_scores"]),    # n x topk_store
        target_exact_rank=np.asarray(buf["target_exact_rank"], dtype=np.int64),
        target_in_catalog=np.asarray(buf["target_in_catalog"], dtype=np.int64),
        target_codes=target_codes,                             # n x L (token space)
        beam_codes=beam_codes,                                 # n x beam_k x L (token space)
        hist_len=hist_len,                                     # n
        n_catalog=np.int64(n_catalog),
    )
    ray.shutdown()
    print(f"wrote {npz} (users={n}, n_catalog={n_catalog}, topk_store={args.topk_store}, beam_k={args.beam_k})", flush=True)
    return 0


# ----------------------------------------------------------------------------
# Analyze stage (pure numpy)
# ----------------------------------------------------------------------------

def analyze_stage(args) -> int:
    dump_dir = args.dump_root / args.category
    meta = json.loads((dump_dir / "meta.json").read_text())
    t2i = json.loads((dump_dir / "tuple_to_item.json").read_text())
    Lc = meta["L"]
    n_catalog = meta["n_catalog"]
    out_dir = args.out_dir or DEFAULT_OUT_ROOT
    suffix = "_filtered" if args.filter_seen else ""
    npz = np.load(out_dir / f"exact_catalog_{args.category}_seed{args.seed}{suffix}.npz")

    exact_top = npz["exact_top"]                  # n x S (catalog idx)
    target_exact_rank = npz["target_exact_rank"]  # n
    in_cat = npz["target_in_catalog"].astype(bool)
    target_codes = npz["target_codes"]            # n x L
    beam_codes = npz["beam_codes"]                # n x beam_k x L
    n = exact_top.shape[0]

    # Map beam codes -> catalog idx per user; map targets -> idx.
    beam_items = np.full(beam_codes.shape[:2], HALLUCINATION, dtype=np.int64)
    for u in range(n):
        for j in range(beam_codes.shape[1]):
            beam_items[u, j] = tuple_to_item_idx(beam_codes[u, j], t2i, Lc)
    target_idx = np.array([tuple_to_item_idx(target_codes[u], t2i, Lc) for u in range(n)], dtype=np.int64)

    def recall_at(k, exact):
        hit = 0
        for u in range(n):
            if target_idx[u] < 0:
                continue
            pool = exact_top[u, :k] if exact else beam_items[u, :k]
            if target_idx[u] in set(int(x) for x in pool if x >= 0):
                hit += 1
        return hit / n

    def coverage_at(k, exact):
        reached = set()
        for u in range(n):
            pool = exact_top[u, :k] if exact else beam_items[u, :k]
            reached.update(int(x) for x in pool if x >= 0)
        return len(reached) / n_catalog

    rows = []
    for k in (5, 10, 20):
        rows.append({
            "k": k,
            "recall_exact": round(recall_at(k, True), 5),
            "recall_beam": round(recall_at(k, False), 5),
            "coverage_exact": round(coverage_at(k, True), 5),
            "coverage_beam": round(coverage_at(k, False), 5),
        })

    # Beam-unreachable targets: of users whose target the model's beam (top beam_k)
    # never surfaces, what exact rank does the model give the target? High exact
    # rank (small number) => the model KNOWS it but the beam loses it => SEARCH error.
    beam_sets = [set(int(x) for x in beam_items[u] if x >= 0) for u in range(n)]
    miss = np.array([in_cat[u] and (target_idx[u] not in beam_sets[u]) for u in range(n)])
    miss_ranks = target_exact_rank[miss & (target_exact_rank > 0)]
    valid_ranks = target_exact_rank[in_cat & (target_exact_rank > 0)]

    def pct_within(ranks, r):
        return float((ranks <= r).mean()) if ranks.size else float("nan")

    summary = {
        "category": args.category, "seed": args.seed, "filter_seen": bool(args.filter_seen),
        "n_users": int(n), "n_targets_in_catalog": int(in_cat.sum()),
        "per_k": rows,
        "target_exact_rank_median_all": float(np.median(valid_ranks)) if valid_ranks.size else None,
        "beam_missed_targets": int(miss.sum()),
        "beam_missed_target_exact_rank_median": float(np.median(miss_ranks)) if miss_ranks.size else None,
        "beam_missed_pct_exact_rank_le_10": pct_within(miss_ranks, 10),
        "beam_missed_pct_exact_rank_le_20": pct_within(miss_ranks, 20),
        "beam_missed_pct_exact_rank_le_100": pct_within(miss_ranks, 100),
    }

    # Pre-registered verdict (Sports threshold; informational for Beauty).
    if rows:
        r10 = next(r for r in rows if r["k"] == 10)
        d_recall = r10["recall_exact"] - r10["recall_beam"]
        d_cov = (r10["coverage_exact"] - r10["coverage_beam"]) * 100.0
        summary["delta_recall@10_exact_minus_beam"] = round(d_recall, 5)
        summary["delta_coverage@10_points"] = round(d_cov, 3)
        summary["verdict"] = "SEARCH-bound" if (d_recall >= 0.01 and d_cov >= 5.0) else "MODEL-bound"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exact_catalog_{args.category}_seed{args.seed}{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


# ----------------------------------------------------------------------------
# selftest (pure numpy)
# ----------------------------------------------------------------------------

def _selftest() -> int:
    L = 4
    # 5 items, distinct codes.
    t2i = {"0,0,0,0": 0, "1,2,3,4": 1, "5,6,7,8": 2, "9,10,11,12": 3, "13,14,15,16": 4}
    table = build_catalog_token_table(t2i, L)
    # token[l] = raw + l*256 + 2
    assert table[1].tolist() == [1 + 2, 2 + 256 + 2, 3 + 512 + 2, 4 + 768 + 2], table[1].tolist()
    # round-trip de-offset
    for key, j in t2i.items():
        assert tuple_to_item_idx(table[j], t2i, L) == j
    # an out-of-band token is a hallucination
    bad = table[0].copy(); bad[0] = 1  # below n_special=2 -> raw -1
    assert tuple_to_item_idx(bad, t2i, L) == HALLUCINATION
    # ranking
    scores = np.array([0.0, 5.0, -1.0, 2.0, 3.0])  # item 1 best, then 4, 3, 0, 2
    top, rank = topk_and_target_rank(scores, target_idx=3, k=3)
    assert top.tolist() == [1, 4, 3], top.tolist()
    assert rank == 3, rank          # item 3 is the 3rd best
    _, rank_best = topk_and_target_rank(scores, target_idx=1, k=2)
    assert rank_best == 1, rank_best
    _, rank_none = topk_and_target_rank(scores, target_idx=-1, k=2)
    assert rank_none == -1
    print("OK: exact_catalog selftest passed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Exact full-catalog teacher-forced scoring for MARIUS")
    sub = p.add_subparsers(dest="stage", required=False)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--category", default="Sports_and_Outdoors")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    common.add_argument("--out-dir", type=Path, default=None)
    common.add_argument("--filter-seen", action="store_true",
                        help="mask the user's history items before ranking (mirrors "
                             "search() filter_preds), for an apples-to-apples exact-vs-beam "
                             "comparison; writes/reads a _filtered artifact")

    pr = sub.add_parser("run", parents=[common], help="GPU: score the catalog and dump")
    pr.add_argument("--category-slug", required=True)
    pr.add_argument("--quant-method", default=None, help="informational; support tables are read from dump-root")
    pr.add_argument("--output-root", type=Path, default=None)
    pr.add_argument("--results-dir", type=Path, default=None)
    pr.add_argument("--max-users", type=int, default=1000, help="cap users scored (None for all)")
    pr.add_argument("--chunk-size", type=int, default=1024, help="catalog candidates per depth_forward chunk")
    pr.add_argument("--eval-batch-size", type=int, default=32)
    pr.add_argument("--beam-k", type=int, default=20, help="beam width/depth for the comparison search()")
    pr.add_argument("--topk-store", type=int, default=100, help="exact top-K stored per user")

    pa = sub.add_parser("analyze", parents=[common], help="CPU: decomposition table from the npz")

    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if args.stage == "run":
        return run_stage(args)
    if args.stage == "analyze":
        return analyze_stage(args)
    p.error("specify a stage: run | analyze (or --selftest)")


if __name__ == "__main__":
    raise SystemExit(main())
