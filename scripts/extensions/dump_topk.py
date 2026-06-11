#!/usr/bin/env python3
"""Dump ranked Top-K recommendation lists for the beyond-accuracy extension.

Runs the EXISTING evaluation (the authors' search() methods, unchanged) over the
test split for the trained SASRec++ and MARIUS+COSETTE checkpoints, and records,
per test user, the ranked Top-K it produces. It also writes the small support
tables the offline notebook needs (train item popularity, item content
embeddings, the semantic-ID -> item lookup, and metadata).

It does NOT modify any model, loss, or eval code: it attaches a read-only
Lightning callback to trainer.test and re-invokes net.search(batch, K). This is
the only GPU step the extension needs; it is one cheap eval pass per (model,
seed), the same order of cost as scripts/benchmark_extensions.py.

Outputs (under --out-dir, default reports/extensions/topk/<category>/):
  meta.json                         {category, n_special, n_catalog, L, K}
  popularity.npy                    int64[n_catalog]  train interaction counts
  embeddings.npy                    float32[n_catalog, d]  item content embeddings
  tuple_to_item.json                {"c0,c1,..": catalog_idx}  (MARIUS code lookup)
  {model}_seed{seed}_topk.npz       raw Top-K per test user (see below)

npz contents (catalog_idx = vocab_id - n_special):
  SASRec (dense):  topk_items int64[U, K]  (raw vocab ids),  target_item int64[U],  hist_len int64[U]
  MARIUS (gen):    topk_codes int64[U, K, L],  target_codes int64[U, L],  hist_len int64[U]

Optional flags (defaults reproduce the original behavior exactly):
  --n-results N    dump depth (default 20). N != 20 writes {model}_seed{seed}_topk{N}.npz
                   so the validated 20-deep dumps are never clobbered. meta.json's "K"
                   records the depth used when support tables are (re)written; it is
                   informational only and never touched under --no-support.
  --with-scores    adds a "scores" float32[U, N] key: MARIUS candidate joint log-probs
                   (scoring.score_marius_tuples, teacher-forced inside the same bf16
                   autocast as search) or SASRec logits (scoring.score_sasrec_topk).
                   These feed the score-weighted MBR re-ranking in mbr.py.

Usage (Snellius, after locating checkpoints on scratch):
  python scripts/extensions/dump_topk.py \
      --category Beauty --category-slug beauty --seed 42 \
      --emb-method sentence-t5-xl --quant-method COSETTE_128d_256x4_f958-col \
      --output-root "$SCRATCH/cosette_marius/outputs"
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
from pathlib import Path

import fsspec
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as L
import ray
import torch
from omegaconf import OmegaConf

from scripts.benchmark_extensions import get_best_checkpoint, run_directory_for_seed
from scripts.extensions.scoring import score_marius_tuples, score_sasrec_topk
from src.models import SpecialTokens
from src.utils.tools import patch_fsspec

PAD = SpecialTokens.PAD.value
N_SPECIAL = len(SpecialTokens)
K = 20  # default dump depth; override with --n-results


class TopKCollector(L.Callback):
    """Read-only: re-run search() on each test batch and stash the ranked Top-K."""

    def __init__(self, mode, limit_batches=None, n_results=K, with_scores=False):
        self.mode = mode
        self.limit_batches = limit_batches
        self.n_results = n_results
        self.with_scores = with_scores
        self.topk = []
        self.target = []
        self.hist_len = []
        self.scores = []

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.limit_batches is not None and batch_idx >= self.limit_batches:
            return
        # The model trains/evals under bf16-mixed autocast; the LitModule's own
        # test_step runs search() inside that autocast scope. Callback hooks fire
        # OUTSIDE it, and running the generative beam search in fp32 collapses the
        # depth decoder to PAD (degenerate all-zero codes). Re-enter autocast so
        # the dumped Top-K matches the metrics the evaluator reports.
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() \
            else contextlib.nullcontext()
        with autocast:
            gen = pl_module.net.search(batch, n_results=self.n_results)  # dense: B x N ; gen: B x N x L
            if self.with_scores:
                # Score the dumped candidates inside the SAME autocast scope so
                # the scores match the arithmetic that produced the ranking.
                if self.mode == "dense":
                    sc = score_sasrec_topk(pl_module.net, batch, gen)
                else:
                    sc = score_marius_tuples(pl_module.net, batch["input"], gen)
                self.scores.append(sc.detach().float().cpu().numpy())
        target = batch["target"]
        self.topk.append(gen.detach().cpu().numpy())

        if self.mode == "dense":
            tgt = target if target.dim() == 1 else target[:, -1]
            self.target.append(tgt.detach().cpu().numpy())
            query = batch["query"]
            self.hist_len.append((query != PAD).sum(dim=1).detach().cpu().numpy())
        else:  # generative
            tgt = target[:, -1, :] if target.dim() == 3 else target
            self.target.append(tgt.detach().cpu().numpy())
            inp = batch["input"]
            valid = (inp != PAD).any(dim=2)
            self.hist_len.append(valid.sum(dim=1).detach().cpu().numpy())

    def stacked(self):
        return (
            np.concatenate(self.topk, axis=0),
            np.concatenate(self.target, axis=0),
            np.concatenate(self.hist_len, axis=0),
            np.concatenate(self.scores, axis=0) if self.scores else None,
        )


def run_topk(cfg, ckpt_path, *, mode, limit_batches=None, n_results=K,
             with_scores=False, eval_batch_size=None):
    patch_fsspec()
    # datamodule.setup("test") loads both the valid and test splits (see
    # src/data/datamodule.py), so both must be instantiated even though the
    # TopKCollector only runs on test batches.
    cfg.data.ray_datasets.which = ["valid", "test"]
    # The test dataloader batches at datamodule.valid_batch_size; a wide beam
    # (--n-results 100) blows the beam-expansion tensor up ~n_results-fold, so a
    # batch sized for the 20-wide beam OOMs at 100. Shrinking the eval batch is
    # a pure memory/throughput trade with no effect on the dumped Top-K. Left
    # unset (default), the batch size is exactly the checkpoint's, so the
    # validated 20-deep dumps are reproduced bit-for-bit.
    if eval_batch_size is not None:
        cfg.data.datamodule.valid_batch_size = int(eval_batch_size)
    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    model.full_hydra_config = cfg

    collector = TopKCollector(mode=mode, limit_batches=limit_batches,
                              n_results=n_results, with_scores=with_scores)
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[collector],
    )
    protocol = str(cfg.paths.protocol)
    ckpt_uri = ckpt_path if "://" in ckpt_path else f"{protocol}://{ckpt_path}"
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_uri)
    return collector.stacked()


def write_support_tables(cfg, *, category, emb_method, quant_method, out_dir, n_results=K):
    """Train popularity, content embeddings, semantic-ID lookup, and metadata."""
    patch_fsspec()
    fs = fsspec.filesystem(cfg.paths.protocol)
    paths = cfg.paths

    # product_id -> catalog index (vocab id minus the special-token offset).
    items_path = paths.unique_items_tplt.format(category=category)
    with fs.open(items_path, "rb") as f:
        items = pickle.load(f)
    product_to_catalog = {p: i for i, p in enumerate(items)}
    n_catalog = len(items)

    # Train popularity: count product occurrences across train timelines.
    train_path = paths.timelines_tplt.format(category=category, split="train")
    tdf = pd.read_parquet(train_path, filesystem=fs)
    popularity = np.zeros(n_catalog, dtype=np.int64)
    for timeline in tdf["timeline"].values:
        for product in timeline:
            j = product_to_catalog.get(product)
            if j is not None:
                popularity[j] += 1

    # Content embeddings, ordered by catalog index.
    emb_path = paths.embeddings_tplt.format(emb_method=emb_method, category=category)
    edf = pd.read_parquet(emb_path, filesystem=fs)
    d = len(edf["embedding"].iloc[0])
    embeddings = np.zeros((n_catalog, d), dtype=np.float32)
    for product, emb in zip(edf["product_id"].values, edf["embedding"].values):
        j = product_to_catalog.get(product)
        if j is not None:
            embeddings[j] = np.asarray(emb, dtype=np.float32)

    # Semantic-ID tuple -> catalog index (from the post-collision -col table).
    sid_path = paths.semantic_ids_tplt.format(
        emb_method=emb_method, category=category, quant_method=quant_method
    )
    sdf = pd.read_parquet(sid_path, filesystem=fs)
    code_cols = sorted([c for c in sdf.columns if c.startswith("L")])
    tuple_to_item = {}
    for _, row in sdf.iterrows():
        j = product_to_catalog.get(row["product_id"])
        if j is None:
            continue
        key = ",".join(str(int(row[c])) for c in code_cols)
        tuple_to_item[key] = int(j)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "popularity.npy", popularity)
    np.save(out_dir / "embeddings.npy", embeddings)
    (out_dir / "tuple_to_item.json").write_text(json.dumps(tuple_to_item))
    (out_dir / "meta.json").write_text(json.dumps({
        "category": category, "n_special": N_SPECIAL, "n_catalog": n_catalog,
        "L": len(code_cols), "K": n_results,
    }, indent=2))
    print(f"Support tables written to {out_dir} (n_catalog={n_catalog}, d={d}, L={len(code_cols)})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump Top-K lists + support tables for beyond-accuracy analysis")
    parser.add_argument("--category", required=True)
    parser.add_argument("--category-slug", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["sasrec", "marius", "both"], default="both",
                        help="dump one method (use when sasrec/marius live under different output roots)")
    parser.add_argument("--no-support", action="store_true",
                        help="skip writing the shared support tables (popularity/embeddings/tuple lookup)")
    parser.add_argument("--emb-method", default="sentence-t5-xl")
    parser.add_argument("--quant-method", default=None, help="the -col COSETTE id, e.g. COSETTE_128d_256x4_<id>-col; required unless --no-support")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-results", type=int, default=K,
                        help=f"dump depth N (default {K}); N != {K} writes "
                             "{model}_seed{seed}_topk{N}.npz so the validated "
                             f"{K}-deep dumps are never clobbered")
    parser.add_argument("--with-scores", action="store_true",
                        help="also store per-candidate model scores: MARIUS joint "
                             "log-probs (scoring.score_marius_tuples), SASRec logits "
                             "(scoring.score_sasrec_topk), as float32[U, N] 'scores'")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="override the test dataloader batch size (datamodule "
                             "valid_batch_size). Needed for wide beams: --n-results "
                             "100 OOMs at the checkpoint's batch 256 on a 40GB A100; "
                             "32-64 fits. Unset reproduces the validated dumps exactly.")
    parser.add_argument("--smoke", action="store_true", help="only a few batches, to validate shapes cheaply")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_root = args.output_root or Path(os.environ.get("OUTPUT_ROOT", repo_root / "outputs"))
    results_dir = args.results_dir or (
        output_root / "results" if (output_root / "results").exists() else repo_root / "reports" / "results"
    )
    out_dir = args.out_dir or (repo_root / "reports" / "extensions" / "topk" / args.category)
    limit = 2 if args.smoke else None

    methods = [("sasrec", "dense"), ("marius", "generative")]
    if args.method != "both":
        methods = [m for m in methods if m[0] == args.method]
    write_support = not args.no_support
    if write_support and not args.quant_method:
        parser.error("--quant-method is required unless --no-support is set")

    ray.init(ignore_reinit_error=True)
    fsspec.filesystem("file")

    support_done = False
    for method, mode in methods:
        run_directory = run_directory_for_seed(results_dir, method=method, slug=args.category_slug, seed=args.seed)
        cfg, ckpt_path = get_best_checkpoint(output_root / "models", run_directory)

        if write_support and not support_done:
            write_support_tables(
                cfg, category=args.category, emb_method=args.emb_method,
                quant_method=args.quant_method, out_dir=out_dir, n_results=args.n_results,
            )
            support_done = True

        topk, target, hist_len, scores = run_topk(
            cfg, ckpt_path, mode=mode, limit_batches=limit,
            n_results=args.n_results, with_scores=args.with_scores,
            eval_batch_size=args.eval_batch_size)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.n_results == K else str(args.n_results)
        npz = out_dir / f"{method}_seed{args.seed}_topk{suffix}.npz"
        arrays = {}
        if mode == "dense":
            arrays.update(topk_items=topk.astype(np.int64), target_item=target.astype(np.int64))
        else:
            arrays.update(topk_codes=topk.astype(np.int64), target_codes=target.astype(np.int64))
        arrays["hist_len"] = hist_len.astype(np.int64)
        if scores is not None:
            arrays["scores"] = scores.astype(np.float32)
        np.savez_compressed(npz, **arrays)
        print(f"{method} seed {args.seed}: wrote {npz} (users={topk.shape[0]}, topk shape={topk.shape}, "
              f"scores={'yes' if scores is not None else 'no'})", flush=True)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
