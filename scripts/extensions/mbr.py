#!/usr/bin/env python3
"""Minimum Bayes Risk (MBR) re-ranking of MARIUS's dumped beam candidates.

Numpy-only (no torch): runs locally on the committed Top-K dumps under
reports/extensions/topk/<category>/. For each test user the MARIUS dump holds
the beam's ranked candidate list as C semantic-ID code tuples (token space,
int64[U, C, L]). MBR re-scores candidate i by its expected utility against the
whole candidate set:

    score_i = sum_j w_j * U(i, j)

with the utility U(i, j) = (number of shared leading code levels) / L, i.e.
prefix-overlap depth in the RQ tree (identical tuple = 1.0). Because the
per-level token offset (token[l] = raw_code[l] + l*K_cb + n_special, see
src/data/marius.py) is a per-level bijection, prefix overlap computed directly
in token space equals prefix overlap on raw codes, so no de-offset is needed
for the utility itself.

Weights:
  uniform        w_j = 1/C. Runnable on the existing dumps (no scores stored).
  scores         w_j = softmax(scores_j / tau) from the npz "scores" key
                 (candidate log-probs, written by dump_topk.py --with-scores).
  auto (default) "scores" when the npz has them, else "uniform".

Modes:
  mbr            re-rank all C candidates by MBR score.
  mbr_topm       keep the first m candidates (by original beam order) as the
                 candidate pool and re-rank only within it; positions m..C-1
                 keep their original order. MBR scores are still computed
                 against ALL C candidates (full evidence set). m = C reproduces
                 plain "mbr".

Ties in the MBR score are broken by the ORIGINAL beam order (stable sort), so
the re-rank is anchored to the model's ranking wherever consensus is silent.

Item mapping for the metrics reuses the validated loaders in
scripts/extensions/compute_beyond_accuracy.py (load_support, load_model_recs);
the re-rank itself is a per-user permutation applied to those catalog-index
lists, so the validated token de-offset / hallucination handling is reused
unchanged. Metrics reuse scripts/extensions/beyond_accuracy.py.

Usage:
  python scripts/extensions/mbr.py --category Beauty --seeds "42 43 44"
  python scripts/extensions/mbr.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extensions import beyond_accuracy as ba  # noqa: E402
from scripts.extensions.compute_beyond_accuracy import load_model_recs, load_support  # noqa: E402

DEFAULT_K = 20  # depth of the validated dumps ({model}_seed{seed}_topk.npz)


# --------------------------------------------------------------------------- #
# Core MBR machinery (pure numpy)
# --------------------------------------------------------------------------- #
def prefix_overlap_utility(codes):
    """Pairwise prefix-overlap utility between a user's candidates.

    codes: int64[U, C, L] code tuples (token or raw space, must be consistent).
    Returns float64[U, C, C] with entry (u, i, j) = (shared leading levels)/L.
    """
    codes = np.asarray(codes)
    U_, C, L = codes.shape
    eq = codes[:, :, None, :] == codes[:, None, :, :]  # U x C x C x L
    # A level counts only if all shallower levels also match (prefix property).
    prefix = np.cumprod(eq, axis=-1)
    return prefix.sum(axis=-1).astype(np.float64) / float(L)


def candidate_weights(n_candidates, scores=None, tau=1.0):
    """Per-user candidate weights w[U, C]. Uniform unless log-prob scores given."""
    if scores is None:
        return None  # uniform; handled without materializing U x C
    s = np.asarray(scores, dtype=np.float64) / float(tau)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def mbr_scores(codes, scores=None, tau=1.0, user_chunk=8192):
    """MBR expected-utility score per candidate: float64[U, C].

    Computed in user-chunks so the O(U*C*C*L) prefix-overlap intermediate stays
    bounded: at depth C=100 the full-U cumprod intermediate is ~63 GB, which OOMs
    a 96 GB node. Chunking is numerically identical to the unchunked computation
    (each user's score depends only on that user's own candidates)."""
    codes = np.asarray(codes)
    U, C, _L = codes.shape
    scores = None if scores is None else np.asarray(scores)
    out = np.empty((U, C), dtype=np.float64)
    for a in range(0, U, user_chunk):
        b = min(a + user_chunk, U)
        util = prefix_overlap_utility(codes[a:b])  # chunk x C x C
        w = candidate_weights(C, scores=None if scores is None else scores[a:b], tau=tau)
        out[a:b] = util.mean(axis=2) if w is None else np.einsum("uij,uj->ui", util, w)
    return out


def mbr_order(codes, scores=None, tau=1.0, mode="mbr", topm=None):
    """Return the re-ranking permutation int64[U, C] (stable: ties keep beam order)."""
    s = mbr_scores(codes, scores=scores, tau=tau)
    C = s.shape[1]
    if mode == "mbr":
        return np.argsort(-s, axis=1, kind="stable")
    if mode == "mbr_topm":
        m = C if topm is None else int(topm)
        if not (1 <= m <= C):
            raise ValueError(f"topm must be in [1, {C}], got {m}")
        head = np.argsort(-s[:, :m], axis=1, kind="stable")
        tail = np.broadcast_to(np.arange(m, C), (s.shape[0], C - m))
        return np.concatenate([head, tail], axis=1).astype(np.int64)
    raise ValueError(f"unknown mode: {mode}")


def apply_order(recs, order):
    """Permute per-user recommendation lists (lists of catalog idx) by `order`."""
    return [[row[j] for j in perm] for row, perm in zip(recs, order)]


# --------------------------------------------------------------------------- #
# Metrics (NDCG is the one thing beyond_accuracy.py does not provide)
# --------------------------------------------------------------------------- #
def ndcg_at_k(recs, targets, k):
    """Leave-one-out NDCG@k: 1/log2(rank+2) if the held-out target is at 0-based
    `rank` within the Top-k, else 0; averaged over users with a known target.
    Sentinel slots (-1) occupy positions, consistent with ba.recall_at_k."""
    vals = []
    for row, tgt in zip(recs, targets):
        if tgt is None or tgt < 0:
            continue
        top = list(row[:k])
        vals.append(1.0 / np.log2(top.index(tgt) + 2) if tgt in top else 0.0)
    return float(np.mean(vals)) if vals else 0.0


METRICS = ["recall", "ndcg", "coverage", "gini", "arp", "aplt", "tail_recall",
           "novelty", "entropy_norm", "hallucination_rate"]


def evaluate(recs, targets, pop, n_catalog, k):
    m = ba.compute_all(recs, pop, n_catalog, item_emb=None, targets=targets, k=k)
    m["ndcg"] = ndcg_at_k(recs, targets, k)
    return {name: m[name] for name in METRICS}


# --------------------------------------------------------------------------- #
# Dump loading
# --------------------------------------------------------------------------- #
def load_marius_dump(dump_dir: Path, seed: int, meta: dict, t2i: dict, n_results: int = DEFAULT_K):
    """Return (topk_codes[U,C,L] token space, recs, targets, scores or None).

    For the validated 20-deep dumps this delegates the item mapping to
    compute_beyond_accuracy.load_model_recs. For deeper dumps written by
    dump_topk.py --n-results N (file suffix topk{N}.npz) it applies the
    identical de-offset + lookup, kept in sync with load_model_recs.
    """
    suffix = "" if n_results == DEFAULT_K else str(n_results)
    npz_path = dump_dir / f"marius_seed{seed}_topk{suffix}.npz"
    npz = np.load(npz_path)
    codes = npz["topk_codes"].astype(np.int64)
    if n_results == DEFAULT_K:
        recs, targets = load_model_recs(dump_dir, "marius", seed, meta, t2i)
    else:
        # Mirrors load_model_recs's marius branch exactly (same K_cb derivation).
        ns, L = meta["n_special"], meta["L"]
        K_cb = max(int(x) for key in t2i for x in key.split(",")) + 1
        level_off = [l * K_cb + ns for l in range(L)]

        def to_item(code):
            raw = [int(c) - level_off[l] for l, c in enumerate(code)]
            if any(r < 0 or r >= K_cb for r in raw):
                return ba.HALLUCINATION
            return t2i.get(",".join(str(r) for r in raw), ba.HALLUCINATION)

        recs = [[to_item(code) for code in row] for row in codes]
        targets = [to_item(code) for code in npz["target_codes"]]
    scores = npz["scores"].astype(np.float64) if "scores" in npz.files else None
    return codes, recs, targets, scores


# --------------------------------------------------------------------------- #
# Per-category run + report
# --------------------------------------------------------------------------- #
def run_category(dump_dir: Path, seeds, k, mode, topm, weights, tau, n_results=DEFAULT_K):
    meta, pop, _emb, t2i = load_support(dump_dir)
    rows = []
    weight_used = None
    for seed in seeds:
        codes, recs, targets, scores = load_marius_dump(dump_dir, seed, meta, t2i, n_results)
        if weights == "uniform":
            use_scores = None
        elif weights == "scores":
            if scores is None:
                raise SystemExit(f"--weights scores requested but no 'scores' key in seed {seed} dump")
            use_scores = scores
        else:  # auto
            use_scores = scores
        weight_used = "scores" if use_scores is not None else "uniform"
        order = mbr_order(codes, scores=use_scores, tau=tau, mode=mode, topm=topm)
        recs_mbr = apply_order(recs, order)
        for variant, r in (("original", recs), ("mbr", recs_mbr)):
            m = evaluate(r, targets, pop, meta["n_catalog"], k)
            m.update({"seed": seed, "variant": variant})
            rows.append(m)
    df = pd.DataFrame(rows)
    return df, meta, weight_used


def summarize(df):
    """Mean over seeds per variant + delta (mbr - original). Returns (means, delta)."""
    means = df.groupby("variant")[METRICS].mean()
    delta = means.loc["mbr"] - means.loc["original"]
    return means, delta


def fmt_table(means, delta):
    lines = ["| metric | original | mbr | delta |", "|---|---|---|---|"]
    for m in METRICS:
        o, b = means.loc["original", m], means.loc["mbr", m]
        lines.append(f"| {m} | {o:.4f} | {b:.4f} | {delta[m]:+.4f} |")
    return "\n".join(lines)


def fmt_per_seed(df):
    cols = ["seed", "variant"] + METRICS
    sub = df[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    lines = [header, sep]
    for _, r in sub.iterrows():
        vals = [str(int(r["seed"])), r["variant"]] + [f"{r[m]:.4f}" for m in METRICS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_path: Path, category, df, means, delta, k, mode, topm, weight_used, tau, n_results):
    tau_str = f"{tau}" if weight_used == "scores" else "n/a (uniform weights)"
    if weight_used == "scores":
        scope = """## Scope

This run uses SCORE-WEIGHTED MBR: w_j = softmax(log p_j / tau) over the
model's own candidate log-probabilities (the "scores" key of the dumps),
so confident candidates dominate the consensus. Lower tau concentrates the
weights toward the beam's argmax (tau -> 0 recovers the original order);
higher tau approaches the uniform-weight lower bound. Compare against the
uniform variant to see how much accuracy the confidence weighting recovers."""
    else:
        scope = """## Scope and expectation management

This run uses UNIFORM candidate weights (w_j = 1/C) because the dumps carry
no candidate scores. Uniform-weight MBR ignores the model's confidence
entirely: a candidate from a large cluster of mutually similar beam entries
is promoted even if every member of that cluster has low probability. It can
therefore HURT accuracy, and the numbers below should be read as a LOWER
BOUND on what score-weighted MBR (w_j = softmax(log p_j / tau), enabled
automatically once dumps carry a "scores" key) can do.

The secondary question the same run answers: does consensus re-ranking at
least move the DIVERSITY metrics (coverage, Gini, ARP, APLT) in the right
direction, i.e. is the MBR-consensus pick less popularity-peaked than the
beam's raw log-prob order?"""
    body = f"""# MBR re-ranking pilot: {category}

MBR (Minimum Bayes Risk) re-ranking of MARIUS's dumped beam candidates
(C = {n_results} code tuples per user), utility = prefix-overlap depth in the
RQ tree (shared leading code levels / 4). Candidate i is re-scored by
sum_j w_j * U(i, j) and the list is re-sorted (stable; ties keep the original
beam order). Metrics at k = {k}, seeds {sorted(df['seed'].unique().tolist())},
mode = {mode}{f" (m = {topm})" if mode == "mbr_topm" else ""},
weights = {weight_used}, tau = {tau_str}.

{scope}

## Mean over seeds (k = {k})

{fmt_table(means, delta)}

Delta = mbr - original. For recall/ndcg/coverage/aplt/tail_recall/novelty/
entropy_norm higher is better; for gini/arp/hallucination_rate lower is better.

## Per seed

{fmt_per_seed(df)}

Generated by scripts/extensions/mbr.py (numpy-only; reuses the validated
loaders in compute_beyond_accuracy.py and the metric suite in
beyond_accuracy.py).
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    # 1) Utility matrix by hand. L=4, offsets irrelevant (token-space invariance).
    codes = np.array([[
        [9, 9, 9, 9],     # A
        [9, 9, 9, 1],     # B: shares 3 levels with A
        [9, 9, 2, 9],     # C: shares 2 with A/B (level-3 match does not count: not a prefix)
        [5, 9, 9, 9],     # D: shares 0 (level-0 differs)
    ]])
    util = prefix_overlap_utility(codes)[0]
    expect = np.array([
        [1.00, 0.75, 0.50, 0.00],
        [0.75, 1.00, 0.50, 0.00],
        [0.50, 0.50, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ])
    assert np.allclose(util, expect), util
    # 2) Brute-force cross-check of vectorized utility on random codes.
    rng = np.random.default_rng(0)
    rc = rng.integers(0, 4, size=(5, 6, 4))
    fast = prefix_overlap_utility(rc)
    for u in range(5):
        for i in range(6):
            for j in range(6):
                d = 0
                for l in range(4):
                    if rc[u, i, l] == rc[u, j, l]:
                        d += 1
                    else:
                        break
                assert fast[u, i, j] == d / 4.0
    # 3) Uniform MBR demotes the consensus outlier D and keeps A first
    #    (A and B tie on consensus 0.5625 vs C 0.5; stable sort keeps A before B).
    order = mbr_order(codes, mode="mbr")[0]
    assert order.tolist() == [0, 1, 2, 3], order
    codes_out_first = codes[:, [3, 0, 1, 2], :]  # outlier D leads the beam
    order = mbr_order(codes_out_first, mode="mbr")[0]
    assert order.tolist() == [1, 2, 3, 0], order  # A, B, C promoted over D
    # 4) Score weights: peaked scores on the outlier keep it on top at small tau.
    scores = np.array([[50.0, 0.0, 0.0, 0.0]])
    order = mbr_order(codes_out_first, scores=scores, tau=1.0, mode="mbr")[0]
    assert order[0] == 0, order
    # 5) mbr_topm: m = C equals plain mbr; m=2 only permutes the first two slots.
    o_all = mbr_order(codes_out_first, mode="mbr")[0]
    o_topm = mbr_order(codes_out_first, mode="mbr_topm", topm=4)[0]
    assert o_all.tolist() == o_topm.tolist()
    o2 = mbr_order(codes_out_first, mode="mbr_topm", topm=2)[0]
    assert sorted(o2[:2].tolist()) == [0, 1] and o2[2:].tolist() == [2, 3], o2
    # 6) apply_order is a pure permutation.
    recs = [[10, 20, 30, 40]]
    assert apply_order(recs, np.array([[1, 2, 3, 0]])) == [[20, 30, 40, 10]]
    # 7) ndcg sanity: target at rank 0 -> 1.0; rank 1 -> 1/log2(3); miss -> 0; skip -1.
    assert ndcg_at_k([[7, 8], [9, 7], [1, 2], [3, 4]], [7, 7, 7, -1], k=2) == \
        float(np.mean([1.0, 1.0 / np.log2(3), 0.0]))
    print("OK: selftest passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MBR re-ranking of MARIUS dumped beams")
    parser.add_argument("--dump-root", type=Path, default=REPO / "reports" / "extensions" / "topk")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--seeds", default="42 43 44")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-results", type=int, default=DEFAULT_K,
                        help="dump depth C; != 20 reads {model}_seed{s}_topk{C}.npz")
    parser.add_argument("--mode", choices=["mbr", "mbr_topm"], default="mbr")
    parser.add_argument("--topm", type=int, default=None, help="m for mbr_topm")
    parser.add_argument("--weights", choices=["auto", "uniform", "scores"], default="auto")
    parser.add_argument("--tau", type=float, default=1.0, help="softmax temperature for score weights")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    dump_dir = args.dump_root / args.category
    seeds = [int(s) for s in args.seeds.split()]
    df, meta, weight_used = run_category(
        dump_dir, seeds, args.k, args.mode, args.topm, args.weights, args.tau, args.n_results)
    means, delta = summarize(df)

    print(f"{args.category}: mode={args.mode} weights={weight_used} k={args.k} C={args.n_results}")
    print(fmt_table(means, delta))

    out = args.out or (REPO / "reports" / "extensions" / "mbr" / f"mbr_{args.category}.md")
    write_report(out, meta["category"], df, means, delta, args.k, args.mode,
                 args.topm, weight_used, args.tau, args.n_results)
    csv = out.with_suffix(".csv")
    df.to_csv(csv, index=False)
    print(f"\nWrote {out}\nWrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
