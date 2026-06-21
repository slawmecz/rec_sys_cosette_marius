#!/usr/bin/env python3
"""Directional-structure gate for order-aware COSETTE (Tier D, zero-GPU).

The exact-scoring oracle showed the catalog collapse is MODEL-bound: the model
under-ranks tail items, the beam is near-optimal, so the lever is upstream in the
tokenization/representation (see .context/extension-plan-2026-06-16.md). COSETTE
builds its codes from SYMMETRIC co-occurrence (items that appear together in a
timeline, order-agnostic). This gate tests, BEFORE committing to a tokenizer
retrain, whether a DIRECTIONAL (next-item / successor) signal carries next-item-
predictive structure that COSETTE's symmetric codes discard.

The decoder picks the second code digit conditioned on the first (the L1 family,
256 of them). So the principled comparison granularity is G = 256 groups. We
compare, at matched G, how much of the next-item's entropy each item-grouping
explains (normalized mutual information I(group(current); next_item) / H(next)):

  cosette_l1   the item's actual COSETTE first digit (256 groups) -- the real baseline
  direction    256-way clustering of a directed PPMI+SVD embedding of next-item
               transitions (items with similar SUCCESSORS cluster together)
  symmetric    same pipeline on the SYMMETRIZED transitions (T + T.T) -- isolates
               that DIRECTION, not just "any learned clustering", is the lever
  random       random 256-way grouping -- the chance floor

Metric A is over all transitions; Metric B restricts to transitions whose NEXT
item is in MARIUS's consistently-unreached set (recomputed from the committed
depth-20 dumps), i.e. the buried items a directional code would need to help.

GREENLIGHT (pre-registered): pursue directional COSETTE iff, on Metric A,
  frac(direction) - frac(cosette_l1) >= 0.05  AND  frac(direction) >= 1.2 * frac(symmetric).
Otherwise shelve the retrain and report a controlled null. Either outcome is a
publishable result (the lever is / is not directional structure).

numpy/scipy/pandas only (no torch, no GPU). Reads the train timelines parquet and
the items pickle from the dataset (paths passed in; the job script finds them),
plus the committed support tables + MARIUS dumps under the dump-dir.

  python scripts/extensions/direction_gate.py \
      --category Sports_and_Outdoors \
      --timelines /path/Sports_and_Outdoors.train.parquet \
      --items /path/Sports_and_Outdoors_items.pkl
  python scripts/extensions/direction_gate.py --selftest
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extensions import beyond_accuracy as ba  # noqa: E402
from scripts.extensions.compute_beyond_accuracy import load_model_recs, load_support  # noqa: E402

GREENLIGHT_ABS_MARGIN = 0.05   # frac(direction) - frac(cosette_l1)
GREENLIGHT_SYM_RATIO = 1.2     # frac(direction) / frac(symmetric)


# --------------------------------------------------------------------------- #
# Transition / co-occurrence structure (pure numpy/scipy)
# --------------------------------------------------------------------------- #
def transitions_from_timelines(timelines, product_to_catalog):
    """Consecutive (current, next) catalog-index pairs across all train timelines."""
    cur, nxt = [], []
    for tl in timelines:
        idx = [product_to_catalog[p] for p in tl if p in product_to_catalog]
        for a, b in zip(idx[:-1], idx[1:]):
            cur.append(a)
            nxt.append(b)
    return np.asarray(cur, dtype=np.int64), np.asarray(nxt, dtype=np.int64)


def ppmi_svd_embedding(cur, nxt, n_catalog, dim, seed):
    """Directed PPMI(item -> next item) reduced to `dim` dims via truncated SVD.

    Rows index the source item; the embedding clusters items with similar
    successor distributions. PPMI is nonzero only where a transition was observed,
    so the matrix stays as sparse as the transition counts.
    """
    T = coo_matrix((np.ones(len(cur)), (cur, nxt)), shape=(n_catalog, n_catalog)).tocsr()
    T.sum_duplicates()
    N = T.data.sum()
    row = np.asarray(T.sum(axis=1)).ravel()  # source marginals
    col = np.asarray(T.sum(axis=0)).ravel()  # target marginals
    coo = T.tocoo()
    # pmi = log( c * N / (row_i * col_j) ); ppmi = max(pmi, 0)
    with np.errstate(divide="ignore"):
        pmi = np.log((coo.data * N) / (row[coo.row] * col[coo.col]))
    ppmi = np.maximum(pmi, 0.0)
    P = coo_matrix((ppmi, (coo.row, coo.col)), shape=T.shape).tocsr()
    P.eliminate_zeros()
    k = min(dim, min(P.shape) - 1)
    # svds needs a nonzero operator; guard the degenerate all-zero case.
    if P.nnz == 0 or k < 1:
        return np.zeros((n_catalog, max(1, dim)), dtype=np.float64), T
    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(min(P.shape))
    U, S, _ = svds(P, k=k, v0=v0)
    return U * np.sqrt(S), T


def kmeans_grouping(emb, G, seed):
    """Group items into G clusters (scipy kmeans2, ++ init). Returns int[n_catalog]."""
    from scipy.cluster.vq import kmeans2
    n = emb.shape[0]
    G = min(G, n)
    # whiten lightly for stable k-means; guard zero-variance dims.
    std = emb.std(axis=0)
    std[std == 0] = 1.0
    _centroids, labels = kmeans2(emb / std, G, minit="++", seed=seed, missing="warn")
    return labels.astype(np.int64)


# --------------------------------------------------------------------------- #
# Mutual information of a grouping with the next item
# --------------------------------------------------------------------------- #
def entropy_of(values, n_support):
    counts = np.bincount(values, minlength=n_support).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def mi_fraction(grp, cur, nxt, n_catalog, h_next):
    """I(group(current); next_item) and its fraction of H(next_item)."""
    g_cur = grp[cur]
    G = int(grp.max()) + 1
    J = coo_matrix((np.ones(len(cur)), (g_cur, nxt)), shape=(G, n_catalog)).tocoo()
    J.sum_duplicates()
    J = J.tocoo()
    N = J.data.sum()
    p_g = np.asarray(coo_matrix((J.data, (J.row, J.col)), shape=(G, n_catalog)).sum(axis=1)).ravel()
    p_j = np.asarray(coo_matrix((J.data, (J.row, J.col)), shape=(G, n_catalog)).sum(axis=0)).ravel()
    # MI = sum p(g,j) log( p(g,j) / (p(g) p(j)) ), over observed cells.
    pgj = J.data / N
    mi = float((pgj * np.log((J.data * N) / (p_g[J.row] * p_j[J.col]))).sum())
    return mi, (mi / h_next if h_next > 0 else float("nan"))


# --------------------------------------------------------------------------- #
# MARIUS consistently-unreached set (from the committed depth-20 dumps)
# --------------------------------------------------------------------------- #
def marius_unreached(dump_dir, meta, t2i, n_catalog, k=10):
    seeds = [42, 43, 44]
    reached = np.zeros(n_catalog, dtype=bool)
    used = []
    for s in seeds:
        if not (dump_dir / f"marius_seed{s}_topk.npz").exists():
            continue
        used.append(s)
        recs, _ = load_model_recs(dump_dir, "marius", s, meta, t2i)
        for row in recs:
            for it in row[:k]:
                if it is not ba.HALLUCINATION and it >= 0:
                    reached[it] = True
    unreached = np.where(~reached)[0]
    return set(int(x) for x in unreached), used


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_gate(category, timelines_path, items_path, dump_dir, dim, G, seed):
    with open(items_path, "rb") as f:
        items = pickle.load(f)
    product_to_catalog = {p: i for i, p in enumerate(items)}
    n_catalog = len(items)

    meta, _pop, _emb, t2i = load_support(dump_dir)
    assert meta["n_catalog"] == n_catalog, (meta["n_catalog"], n_catalog)
    cosette_l1 = np.full(n_catalog, -1, dtype=np.int64)
    for key, j in t2i.items():
        cosette_l1[int(j)] = int(key.split(",")[0])  # first code digit

    tdf = pd.read_parquet(timelines_path, columns=["timeline"])
    cur, nxt = transitions_from_timelines(tdf["timeline"].values, product_to_catalog)

    emb_dir, _T = ppmi_svd_embedding(cur, nxt, n_catalog, dim, seed)
    emb_sym, _ = ppmi_svd_embedding(np.concatenate([cur, nxt]), np.concatenate([nxt, cur]),
                                    n_catalog, dim, seed)  # T + T.T
    grp = {
        "cosette_l1": cosette_l1,
        "direction": kmeans_grouping(emb_dir, G, seed),
        "symmetric": kmeans_grouping(emb_sym, G, seed),
        "random": np.random.default_rng(seed).integers(0, G, size=n_catalog),
    }

    def metric(cur_, nxt_):
        h = entropy_of(nxt_, n_catalog)
        out = {}
        for name, g in grp.items():
            mi, frac = mi_fraction(g, cur_, nxt_, n_catalog, h)
            out[name] = {"mi": round(mi, 5), "frac_of_H_next": round(frac, 5),
                         "n_groups": int(g.max()) + 1}
        out["_H_next"] = round(h, 5)
        out["_n_transitions"] = int(len(cur_))
        return out

    metric_a = metric(cur, nxt)

    unreached, used_seeds = marius_unreached(dump_dir, meta, t2i, n_catalog)
    tail = np.array([n in unreached for n in nxt], dtype=bool)
    metric_b = metric(cur[tail], nxt[tail]) if tail.sum() > 50 else {"_skipped": "too few tail transitions"}

    fa_dir = metric_a["direction"]["frac_of_H_next"]
    fa_cos = metric_a["cosette_l1"]["frac_of_H_next"]
    fa_sym = metric_a["symmetric"]["frac_of_H_next"]
    greenlight = (fa_dir - fa_cos >= GREENLIGHT_ABS_MARGIN) and (fa_dir >= GREENLIGHT_SYM_RATIO * fa_sym)

    return {
        "category": category, "n_catalog": n_catalog, "n_transitions": int(len(cur)),
        "svd_dim": dim, "n_groups_G": G, "seed": seed,
        "unreached_seeds_used": used_seeds, "n_unreached_next_transitions": int(tail.sum()),
        "metric_a_all_transitions": metric_a,
        "metric_b_unreached_next": metric_b,
        "delta_dir_minus_cosette": round(fa_dir - fa_cos, 5),
        "ratio_dir_over_symmetric": round(fa_dir / fa_sym, 4) if fa_sym > 0 else None,
        "verdict": "GREENLIGHT directional COSETTE" if greenlight else "SHELVE (controlled null)",
    }


def _selftest() -> int:
    # MI sanity: a grouping that perfectly determines next -> frac ~ 1; random -> ~0.
    n_catalog = 40
    rng = np.random.default_rng(0)
    cur = rng.integers(0, n_catalog, size=4000)
    # next item determined by current item's "class" (item % 4) -> 4 deterministic targets
    nxt = (cur % 4) * 10 + 0  # maps to one of 4 items {0,10,20,30}
    h = entropy_of(nxt, n_catalog)
    perfect = (np.arange(n_catalog) % 4)               # groups == the deterministic class
    _, frac_perfect = mi_fraction(perfect, cur, nxt, n_catalog, h)
    _, frac_random = mi_fraction(rng.integers(0, 4, n_catalog), cur, nxt, n_catalog, h)
    assert frac_perfect > 0.98, frac_perfect
    assert frac_random < 0.3, frac_random
    # entropy_of: uniform over 4 items -> log 4
    assert abs(entropy_of((np.arange(4000) % 4), 4) - np.log(4)) < 1e-6
    # ppmi/svd/kmeans pipeline runs and yields G groups on small directed data.
    emb, _ = ppmi_svd_embedding(cur, nxt, n_catalog, dim=8, seed=0)
    g = kmeans_grouping(emb, G=4, seed=0)
    assert g.shape == (n_catalog,) and g.max() <= 3
    print("OK: direction_gate selftest passed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Directional-structure gate for order-aware COSETTE")
    p.add_argument("--category", default="Sports_and_Outdoors")
    p.add_argument("--timelines", type=Path, help="train timelines parquet ({Category}.train.parquet)")
    p.add_argument("--items", type=Path, help="items pickle ({Category}_items.pkl)")
    p.add_argument("--dump-dir", type=Path, default=None,
                   help="reports/extensions/topk/<category> (support tables + marius dumps)")
    p.add_argument("--dim", type=int, default=32, help="SVD embedding dimension")
    p.add_argument("--groups", type=int, default=256, help="G: comparison granularity (256 = L1 codebook)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=REPO / "reports" / "extensions" / "direction_gate")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()
    if not (args.timelines and args.items):
        p.error("--timelines and --items are required (the job script finds them under DATA_ROOT)")

    dump_dir = args.dump_dir or (REPO / "reports" / "extensions" / "topk" / args.category)
    res = run_gate(args.category, args.timelines, args.items, dump_dir, args.dim, args.groups, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"direction_gate_{args.category}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print(f"\nWrote {args.out_dir}/direction_gate_{args.category}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
