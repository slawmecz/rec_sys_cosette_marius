#!/usr/bin/env python3
"""Compute the beyond-accuracy table from dumped Top-K lists (non-interactive).

Reads the artifacts written by scripts/extensions/dump_topk.py under
reports/extensions/topk/<category>/ (meta.json, popularity.npy, embeddings.npy,
tuple_to_item.json, {model}_seed{seed}_topk.npz) and writes a tidy CSV of the
beyond-accuracy suite (coverage, Gini, entropy, ARP, APLT, novelty, ILD,
hallucination rate, tail-recall, recall) per (category, model, k), averaged over
seeds. This is the headless counterpart to notebooks/beyond_accuracy.ipynb so it
can run on Snellius without Jupyter.

  python scripts/extensions/compute_beyond_accuracy.py \
      --category Beauty --seeds "42 43 44" --ks "10 20"

Run the built-in check with --selftest (fabricates tiny dumps in a temp dir).
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


def load_support(dump_dir: Path):
    meta = json.loads((dump_dir / "meta.json").read_text())
    pop = np.load(dump_dir / "popularity.npy")
    emb_path = dump_dir / "embeddings.npy"
    emb = np.load(emb_path) if emb_path.exists() else None
    t2i = json.loads((dump_dir / "tuple_to_item.json").read_text())
    return meta, pop, emb, t2i


def load_model_recs(dump_dir: Path, model: str, seed: int, meta: dict, t2i: dict):
    npz = np.load(dump_dir / f"{model}_seed{seed}_topk.npz")
    ns = meta["n_special"]
    if model == "sasrec":
        recs = [[int(v) - ns if int(v) >= ns else ba.HALLUCINATION for v in row] for row in npz["topk_items"]]
        targets = [int(t) - ns if int(t) >= ns else ba.HALLUCINATION for t in npz["target_item"]]
    else:
        recs = [[t2i.get(",".join(str(int(c)) for c in code), ba.HALLUCINATION) for code in row]
                for row in npz["topk_codes"]]
        targets = [t2i.get(",".join(str(int(c)) for c in code), ba.HALLUCINATION) for code in npz["target_codes"]]
    return recs, targets


def compute_table(dump_dir: Path, models, seeds, ks) -> pd.DataFrame:
    meta, pop, emb, t2i = load_support(dump_dir)
    rows = []
    for model in models:
        for seed in seeds:
            npz_path = dump_dir / f"{model}_seed{seed}_topk.npz"
            if not npz_path.exists():
                print(f"  (skip {model} seed {seed}: no dump)", flush=True)
                continue
            recs, targets = load_model_recs(dump_dir, model, seed, meta, t2i)
            for k in ks:
                m = ba.compute_all(recs, pop, meta["n_catalog"], item_emb=emb, targets=targets, k=k)
                m.update({"category": meta["category"], "model": model, "seed": seed})
                rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    meta_cols = ["category", "model", "seed", "k"]
    metric_cols = [c for c in df.columns if c not in meta_cols]
    agg = df.groupby(["category", "model", "k"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns]
    return agg.reset_index()


def _selftest() -> int:
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "Beauty"
        d.mkdir(parents=True)
        n_special, n_catalog, L, K, U = 2, 50, 4, 20, 200
        (d / "meta.json").write_text(json.dumps(
            {"category": "Beauty", "n_special": n_special, "n_catalog": n_catalog, "L": L, "K": K}))
        pop = (rng.integers(1, 500, n_catalog)).astype(np.int64)
        np.save(d / "popularity.npy", pop)
        np.save(d / "embeddings.npy", rng.normal(size=(n_catalog, 8)).astype(np.float32))
        # tuple_to_item: give each catalog item a distinct code tuple
        codes = rng.integers(0, 256, size=(n_catalog, L))
        t2i = {",".join(str(int(c)) for c in codes[i]): i for i in range(n_catalog)}
        (d / "tuple_to_item.json").write_text(json.dumps(t2i))
        for seed in (42, 43):
            np.savez_compressed(
                d / f"sasrec_seed{seed}_topk.npz",
                topk_items=(rng.integers(n_special, n_special + n_catalog, size=(U, K))).astype(np.int64),
                target_item=(rng.integers(n_special, n_special + n_catalog, size=U)).astype(np.int64),
                hist_len=(rng.integers(1, 50, U)).astype(np.int64))
            # marius: real items + ~10% out-of-vocab tuples (hallucinations)
            pick = rng.integers(0, n_catalog, size=(U, K))
            topk_codes = codes[pick]
            halluc = rng.random((U, K)) < 0.1
            topk_codes[halluc] = rng.integers(0, 256, size=(halluc.sum(), L))
            np.savez_compressed(
                d / f"marius_seed{seed}_topk.npz",
                topk_codes=topk_codes.astype(np.int64),
                target_codes=codes[rng.integers(0, n_catalog, size=U)].astype(np.int64),
                hist_len=(rng.integers(1, 50, U)).astype(np.int64))
        out = compute_table(d, ["sasrec", "marius"], [42, 43], [10, 20])
        assert not out.empty and len(out) == 4, out
        assert "coverage_mean" in out.columns and "hallucination_rate_mean" in out.columns
        mar = out[(out.model == "marius") & (out.k == 10)]["hallucination_rate_mean"].iloc[0]
        assert 0.03 < mar < 0.2, f"hallucination rate off: {mar}"
        print(out.to_string(index=False))
        print("\nOK: selftest passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Beyond-accuracy table from Top-K dumps")
    parser.add_argument("--dump-root", type=Path, default=REPO / "reports" / "extensions" / "topk")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--models", default="sasrec marius")
    parser.add_argument("--seeds", default="42 43 44")
    parser.add_argument("--ks", default="10 20")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    dump_dir = args.dump_root / args.category
    models = args.models.split()
    seeds = [int(s) for s in args.seeds.split()]
    ks = [int(k) for k in args.ks.split()]

    df = compute_table(dump_dir, models, seeds, ks)
    if df.empty:
        print(f"No dumps found under {dump_dir}. Run jobs/22_dump_topk_*.sbatch first.", flush=True)
        return 1
    out = args.out or (REPO / "reports" / "extensions" / f"beyond_accuracy_{args.category}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False), flush=True)
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
