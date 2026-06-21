#!/usr/bin/env python
"""Aggregate Arts_Crafts_and_Sewing 5-seed test metrics vs the paper (Table 3).

Reads the per-seed run directories recorded in runs/seed_<seed>.{marius,sasrec},
loads the metrics.pkl written by src/test.py for each run, and prints mean +/- std
for R@5, NDCG@5, R@10, NDCG@10 next to the paper targets.

R@k == HR@k here (one held-out target per user). Metrics are logged by the
LightningModule as 'test/<category>/<HR@k|NDCG@k>'.

Usage:
  PROJECT_ROOT=... DATA_ROOT=... python scripts/aggregate_arts.py
  python scripts/aggregate_arts.py --data-root /scratch-shared/scur1250/cosette_arts
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import statistics
from pathlib import Path

CATEGORY = "Arts_Crafts_and_Sewing"
# (display name, logged metric suffix)
METRICS = [("R@5", "HR@5"), ("NDCG@5", "NDCG@5"), ("R@10", "HR@10"), ("NDCG@10", "NDCG@10")]

# Paper Table 3 (Arts, Amazon 2023); reference single-seed got MARIUS R@10 5.04.
PAPER = {
    "MARIUS": {"R@5": 3.49, "NDCG@5": 2.37, "R@10": 5.30, "NDCG@10": 2.95},
    "SASRec": {"R@5": 3.51, "NDCG@5": 2.42, "R@10": 5.09, "NDCG@10": 2.93},
}


def find_metrics_pkl(model_root: Path, run_dir: str):
    hits = sorted(glob.glob(str(model_root / run_dir / "**" / "metrics.pkl"), recursive=True))
    return hits[0] if hits else None


def load_test_metrics(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    tm = data["test_metrics"][0]  # trainer.test returns a list with one dict
    out = {}
    for disp, suffix in METRICS:
        exact = f"test/{CATEGORY}/{suffix}"
        key = exact if exact in tm else next(
            (k for k in tm if k.startswith("test/") and k.endswith("/" + suffix)), None
        )
        if key is not None:
            v = float(tm[key])
            out[disp] = v * 100.0 if v <= 1.0 else v  # normalise to percent
    return out


def collect(model_root: Path, runs_dir: Path, suffix: str) -> dict:
    rows = {}
    for f in sorted(glob.glob(str(runs_dir / f"seed_*.{suffix}"))):
        seed = int(os.path.basename(f).split("_")[1].split(".")[0])
        run_dir = Path(f).read_text().strip()
        if not run_dir:
            continue
        pkl = find_metrics_pkl(model_root, run_dir)
        if not pkl:
            print(f"  [warn] no metrics.pkl yet for seed {seed} ({run_dir})")
            continue
        rows[seed] = load_test_metrics(pkl)
    return rows


def print_table(model: str, rows: dict) -> None:
    paper = PAPER[model]
    print(f"\n=== {model} (Arts) — {len(rows)} seed(s): {sorted(rows)} ===")
    print(f"{'metric':>9} | {'ours mean ± std':>17} | {'paper':>6} | {'delta_pp':>8}")
    print("-" * 50)
    for disp, _ in METRICS:
        vals = [r[disp] for r in rows.values() if disp in r]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{disp:>9} | {mean:8.2f} ± {std:5.2f}  | {paper[disp]:6.2f} | {mean - paper[disp]:+8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "/scratch-shared/scur1250/cosette_arts"))
    ap.add_argument("--runs-dir", default=os.path.join(os.environ.get("PROJECT_ROOT", "."), "runs"))
    args = ap.parse_args()

    model_root = Path(args.data_root) / "models"
    runs_dir = Path(args.runs_dir)

    marius = collect(model_root, runs_dir, "marius")
    sasrec = collect(model_root, runs_dir, "sasrec")

    if marius:
        print_table("MARIUS", marius)
    if sasrec:
        print_table("SASRec", sasrec)
    if not marius and not sasrec:
        print("No results found. Expected runs/seed_*.{marius,sasrec} and metrics.pkl "
              f"under {model_root}/<run_dir>/.")


if __name__ == "__main__":
    main()
