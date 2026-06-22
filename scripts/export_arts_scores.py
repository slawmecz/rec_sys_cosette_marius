#!/usr/bin/env python
"""Export Arts_Crafts_and_Sewing (Amazon 2023) per-seed test metrics to the
standard score jsonl the replication notebook reads.

Mirrors scripts/aggregate_arts.py (reads the run directories recorded in
runs/seed_<seed>.{marius,sasrec} and the metrics.pkl written by src/test.py),
but emits one jsonl record per seed in the same schema as the Beauty/Sports
exporter, into:

  reports/results/marius_arts_5seed_full_scores.jsonl
  reports/results/sasrec_arts_5seed_full_scores.jsonl

Re-run after each new seed finishes to extend the notebook tables and plots.
Safety: a method's jsonl is only rewritten when at least one seed is found, so
running this where DATA_ROOT is not mounted (e.g. a laptop) never clobbers the
committed cache.

Usage:
  source jobs/arts2023/env_arts.sh && python scripts/export_arts_scores.py
  python scripts/export_arts_scores.py --data-root /scratch-shared/scur1250/cosette_arts
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

from scripts.aggregate_arts import (
    CATEGORY,
    METRICS,
    find_metrics_pkl,
    load_test_metrics,
)

METRIC_COLUMNS = [disp for disp, _ in METRICS]


def collect_records(model_root: Path, runs_dir: Path, suffix: str) -> list[dict]:
    """One score record per seed with a recorded metrics.pkl."""
    records: list[dict] = []
    for f in sorted(glob.glob(str(runs_dir / f"seed_*.{suffix}"))):
        seed = int(os.path.basename(f).split("_")[1].split(".")[0])
        run_dir = Path(f).read_text().strip()
        if not run_dir:
            continue
        pkl = find_metrics_pkl(model_root, run_dir)
        if not pkl:
            print(f"  [warn] no metrics.pkl yet for seed {seed} ({run_dir})")
            continue
        metrics = load_test_metrics(pkl)
        record = {
            "status": "ok",
            "mode": "full",
            "category": CATEGORY,
            "seed": seed,
            "run_directory": run_dir,
            "timestamp": datetime.fromtimestamp(os.path.getmtime(pkl)).isoformat(
                timespec="seconds"
            ),
        }
        record.update({m: metrics[m] for m in METRIC_COLUMNS if m in metrics})
        records.append(record)
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    if not records:
        print(f"  [skip] no seeds found; left {out_path.name} untouched")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    seeds = sorted(r["seed"] for r in records)
    print(f"  [ok] wrote {len(records)} seed(s) {seeds} -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", "/scratch-shared/scur1250/cosette_arts"),
    )
    ap.add_argument(
        "--runs-dir",
        default=os.path.join(os.environ.get("PROJECT_ROOT", "."), "runs"),
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.join(os.environ.get("PROJECT_ROOT", "."), "reports", "results"),
    )
    args = ap.parse_args()

    model_root = Path(args.data_root) / "models"
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)

    for method in ("marius", "sasrec"):
        print(f"{method}:")
        records = collect_records(model_root, runs_dir, method)
        write_jsonl(records, out_dir / f"{method}_arts_5seed_full_scores.jsonl")


if __name__ == "__main__":
    main()
