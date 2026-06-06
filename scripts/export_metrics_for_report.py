#!/usr/bin/env python3
"""
Copy Table 5 scores and validation curves into reports/ for the notebook.

  source jobs/wandb.env
  conda activate recsys
  python scripts/export_metrics_for_report.py --copy-results
  python scripts/export_metrics_for_report.py --source wandb

Without WandB API: --source checkpoints (sparse validation points).

Writes reports/results/ (jsonl) and reports/metrics/ (CSV).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reporting.constants import CATEGORIES, EXPECTED_SEEDS

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get(
        "OUTPUT_ROOT",
        os.environ.get("SCRATCH", "/home/scur1266/scratch") + "/cosette_marius/outputs",
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        "DATA_ROOT",
        os.environ.get("SCRATCH", "/home/scur1266/scratch") + "/cosette_marius/data",
    )
)

REPORTS_DIR = REPO_ROOT / "reports"
RESULTS_OUT = REPORTS_DIR / "results"
METRICS_OUT = REPORTS_DIR / "metrics"

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "cosette-and-marius")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")

# Lightning / COSETTE metric keys to export (validation + optional train).
TRAIN_METRICS = ["train/{category}/loss"]
VAL_METRICS = [
    "valid/{category}/HR@5",
    "valid/{category}/NDCG@5",
    "valid/{category}/HR@10",
    "valid/{category}/NDCG@10",
]
COSETTE_METRICS = [
    "train_loss",
    "lr",
    "collision_rate",
    "mean_collisions",
    "max_collisions",
]

RUN_PATTERNS = {
    "sasrec": re.compile(r"^SASRec_(?P<category>.+?)_seed(?P<seed>\d+)_"),
    "marius": re.compile(r"^MARIUS_small_(?P<category>.+?)_seed(?P<seed>\d+)_"),
    "cosette": re.compile(r"^COSETTE_"),
}


def _ensure_dirs() -> None:
    RESULTS_OUT.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.mkdir(parents=True, exist_ok=True)


def copy_results(output_root: Path) -> list[Path]:
    """Copy final score jsonl + latest summary tables into reports/results/."""
    src = output_root / "results"
    if not src.exists():
        print(f"[copy-results] missing {src}", file=sys.stderr)
        return []

    copied: list[Path] = []
    patterns = [
        "*_5seed_full_scores.jsonl",
        "*_5seed_summary_full_latest.txt",
        "*_marius_5seed_summary_full_latest.txt",
    ]
    for pattern in patterns:
        for path in sorted(src.glob(pattern)):
            dest = RESULTS_OUT / path.name
            shutil.copy2(path, dest)
            copied.append(dest)
            print(f"[copy-results] {path.name} -> {dest}")

    return copied


def _write_csv_rows(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _subsample_history(history: list[dict[str, Any]], key: str, every: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, row in enumerate(history):
        if key not in row or row[key] is None:
            continue
        step = row.get("_step", row.get("step", row.get("epoch", i)))
        if step is None:
            continue
        if i == 0 or i == len(history) - 1 or int(step) % every == 0:
            out.append(row)
    return out


def _parse_run_name(name: str) -> dict[str, str] | None:
    for method, pattern in RUN_PATTERNS.items():
        match = pattern.search(name)
        if not match:
            continue
        info: dict[str, str] = {"method": method, "run_name": name}
        if method in {"sasrec", "marius"}:
            info["category"] = match.group("category")
            info["seed"] = match.group("seed")
        return info
    return None


def export_wandb(
    *,
    entity: str,
    project: str,
    subsample_train: int = 500,
) -> list[Path]:
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required for --source wandb.\n"
            "Use the recsys conda env, e.g.:\n"
            "  source ~/miniconda3/etc/profile.d/conda.sh && conda activate recsys\n"
            "  source jobs/wandb.env\n"
            "  python scripts/export_metrics_for_report.py --source wandb"
        ) from exc

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path)
    written: list[Path] = []

    for run in runs:
        meta = _parse_run_name(run.name)
        if meta is None:
            continue

        history = run.history(samples=10000, pandas=False)
        if not history:
            continue

        method = meta["method"]
        rows: list[dict[str, Any]] = []

        if method == "cosette":
            for row in history:
                epoch = row.get("epoch")
                if epoch is None:
                    continue
                for metric in COSETTE_METRICS:
                    if metric not in row or row[metric] is None:
                        continue
                    rows.append(
                        {
                            "method": "cosette",
                            "category": run.name,
                            "seed": "",
                            "step": int(epoch),
                            "metric": metric,
                            "value": float(row[metric]),
                            "run_name": run.name,
                            "run_id": run.id,
                        }
                    )
            out_name = f"cosette_{run.name}.csv"
        else:
            category = meta["category"]
            seed = meta["seed"]
            val_keys = [k.format(category=category) for k in VAL_METRICS]
            train_key = TRAIN_METRICS[0].format(category=category)

            val_rows = [r for r in history if any(k in r and r[k] is not None for k in val_keys)]
            for row in val_rows:
                step = row.get("_step", row.get("step", row.get("trainer/global_step")))
                if step is None:
                    continue
                for key in val_keys:
                    if key not in row or row[key] is None:
                        continue
                    short = key.split("/", 2)[-1]
                    rows.append(
                        {
                            "method": method,
                            "category": category,
                            "seed": seed,
                            "step": int(step),
                            "metric": short,
                            "value": float(row[key]),
                            "run_name": run.name,
                            "run_id": run.id,
                        }
                    )

            train_rows = _subsample_history(
                [r for r in history if train_key in r and r[train_key] is not None],
                train_key,
                subsample_train,
            )
            for row in train_rows:
                step = row.get("_step", row.get("step", row.get("trainer/global_step", 0)))
                rows.append(
                    {
                        "method": method,
                        "category": category,
                        "seed": seed,
                        "step": int(step),
                        "metric": "train_loss",
                        "value": float(row[train_key]),
                        "run_name": run.name,
                        "run_id": run.id,
                    }
                )
            out_name = f"{method}_{category}_seed{seed}.csv"

        out_path = METRICS_OUT / out_name
        n = _write_csv_rows(out_path, rows)
        if n:
            written.append(out_path)
            print(f"[wandb] {run.name}: {n} rows -> {out_path.name}")

    return written


def export_checkpoints(output_root: Path) -> list[Path]:
    """Sparse validation metrics from Ray checkpoint_manager_snapshot.json."""
    models_dir = output_root / "models"
    if not models_dir.exists():
        print(f"[checkpoints] missing {models_dir}", file=sys.stderr)
        return []

    written: list[Path] = []
    for snap_path in sorted(models_dir.glob("*/checkpoint_manager_snapshot.json")):
        run_dir = snap_path.parent.name
        meta = _parse_run_name(run_dir)
        if meta is None or meta["method"] not in {"sasrec", "marius"}:
            continue

        with snap_path.open() as handle:
            data = json.load(handle)

        rows: list[dict[str, Any]] = []
        for ckpt in data.get("checkpoint_results", []):
            metrics = ckpt.get("metrics", {})
            step = int(metrics.get("step", 0))
            for key, value in metrics.items():
                if not key.startswith("valid/"):
                    continue
                short = key.split("/", 2)[-1]
                rows.append(
                    {
                        "method": meta["method"],
                        "category": meta["category"],
                        "seed": meta["seed"],
                        "step": step,
                        "metric": short,
                        "value": float(value),
                        "run_name": run_dir,
                        "run_id": "",
                    }
                )

        out_path = METRICS_OUT / f"{meta['method']}_{meta['category']}_seed{meta['seed']}_checkpoints.csv"
        n = _write_csv_rows(out_path, rows)
        if n:
            written.append(out_path)
            print(f"[checkpoints] {run_dir}: {n} rows -> {out_path.name}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy-results",
        action="store_true",
        help="Copy score jsonl and latest summary tables to reports/results/",
    )
    parser.add_argument(
        "--source",
        choices=("wandb", "checkpoints"),
        help="Export learning-curve CSVs from WandB or local checkpoint snapshots",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Training output root (scores, models)",
    )
    parser.add_argument(
        "--entity",
        default=WANDB_ENTITY,
        help="WandB entity (or set WANDB_ENTITY)",
    )
    parser.add_argument(
        "--project",
        default=WANDB_PROJECT,
        help="WandB project name",
    )
    parser.add_argument(
        "--subsample-train",
        type=int,
        default=500,
        help="Keep train loss every N steps when exporting from WandB",
    )
    args = parser.parse_args()

    if not args.copy_results and args.source is None:
        parser.error("Pass --copy-results and/or --source wandb|checkpoints")

    _ensure_dirs()

    if args.copy_results:
        copy_results(args.output_root)

    if args.source == "wandb":
        if not args.entity:
            print(
                "Warning: WANDB_ENTITY not set; trying project path only.",
                file=sys.stderr,
            )
        export_wandb(
            entity=args.entity,
            project=args.project,
            subsample_train=args.subsample_train,
        )
    elif args.source == "checkpoints":
        export_checkpoints(args.output_root)

    print(f"\nReports directory: {REPORTS_DIR}")
    print(f"  results: {RESULTS_OUT}")
    print(f"  metrics: {METRICS_OUT}")
    print(f"Expected seeds: {EXPECTED_SEEDS}")
    print(f"Categories: {', '.join(CATEGORIES)}")


if __name__ == "__main__":
    main()
