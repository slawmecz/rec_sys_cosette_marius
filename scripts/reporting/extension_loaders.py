"""Load extension artifacts for extended_report.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .constants import CATEGORIES, METRIC_COLUMNS, PAPER_MARIUS_COSETTE, PAPER_SASREC_PP
from .loaders import load_scores_jsonl, summarize_method


def extensions_dir(repo_root: Path) -> Path:
    return repo_root / "reports" / "extensions"


def load_inference_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("seconds", "peak_mem_gb"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_wall_clock(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_model_vocab(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def val_test_gap_table(results_dir: Path, metrics_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for category, meta in CATEGORIES.items():
        slug = meta["slug"]
        for method_key, paper_dict, fname_prefix in (
            ("SASRec++", PAPER_SASREC_PP, "sasrec"),
            ("MARIUS + COSETTE", PAPER_MARIUS_COSETTE, "marius"),
        ):
            test_records = load_scores_jsonl(results_dir / f"{fname_prefix}_{slug}_5seed_full_scores.jsonl")
            test_summary = summarize_method(test_records)
            if test_summary is None:
                continue
            test_r10 = test_summary["R@10_mean"]

            val_frames = []
            for seed in test_summary["seeds"]:
                csv_path = metrics_dir / f"{fname_prefix}_{category}_seed{seed}.csv"
                if not csv_path.exists():
                    continue
                part = pd.read_csv(csv_path)
                part = part[(part["metric"] == "HR@10")]
                if part.empty:
                    continue
                val_frames.append(part.groupby("step")["value"].mean().max())
            val_r10 = 100.0 * sum(val_frames) / len(val_frames) if val_frames else float("nan")

            rows.append(
                {
                    "category": category,
                    "method": method_key,
                    "val_R@10_mean": val_r10,
                    "test_R@10_mean": test_r10,
                    "gap_pp": val_r10 - test_r10,
                    "paper_test_R@10": paper_dict[category]["R@10"][0],
                }
            )
    return pd.DataFrame(rows)


def per_seed_spread(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for category, meta in CATEGORIES.items():
        slug = meta["slug"]
        for method_key, fname_prefix in (
            ("SASRec++", "sasrec"),
            ("MARIUS + COSETTE", "marius"),
        ):
            for record in load_scores_jsonl(results_dir / f"{fname_prefix}_{slug}_5seed_full_scores.jsonl"):
                rows.append(
                    {
                        "category": category,
                        "method": method_key,
                        "seed": record["seed"],
                        **{m: float(record[m]) for m in METRIC_COLUMNS if m in record},
                    }
                )
    return pd.DataFrame(rows)


def cosette_collision_df(metrics_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(metrics_dir.glob("cosette_*.csv")):
        df = pd.read_csv(path)
        part = df[df["metric"] == "collision_rate"].copy()
        if part.empty:
            continue
        part["source"] = path.name
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
