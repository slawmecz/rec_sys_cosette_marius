#!/usr/bin/env python3
"""Compute COSETTE codebook usage from semantic-ID parquets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(
    "/home/scur1266/scratch/cosette_marius/data/data/embeddings/sentence-t5-xl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "extensions"

DATASETS = {
    "Beauty": {
        "quant": "COSETTE_128d_256x4_f958",
        "n_items": 12_101,
    },
    "Sports_and_Outdoors": {
        "quant": "COSETTE_128d_256x4_8ed1",
        "n_items": 18_357,
    },
}
CODEBOOK_SIZE = 256
LEVEL_COLS = ["L0", "L1", "L2", "L3"]


def _distribution_stats(values: np.ndarray, *, codebook_size: int) -> dict[str, float]:
    counts = np.bincount(values.astype(np.int64), minlength=codebook_size)
    counts = counts[:codebook_size]
    total = int(counts.sum())
    used = int((counts > 0).sum())
    probs = counts[counts > 0] / total if total else np.array([])
    entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0
    perplexity = float(math.exp(entropy)) if probs.size else 0.0
    top1_load = float(counts.max() / total) if total else 0.0
    return {
        "codes_used": used,
        "codebook_size": codebook_size,
        "usage_ratio": used / codebook_size,
        "entropy": entropy,
        "perplexity": perplexity,
        "top1_load": top1_load,
        "n_items": total,
    }


def _sid_stats(df: pd.DataFrame) -> dict[str, float]:
    tuples = df[LEVEL_COLS].astype(str).agg("-".join, axis=1)
    counts = tuples.value_counts()
    n_items = len(df)
    unique_sids = int(counts.shape[0])
    collision_rate = 1.0 - unique_sids / n_items if n_items else 0.0
    probs = counts.to_numpy() / n_items
    sid_entropy = float(-(probs * np.log(probs)).sum()) if n_items else 0.0
    return {
        "unique_full_sids": unique_sids,
        "collision_rate": collision_rate,
        "max_items_per_sid": float(counts.max()),
        "mean_items_per_sid": float(counts.mean()),
        "sid_entropy": sid_entropy,
    }


def analyze_parquet(path: Path, *, category: str, variant: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    missing = [c for c in LEVEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    level_rows: list[dict] = []
    for level in LEVEL_COLS:
        stats = _distribution_stats(df[level].to_numpy(), codebook_size=CODEBOOK_SIZE)
        level_rows.append(
            {
                "category": category,
                "variant": variant,
                "level": level,
                **stats,
            }
        )

    sid = _sid_stats(df)
    summary = {
        "category": category,
        "variant": variant,
        "parquet": str(path),
        "n_items": len(df),
        **sid,
        "mean_level_usage_ratio": float(np.mean([r["usage_ratio"] for r in level_rows])),
        "mean_level_perplexity": float(np.mean([r["perplexity"] for r in level_rows])),
        "mean_level_top1_load": float(np.mean([r["top1_load"] for r in level_rows])),
    }
    return pd.DataFrame(level_rows), summary


def _short_category(name: str) -> str:
    return "Sports" if name == "Sports_and_Outdoors" else name


def save_plots(summary_df: pd.DataFrame, per_level: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary = summary_df.copy()
    summary["dataset"] = summary["category"].map(_short_category)
    per_level = per_level.copy()
    per_level["dataset"] = per_level["category"].map(_short_category)

    # 1) Full-SID collision rate before vs after -col
    fig, ax = plt.subplots(figsize=(7, 4))
    datasets = summary["dataset"].unique()
    x = np.arange(len(datasets))
    width = 0.35
    before = [
        summary.loc[
            (summary["dataset"] == d) & (summary["variant"] == "before_col"),
            "collision_rate",
        ].iloc[0]
        * 100
        for d in datasets
    ]
    after = [
        summary.loc[
            (summary["dataset"] == d) & (summary["variant"] == "after_col"),
            "collision_rate",
        ].iloc[0]
        * 100
        for d in datasets
    ]
    ax.bar(x - width / 2, before, width, label="before -col", color="#d95f02")
    ax.bar(x + width / 2, after, width, label="after -col", color="#1b9e77")
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Full-SID collision rate (%)")
    ax.set_title("COSETTE full semantic-ID collisions")
    ax.legend()
    fig.tight_layout()
    path = fig_dir / "codebook_collision_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2) Per-level top-1 code load (dominant code share)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, datasets):
        sub = per_level[per_level["dataset"] == dataset]
        levels = LEVEL_COLS
        x = np.arange(len(levels))
        for i, variant in enumerate(["before_col", "after_col"]):
            vals = sub.loc[sub["variant"] == variant, "top1_load"].to_numpy() * 100
            offset = -width / 2 if variant == "before_col" else width / 2
            label = "before -col" if variant == "before_col" else "after -col"
            color = "#d95f02" if variant == "before_col" else "#1b9e77"
            ax.bar(x + offset, vals, width, label=label, color=color)
        ax.set_xticks(x, levels)
        ax.set_title(dataset)
        ax.set_xlabel("Quantization level")
        ax.set_ylabel("Top-1 code load (%)")
    axes[0].legend()
    fig.suptitle("Most frequent code share per level", y=1.02)
    fig.tight_layout()
    path = fig_dir / "codebook_top1_load_by_level.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 3) Per-level perplexity (code distribution spread)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, datasets):
        sub = per_level[per_level["dataset"] == dataset]
        levels = LEVEL_COLS
        x = np.arange(len(levels))
        for variant, color, label in [
            ("before_col", "#d95f02", "before -col"),
            ("after_col", "#1b9e77", "after -col"),
        ]:
            vals = sub.loc[sub["variant"] == variant, "perplexity"].to_numpy()
            offset = -width / 2 if variant == "before_col" else width / 2
            ax.bar(x + offset, vals, width, label=label, color=color)
        ax.set_xticks(x, levels)
        ax.set_title(dataset)
        ax.set_xlabel("Quantization level")
        ax.set_ylabel("Perplexity")
    axes[0].legend()
    fig.suptitle("Per-level code distribution perplexity", y=1.02)
    fig.tight_layout()
    path = fig_dir / "codebook_perplexity_by_level.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    # 4) SID entropy + unique SID count (summary panel)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ds_x = np.arange(len(datasets))
    for ax, metric, title, ylabel, scale in [
        (axes[0], "sid_entropy", "SID sequence entropy", "Entropy (nats)", 1.0),
        (axes[1], "unique_full_sids", "Unique full SIDs", "Count", 1.0),
    ]:
        before_vals = [
            summary.loc[
                (summary["dataset"] == d) & (summary["variant"] == "before_col"),
                metric,
            ].iloc[0]
            * scale
            for d in datasets
        ]
        after_vals = [
            summary.loc[
                (summary["dataset"] == d) & (summary["variant"] == "after_col"),
                metric,
            ].iloc[0]
            * scale
            for d in datasets
        ]
        ax.bar(ds_x - width / 2, before_vals, width, label="before -col", color="#d95f02")
        ax.bar(ds_x + width / 2, after_vals, width, label="after -col", color="#1b9e77")
        ax.set_xticks(ds_x, datasets)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle("Full semantic-ID diversity", y=1.02)
    fig.tight_layout()
    path = fig_dir / "codebook_sid_diversity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    return written


def save_token_distribution(
    parquet_jobs: list[tuple[str, str, Path]], out_dir: Path
) -> list[Path]:
    import matplotlib.pyplot as plt

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    count_rows: list[dict] = []

    for category, variant, path in parquet_jobs:
        df = pd.read_parquet(path)
        dataset = _short_category(category)
        for level in LEVEL_COLS:
            counts = np.bincount(df[level].to_numpy(dtype=np.int64), minlength=CODEBOOK_SIZE)[
                :CODEBOOK_SIZE
            ]
            total = int(counts.sum())
            for code, cnt in enumerate(counts):
                if cnt == 0:
                    continue
                count_rows.append(
                    {
                        "category": category,
                        "dataset": dataset,
                        "variant": variant,
                        "level": level,
                        "code": code,
                        "count": int(cnt),
                        "fraction": cnt / total if total else 0.0,
                    }
                )

    # Combined figure: rows = datasets, cols = L0–L3 (after -col)
    after_jobs = [(c, v, p) for c, v, p in parquet_jobs if v == "after_col"]
    if after_jobs:
        n_rows = len(after_jobs)
        fig, axes = plt.subplots(n_rows, len(LEVEL_COLS), figsize=(14, 3.2 * n_rows), sharey=True)
        if n_rows == 1:
            axes = np.array([axes])
        for row, (category, _variant, path) in enumerate(after_jobs):
            df = pd.read_parquet(path)
            dataset = _short_category(category)
            for col, level in enumerate(LEVEL_COLS):
                ax = axes[row, col]
                counts = np.bincount(
                    df[level].to_numpy(dtype=np.int64), minlength=CODEBOOK_SIZE
                )[:CODEBOOK_SIZE]
                ax.bar(
                    np.arange(CODEBOOK_SIZE),
                    counts,
                    width=1.0,
                    color="#1b9e77",
                    alpha=0.85,
                )
                ax.set_title(f"{dataset} — {level}")
                ax.set_xlabel("Code index")
                if col == 0:
                    ax.set_ylabel("Item count")
        fig.suptitle("COSETTE token distribution per code (after -col)", y=1.01)
        fig.tight_layout()
        path_out = fig_dir / "token_distribution.png"
        fig.savefig(path_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path_out)

    counts_path = out_dir / "token_distribution_counts.csv"
    pd.DataFrame(count_rows).to_csv(counts_path, index=False)
    written.append(counts_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root with sentence-t5-xl/<category>/*.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip parquet scan; plot from existing CSVs in --out-dir",
    )
    args = parser.parse_args()

    if args.plots_only:
        summary_path = args.out_dir / "codebook_usage_summary.csv"
        per_level_path = args.out_dir / "codebook_usage_per_level.csv"
        summary_df = pd.read_csv(summary_path)
        per_level = pd.read_csv(per_level_path)
    else:
        per_level_frames: list[pd.DataFrame] = []
        summaries: list[dict] = []

    parquet_jobs: list[tuple[str, str, Path]] = []

    if not args.plots_only:
        for category, meta in DATASETS.items():
            quant = meta["quant"]
            category_dir = args.data_root / category
            for suffix, variant in [("", "before_col"), ("-col", "after_col")]:
                path = category_dir / f"{quant}{suffix}.parquet"
                if not path.exists():
                    raise FileNotFoundError(path)
                parquet_jobs.append((category, variant, path))
                level_df, summary = analyze_parquet(
                    path, category=category, variant=variant
                )
                per_level_frames.append(level_df)
                summaries.append(summary)

        per_level = pd.concat(per_level_frames, ignore_index=True)
        summary_df = pd.DataFrame(summaries)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_level_path = args.out_dir / "codebook_usage_per_level.csv"
        summary_path = args.out_dir / "codebook_usage_summary.csv"
        json_path = args.out_dir / "codebook_usage.json"

        per_level.to_csv(per_level_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        json_path.write_text(
            json.dumps(
                {
                    "summary": summaries,
                    "per_level": per_level.to_dict(orient="records"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Wrote {per_level_path}")
        print(f"Wrote {summary_path}")
        print(f"Wrote {json_path}")
        print()
        print(summary_df.to_string(index=False))

    plot_paths = save_plots(summary_df, per_level, args.out_dir)
    for path in plot_paths:
        print(f"Wrote {path}")

    if parquet_jobs:
        token_paths = save_token_distribution(parquet_jobs, args.out_dir)
    else:
        parquet_jobs = []
        for category, meta in DATASETS.items():
            quant = meta["quant"]
            category_dir = args.data_root / category
            for suffix, variant in [("", "before_col"), ("-col", "after_col")]:
                path = category_dir / f"{quant}{suffix}.parquet"
                if path.exists():
                    parquet_jobs.append((category, variant, path))
        token_paths = save_token_distribution(parquet_jobs, args.out_dir)
    for path in token_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
