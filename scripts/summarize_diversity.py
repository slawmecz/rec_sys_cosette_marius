#!/usr/bin/env python3
"""Aggregate per-run JSON files (written by eval_diversity.py) across seeds and
produce a comparison table + plot for MARIUS vs SASRec recommendation skew and
diversity (Gini index + Shannon entropy).
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_runs(input_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(input_dir.glob("*.json"))]


def to_rows(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        if run["mode"] == "generative":
            for level, (g, e) in enumerate(zip(run["gini_per_level"], run["entropy_per_level"])):
                rows.append(
                    {
                        "method": run["method"],
                        "seed": run["seed"],
                        "level": str(level),
                        "gini": g,
                        "entropy": e,
                        "n_total": run["k_per_level"],
                        "valid_HR10": run["valid_HR10"],
                    }
                )
        else:
            rows.append(
                {
                    "method": run["method"],
                    "seed": run["seed"],
                    "level": "item",
                    "gini": run["gini"],
                    "entropy": run["entropy"],
                    "n_total": run["n_items"],
                    "valid_HR10": run["valid_HR10"],
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "seed", "level", "gini", "entropy", "n_total", "valid_HR10"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["method"], r["level"], r["seed"])):
            writer.writerow(row)


def grouped_stats(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(method, level) -> {gini: (mean, std, n), entropy: (mean, std, n)} across seeds."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["level"]), []).append(row)
    return {
        key: {
            "gini": (
                float(np.mean([r["gini"] for r in vals])),
                float(np.std([r["gini"] for r in vals])),
                len(vals),
            ),
            "entropy": (
                float(np.mean([r["entropy"] for r in vals])),
                float(np.std([r["entropy"] for r in vals])),
                len(vals),
            ),
        }
        for key, vals in groups.items()
    }


def write_summary_txt(rows: list[dict], stats: dict, category: str, path: Path) -> None:
    seeds = sorted({row["seed"] for row in rows})
    methods = sorted({row["method"] for row in rows})

    lines = [
        f"{category} - Recommendation diversity: Gini index + Shannon entropy (test set, real checkpoints)",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Seeds: {seeds}",
        "",
        "Gini: higher = more concentrated/popularity-skewed recommendations.",
        "Entropy (nats): higher = more diverse/uniform recommendations.",
        "MARIUS metrics are reported per RVQ level (conditioned on preceding levels);",
        "SASRec metrics are a single flat value over recommended item ids.",
        "",
    ]

    header = f"{'method':<10} | {'level':<6} | {'gini mean +/- std':<22} | {'entropy mean +/- std (nats)':<28}"
    lines.append(header)
    lines.append("-" * len(header))
    for method in methods:
        levels = sorted({lvl for (m, lvl) in stats if m == method}, key=lambda x: (x != "item", x))
        for level in levels:
            s = stats[(method, level)]
            gm, gs, n = s["gini"]
            em, es, _ = s["entropy"]
            lines.append(
                f"{method:<10} | {level:<6} | {gm:.4f} +/- {gs:.4f}        | {em:.4f} +/- {es:.4f}  (n={n})"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_plot(stats: dict, category: str, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot", flush=True)
        return False

    marius_levels = sorted(
        (lvl for (m, lvl) in stats if m == "MARIUS"), key=int
    )
    fig, (ax_gini, ax_ent) = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, ylabel in [
        (ax_gini, "gini", "Gini index"),
        (ax_ent, "entropy", "Shannon entropy (nats)"),
    ]:
        if marius_levels:
            means = [stats[("MARIUS", lvl)][metric][0] for lvl in marius_levels]
            stds = [stats[("MARIUS", lvl)][metric][1] for lvl in marius_levels]
            ax.errorbar(
                [int(lvl) for lvl in marius_levels],
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                label="MARIUS (per RVQ level)",
            )

        if ("SASRec", "item") in stats:
            mean, std, _ = stats[("SASRec", "item")][metric]
            ax.axhline(mean, color="firebrick", linestyle="--", label=f"SASRec = {mean:.3f}")
            ax.fill_between(ax.get_xlim(), mean - std, mean + std, color="firebrick", alpha=0.15)

        ax.set_xlabel("RVQ level (MARIUS only)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{category}: {ylabel}")
        ax.set_xticks(range(len(marius_levels)))
        ax.legend()

    fig.suptitle(f"{category}: recommendation diversity across seeds")
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Summarize diversity results (Gini + entropy) across seeds")
    parser.add_argument("--category", default="Arts_Crafts_and_Sewing")
    parser.add_argument("--input-dir", type=Path, default=Path("reports/diversity"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diversity"))
    args = parser.parse_args()

    input_dir = args.input_dir / args.category
    runs = load_runs(input_dir)
    if not runs:
        raise SystemExit(f"No per-run JSON files found under {input_dir}")

    rows = to_rows(runs)
    stats = grouped_stats(rows)

    csv_path = args.output_dir / f"{args.category}_diversity_table.csv"
    txt_path = args.output_dir / f"{args.category}_diversity_summary.txt"
    png_path = args.output_dir / f"{args.category}_diversity_comparison.png"

    write_csv(rows, csv_path)
    write_summary_txt(rows, stats, args.category, txt_path)
    plotted = write_plot(stats, args.category, png_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {png_path}" if plotted else "Skipped plot (matplotlib unavailable)")
    print()
    print(txt_path.read_text())


if __name__ == "__main__":
    main()
