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
        ild = run.get("ild")
        if run["mode"] == "generative":
            support = run.get("support_per_level") or [{}] * len(run["gini_per_level"])
            for level, (g, e, sup) in enumerate(
                zip(run["gini_per_level"], run["entropy_per_level"], support)
            ):
                rows.append(
                    {
                        "method": run["method"],
                        "seed": run["seed"],
                        "level": str(level),
                        "gini": g,
                        "entropy": e,
                        # ILD is a single list-level scalar, not per RVQ level; attach to level 0 row only
                        "ild": ild if level == 0 else None,
                        "n_total": run["k_per_level"],
                        # Support diagnostics for interpreting the conditional Gini/entropy.
                        "n_groups": sup.get("n_groups"),
                        "mean_group_size": sup.get("mean_group_size"),
                        "mean_distinct_codes": sup.get("mean_distinct_codes"),
                        "total_distinct_codes": sup.get("total_distinct_codes"),
                        "valid_HR10": run["valid_HR10"],
                    }
                )
            # Item-level row (full semantic-ID tuple as one item) for a like-for-like
            # comparison with SASRec's item-level Gini/entropy.
            if "gini" in run:
                rows.append(
                    {
                        "method": run["method"],
                        "seed": run["seed"],
                        "level": "item",
                        "gini": run["gini"],
                        "entropy": run["entropy"],
                        # Binary item-level ILD (comparable to SASRec); the level-0
                        # row carries the code-level Hamming ILD instead.
                        "ild": run.get("item_ild"),
                        "category_ild": run.get("category_ild"),
                        "unknown_category_frac": run.get("unknown_category_frac"),
                        "n_total": run["n_items"],
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
                    "ild": ild,
                    "category_ild": run.get("category_ild"),
                    "unknown_category_frac": run.get("unknown_category_frac"),
                    "n_total": run["n_items"],
                    "valid_HR10": run["valid_HR10"],
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "seed",
        "level",
        "gini",
        "entropy",
        "ild",
        "category_ild",
        "unknown_category_frac",
        "n_total",
        "n_groups",
        "mean_group_size",
        "mean_distinct_codes",
        "total_distinct_codes",
        "valid_HR10",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["method"], r["level"], r["seed"])):
            writer.writerow(row)


def grouped_stats(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """(method, level) -> {gini, entropy, ild: (mean, std, n)} across seeds."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["level"]), []).append(row)

    result = {}
    for key, vals in groups.items():
        entry = {
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
        ild_vals = [r["ild"] for r in vals if r.get("ild") is not None]
        if ild_vals:
            entry["ild"] = (float(np.mean(ild_vals)), float(np.std(ild_vals)), len(ild_vals))
        cat_ild_vals = [r["category_ild"] for r in vals if r.get("category_ild") is not None]
        if cat_ild_vals:
            entry["category_ild"] = (
                float(np.mean(cat_ild_vals)),
                float(np.std(cat_ild_vals)),
                len(cat_ild_vals),
            )
        unk_vals = [r["unknown_category_frac"] for r in vals if r.get("unknown_category_frac") is not None]
        if unk_vals:
            entry["unknown_category_frac"] = (
                float(np.mean(unk_vals)),
                float(np.std(unk_vals)),
                len(unk_vals),
            )
        # Support diagnostics (generative per-level rows only): mean across seeds.
        for field in ("mean_group_size", "mean_distinct_codes", "total_distinct_codes"):
            field_vals = [r[field] for r in vals if r.get(field) is not None]
            if field_vals:
                entry[field] = float(np.mean(field_vals))
        result[key] = entry
    return result


def write_summary_txt(rows: list[dict], stats: dict, category: str, path: Path) -> None:
    seeds = sorted({row["seed"] for row in rows})
    methods = sorted({row["method"] for row in rows})

    lines = [
        f"{category} - Recommendation diversity: Gini + Entropy + ILD (test set, real checkpoints)",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Seeds: {seeds}",
        "",
        "Gini: higher = more concentrated/popularity-skewed recommendations.",
        "Entropy (nats): higher = more diverse/uniform recommendations.",
        "ILD (item rows): binary intra-list diversity (fraction of distinct item pairs per user); same computation for both methods, so directly comparable.",
        "ILD (MARIUS level-0 row): normalized Hamming distance over RVQ codes - code-level diversity, not comparable to the binary item-level ILD.",
        "CatILD: binary intra-list diversity over category labels (fraction of recommended pairs in different categories) - the Gini-Simpson diversity index; same computation for both methods, so directly comparable. For MARIUS it is currently depressed by the high unknown-category fraction.",
        "MARIUS Gini/Entropy are per RVQ level (conditioned on preceding levels); ILD/CatILD are single list-level values.",
        "SASRec metrics are a single flat value over recommended item ids.",
        "",
    ]

    lines.append(
        "Support columns (MARIUS per-level only): grp = mean recs per conditioning group, "
        "codes = mean distinct codes used per group. Small grp/codes at deeper levels means a "
        "low Gini there can be a support artifact rather than genuine uniformity."
    )
    lines.append("")

    # CatILD sanity check: if a method's recs largely fail the category lookup
    # (high unknown fraction), its CatILD collapses toward 0 for a spurious reason
    # (all-unknown pairs "match"). Near-zero here means a low CatILD is genuine.
    unk_lines = []
    for method in methods:
        s = stats.get((method, "item"))
        if s and "unknown_category_frac" in s:
            um, us, _ = s["unknown_category_frac"]
            unk_lines.append(f"  {method:<8}: {um:.4f} +/- {us:.4f}")
    if unk_lines:
        lines.append("Unknown-category fraction of recommended items (CatILD sanity check; lower = more trustworthy):")
        lines.extend(unk_lines)
        lines.append("")

    header = (
        f"{'method':<10} | {'level':<6} | {'gini mean +/- std':<22} | {'entropy (nats)':<22} | "
        f"{'ILD':<20} | {'CatILD':<20} | {'grp':>7} | {'codes':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for method in methods:
        levels = sorted({lvl for (m, lvl) in stats if m == method}, key=lambda x: (x != "item", x))
        for level in levels:
            s = stats[(method, level)]
            gm, gs, n = s["gini"]
            em, es, _ = s["entropy"]
            if "ild" in s:
                im, is_, _ = s["ild"]
                ild_str = f"{im:.4f} +/- {is_:.4f}"
            else:
                ild_str = "-"
            if "category_ild" in s:
                cim, cis, _ = s["category_ild"]
                cat_ild_str = f"{cim:.4f} +/- {cis:.4f}"
            else:
                cat_ild_str = "-"
            grp = f"{s['mean_group_size']:.2f}" if "mean_group_size" in s else "-"
            codes = f"{s['mean_distinct_codes']:.2f}" if "mean_distinct_codes" in s else "-"
            lines.append(
                f"{method:<10} | {level:<6} | {gm:.4f} +/- {gs:.4f}        | "
                f"{em:.4f} +/- {es:.4f}  | {ild_str:<20} | {cat_ild_str:<20} | {grp:>7} | {codes:>7}  (n={n})"
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
        (lvl for (m, lvl) in stats if m == "MARIUS" and lvl != "item"), key=int
    )
    fig, ((ax_gini, ax_ent), (ax_cat, ax_ild)) = plt.subplots(2, 2, figsize=(13, 9))

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

    # Category ILD (Gini-Simpson over recommended categories): same computation for
    # both methods, so MARIUS vs SASRec is directly comparable. Unlike binary item
    # ILD it does not saturate, since distinct items frequently share a category.
    cat_methods, cat_means, cat_stds = [], [], []
    for method, key in [("MARIUS", ("MARIUS", "item")), ("SASRec", ("SASRec", "item"))]:
        if key in stats and "category_ild" in stats[key]:
            m, s, _ = stats[key]["category_ild"]
            cat_methods.append(method)
            cat_means.append(m)
            cat_stds.append(s)
    if cat_methods:
        colors = ["steelblue" if m == "MARIUS" else "firebrick" for m in cat_methods]
        bars = ax_cat.bar(cat_methods, cat_means, yerr=cat_stds, capsize=6, color=colors, alpha=0.85)
        for bar, val in zip(bars, cat_means):
            ax_cat.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=10)
    ax_cat.set_ylabel("Category ILD (fraction of cross-category pairs)")
    ax_cat.set_title(f"{category}: Category ILD (higher = more diverse)")

    # ILD: the binary item-level ILD saturates at 1.0 for both methods (top-K lists
    # never repeat an item), so the discriminating signal is MARIUS's code-level
    # (normalized Hamming over RVQ codes) ILD on the level-0 row.
    ild_labels, ild_means, ild_stds, ild_colors = [], [], [], []
    if ("MARIUS", "0") in stats and "ild" in stats[("MARIUS", "0")]:
        m, s, _ = stats[("MARIUS", "0")]["ild"]
        ild_labels.append("MARIUS\n(code-level)")
        ild_means.append(m)
        ild_stds.append(s)
        ild_colors.append("steelblue")
    for method in ("MARIUS", "SASRec"):
        key = (method, "item")
        if key in stats and "ild" in stats[key]:
            m, s, _ = stats[key]["ild"]
            ild_labels.append(f"{method}\n(item, binary)")
            ild_means.append(m)
            ild_stds.append(s)
            ild_colors.append("steelblue" if method == "MARIUS" else "firebrick")
    if ild_labels:
        bars = ax_ild.bar(ild_labels, ild_means, yerr=ild_stds, capsize=6, color=ild_colors, alpha=0.85)
        for bar, val in zip(bars, ild_means):
            ax_ild.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=10)
    ax_ild.set_ylabel("ILD")
    ax_ild.set_title(f"{category}: Intra-List Diversity (item ILD saturates at 1.0)")

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
