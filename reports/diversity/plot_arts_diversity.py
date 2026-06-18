import sys
sys.path.insert(0, "/gpfs/home5/scur1273/.local/lib/python3.9/site-packages")
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

df = pd.read_csv("Arts_Crafts_and_Sewing_diversity_table.csv")

# ── colour palette ────────────────────────────────────────────────────────────
MARIUS_COLOR = "#4C72B0"
SASREC_COLOR = "#DD8452"
LEVEL_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# ── aggregate per (method, level) ────────────────────────────────────────────
agg = (
    df.groupby(["method", "level"])
    .agg(
        gini_mean=("gini", "mean"),
        gini_std=("gini", "std"),
        entropy_mean=("entropy", "mean"),
        entropy_std=("entropy", "std"),
        ild_mean=("ild", "mean"),
        ild_std=("ild", "std"),
    )
    .reset_index()
)

marius = agg[agg["method"] == "MARIUS"].copy()
marius["level"] = marius["level"].astype(str)
sasrec  = agg[agg["method"] == "SASRec"].copy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Arts, Crafts & Sewing — Recommendation Diversity\n(5 seeds; error bars = ±1 std)",
    fontsize=13, fontweight="bold", y=1.02,
)

# ── helper: draw one group-bar panel ─────────────────────────────────────────
def grouped_bars(ax, marius_vals, marius_errs, sasrec_val, sasrec_err,
                 ylabel, title, ylim=None):
    levels = [f"L{l}" for l in marius["level"]]
    x = np.arange(len(levels))
    w = 0.35

    bars_m = ax.bar(x - w/2, marius_vals, w, yerr=marius_errs,
                    color=LEVEL_COLORS[:len(levels)], capsize=4,
                    label="MARIUS (per level)", zorder=3)
    # SASRec single value drawn as a horizontal dashed line + shaded band
    ax.axhline(sasrec_val, color=SASREC_COLOR, linewidth=2,
               linestyle="--", label=f"SASRec  ({sasrec_val:.4f})", zorder=4)
    ax.fill_between([-0.5, len(levels) - 0.5],
                    sasrec_val - sasrec_err, sasrec_val + sasrec_err,
                    color=SASREC_COLOR, alpha=0.15, zorder=2)

    ax.set_xticks(x - w/2)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, loc="best")

# Panel 1 — Gini coefficient
grouped_bars(
    axes[0],
    marius["gini_mean"].values, marius["gini_std"].values,
    sasrec["gini_mean"].values[0], sasrec["gini_std"].values[0],
    ylabel="Gini (↑ = more concentrated)",
    title="Gini Coefficient",
    ylim=(0.90, 1.002),
)

# Panel 2 — Entropy
grouped_bars(
    axes[1],
    marius["entropy_mean"].values, marius["entropy_std"].values,
    sasrec["entropy_mean"].values[0], sasrec["entropy_std"].values[0],
    ylabel="Entropy in nats (↑ = more diverse)",
    title="Entropy",
)

# Panel 3 — ILD (only level 0 for MARIUS)
ax3 = axes[2]
ild_row = marius[marius["level"] == "0"].iloc[0]
bars = ax3.bar(
    [0], [ild_row["ild_mean"]], 0.4,
    yerr=[[ild_row["ild_std"]], [ild_row["ild_std"]]],
    color=MARIUS_COLOR, capsize=6, label="MARIUS L0", zorder=3,
)
sr = sasrec.iloc[0]
ax3.bar(
    [0.55], [sr["ild_mean"]], 0.4,
    yerr=[[sr["ild_std"] if not np.isnan(sr["ild_std"]) else 0],
          [sr["ild_std"] if not np.isnan(sr["ild_std"]) else 0]],
    color=SASREC_COLOR, capsize=6, label="SASRec", zorder=3,
)
ax3.set_xticks([0, 0.55])
ax3.set_xticklabels(["MARIUS\n(L0 only)", "SASRec"], fontsize=10)
ax3.set_ylabel("ILD (↑ = more diverse within list)", fontsize=10)
ax3.set_title("Intra-List Diversity (ILD)", fontsize=11, fontweight="bold")
ax3.yaxis.grid(True, linestyle=":", alpha=0.6)
ax3.set_axisbelow(True)
ax3.legend(fontsize=8)

# ── per-seed scatter (tiny, below main panels) ───────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
fig2.suptitle(
    "Arts, Crafts & Sewing — Per-seed variance across 5 seeds",
    fontsize=12, fontweight="bold", y=1.01,
)
metrics = [
    ("gini",    "Gini coefficient",    axes2[0]),
    ("entropy", "Entropy (nats)",      axes2[1]),
    ("ild",     "ILD",                 axes2[2]),
]
seeds = sorted(df["seed"].unique())
seed_markers = ["o", "s", "D", "^", "v"]

for col, label, ax in metrics:
    for i, seed in enumerate(seeds):
        sub = df[df["seed"] == seed]
        mdf = sub[sub["method"] == "MARIUS"]
        sdf = sub[sub["method"] == "SASRec"]
        level_labels = [f"M-L{l}" for l in mdf["level"]]
        vals_m = mdf[col].values
        # drop NaN
        valid = ~np.isnan(vals_m)
        ax.scatter(np.array(level_labels)[valid], vals_m[valid],
                   marker=seed_markers[i], color=MARIUS_COLOR,
                   alpha=0.7, s=60, zorder=3)
        if not sdf[col].isna().all():
            ax.scatter(["SASRec"], sdf[col].dropna().values[:1],
                       marker=seed_markers[i], color=SASREC_COLOR,
                       alpha=0.7, s=60, zorder=3)

    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8)

# shared legend for seeds
handles = [
    mpatches.Patch(color=MARIUS_COLOR, label="MARIUS"),
    mpatches.Patch(color=SASREC_COLOR, label="SASRec"),
]
fig2.legend(handles=handles, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.04), fontsize=9)

fig.tight_layout()
fig2.tight_layout()
fig.savefig("Arts_Crafts_and_Sewing_diversity_bars.png",  dpi=150, bbox_inches="tight")
fig2.savefig("Arts_Crafts_and_Sewing_diversity_seeds.png", dpi=150, bbox_inches="tight")
print("Saved Arts_Crafts_and_Sewing_diversity_bars.png")
print("Saved Arts_Crafts_and_Sewing_diversity_seeds.png")