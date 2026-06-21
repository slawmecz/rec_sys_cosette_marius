#!/usr/bin/env python3
"""Figure library for the RQ1-4 extension story.

Single source of truth for the extension's data loaders and figures. Everything
loads committed artifacts under reports/, so it re-executes anywhere the repo is
checked out (no GPU, no torch). The story notebook (build_extension_story_notebook.py)
imports these functions; running this file as a script saves every figure to
reports/figures/ as PNG, SVG, and PDF for direct use in the paper.

    python3 scripts/extensions/extension_figures.py          # save all figures
    python3 scripts/extensions/extension_figures.py rq4      # save just F1 (RQ4 Pareto)

Set EXTFIG_PAPER=1 to suppress the figure-level title (so the LaTeX caption is the
only title) -- used to render the title-free PDFs under latex/figures/.

Numbers are verified against EXTENSION_RESULTS.md. Recall is plotted in percent
(beyond-accuracy CSVs store it as a fraction; the 5-seed jsonls store percent).
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve()
while not (REPO / "scripts" / "extensions" / "beyond_accuracy.py").exists() and REPO != REPO.parent:
    REPO = REPO.parent
RES = REPO / "reports" / "results"
FAITH = REPO / "reports" / "paper_faithful_rerun"
EXT = REPO / "reports" / "extensions"
FIG = REPO / "reports" / "figures"
ARTS_DIR = EXT / "topk_arts"

ARTS = "Arts_Crafts_and_Sewing"
ARTS_SEEDS = [42, 44, 46, 48, 50]
N_CATALOG = {"Beauty": 12101, "Sports_and_Outdoors": 18357, "Arts_Crafts_and_Sewing": 89958}
SHORT = {"Beauty": "Beauty", "Sports_and_Outdoors": "Sports", "Arts_Crafts_and_Sewing": "Arts (90k)"}

# --------------------------------------------------------------------------- #
# Style (shared muted palette + tidy helpers used across the project notebooks)
# --------------------------------------------------------------------------- #
SAS, MAR = "#4c72b0", "#c44e52"            # the two models
NEUTRAL, ACCENT, MUTE = "#8c8c8c", "#55a868", "#b0b0b0"
COL = {"sasrec": SAS, "marius": MAR}
# RQ4 mitigation-arm palette
ARM = {
    "pmi": "#dd8452",       # post-processing (decode re-rank)
    "mbr": "#937860",       # beam-stage re-rank
    "logitadj": "#8172b3",  # in-processing (train loss)
    "distill": "#da8bc3",   # cross-paradigm distillation
}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.bbox": "tight", "figure.facecolor": "white",
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "axes.titlelocation": "left", "axes.titlepad": 8,
})


def tidy(ax, ygrid=True):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color="#e6e6e6", lw=0.8)
    return ax


def labels(ax, xs, vals, fmt="{:.2f}", dy=0.0, color="#555"):
    for x, v in zip(np.atleast_1d(xs), np.atleast_1d(vals)):
        ax.annotate(fmt.format(v), (x, v + dy), ha="center", va="bottom",
                    fontsize=7.5, color=color)


def _suptitle(fig, *args, **kwargs):
    """Figure-level title, suppressed when EXTFIG_PAPER is set so the paper PDFs
    carry no baked-in title (the LaTeX \\caption describes the figure instead).
    The notebook (no env var) keeps the self-documenting titles."""
    if os.environ.get("EXTFIG_PAPER"):
        return
    fig.suptitle(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def jsonl_r10(path):
    """(mean, std) of R@10 (percent) over the ok-seeds of a 5seed scores jsonl."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    vals = [r["R@10"] for r in rows if r.get("status") == "ok"]
    return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)


def load_beyond_csv(path):
    """A beyond-accuracy CSV (category, model, k, <metric>_mean, <metric>_std, ...)."""
    return pd.read_csv(path)


def arts_beyond():
    """5-seed mean beyond-accuracy for Arts. Returns df with columns
    [model, k, <metric>] where <metric> is the cross-seed mean of <metric>_mean."""
    frames = []
    for s in ARTS_SEEDS:
        p = ARTS_DIR / f"seed{s}" / f"beyond_accuracy_{ARTS}_seed{s}.csv"
        if p.exists():
            d = load_beyond_csv(p)
            d["seed"] = s
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no Arts beyond-accuracy CSVs under {ARTS_DIR}")
    df = pd.concat(frames, ignore_index=True)
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    agg = df.groupby(["model", "k"])[mean_cols].mean().reset_index()
    agg.columns = [c[:-5] if c.endswith("_mean") else c for c in agg.columns]
    return agg


def arts_reach():
    """5-seed mean Chao1 coverage ceiling for Arts, per (model, k)."""
    frames = []
    for s in ARTS_SEEDS:
        p = ARTS_DIR / f"seed{s}" / "reach_estimator" / f"reach_estimator_{ARTS}.csv"
        if p.exists():
            frames.append(load_beyond_csv(p))
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["model", "k"])[["observed_coverage_mean", "chao1_coverage_mean"]].mean()


def arts_pmi(prior):
    """5-seed mean PMI Pareto for a prior. Returns df indexed by alpha."""
    files = sorted(glob.glob(str(ARTS_DIR / "seed*" / f"pmi_{ARTS}_{prior}_demand_pareto.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.groupby("alpha").mean(numeric_only=True)


def arts_mbr():
    """5-seed mean MBR sweep. Returns (baseline_dict, df indexed by tau of the mbr rows)."""
    rows = []
    for tau in [0.25, 0.5, 1.0, 2.0]:
        files = sorted(glob.glob(str(ARTS_DIR / "seed*" / f"mbr_{ARTS}_tau{tau}.csv")))
        if not files:
            continue
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        mbr = d[d.variant == "mbr"].mean(numeric_only=True)
        mbr["tau"] = tau
        rows.append(mbr)
    base_files = sorted(glob.glob(str(ARTS_DIR / "seed*" / f"mbr_{ARTS}_tau1.0.csv")))
    bd = pd.concat([pd.read_csv(f) for f in base_files], ignore_index=True)
    base = bd[bd.variant == "original"].mean(numeric_only=True).to_dict()
    return base, pd.DataFrame(rows).set_index("tau")


def logitadj(slug, cat, taus=(0.5, 1.0, 1.5), k=10):
    """Train-time logit-adjustment beyond-accuracy for one dataset.
    Returns (baseline_row_dict, df indexed by tau). tau=0 is the committed baseline."""
    base = load_beyond_csv(EXT / f"beyond_accuracy_{cat}.csv")
    base = base[(base.model == "marius") & (base.k == k)].iloc[0]
    rows = []
    for tau in taus:
        p = EXT / f"topk_logitadj_{slug}_tau{tau}" / f"beyond_accuracy_{cat}_logitadj.csv"
        if not p.exists():
            continue
        d = load_beyond_csv(p)
        r = d[(d.model == "marius") & (d.k == k)].iloc[0].to_dict()
        r["tau"] = tau
        rows.append(r)
    return base, pd.DataFrame(rows).set_index("tau")


def oracle(path):
    d = json.loads(Path(path).read_text())
    perk = {row["k"]: row for row in d["per_k"]}
    d["_perk"] = perk
    return d


def oracle_ranks(npz_path):
    arr = np.load(npz_path)["target_exact_rank"]
    return arr[arr >= 0]


def distill():
    d = load_beyond_csv(EXT / "distill" / "distill_beyond_accuracy.csv")
    base = d[d.variant == "baseline"].iloc[0]
    dist = d[d.variant == "distill"].iloc[0]
    return base, dist


def arts_conditional():
    """Per-RVQ-level conditional Gini/entropy of MARIUS's recommended semantic IDs
    (5-seed mean), plus the SASRec flat reference. Source: the maks_new_metrics
    instrument output committed under reports/diversity/."""
    df = pd.read_csv(REPO / "reports" / "diversity" / f"{ARTS}_diversity_table.csv")
    mar = df[df.method == "MARIUS"].copy()
    mar["level"] = mar["level"].astype(int)
    g = mar.groupby("level").agg(
        gini=("gini", "mean"), gini_sd=("gini", "std"),
        ent=("entropy", "mean"), ent_sd=("entropy", "std")).reset_index()
    sas = df[df.method == "SASRec"]
    return g, float(sas["gini"].mean()), float(sas["entropy"].mean())


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_rq1_reproduction():
    """RQ1: SASRec++ vs MARIUS+COSETTE across scales; the small-to-large flip."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    # (panel title, sasrec (mean,std), marius (mean,std), paper marius)
    bsas = jsonl_r10(FAITH / "beauty_d32__scores.jsonl")
    ssas = jsonl_r10(FAITH / "sports_d64__scores.jsonl")
    panels = [
        ("Beauty (12k)", bsas, jsonl_r10(RES / "marius_beauty_5seed_full_scores.jsonl"), 10.02),
        ("Sports (18k)", ssas, jsonl_r10(RES / "marius_sports_5seed_full_scores.jsonl"), 6.72),
        ("Arts (90k)", jsonl_r10(RES / "sasrec_arts_5seed_full_scores.jsonl"),
         jsonl_r10(RES / "marius_arts_5seed_full_scores.jsonl"), None),
    ]
    for ax, (title, sas, mar, paper) in zip(axes, panels):
        means = [sas[0], mar[0]]
        errs = [sas[1], mar[1]]
        ax.bar([0, 1], means, 0.6, yerr=errs, color=[SAS, MAR], ecolor="#999",
               capsize=3, error_kw={"lw": 1})
        labels(ax, [0, 1], means, dy=max(errs) + 0.08)
        if paper:
            ax.axhline(paper, color="#bbb", ls=(0, (4, 3)), lw=1)
            ax.text(1.4, paper + 0.05, "paper", ha="right", fontsize=7.5, color="#999")
        flip = mar[0] > sas[0]
        ax.set_title(f"{title}   " + ("MARIUS > SASRec" if flip else "MARIUS < SASRec"),
                     color=(MAR if flip else "#444"))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SASRec++", "MARIUS\n+COSETTE"])
        ax.set_ylim(0, max(means) * 1.25 + 1)
        tidy(ax)
    axes[0].set_ylabel("test Recall@10 (%)")
    _suptitle(fig, "RQ1  Reproducibility: the paper's claim reproduces only at scale "
                 "(Arts: MARIUS > SASRec, perm p=0.008)", x=0.5, y=1.02, fontsize=10.5)
    plt.tight_layout()
    return fig


def fig_rq2_beyond_accuracy():
    """RQ2 (F2): beyond-accuracy, MARIUS vs SASRec, small vs large.
    Coverage and Gini show aggregate collapse; ILD and APLT show per-list diversity is fine."""
    cats = ["Beauty", "Sports_and_Outdoors", "Arts_Crafts_and_Sewing"]
    b14 = {c: load_beyond_csv(EXT / f"beyond_accuracy_{c}.csv") for c in cats[:2]}
    arts = arts_beyond()

    def val(cat, model, metric):
        if cat == ARTS:
            r = arts[(arts.model == model) & (arts.k == 10)].iloc[0]
            return float(r[metric])
        d = b14[cat]
        return float(d[(d.model == model) & (d.k == 10)].iloc[0][f"{metric}_mean"])

    panels = [("coverage", "catalog coverage (higher=broader)", "{:.2f}", 0.02),
              ("gini", "Gini of exposure (lower=less concentrated)", "{:.2f}", 0.02),
              ("ild", "intra-list diversity (per-list)", "{:.3f}", 0.004),
              ("aplt", "avg % long-tail per list (higher=more tail)", "{:.2f}", 0.006)]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    x = np.arange(len(cats))
    for ax, (metric, title, fmt, dy) in zip(axes, panels):
        for i, m in enumerate(["sasrec", "marius"]):
            vals = [val(c, m, metric) for c in cats]
            bars = ax.bar(x + (i - 0.5) * 0.36, vals, 0.34, color=COL[m],
                          label=m.upper() if ax is axes[0] else None)
            labels(ax, x + (i - 0.5) * 0.36, vals, fmt=fmt, dy=dy)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[c] for c in cats], fontsize=8)
        ax.set_title(title, fontsize=9.5)
        tidy(ax)
    axes[0].legend(loc="upper right")
    _suptitle(fig, "RQ2  MARIUS reaches fewer distinct items in aggregate (coverage, Gini) "
                 "yet its lists are as diverse and more tail-leaning (ILD, APLT)",
                 x=0.5, y=1.02, fontsize=10.5)
    plt.tight_layout()
    return fig


def fig_rq3_oracle():
    """RQ3 (F3): exact full-catalog oracle. Targets are buried by the model's ranking
    (left); exact recall ~= beam recall => model-bound, not a search artifact (right)."""
    # Both rows use the FILTERED protocol (history items masked before ranking), so the
    # exact-vs-beam comparison is apples-to-apples and exact == beam holds on the right panel.
    specs = [
        ("Arts MARIUS", oracle(ARTS_DIR / "seed42" / "exact_catalog" /
                               f"exact_catalog_{ARTS}_seed42_filtered.json"),
         oracle_ranks(ARTS_DIR / "seed42" / "exact_catalog" /
                      f"exact_catalog_{ARTS}_seed42_filtered.npz"), MAR, "-"),
        ("Sports MARIUS", oracle(EXT / "exact_catalog" /
                                 f"exact_catalog_Sports_and_Outdoors_seed42_filtered.json"),
         oracle_ranks(EXT / "exact_catalog" / "exact_catalog_Sports_and_Outdoors_seed42_filtered.npz"),
         "#c98a8c", "--"),
    ]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 3.8))

    # Left: CDF of the held-out target's exact full-catalog rank (log-x).
    for name, _, ranks, c, ls in specs:
        xs = np.sort(ranks)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axL.plot(xs, ys, color=c, ls=ls, lw=1.8,
                 label=f"{name} (median {np.median(ranks):.0f})")
    axL.axvspan(1, 10, color="#e9f2e6", zorder=0)
    axL.text(10, 0.02, "exact top-10", fontsize=7.5, color=ACCENT, ha="left")
    axL.set_xscale("log")
    axL.set_xlabel("exact full-catalog rank of the true next item (log)")
    axL.set_ylabel("fraction of users <= rank")
    axL.set_title("Targets sit deep in the model's exact ranking\n"
                  "(median exact rank ~1000-1956 of the full catalog)")
    axL.legend(loc="lower right")
    tidy(axL, ygrid=False)
    axL.grid(True, color="#eee", lw=0.7)

    # Right: exact vs beam recall@10 -> the gap is ~0 => MODEL-bound.
    names = [s[0] for s in specs]
    xb = np.arange(len(specs))
    rex = [100 * s[1]["_perk"][10]["recall_exact"] for s in specs]
    rbe = [100 * s[1]["_perk"][10]["recall_beam"] for s in specs]
    axR.bar(xb - 0.2, rbe, 0.38, color=MUTE, label="beam search")
    axR.bar(xb + 0.2, rex, 0.38, color="#3c6e47", label="exact oracle (full catalog)")
    labels(axR, xb - 0.2, rbe, fmt="{:.1f}", dy=0.06)
    labels(axR, xb + 0.2, rex, fmt="{:.1f}", dy=0.06)
    axR.set_xticks(xb)
    axR.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
    axR.set_ylabel("Recall@10 (%)")
    axR.set_title("Exact oracle ~= beam => the collapse is the\nmodel's ranking, not the beam search")
    axR.set_ylim(0, max(rex + rbe) * 1.3)
    axR.legend(loc="upper right")
    tidy(axR)

    _suptitle(fig, "RQ2.2  The popularity collapse is model-bound: exact full-catalogue "
                 "scoring matches the beam (filtered, seed 42)", x=0.5, y=1.02, fontsize=10.5)
    plt.tight_layout()
    return fig


def fig_rq4_pareto():
    """RQ4 (F1): the accuracy-vs-coverage trade-off across all mitigation arms.
    Panel A: absolute Pareto on Arts (90k). Panel B: normalized %-change vs each arm's
    own baseline, so all arms (across datasets) share one axis -- every lever is a
    bounded dial; distillation is the extreme, and (per RQ3) none lifts the ceiling."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # ---- Panel A: absolute Arts Pareto ---------------------------------- #
    ab = arts_beyond()
    base = ab[(ab.model == "marius") & (ab.k == 10)].iloc[0]
    sas = ab[(ab.model == "sasrec") & (ab.k == 10)].iloc[0]
    bx, by = 100 * base["recall"], base["coverage"]

    for prior, c, mk in [("cond2", ARM["pmi"], "o"), ("pair", "#e3a36f", "s"),
                         ("item", "#c66f3a", "^")]:
        d = arts_pmi(prior)
        axA.plot(100 * d["recall"], d["coverage"], "-", color=c, lw=1.2, marker=mk,
                 ms=4, label=f"PMI ({prior})")
    base_mbr, dmbr = arts_mbr()
    axA.plot(100 * dmbr["recall"], dmbr["coverage"], "-", color=ARM["mbr"], lw=1.2,
             marker="D", ms=4, label="MBR (tau sweep)")
    axA.scatter([bx], [by], s=90, color=MAR, zorder=5, edgecolor="white", lw=1)
    axA.annotate("MARIUS\nbaseline", (bx, by), textcoords="offset points",
                 xytext=(6, -2), fontsize=7.5, color=MAR)
    axA.scatter([100 * sas["recall"]], [sas["coverage"]], s=90, color=SAS, marker="*",
                zorder=5, edgecolor="white", lw=0.5)
    axA.annotate("SASRec++", (100 * sas["recall"], sas["coverage"]),
                 textcoords="offset points", xytext=(6, 2), fontsize=7.5, color=SAS)
    axA.set_xlabel("Recall@10 (%)")
    axA.set_ylabel("catalog coverage")
    axA.set_title("A. Arts (90k): post-hoc re-rank trades recall for coverage")
    axA.legend(loc="lower left", fontsize=7.8)
    tidy(axA, ygrid=False)
    axA.grid(True, color="#eee", lw=0.7)

    # ---- Panel B: normalized %-change, all arms across datasets --------- #
    def pct(v, b):
        return 100 * (v - b) / b

    # PMI cond2 (Arts) trajectory
    d = arts_pmi("cond2")
    axB.plot([pct(r, base["recall"]) * 1 for r in d["recall"]],
             [pct(c, base["coverage"]) for c in d["coverage"]],
             "-o", color=ARM["pmi"], lw=1.2, ms=4, label="PMI cond2 (Arts)")
    # PMI item (Arts) -- strongest coverage lever
    di = arts_pmi("item")
    axB.plot([pct(r, base["recall"]) for r in di["recall"]],
             [pct(c, base["coverage"]) for c in di["coverage"]],
             "-^", color="#c66f3a", lw=1.0, ms=4, label="PMI item (Arts)")
    # MBR (Arts)
    axB.plot([pct(r, base_mbr["recall"]) for r in dmbr["recall"]],
             [pct(c, base_mbr["coverage"]) for c in dmbr["coverage"]],
             "-D", color=ARM["mbr"], lw=1.2, ms=4, label="MBR (Arts)")
    # logit-adj (Beauty, Sports) -- train-time
    for slug, cat, mk in [("beauty", "Beauty", "v"), ("sports", "Sports_and_Outdoors", "P")]:
        b, dla = logitadj(slug, cat)
        axB.plot([pct(dla.loc[t, "recall_mean"], b["recall_mean"]) for t in dla.index],
                 [pct(dla.loc[t, "coverage_mean"], b["coverage_mean"]) for t in dla.index],
                 "-", color=ARM["logitadj"], lw=1.0, marker=mk, ms=5,
                 label=f"logit-adj ({SHORT[cat]})")
    # distillation (Sports) -- the extreme
    db, dd = distill()
    axB.scatter([pct(dd["recall_mean"], db["recall_mean"])],
                [pct(dd["coverage_mean"], db["coverage_mean"])],
                s=140, color=ARM["distill"], marker="*", zorder=5, edgecolor="white",
                lw=0.6, label="distillation (Sports)")
    axB.annotate("distillation\n(+123% cov, -22% R@10)\nbut targets ranked DEEPER",
                 (pct(dd["recall_mean"], db["recall_mean"]),
                  pct(dd["coverage_mean"], db["coverage_mean"])),
                 textcoords="offset points", xytext=(-8, -38), fontsize=7,
                 color=ARM["distill"], ha="center")
    axB.axhline(0, color="#bbb", lw=0.8)
    axB.axvline(0, color="#bbb", lw=0.8)
    axB.scatter([0], [0], s=60, color="#333", zorder=6)
    axB.annotate("each arm's\nbaseline", (0, 0), textcoords="offset points",
                 xytext=(6, 6), fontsize=7, color="#333")
    axB.set_xlabel("change in Recall@10 (%)")
    axB.set_ylabel("change in catalog coverage (%)")
    axB.set_title("B. All arms, normalized: every lever is a bounded dial\n"
                  "(up-left = more coverage, less recall)")
    axB.legend(loc="upper right", fontsize=7.3)
    tidy(axB, ygrid=False)
    axB.grid(True, color="#eee", lw=0.7)

    _suptitle(fig, "RQ4  Mitigation across pipeline stages: a clean accuracy-vs-coverage "
                 "trade-off; no lever moves the model-bound ceiling", x=0.5, y=1.02,
                 fontsize=10.5)
    plt.tight_layout()
    return fig


def fig_rq3_conditional_diversity():
    """RQ2.2 refinement: where in the RVQ hierarchy the model-bound concentration sits.
    MARIUS's recommended semantic IDs are near-degenerate at the coarse levels and branch
    only at the leaf. MARIUS-only on the per-level axes: SASRec has no RVQ levels, and its
    entropy support (89,958 items) differs from MARIUS's (256 codes/level), so a per-level
    overlay would be misleading; SASRec is described in the caption instead."""
    g, _sas_gini, _sas_ent = arts_conditional()
    lv = g["level"].values
    eff = np.exp(g["ent"].values)  # effective number of codes = exp(conditional entropy)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    axA.errorbar(lv, g["gini"], yerr=g["gini_sd"], fmt="-o", color=MAR, lw=1.6, ms=6,
                 capsize=3, ecolor="#999")
    labels(axA, lv, g["gini"], fmt="{:.3f}", dy=0.003)
    axA.set_xticks(lv); axA.set_xticklabels([f"L{int(l)}" for l in lv])
    axA.set_ylim(0.90, 1.005)
    axA.set_xlabel("RVQ level (coarse to fine)"); axA.set_ylabel("conditional Gini")
    axA.set_title("A. Recommended codes near-degenerate at coarse levels")
    tidy(axA, ygrid=False); axA.grid(True, color="#eee", lw=0.7)

    axB.bar(lv, eff, 0.6, color=MAR)
    labels(axB, lv, eff, fmt="{:.1f}", dy=0.4)
    axB.set_xticks(lv); axB.set_xticklabels([f"L{int(l)}" for l in lv])
    axB.set_xlabel("RVQ level (coarse to fine)")
    axB.set_ylabel("effective codes (exp of conditional entropy)")
    axB.set_title("B. MARIUS branches only at the leaf level")
    axB.set_ylim(0, max(eff) * 1.25)
    tidy(axB)

    _suptitle(fig, "RQ2.2  Conditional code usage: MARIUS funnels into one coarse semantic "
                 "bucket and branches only at the leaf", x=0.5, y=1.02, fontsize=10.5)
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Registry + save
# --------------------------------------------------------------------------- #
FIGURES = {
    "rq1": ("fig_rq1_reproduction", fig_rq1_reproduction),
    "rq2": ("fig_rq2_beyond_accuracy", fig_rq2_beyond_accuracy),
    "rq3": ("fig_rq3_oracle", fig_rq3_oracle),
    "rq4": ("fig_rq4_pareto", fig_rq4_pareto),  # F1
    "rq3cond": ("fig_rq3_conditional_diversity", fig_rq3_conditional_diversity),
}


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    print(f"wrote {FIG / name}.png/.svg/.pdf")


def save_all(which=None):
    keys = [which] if which else list(FIGURES)
    for k in keys:
        name, fn = FIGURES[k]
        save_fig(fn(), name)
        plt.close("all")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg and arg not in FIGURES:
        sys.exit(f"unknown figure '{arg}'; choose from {list(FIGURES)}")
    save_all(arg)
