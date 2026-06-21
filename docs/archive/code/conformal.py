#!/usr/bin/env python3
"""Split-conformal recommendation sets on the frozen Top-K dumps (feasibility pilot).

Question piloted here: can we wrap the frozen MARIUS / SASRec++ rankers in
distribution-free conformal prediction sets, and is the resulting set size a
useful, calibrated uncertainty signal? This runs entirely on the local dumps
written by scripts/extensions/dump_topk.py (depth K = 20), so the guaranteeable
coverage is capped at each model's Recall@20.

Method (standard split conformal on rank nonconformity)
--------------------------------------------------------
Per (model, category, seed):
  * load the ranked top-20 catalog items and the held-out target per user via
    the validated loaders in scripts/extensions/compute_beyond_accuracy.py;
  * nonconformity score of a user = 1-based rank of the target in the list
    (21 = absent, treated as infinity);
  * split users 50/50 into calibration / evaluation halves with a fixed
    numpy rng (default seed 0; identical split for all models and seeds of a
    category because the user count is fixed);
  * for a target coverage c, the conformal set is the top k* of the ranking,
    where k* is the ceil((n_cal + 1) * c)-th smallest calibration rank. If
    k* > 20 the coverage is unachievable at this dump depth. By the standard
    split-conformal quantile lemma, P(target in top-k*) >= c on exchangeable
    data; rank ties only make this conservative.

Mondrian variant: calibrate k* separately inside history-length buckets
(hist <= 5, 6-15, >= 16), giving per-bucket (hence per-user-varying) set sizes
with the same per-bucket guarantee. History length is read from the raw npz;
the MARIUS dumps store hist_len = sasrec hist_len + 1 (one extra sequence
token; verified exactly on Beauty and Sports), so 1 is subtracted for MARIUS
to put both models on the same "number of history interactions" scale. The
script asserts that relationship whenever the matching SASRec dump exists.

Outputs (under --out-dir, default reports/extensions/conformal/):
  split_conformal_<category>.csv   per (model, seed, coverage) rows
  mondrian_<category>.csv          per (model, seed, coverage, bucket) rows
  conformal_<category>.md          per-category report (seed means)
  conformal_summary.md             cross-category headline tables (rebuilt
                                   from whatever CSVs are present)

Usage:
  python3 scripts/extensions/conformal.py --category Beauty
  python3 scripts/extensions/conformal.py --category Sports_and_Outdoors
  python3 scripts/extensions/conformal.py --selftest

Pure numpy/pandas; no torch anywhere on this code path.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extensions import beyond_accuracy as ba  # noqa: E402
from scripts.extensions.compute_beyond_accuracy import load_support, load_model_recs  # noqa: E402

DEPTH = 20                  # DEFAULT depth (validated dumps); the live depth is
                            # read per-dump from the topk npz array shape, so this
                            # is only the fallback used by the standalone selftest.
ABSENT = DEPTH + 1          # rank sentinel at the default depth (selftest only)
BUCKETS = [(0, 5, "hist<=5"), (6, 15, "hist6-15"), (16, 10 ** 9, "hist>=16")]
DEFAULT_COVERAGES = (0.04, 0.06, 0.08, 0.10, 0.12)
DEFAULT_MONDRIAN = (0.08, 0.06)


# --------------------------------------------------------------------------- #
# Core conformal machinery
# --------------------------------------------------------------------------- #
def ranks_from_recs(recs, targets, depth=DEPTH):
    """1-based rank of the target in each user's ranked list.

    Returns int array: rank in [1, depth], ABSENT (= depth + 1) if the target is
    not in the list, and -1 for users with an invalid target (excluded from
    everything downstream). Hallucination sentinels (-1) in the lists simply
    occupy a slot and can never match a valid target.
    """
    out = np.full(len(recs), depth + 1, dtype=np.int64)
    for i, (row, tgt) in enumerate(zip(recs, targets)):
        if tgt is None or int(tgt) < 0:
            out[i] = -1
            continue
        for j, it in enumerate(row[:depth]):
            if it == tgt:
                out[i] = j + 1
                break
    return out


def split_indices(n_users, split_seed=0):
    """Fixed 50/50 calibration / evaluation split (first half of a permutation)."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_users)
    half = n_users // 2
    return perm[:half], perm[half:]


def conformal_k_star(cal_ranks, coverage, depth=DEPTH):
    """Split-conformal threshold: the m-th smallest calibration rank with
    m = ceil((n_cal + 1) * coverage). Returns (k_star, m); k_star = depth + 1
    (> depth) means the coverage is unachievable at this dump depth."""
    cal = np.asarray(cal_ranks)
    n = cal.size
    m = int(math.ceil((n + 1) * coverage))
    if m > n:
        return depth + 1, m
    k = int(np.sort(cal)[m - 1])
    return k, m


def mean_effective_set_size(recs, idx, k):
    """Mean number of REAL items (sentinel -1 excluded) in the top-k of each
    selected user's list. For SASRec this equals k; for MARIUS hallucinated
    slots shrink the usable set."""
    if k <= 0:
        return 0.0
    vals = [sum(1 for it in recs[i][:k] if it is not None and it >= 0) for i in idx]
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# Per-seed analysis
# --------------------------------------------------------------------------- #
def canonical_hist(npz, model, sasrec_hist=None):
    """History length on the shared 'number of history interactions' scale.

    The MARIUS dump stores hist_len exactly one larger than the SASRec dump for
    the same users (one extra sequence token); subtract it. When the matching
    SASRec hist is supplied, the +1 relationship is asserted.
    """
    h = npz["hist_len"].astype(np.int64)
    if model == "marius":
        h = h - 1
        if sasrec_hist is not None and not np.array_equal(h, sasrec_hist):
            # The SASRec seed-43 Sports dump stores one 556-user block in a
            # permuted order (same users; sorted multisets agree). Each npz is
            # internally consistent, so per-model analysis is unaffected; only
            # flag a real mismatch in the user POPULATION.
            if not np.array_equal(np.sort(h), np.sort(sasrec_hist)):
                raise ValueError("MARIUS hist_len - 1 does not match SASRec "
                                 "hist_len even as a multiset; dumps disagree "
                                 "on the user population")
            n_perm = int(np.sum(h != sasrec_hist))
            print(f"  (note: {n_perm} users appear in a different order in the "
                  "matching SASRec dump; multiset identical, per-model analysis "
                  "unaffected)", flush=True)
    return h


def analyze_seed(recs, ranks, n_hist, cal_idx, eval_idx, coverages,
                 mondrian_coverages, depth=DEPTH):
    """All split-conformal and Mondrian rows for one (model, seed). Returns
    (split_rows, mondrian_rows) as lists of dicts (no model/seed keys yet).

    `depth` is the live dump depth; the ceiling is Recall@depth and k* is
    achievable only when k* <= depth. The ceiling columns are named depth-
    agnostically (ceiling_recall_eval/all, bucket_recall_eval); read the actual
    depth from the 'depth' column run_category stamps on each row."""
    valid = ranks >= 0
    cal = cal_idx[valid[cal_idx]]
    ev = eval_idx[valid[eval_idx]]
    cal_r, ev_r = ranks[cal], ranks[ev]
    ceiling_eval = float(np.mean(ev_r <= depth))   # achievable coverage ceiling
    ceiling_all = float(np.mean(ranks[valid] <= depth))

    split_rows = []
    for c in coverages:
        k, m = conformal_k_star(cal_r, c, depth=depth)
        achievable = k <= depth
        emp = float(np.mean(ev_r <= k)) if achievable else float("nan")
        eff = mean_effective_set_size(recs, ev, k) if achievable else float("nan")
        split_rows.append({
            "target_coverage": c, "n_cal": int(cal.size), "n_eval": int(ev.size),
            "m": m, "k_star": k, "achievable": achievable,
            "empirical_coverage": emp, "effective_set_size": eff,
            "ceiling_recall_eval": ceiling_eval, "ceiling_recall_all": ceiling_all,
        })

    mond_rows = []
    for c in mondrian_coverages:
        for lo, hi, label in BUCKETS:
            bc = cal[(n_hist[cal] >= lo) & (n_hist[cal] <= hi)]
            be = ev[(n_hist[ev] >= lo) & (n_hist[ev] <= hi)]
            k, m = conformal_k_star(ranks[bc], c, depth=depth)
            achievable = k <= depth
            emp = float(np.mean(ranks[be] <= k)) if (achievable and be.size) else float("nan")
            ceiling_b = float(np.mean(ranks[be] <= depth)) if be.size else float("nan")
            mond_rows.append({
                "target_coverage": c, "bucket": label,
                "n_cal": int(bc.size), "n_eval": int(be.size), "m": m,
                "k_star": k, "achievable": achievable,
                "empirical_coverage": emp, "bucket_recall_eval": ceiling_b,
            })
    return split_rows, mond_rows


def _dump_suffix(n_results: int) -> str:
    """File suffix for {model}_seed{seed}_topk{suffix}.npz (matches dump_topk.py)."""
    return "" if n_results == DEPTH else str(n_results)


def load_recs_at_depth(dump_dir: Path, model: str, seed: int, meta: dict, t2i: dict,
                       n_results: int = DEPTH):
    """(recs, targets) for an arbitrary dump depth.

    For the validated depth-20 dumps this delegates to the shared
    compute_beyond_accuracy.load_model_recs. For deeper dumps written by
    dump_topk.py --n-results N (file suffix topk{N}.npz) it applies the
    identical de-offset + lookup, kept byte-for-byte in sync with that loader
    (and with mbr.load_marius_dump)."""
    if n_results == DEPTH:
        return load_model_recs(dump_dir, model, seed, meta, t2i)
    npz = np.load(dump_dir / f"{model}_seed{seed}_topk{_dump_suffix(n_results)}.npz")
    ns = meta["n_special"]
    if model == "sasrec":
        recs = [[int(v) - ns if int(v) >= ns else ba.HALLUCINATION for v in row]
                for row in npz["topk_items"]]
        targets = [int(t) - ns if int(t) >= ns else ba.HALLUCINATION
                   for t in npz["target_item"]]
        return recs, targets
    L = meta["L"]
    K = max(int(x) for key in t2i for x in key.split(",")) + 1
    level_off = [l * K + ns for l in range(L)]

    def to_item(code):
        raw = [int(c) - level_off[l] for l, c in enumerate(code)]
        if any(r < 0 or r >= K for r in raw):
            return ba.HALLUCINATION
        return t2i.get(",".join(str(r) for r in raw), ba.HALLUCINATION)

    recs = [[to_item(code) for code in row] for row in npz["topk_codes"]]
    targets = [to_item(code) for code in npz["target_codes"]]
    return recs, targets


def run_category(dump_dir: Path, models, seeds, coverages, mondrian_coverages,
                 split_seed=0, n_results=DEPTH):
    """Returns (split_df, mondrian_df) over all (model, seed) found in dump_dir.

    The dump depth is read from the topk npz array shape (axis 1), not the
    DEPTH constant, so the same code runs on the depth-20 or the depth-100
    dumps; the per-category report stamps the depth it actually used."""
    meta, _pop, _emb, t2i = load_support(dump_dir)
    suffix = _dump_suffix(n_results)
    split_all, mond_all = [], []
    for model in models:
        for seed in seeds:
            npz_path = dump_dir / f"{model}_seed{seed}_topk{suffix}.npz"
            if not npz_path.exists():
                print(f"  (skip {model} seed {seed}: no dump)", flush=True)
                continue
            recs, targets = load_recs_at_depth(dump_dir, model, seed, meta, t2i, n_results)
            npz = np.load(npz_path)
            # Live depth from the dumped Top-K array shape (sasrec: [U, K];
            # marius: [U, K, L]); both models in a category share this depth.
            arr = npz["topk_items"] if model == "sasrec" else npz["topk_codes"]
            depth = int(arr.shape[1])
            sas_path = dump_dir / f"sasrec_seed{seed}_topk{suffix}.npz"
            sas_hist = np.load(sas_path)["hist_len"].astype(np.int64) if (
                model == "marius" and sas_path.exists()) else None
            n_hist = canonical_hist(npz, model, sasrec_hist=sas_hist)
            ranks = ranks_from_recs(recs, targets, depth=depth)
            cal_idx, eval_idx = split_indices(len(recs), split_seed)
            srows, mrows = analyze_seed(recs, ranks, n_hist, cal_idx, eval_idx,
                                        coverages, mondrian_coverages, depth=depth)
            base = {"category": meta["category"], "model": model, "seed": seed,
                    "depth": depth}
            split_all += [{**base, **r} for r in srows]
            mond_all += [{**base, **r} for r in mrows]
    return pd.DataFrame(split_all), pd.DataFrame(mond_all)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt_k(sub: pd.DataFrame, depth: int = DEPTH) -> str:
    """Seed-aggregate display of k_star: mean if all seeds achievable, else >depth."""
    n_seeds = len(sub)
    n_ach = int(sub["achievable"].sum())
    if n_ach < n_seeds:
        return f">{depth} ({n_ach}/{n_seeds} seeds achievable)"
    ks = sub["k_star"].astype(float)
    per_seed = "/".join(str(int(k)) for k in sub["k_star"])
    return f"{ks.mean():.2f} ({per_seed})"


def _fmt_cov(sub: pd.DataFrame, col="empirical_coverage") -> str:
    vals = sub.loc[sub["achievable"], col].astype(float)
    if vals.empty:
        return "n/a"
    s = f"{vals.mean():.4f}"
    if len(vals) > 1:
        s += f" +/- {vals.std(ddof=1):.4f}"
    return s


def _depth_of(df: pd.DataFrame) -> int:
    """Live dump depth stamped by run_category; falls back to DEPTH."""
    if "depth" in df.columns and not df["depth"].empty:
        return int(df["depth"].iloc[0])
    return DEPTH


def split_table_md(split_df: pd.DataFrame, models) -> str:
    depth = _depth_of(split_df)
    lines = ["| target c | " + " | ".join(
        f"{m}: k* | {m}: emp. cov." for m in models) + " |"]
    lines.append("|" + "---|" * (1 + 2 * len(models)))
    for c in sorted(split_df["target_coverage"].unique()):
        cells = [f"{c:.2f}"]
        for m in models:
            sub = split_df[(split_df["model"] == m) & (split_df["target_coverage"] == c)]
            if sub.empty:
                cells += ["-", "-"]
            else:
                cells += [_fmt_k(sub, depth), _fmt_cov(sub)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mondrian_table_md(mond_df: pd.DataFrame, models, coverage) -> str:
    depth = _depth_of(mond_df)
    sub_c = mond_df[np.isclose(mond_df["target_coverage"], coverage)]
    lines = ["| bucket | n_cal | bucket Recall@%d (eval) | " % depth + " | ".join(
        f"{m}: k* | {m}: emp. cov." for m in models) + " |"]
    lines.append("|" + "---|" * (3 + 2 * len(models)))
    for _lo, _hi, label in BUCKETS:
        sub_b = sub_c[sub_c["bucket"] == label]
        if sub_b.empty:
            continue
        n_cal = int(sub_b["n_cal"].mean())
        cells = [label, str(n_cal)]
        ceil_parts = []
        for m in models:
            s = sub_b[sub_b["model"] == m]
            ceil_parts.append(f"{m[:3]} {s['bucket_recall_eval'].mean():.3f}" if not s.empty else "")
        cells.append(", ".join(p for p in ceil_parts if p))
        for m in models:
            s = sub_b[sub_b["model"] == m]
            if s.empty:
                cells += ["-", "-"]
            else:
                cells += [_fmt_k(s, depth), _fmt_cov(s)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_category_md(out_dir: Path, category: str, split_df, mond_df, models,
                      mondrian_coverages) -> Path:
    sd = split_df[split_df["category"] == category]
    md = mond_df[mond_df["category"] == category]
    depth = _depth_of(sd)
    n_cal = int(sd["n_cal"].iloc[0])
    n_eval = int(sd["n_eval"].iloc[0])
    seeds = sorted(sd["seed"].unique())
    lines = [
        f"# Conformal recommendation sets: {category}",
        "",
        f"Split conformal on rank nonconformity over the frozen depth-{depth} Top-K",
        f"dumps (reports/extensions/topk/{category}). Users split 50/50 into",
        f"calibration (n={n_cal}) / evaluation (n={n_eval}) with numpy rng seed 0;",
        f"seeds {', '.join(str(s) for s in seeds)}; values below are seed means",
        "(per-seed k* in parentheses).",
        "",
        "## Feasibility ceiling",
        "",
        f"With a single held-out positive and a depth-{depth} list, the largest coverage",
        f"any conformal wrapper can guarantee is the model's Recall@{depth}:",
        "",
        f"| model | Recall@{depth} (eval half) | Recall@{depth} (all users) |",
        "|---|---|---|",
    ]
    for m in models:
        s = sd[sd["model"] == m]
        if s.empty:
            continue
        one = s.drop_duplicates(subset=["seed"])
        lines.append(f"| {m} | {one['ceiling_recall_eval'].mean():.4f} | "
                     f"{one['ceiling_recall_all'].mean():.4f} |")
    lines += [
        "",
        f"Targets above this ceiling are marked unachievable (>{depth}). Conventional",
        "90% coverage guarantees are impossible for next-item hit at this depth.",
        "",
        "## Split conformal: target coverage -> set size k* -> empirical coverage",
        "",
        split_table_md(sd, models),
        "",
        "Empirical coverage is computed on the held-out evaluation half; the",
        "conformal guarantee is empirical coverage >= target c (rank ties make it",
        "conservative, so overshoot is expected). The guarantee is marginal over",
        "the calibration draw: with a single fixed split the eval estimate has a",
        f"binomial s.e. of about {math.sqrt(0.08 * 0.92 / n_eval):.4f} at c=0.08,",
        "so undershoot of a few thousandths is within noise.",
    ]
    mar = sd[(sd["model"] == "marius") & sd["achievable"]]
    if not mar.empty:
        eff = mar.groupby("target_coverage").agg(
            k=("k_star", "mean"), eff=("effective_set_size", "mean"), n=("seed", "size"))
        full = eff[eff["n"] == eff["n"].max()]
        if not full.empty:
            lines += ["", "MARIUS effective set sizes (real items only; hallucinated",
                      "slots in the top-k* are wasted):", ""]
            lines.append("| target c | k* (mean) | real items in set (mean) |")
            lines.append("|---|---|---|")
            for c, row in full.iterrows():
                lines.append(f"| {c:.2f} | {row['k']:.2f} | {row['eff']:.2f} |")
    for c in mondrian_coverages:
        lines += [
            "",
            f"## Mondrian conformal at target c = {c:.2f} (history-length buckets)",
            "",
            mondrian_table_md(md, models, c),
        ]
    lines += [
        "",
        "Mondrian k* is calibrated inside each bucket, so each bucket carries its",
        "own >= c guarantee and the per-user set size varies with history length.",
        "",
        f"CSV: split_conformal_{category}.csv, mondrian_{category}.csv.",
        "",
    ]
    path = out_dir / f"conformal_{category}.md"
    path.write_text("\n".join(lines))
    return path


def _headline_lines(sd: pd.DataFrame, md_df: pd.DataFrame, models, primary) -> list:
    """Computed (not hand-written) headline bullets for the summary."""
    depth = _depth_of(sd)
    lines = ["## Headline", ""]
    if len(models) == 2:
        a, b = models
        for cat in sorted(sd["category"].unique()):
            s = sd[sd["category"] == cat]
            rows = []
            for c in sorted(s["target_coverage"].unique()):
                ka = s[(s["model"] == a) & (s["target_coverage"] == c)]
                kb = s[(s["model"] == b) & (s["target_coverage"] == c)]
                if ka.empty or kb.empty:
                    continue
                if ka["achievable"].all() and kb["achievable"].all():
                    rows.append((c, ka["k_star"].mean(), kb["k_star"].mean()))
            if rows:
                n_le = sum(1 for _c, x, y in rows if x <= y)
                c0, x0, y0 = rows[-1]
                lines.append(
                    f"- {cat}: {a} needs a set no larger than {b} at {n_le}/{len(rows)}"
                    f" jointly achievable targets (largest, c={c0:.2f}: mean k*"
                    f" {x0:.2f} vs {y0:.2f}).")
            else:
                lines.append(f"- {cat}: no target coverage is achievable for both "
                             f"models at depth {depth}.")
    for cat in sorted(md_df["category"].unique()):
        m = md_df[md_df["category"] == cat]
        cold_label, warm_label = BUCKETS[0][2], BUCKETS[-1][2]
        # Largest Mondrian coverage at which every bucket is achievable for all
        # seeds of the first model; fall back toward smaller coverages.
        for c in sorted(m["target_coverage"].unique(), reverse=True):
            sub = m[np.isclose(m["target_coverage"], c) & (m["model"] == models[0])]
            if not sub.empty and sub["achievable"].all():
                kc = sub[sub["bucket"] == cold_label]["k_star"].mean()
                kw = sub[sub["bucket"] == warm_label]["k_star"].mean()
                direction = ("cold users need the larger sets" if kc - kw > 1 else
                             "set size does NOT shrink with history" if kw - kc > 1
                             else "approximately flat across history")
                note = ""
                if c < primary - 1e-9:
                    note = f" (c={primary:.2f} exceeds at least one bucket ceiling)"
                lines.append(
                    f"- {cat} Mondrian at c={c:.2f}: {models[0]} k* {kc:.2f}"
                    f" ({cold_label}) vs {kw:.2f} ({warm_label}); {direction}{note}.")
                break
        else:
            lines.append(f"- {cat} Mondrian: no tested coverage is achievable in "
                         f"every bucket at depth {depth}.")
    lines.append("")
    return lines


def write_summary_md(out_dir: Path, models, mondrian_primary=0.08) -> Path:
    """Rebuild conformal_summary.md from every category CSV present in out_dir."""
    split_files = sorted(out_dir.glob("split_conformal_*.csv"))
    sd = pd.concat([pd.read_csv(f) for f in split_files], ignore_index=True)
    mond_files = sorted(out_dir.glob("mondrian_*.csv"))
    md_df = pd.concat([pd.read_csv(f) for f in mond_files], ignore_index=True)
    depth = _depth_of(sd)
    lines = [
        "# Conformal recommendation sets: feasibility pilot (summary)",
        "",
        "Split conformal prediction on rank nonconformity, wrapped around the",
        f"frozen MARIUS and SASRec++ depth-{depth} Top-K dumps (3 seeds, 50/50",
        "calibration/evaluation split, numpy rng seed 0). Full method and",
        "per-category detail in conformal_<category>.md.",
        "",
        "## Honest feasibility verdict",
        "",
        f"With a single held-out positive per user and lists truncated at depth {depth},",
        f"the largest guaranteeable coverage equals the model's Recall@{depth}:",
        "",
        f"| category | model | coverage ceiling (Recall@{depth}, eval half) |",
        "|---|---|---|",
    ]
    for cat in sorted(sd["category"].unique()):
        for m in models:
            s = sd[(sd["category"] == cat) & (sd["model"] == m)].drop_duplicates(subset=["seed"])
            if s.empty:
                continue
            lines.append(f"| {cat} | {m} | {s['ceiling_recall_eval'].mean():.4f} |")
    lines += [
        "",
        "So conventional 90% (or even 20%) next-item coverage guarantees are",
        "impossible at this depth. The viable framings are:",
        "(a) low-coverage guarantees with tiny sets (quantified below);",
        "(b) Mondrian per-bucket set size as a calibrated difficulty signal",
        "(quantified below);",
        "(c) deeper beams raise the ceiling, with sublinearly diminishing returns",
        "(needs the deeper dumps; see the deeper-beam dependency note).",
        "",
    ]
    lines += _headline_lines(sd, md_df, models, mondrian_primary)
    lines += [
        "## (a) Matched guaranteed coverage: which model needs the smaller set?",
        "",
    ]
    for cat in sorted(sd["category"].unique()):
        lines += [f"### {cat}", "", split_table_md(sd[sd["category"] == cat], models), ""]
    lines += [
        f"## (b) Mondrian set size by history length (target c = {mondrian_primary:.2f})",
        "",
    ]
    for cat in sorted(md_df["category"].unique()):
        lines += [f"### {cat}", "",
                  mondrian_table_md(md_df[md_df["category"] == cat], models, mondrian_primary), ""]
    others = sorted(set(round(c, 4) for c in md_df["target_coverage"].unique())
                    - {round(mondrian_primary, 4)})
    for c in others:
        lines += [f"## Mondrian at target c = {c:.2f} (secondary)", ""]
        for cat in sorted(md_df["category"].unique()):
            lines += [f"### {cat}", "",
                      mondrian_table_md(md_df[md_df["category"] == cat], models, c), ""]
    if depth <= DEPTH:
        lines += [
            "## Deeper-beam dependency",
            "",
            "Raising the coverage ceiling requires deeper ranked lists than the local",
            f"depth-{depth} dumps. The extended dump_topk.py run (--n-results 100, both",
            "marius and sasrec) produces exactly the artifacts needed; re-running this",
            "script on those dumps (it reads the depth from the npz shape) is the only",
            "follow-up.",
            "",
        ]
    else:
        lines += [
            "## Deeper beam (this run)",
            "",
            f"These sets are computed on the depth-{depth} dumps (dump_topk.py",
            f"--n-results {depth}); the ceiling is Recall@{depth}, well above the",
            "depth-20 numbers, so larger coverage targets are now achievable (see the",
            "split-conformal tables above for where k* lands).",
            "",
        ]
    path = out_dir / "conformal_summary.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import json as _json
    import tempfile

    # 1) rank computation on a tiny manual case.
    r = ranks_from_recs([[3, 1, 2], [5, 6, 7], [9, -1, 4]], [1, 9, -1], depth=3)
    assert r.tolist() == [2, 4, -1], r  # depth=3 so absent = 4; invalid target = -1
    r2 = ranks_from_recs([[3, 1, 2]], [3])
    assert r2.tolist() == [1], r2

    # 2) conformal quantile against a brute-force second implementation.
    rng = np.random.default_rng(1)
    cal = rng.integers(1, ABSENT + 1, size=5000)
    for c in (0.04, 0.08, 0.12):
        k, m = conformal_k_star(cal, c)
        brute = next(kk for kk in range(1, ABSENT + 1)
                     if int(np.sum(cal <= kk)) >= m)
        assert k == brute, (c, k, brute)
    k_hi, _ = conformal_k_star(np.full(100, ABSENT), 0.5)
    assert k_hi == ABSENT  # all-miss calibration -> unachievable

    # 3) end-to-end on a fabricated dump with controlled hit-rate structure.
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "Beauty"
        d.mkdir(parents=True)
        n_special, n_catalog, L, U = 2, 500, 4, 6000
        (d / "meta.json").write_text(_json.dumps(
            {"category": "Beauty", "n_special": n_special, "n_catalog": n_catalog,
             "L": L, "K": DEPTH}))
        np.save(d / "popularity.npy", rng.integers(1, 100, n_catalog).astype(np.int64))
        codes = rng.integers(0, 256, size=(n_catalog, L))
        t2i = {",".join(str(int(c)) for c in codes[i]): i for i in range(n_catalog)}
        (d / "tuple_to_item.json").write_text(_json.dumps(t2i))
        K_cb = 256
        level_off = np.arange(L) * K_cb + n_special

        # History-dependent difficulty: cold users hit with per-rank prob 0.004
        # (Recall@20 = 0.08), warm users 0.02 (Recall@20 = 0.40).
        hist = np.where(rng.random(U) < 0.5,
                        rng.integers(4, 6, U),      # cold bucket (<= 5)
                        rng.integers(16, 50, U))    # warm bucket (>= 16)
        per_rank = np.where(hist <= 5, 0.004, 0.02)
        u = rng.random(U)
        rank_true = np.where(u < per_rank * DEPTH,
                             np.floor(u / per_rank).astype(int) + 1, ABSENT)

        targets = rng.integers(0, n_catalog, U)
        items = np.empty((U, DEPTH), dtype=np.int64)
        for i in range(U):
            row = rng.choice(n_catalog, size=DEPTH, replace=False)
            row = row[row != targets[i]][:DEPTH]
            while row.size < DEPTH:
                extra = rng.integers(0, n_catalog)
                if extra != targets[i] and extra not in row:
                    row = np.append(row, extra)
            if rank_true[i] <= DEPTH:
                row[rank_true[i] - 1] = targets[i]
            items[i] = row
        for seed in (42, 43):
            np.savez_compressed(
                d / f"sasrec_seed{seed}_topk.npz",
                topk_items=(items + n_special), target_item=(targets + n_special),
                hist_len=hist.astype(np.int64))
            np.savez_compressed(  # same ranking in MARIUS token space, hist + 1
                d / f"marius_seed{seed}_topk.npz",
                topk_codes=(codes[items] + level_off).astype(np.int64),
                target_codes=(codes[targets] + level_off).astype(np.int64),
                hist_len=(hist + 1).astype(np.int64))

        models = ["sasrec", "marius"]
        sd, md = run_category(d, models, [42, 43], DEFAULT_COVERAGES, DEFAULT_MONDRIAN)
        assert len(sd) == 2 * 2 * len(DEFAULT_COVERAGES), sd.shape
        # Identical rankings dumped in both formats must give identical k*.
        piv = sd.pivot_table(index=["seed", "target_coverage"], columns="model",
                             values="k_star")
        assert (piv["marius"] == piv["sasrec"]).all(), piv
        # Marginal Recall@20 = 0.5*0.08 + 0.5*0.40 = 0.24: c <= 0.12 achievable.
        ach = sd[np.isclose(sd["target_coverage"], 0.12)]
        assert ach["achievable"].all(), ach
        # Empirical coverage must meet the target (allow tiny eval-half noise).
        ok = sd[sd["achievable"]]
        assert (ok["empirical_coverage"] >= ok["target_coverage"] - 0.01).all(), ok
        # Mondrian at 0.08: cold bucket needs a systematically larger set than
        # warm (per-rank hit prob 0.004 vs 0.02 -> k* ~ 20+ vs ~ 4).
        m08 = md[np.isclose(md["target_coverage"], 0.08) & (md["model"] == "sasrec")]
        k_cold = m08[m08["bucket"] == "hist<=5"]["k_star"]
        k_warm = m08[m08["bucket"] == "hist>=16"]["k_star"]
        assert (k_cold.values > k_warm.values).all(), (k_cold, k_warm)
        assert (k_warm <= 6).all(), k_warm
        # No users fabricated in the middle bucket.
        assert (m08[m08["bucket"] == "hist6-15"]["n_cal"] == 0).all()

        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        sd.to_csv(out_dir / "split_conformal_Beauty.csv", index=False)
        md.to_csv(out_dir / "mondrian_Beauty.csv", index=False)
        p1 = write_category_md(out_dir, "Beauty", sd, md, models, DEFAULT_MONDRIAN)
        p2 = write_summary_md(out_dir, models)
        assert p1.exists() and p2.exists()
        assert "unachievable" in p1.read_text()
        print(sd[["model", "seed", "target_coverage", "k_star", "achievable",
                  "empirical_coverage"]].to_string(index=False))
        print("\nOK: conformal selftest passed.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split-conformal recommendation sets from Top-K dumps")
    parser.add_argument("--dump-root", type=Path,
                        default=REPO / "reports" / "extensions" / "topk")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--models", default="sasrec marius")
    parser.add_argument("--seeds", default="42 43 44")
    parser.add_argument("--n-results", type=int, default=DEPTH,
                        help=f"dump depth to read; != {DEPTH} reads "
                             "{model}_seed{seed}_topk{N}.npz. The live depth used "
                             "for the math is still read from the npz array shape.")
    parser.add_argument("--coverages", default=" ".join(str(c) for c in DEFAULT_COVERAGES))
    parser.add_argument("--mondrian-coverages",
                        default=" ".join(str(c) for c in DEFAULT_MONDRIAN))
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO / "reports" / "extensions" / "conformal")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    dump_dir = args.dump_root / args.category
    models = args.models.split()
    seeds = [int(s) for s in args.seeds.split()]
    coverages = [float(c) for c in args.coverages.split()]
    mondrian_coverages = [float(c) for c in args.mondrian_coverages.split()]

    split_df, mond_df = run_category(dump_dir, models, seeds, coverages,
                                     mondrian_coverages, args.split_seed,
                                     n_results=args.n_results)
    if split_df.empty:
        print(f"No dumps found under {dump_dir}.", flush=True)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_csv = args.out_dir / f"split_conformal_{args.category}.csv"
    mond_csv = args.out_dir / f"mondrian_{args.category}.csv"
    split_df.to_csv(split_csv, index=False)
    mond_df.to_csv(mond_csv, index=False)
    md_path = write_category_md(args.out_dir, args.category, split_df, mond_df,
                                models, mondrian_coverages)
    summary_path = write_summary_md(args.out_dir, models, mondrian_coverages[0])

    cols = ["model", "seed", "target_coverage", "k_star", "achievable",
            "empirical_coverage", "effective_set_size"]
    print(split_df[cols].to_string(index=False), flush=True)
    print()
    mcols = ["model", "seed", "target_coverage", "bucket", "n_cal", "k_star",
             "achievable", "empirical_coverage", "bucket_recall_eval"]
    print(mond_df[mcols].to_string(index=False), flush=True)
    print(f"\nWrote {split_csv}\nWrote {mond_csv}\nWrote {md_path}\nWrote {summary_path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
