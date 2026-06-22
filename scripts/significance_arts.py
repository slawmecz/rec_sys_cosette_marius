#!/usr/bin/env python
"""Seed-level statistical significance for the Arts_Crafts_and_Sewing (Amazon 2023)
5-seed reproduction.

Reads the per-seed test scores in
  reports/results/{marius,sasrec}_arts_5seed_full_scores.jsonl
and reports, per metric (R@5, NDCG@5, R@10, NDCG@10):

  Q1  95% confidence intervals on each model's mean (Student-t interval, primary;
      bootstrap reported alongside as a cross-check).
  Q3a MARIUS vs SASRec head-to-head: exact two-sample permutation test
      (C(10,5)=252 relabelings) plus the Welch t-statistic, Holm-corrected
      across the 4 metrics.
  Q2  Ours vs the paper's reported value: effect size and whether the paper
      point estimate lies inside our 95% CI. (The paper publishes no per-seed
      values or variance, so a proper two-sample test against it is impossible;
      CI containment is the honest statement.)

No scipy dependency: confidence intervals use the Student-t interval (with a
bootstrap cross-check), the head-to-head uses an exact permutation test. With
only n=5 seeds the *paired* sign-flip test cannot drop below p=0.0625, so it is
excluded; the two-sample permutation test is the one with power to reach
significance here -- this is also why the per-user paired test (separate script)
is needed for a strong claim.

Usage:
  python scripts/significance_arts.py
  python scripts/significance_arts.py --alpha 0.05 --n-boot 20000
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np

METRICS = ["R@5", "NDCG@5", "R@10", "NDCG@10"]
# Paper Table 11 (Arts, Amazon 2023), single reported value, no variance.
# (Table 3 is the COSETTE model-size study; only the 2014 results in Table 5 are
# stated as a mean over 5 runs.)
PAPER = {
    "MARIUS": {"R@5": 3.49, "NDCG@5": 2.37, "R@10": 5.30, "NDCG@10": 2.95},
    "SASRec": {"R@5": 3.51, "NDCG@5": 2.42, "R@10": 5.09, "NDCG@10": 2.93},
}
# Two-sided Student-t critical values (alpha=0.05) for small df.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def load_scores(results_dir: Path, method: str) -> dict[int, dict]:
    path = results_dir / f"{method}_arts_5seed_full_scores.jsonl"
    return {
        (r := json.loads(line))["seed"]: r
        for line in path.read_text().splitlines()
        if line.strip()
    }


def bootstrap_ci(vals: np.ndarray, n_boot: int, rng: np.random.Generator,
                 alpha: float) -> tuple[float, float]:
    n = len(vals)
    means = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def t_ci(vals: np.ndarray, alpha: float) -> tuple[float, float]:
    """Primary 95% CI: Student-t interval (the convention reviewers expect)."""
    n = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    mean = vals.mean()
    se = vals.std(ddof=1) / math.sqrt(n)
    tc = T_CRIT_95.get(n - 1, 1.96) if abs(alpha - 0.05) < 1e-9 else 1.96
    return float(mean - tc * se), float(mean + tc * se)


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = math.sqrt(va / len(a) + vb / len(b))
    return float("nan") if denom == 0 else float((a.mean() - b.mean()) / denom)


def two_sample_perm_p(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Exact two-sided permutation p for the difference in means."""
    pooled = np.concatenate([a, b])
    n, obs = len(a), abs(a.mean() - b.mean())
    idx = range(len(pooled))
    count = total = 0
    for combo in itertools.combinations(idx, n):
        mask = np.zeros(len(pooled), dtype=bool)
        mask[list(combo)] = True
        diff = abs(pooled[mask].mean() - pooled[~mask].mean())
        count += diff >= obs - 1e-12
        total += 1
    return count / total, total


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min((m - i) * p, 1.0))
        adj[k] = running
    return adj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default=os.path.join(os.environ.get("PROJECT_ROOT", "."),
                                         "reports", "results"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    results_dir = Path(args.results_dir)
    M = load_scores(results_dir, "marius")
    S = load_scores(results_dir, "sasrec")
    seeds = sorted(M)
    arr = {("M", k): np.array([M[s][k] for s in seeds]) for k in METRICS}
    arr.update({("S", k): np.array([S[s][k] for s in seeds]) for k in METRICS})

    print(f"Arts 2023 significance | n={len(seeds)} seeds {seeds} | alpha={args.alpha}\n")

    # Q1 -- confidence intervals (Student-t primary, bootstrap cross-check)
    print("== Q1: 95% CIs (Student-t primary; bootstrap cross-check) ==")
    print(f"{'metric':8} | {'MARIUS mean':11} | {'t CI':16} | {'boot CI':16} | "
          f"{'SASRec mean':11} | {'t CI':16} | {'boot CI':16}")
    print("-" * 110)
    for k in METRICS:
        mv, sv = arr[("M", k)], arr[("S", k)]
        mt, stt = t_ci(mv, args.alpha), t_ci(sv, args.alpha)
        mb, sb = bootstrap_ci(mv, args.n_boot, rng, args.alpha), bootstrap_ci(sv, args.n_boot, rng, args.alpha)
        print(f"{k:8} | {mv.mean():11.3f} | [{mt[0]:6.3f},{mt[1]:6.3f}] | [{mb[0]:6.3f},{mb[1]:6.3f}] | "
              f"{sv.mean():11.3f} | [{stt[0]:6.3f},{stt[1]:6.3f}] | [{sb[0]:6.3f},{sb[1]:6.3f}]")

    # Q3a -- MARIUS vs SASRec
    print("\n== Q3a: MARIUS vs SASRec (exact two-sample permutation + Welch t) ==")
    raw = {k: two_sample_perm_p(arr[("M", k)], arr[("S", k)])[0] for k in METRICS}
    adj = holm(raw)
    print(f"{'metric':8} | {'diff(M-S)':10} | {'Welch t':8} | {'perm p':8} | "
          f"{'Holm p':8} | sig(Holm)?")
    print("-" * 66)
    for k in METRICS:
        mv, sv = arr[("M", k)], arr[("S", k)]
        sig = "YES" if adj[k] < args.alpha else "no"
        winner = "MARIUS" if mv.mean() > sv.mean() else "SASRec"
        tag = f"{sig} ({winner})" if sig == "YES" else sig
        print(f"{k:8} | {mv.mean()-sv.mean():+10.3f} | {welch_t(mv, sv):+7.2f} | "
              f"{raw[k]:8.4f} | {adj[k]:8.4f} | {tag}")
    print("note: paired sign-flip test excluded -- its floor at n=5 is 2/32=0.0625,")
    print("      so it cannot reach alpha=0.05 and carries no evidential value here.")

    # Q2 -- ours vs paper point value
    print("\n== Q2: ours vs paper point value (CI containment; effect size in std) ==")
    print(f"{'model':7} | {'metric':8} | {'ours':7} | {'paper':6} | {'delta':7} | "
          f"{'std away':8} | paper in 95% CI?")
    print("-" * 70)
    for label, full in (("MARIUS", "M"), ("SASRec", "S")):
        for k in METRICS:
            v = arr[(full, k)]
            mean, sd, mu = v.mean(), v.std(ddof=1), PAPER[label][k]
            lo, hi = t_ci(v, args.alpha)
            inside = "yes" if lo <= mu <= hi else "NO"
            std_away = (mean - mu) / sd if sd else float("nan")
            print(f"{label:7} | {k:8} | {mean:7.3f} | {mu:6.2f} | {mean-mu:+7.3f} | "
                  f"{std_away:+8.2f} | {inside}")
    print("\nQ2 caveat: the paper reports a point value only (no per-seed data / variance),")
    print("so this is CI containment, not a two-sample test against the paper.")


if __name__ == "__main__":
    main()
