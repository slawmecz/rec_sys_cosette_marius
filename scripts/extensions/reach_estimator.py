#!/usr/bin/env python3
"""Validated catalog-reachability estimator (Good-Turing / Chao1 + rarefaction).

This is the "instrument" upgrade over reporting a single coverage number (which
SimGR, arXiv:2602.07847, already does). Treating USERS as the sampling unit and
catalog ITEMS as species, it answers: of the items the beam never recommends, how
many are structurally unreachable (no amount of additional users from the same
distribution would surface them) versus merely unsampled?

  observed coverage  = distinct items ever emitted / n_catalog  (full user population)
  Chao1              = lower-bound estimate of the asymptotic reachable richness
  rarefaction curve  = distinct items vs fraction of users (does it saturate?)
  Turing unseen mass = Good-Turing estimate of the probability mass on unseen items

If Chao1 / n_catalog is close to the observed coverage (and the rarefaction curve
has saturated), the unreached set is STRUCTURAL: the decoder cannot reach those
items regardless of more users. That is positive evidence for "sick decoding",
and the estimator + protocol is the scoop-resistant contribution.

Numpy/pandas only (no torch, no GPU): runs locally on the committed depth-20 dumps
under reports/extensions/topk/<category>/. Reuses the validated MARIUS token
de-offset from compute_beyond_accuracy so the item mapping can never drift.

  python scripts/extensions/reach_estimator.py --category Sports_and_Outdoors --ks "10 20"
  python scripts/extensions/reach_estimator.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extensions import beyond_accuracy as ba  # noqa: E402
from scripts.extensions.compute_beyond_accuracy import load_support, load_model_recs  # noqa: E402

RAREFACTION_FRACTIONS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def incidence_counts(recs, k, n_catalog):
    """Per-item user-incidence: how many users have the item in their top-k.

    recs: list (per user) of catalog-index lists (ba.HALLUCINATION == -1 dropped).
    Returns int64[n_catalog] of incidence counts.
    """
    counts = np.zeros(n_catalog, dtype=np.int64)
    for row in recs:
        seen = set()
        for it in row[:k]:
            if it is ba.HALLUCINATION or it < 0:
                continue
            if it not in seen:
                seen.add(it)
                counts[it] += 1
    return counts


def chao1(s_obs, f1, f2):
    """Bias-corrected Chao1 richness estimate (robust when f2 == 0)."""
    return s_obs + (f1 * (f1 - 1.0)) / (2.0 * (f2 + 1.0))


def rarefaction(recs, k, fractions, rng):
    """Distinct items reached vs fraction of users (random subsample, one draw)."""
    n_users = len(recs)
    order = rng.permutation(n_users)
    per_user = []
    for u in order:
        items = {it for it in recs[u][:k] if not (it is ba.HALLUCINATION or it < 0)}
        per_user.append(items)
    out = []
    for frac in fractions:
        m = max(1, int(round(frac * n_users)))
        reached = set().union(*per_user[:m]) if m else set()
        out.append((frac, m, len(reached)))
    return out


def estimate(dump_dir: Path, models, seeds, ks):
    meta, pop, _emb, t2i = load_support(dump_dir)
    n_catalog = meta["n_catalog"]
    rng = np.random.default_rng(0)
    rows = []
    rare_rows = []
    for model in models:
        for seed in seeds:
            if not (dump_dir / f"{model}_seed{seed}_topk.npz").exists():
                print(f"  (skip {model} seed {seed}: no dump)", flush=True)
                continue
            recs, _targets = load_model_recs(dump_dir, model, seed, meta, t2i)
            for k in ks:
                counts = incidence_counts(recs, k, n_catalog)
                s_obs = int((counts > 0).sum())
                f1 = int((counts == 1).sum())
                f2 = int((counts == 2).sum())
                total_emissions = int(counts.sum())
                ch = chao1(s_obs, f1, f2)
                turing_unseen = (f1 / total_emissions) if total_emissions else float("nan")
                rows.append({
                    "category": meta["category"], "model": model, "seed": seed, "k": k,
                    "n_catalog": n_catalog, "n_users": len(recs),
                    "observed_reached": s_obs,
                    "observed_coverage": s_obs / n_catalog,
                    "f1_singletons": f1, "f2_doubletons": f2,
                    "chao1": ch, "chao1_coverage": ch / n_catalog,
                    "est_structural_unreachable": max(0.0, n_catalog - ch),
                    "est_structural_unreachable_frac": max(0.0, n_catalog - ch) / n_catalog,
                    "turing_unseen_mass": turing_unseen,
                })
                for frac, m, reached in rarefaction(recs, k, RAREFACTION_FRACTIONS, rng):
                    rare_rows.append({
                        "category": meta["category"], "model": model, "seed": seed, "k": k,
                        "user_fraction": frac, "n_users_sampled": m,
                        "distinct_reached": reached,
                        "reached_coverage": reached / n_catalog,
                    })
    df = pd.DataFrame(rows)
    rare = pd.DataFrame(rare_rows)
    if df.empty:
        return df, rare, df
    metric_cols = [c for c in df.columns if c not in ("category", "model", "seed", "k")]
    agg = df.groupby(["category", "model", "k"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" if b else a for a, b in agg.columns]
    return df, rare, agg.reset_index()


def _selftest() -> int:
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "Beauty"
        d.mkdir(parents=True)
        n_special, n_catalog, L, U = 2, 100, 4, 400
        (d / "meta.json").write_text(json.dumps(
            {"category": "Beauty", "n_special": n_special, "n_catalog": n_catalog, "L": L, "K": L}))
        np.save(d / "popularity.npy", rng.integers(1, 50, n_catalog).astype(np.int64))
        codes = rng.integers(0, 256, size=(n_catalog, L))
        t2i = {",".join(str(int(c)) for c in codes[i]): i for i in range(n_catalog)}
        (d / "tuple_to_item.json").write_text(json.dumps(t2i))
        K_cb = max(int(x) for key in t2i for x in key.split(",")) + 1
        level_off = np.arange(L) * K_cb + n_special
        # MARIUS only ever reaches items [0, 60): coverage must come out ~0.6 and
        # Chao1 must NOT exceed the catalog. Each reachable item to many users so
        # f1/f2 are small (saturated) -> chao1_coverage close to observed.
        reachable = 60
        pick = rng.integers(0, reachable, size=(U, 20))
        topk_codes = codes[pick] + level_off
        np.savez_compressed(
            d / "marius_seed42_topk.npz",
            topk_codes=topk_codes.astype(np.int64),
            target_codes=(codes[rng.integers(0, n_catalog, U)] + level_off).astype(np.int64),
            hist_len=rng.integers(1, 50, U).astype(np.int64))
        _per, rare, agg = estimate(d, ["marius"], [42], [10, 20])
        cov = agg[(agg.model == "marius") & (agg.k == 20)]["observed_coverage_mean"].iloc[0]
        ch_cov = agg[(agg.model == "marius") & (agg.k == 20)]["chao1_coverage_mean"].iloc[0]
        assert 0.5 < cov <= 0.61, f"coverage off: {cov}"
        assert ch_cov <= 1.01, f"chao1 coverage should not blow past catalog: {ch_cov}"
        assert ch_cov >= cov - 1e-9, f"chao1 must be >= observed: {ch_cov} vs {cov}"
        full = rare[(rare.model == "marius") & (rare.k == 20) & (rare.user_fraction == 1.0)]
        assert int(full["distinct_reached"].iloc[0]) == int(round(cov * n_catalog)), "rarefaction endpoint mismatch"
        print(agg.to_string(index=False))
        print("\nOK: reach_estimator selftest passed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Good-Turing / Chao1 catalog-reachability estimator")
    p.add_argument("--dump-root", type=Path, default=REPO / "reports" / "extensions" / "topk")
    p.add_argument("--category", default="Sports_and_Outdoors")
    p.add_argument("--models", default="marius sasrec")
    p.add_argument("--seeds", default="42 43 44")
    p.add_argument("--ks", default="10 20")
    p.add_argument("--out-dir", type=Path, default=REPO / "reports" / "extensions" / "reach_estimator")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()

    dump_dir = args.dump_root / args.category
    models = args.models.split()
    seeds = [int(s) for s in args.seeds.split()]
    ks = [int(k) for k in args.ks.split()]

    per_seed, rare, agg = estimate(dump_dir, models, seeds, ks)
    if agg.empty:
        print(f"No dumps under {dump_dir}.", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.out_dir / f"reach_estimator_{args.category}_perseed.csv", index=False)
    rare.to_csv(args.out_dir / f"reach_estimator_{args.category}_rarefaction.csv", index=False)
    agg.to_csv(args.out_dir / f"reach_estimator_{args.category}.csv", index=False)
    show = ["category", "model", "k", "observed_coverage_mean", "chao1_coverage_mean",
            "est_structural_unreachable_frac_mean", "f1_singletons_mean", "turing_unseen_mass_mean"]
    print(agg[show].to_string(index=False), flush=True)
    print(f"\nWrote {args.out_dir}/reach_estimator_{args.category}*.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
