#!/usr/bin/env python3
"""Label-bias / logit-adjusted re-ranking of MARIUS's beam candidates (the PMI decoder).

The mechanism-targeted counterpart to the MBR pilot (scripts/extensions/mbr.py).
Our REACH analysis localized the catalog collapse to decode step 2: items lose to
popular siblings sharing their first code digit, so whole rare digit0:digit1 prefix
families are pruned. A locally-normalized autoregressive decoder ranked by joint
log-probability is subject to "label bias" (Andor et al. 2016) and the Bayes-optimal
correction under label shift is logit adjustment (Menon et al. 2021): subtract a
prior term from the score. Here the prior is the within-prefix conditional estimated
from training demand:

    adjusted(tuple) = log p(tuple | history) - alpha * log p_hat(digit1 | digit0)

  prior_kind:
    cond2  log p_hat(d1|d0) = log( D(d0,d1) / D(d0) )         # within-prefix conditional (headline, mechanism-matched)
    pair   log p_hat(d0,d1) = log( D(d0,d1) / D(.) )          # joint prefix-family prior
    item   log p_hat(item)  = log( pop(item) / sum pop )      # generic global-popularity debias (baseline ablation)

D(.) is training demand, either count-weighted (one per item) or demand-weighted
(by training interaction count). Subtracting a NEGATIVE log-prior boosts rare
within-prefix continuations, so larger alpha trades top accuracy for catalog reach.
alpha = 0 reproduces the beam's own ranking on the candidate pool.

This needs the SCORED dumps (the npz "scores" key = joint candidate log-probs,
written by dump_topk.py --with-scores). The committed depth-20 dumps carry no
scores; produce a depth-100 scored dump first:
    python scripts/extensions/dump_topk.py --method marius --no-support \
        --n-results 100 --with-scores --eval-batch-size 32 ... (see jobs/22_*.sbatch)

Numpy/pandas only. Reuses the validated loaders + metric suite via mbr.py and
compute_beyond_accuracy.py, so the token de-offset / item mapping never drifts.

  python scripts/extensions/pmi_rerank.py --category Sports_and_Outdoors \
      --seeds "42 43 44" --n-results 100 --prior cond2 --weighting demand \
      --alphas "0 0.25 0.5 0.75 1.0 1.25 1.5"
  python scripts/extensions/pmi_rerank.py --selftest
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
from scripts.extensions.compute_beyond_accuracy import load_support  # noqa: E402
from scripts.extensions.mbr import (  # noqa: E402
    DEFAULT_K, METRICS, apply_order, evaluate, load_marius_dump,
)

DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


# --------------------------------------------------------------------------- #
# Training-demand priors (pure numpy)
# --------------------------------------------------------------------------- #
def build_prefix_prior(t2i: dict, pop, L: int, k_cb: int, weighting: str = "demand", eps: float = 1.0):
    """Return (logp_cond[k_cb,k_cb], logp_pair[k_cb,k_cb], logp_item[n_catalog]).

    logp_cond[d0,d1] = log p_hat(d1|d0); logp_pair[d0,d1] = log p_hat(d0,d1);
    logp_item[j]     = log p_hat(item j). All Laplace-smoothed by eps so the logs
    are finite. weighting: 'demand' (training interaction counts) or 'count'
    (one unit per item, i.e. catalog-structural prior).
    """
    pop = np.asarray(pop, dtype=np.float64)
    n_catalog = len(t2i)
    Dd0 = np.zeros(k_cb, dtype=np.float64)
    Dd0d1 = np.zeros((k_cb, k_cb), dtype=np.float64)
    w_item = np.zeros(n_catalog, dtype=np.float64)
    for key, j in t2i.items():
        raw = [int(x) for x in key.split(",")]
        d0, d1 = raw[0], raw[1]
        w = float(pop[j]) if weighting == "demand" else 1.0
        Dd0[d0] += w
        Dd0d1[d0, d1] += w
        w_item[int(j)] = w

    logp_cond = np.log((Dd0d1 + eps) / (Dd0[:, None] + eps * k_cb))
    total = Dd0d1.sum()
    logp_pair = np.log((Dd0d1 + eps) / (total + eps * k_cb * k_cb))
    logp_item = np.log((w_item + eps) / (w_item.sum() + eps * n_catalog))
    return logp_cond, logp_pair, logp_item


def prefix_prior_term(codes, recs, logp_cond, logp_pair, logp_item, prior_kind, L, k_cb, n_special):
    """float64[U, C] log-prior per candidate for the chosen prior_kind.

    codes: int64[U,C,L] TOKEN space; recs: per-user catalog-index lists (for 'item').
    Candidates whose digit0/digit1 fall out of [0,k_cb) (or hallucinations for
    'item') get prior 0.0, i.e. no adjustment.
    """
    U, C, _ = codes.shape
    if prior_kind == "item":
        out = np.zeros((U, C), dtype=np.float64)
        for u in range(U):
            for c in range(C):
                it = recs[u][c]
                out[u, c] = logp_item[it] if (it is not ba.HALLUCINATION and it >= 0) else 0.0
        return out
    off0 = 0 * k_cb + n_special
    off1 = 1 * k_cb + n_special
    d0 = codes[:, :, 0] - off0
    d1 = codes[:, :, 1] - off1
    valid = (d0 >= 0) & (d0 < k_cb) & (d1 >= 0) & (d1 < k_cb)
    d0c = np.clip(d0, 0, k_cb - 1)
    d1c = np.clip(d1, 0, k_cb - 1)
    table = logp_cond if prior_kind == "cond2" else logp_pair
    out = table[d0c, d1c]
    return np.where(valid, out, 0.0)


def adjusted_order(scores, prior_term, alpha):
    """Re-ranking permutation int64[U,C] by adjusted = scores - alpha*prior (stable)."""
    adjusted = scores - alpha * prior_term
    return np.argsort(-adjusted, axis=1, kind="stable")


# --------------------------------------------------------------------------- #
# Per-category run + report
# --------------------------------------------------------------------------- #
def run_category(dump_dir, seeds, k, alphas, prior_kind, weighting, eps, n_results):
    meta, pop, _emb, t2i = load_support(dump_dir)
    L, n_special, n_catalog = meta["L"], meta["n_special"], meta["n_catalog"]
    k_cb = max(int(x) for key in t2i for x in key.split(",")) + 1
    logp_cond, logp_pair, logp_item = build_prefix_prior(t2i, pop, L, k_cb, weighting, eps)

    rows = []
    for seed in seeds:
        codes, recs, targets, scores = load_marius_dump(dump_dir, seed, meta, t2i, n_results)
        if scores is None:
            raise SystemExit(
                f"seed {seed} dump has no 'scores' key; pmi_rerank needs a scored dump "
                f"(dump_topk.py --with-scores). Committed depth-20 dumps are unscored.")
        prior_term = prefix_prior_term(codes, recs, logp_cond, logp_pair, logp_item,
                                       prior_kind, L, k_cb, n_special)
        for alpha in alphas:
            order = adjusted_order(scores, prior_term, alpha)
            recs_adj = apply_order(recs, order)
            m = evaluate(recs_adj, targets, pop, n_catalog, k)
            m.update({"seed": seed, "alpha": float(alpha)})
            rows.append(m)
    return pd.DataFrame(rows), meta, k_cb


def summarize(df):
    """Mean over seeds per alpha. Returns a tidy DataFrame indexed by alpha."""
    return df.groupby("alpha")[METRICS].mean().reset_index()


def fmt_table(means):
    cols = ["alpha"] + METRICS
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in means.iterrows():
        vals = [f"{r['alpha']:.2f}"] + [f"{r[m]:.4f}" for m in METRICS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(out_path, category, means, k, prior_kind, weighting, eps, n_results, seeds, k_cb):
    base = means[means["alpha"] == 0.0]
    base_recall = float(base["recall"].iloc[0]) if len(base) else float("nan")
    body = f"""# Label-bias (PMI) decoder pilot: {category}

Logit-adjusted re-ranking of MARIUS's beam candidates (C = {n_results} tuples per
user). adjusted(tuple) = log p(tuple|history) - alpha * log p_hat(prior), with the
prior estimated from training demand. Prior kind = {prior_kind}
(cond2 = within-prefix conditional p_hat(digit1|digit0); pair = joint prefix
p_hat(digit0,digit1); item = global item popularity), weighting = {weighting},
Laplace eps = {eps}, codebook size K_cb = {k_cb}. Metrics at k = {k}, seeds {sorted(seeds)}.
alpha = 0 reproduces the beam's own ranking on the candidate pool (baseline
recall = {base_recall:.4f}).

Interpretation: subtracting a NEGATIVE log-prior boosts rare within-prefix
continuations, so increasing alpha trades top-of-list accuracy for catalog reach.
The decisive question (the gate vs the MBR Pareto): on Sports, is there an alpha at
which recall matches an MBR tau setting while delivering STRICTLY higher coverage
AND tail_recall? If yes, the mechanism-matched prior beats the generic consensus dial.

## Mean over seeds (k = {k})

{fmt_table(means)}

For recall/ndcg/coverage/aplt/tail_recall/novelty/entropy_norm higher is better;
for gini/arp/hallucination_rate lower is better.

Generated by scripts/extensions/pmi_rerank.py (numpy-only; reuses mbr.py loaders +
metric suite and compute_beyond_accuracy.py).
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body)


# --------------------------------------------------------------------------- #
# Selftest (pure numpy, no dumps)
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    L, k_cb, n_special = 4, 4, 2
    # 4 items. Prefix family d0=0 has a popular sibling (d1=0, pop 100) and a rare
    # one (d1=1, pop 1); family d0=1 has one item.
    #   item0: codes 0,0,0,0 pop 100   (common within d0=0)
    #   item1: codes 0,1,0,0 pop 1     (rare within d0=0)
    #   item2: codes 1,0,0,0 pop 50
    #   item3: codes 1,1,0,0 pop 50
    t2i = {"0,0,0,0": 0, "0,1,0,0": 1, "1,0,0,0": 2, "1,1,0,0": 3}
    pop = np.array([100, 1, 50, 50], dtype=np.float64)
    logp_cond, logp_pair, logp_item = build_prefix_prior(t2i, pop, L, k_cb, "demand", eps=1e-6)
    # p_hat(d1=0|d0=0) ~ 100/101 (>> p_hat(d1=1|d0=0) ~ 1/101), so cond prior of the
    # popular sibling is much larger (closer to 0) than the rare one.
    assert logp_cond[0, 0] > logp_cond[0, 1], (logp_cond[0, 0], logp_cond[0, 1])

    def tok(raw):
        return [raw[l] + l * k_cb + n_special for l in range(L)]

    # one user, candidates [item0 (common), item1 (rare)], EQUAL model joint score.
    codes = np.array([[tok([0, 0, 0, 0]), tok([0, 1, 0, 0])]], dtype=np.int64)
    recs = [[0, 1]]
    scores = np.array([[5.0, 5.0]], dtype=np.float64)
    prior = prefix_prior_term(codes, recs, logp_cond, logp_pair, logp_item, "cond2", L, k_cb, n_special)
    # alpha = 0: ties -> stable keeps beam order [0, 1]
    assert adjusted_order(scores, prior, 0.0)[0].tolist() == [0, 1]
    # alpha > 0: rare sibling (item1) gets the bigger boost and leads
    assert adjusted_order(scores, prior, 1.0)[0].tolist() == [1, 0], \
        adjusted_order(scores, prior, 1.0)[0].tolist()
    # item prior: equal-prefix but pop differs -> popular item0 has higher logp_item,
    # subtracting it demotes item0, so order flips the same way.
    prior_item = prefix_prior_term(codes, recs, logp_cond, logp_pair, logp_item, "item", L, k_cb, n_special)
    assert prior_item[0, 0] > prior_item[0, 1]
    assert adjusted_order(scores, prior_item, 1.0)[0].tolist() == [1, 0]
    # out-of-range digit (special token) -> prior 0 (no adjustment)
    bad = np.array([[[0, 0, 0, 0]]], dtype=np.int64)  # token 0 < n_special -> raw d0 = -2
    p = prefix_prior_term(bad, [[ba.HALLUCINATION]], logp_cond, logp_pair, logp_item, "cond2", L, k_cb, n_special)
    assert p[0, 0] == 0.0, p
    print("OK: pmi_rerank selftest passed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Label-bias / logit-adjusted re-ranking of MARIUS beams")
    p.add_argument("--dump-root", type=Path, default=REPO / "reports" / "extensions" / "topk")
    p.add_argument("--category", default="Sports_and_Outdoors")
    p.add_argument("--seeds", default="42 43 44")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-results", type=int, default=100,
                   help="dump depth C; reads {model}_seed{s}_topk{C}.npz (100 -> topk100.npz)")
    p.add_argument("--prior", choices=["cond2", "pair", "item"], default="cond2")
    p.add_argument("--weighting", choices=["demand", "count"], default="demand")
    p.add_argument("--eps", type=float, default=1.0, help="Laplace smoothing for the prior")
    p.add_argument("--alphas", default="0 0.25 0.5 0.75 1.0 1.25 1.5")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()

    dump_dir = args.dump_root / args.category
    seeds = [int(s) for s in args.seeds.split()]
    alphas = [float(a) for a in args.alphas.split()]
    df, meta, k_cb = run_category(dump_dir, seeds, args.k, alphas, args.prior,
                                  args.weighting, args.eps, args.n_results)
    means = summarize(df)

    print(f"{args.category}: prior={args.prior} weighting={args.weighting} k={args.k} C={args.n_results}")
    print(fmt_table(means))

    stem = f"pmi_{args.category}_{args.prior}_{args.weighting}"
    out = args.out or (REPO / "reports" / "extensions" / "pmi" / f"{stem}.md")
    write_report(out, meta["category"], means, args.k, args.prior, args.weighting,
                 args.eps, args.n_results, seeds, k_cb)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)
    means.to_csv(out.with_name(f"{stem}_pareto.csv"), index=False)
    print(f"\nWrote {out}\nWrote {out.with_suffix('.csv')}\nWrote {out.with_name(stem + '_pareto.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
