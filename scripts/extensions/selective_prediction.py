#!/usr/bin/env python3
"""Selective recommendation via the frozen MARIUS joint log-prob (zero-GPU).

A positive-outcome analysis: MARIUS's autoregressive decode produces a NORMALIZED
joint code-tuple log-probability that is comparable across users, so it doubles as a
per-user reliability signal. Ranking users by that confidence and abstaining on the
low-confidence tail yields a steep risk-coverage gain -- a capability the discriminative
SASRec++ dot-product (an unnormalized score) does not provide by construction.

This script is read-only and torch-free: it operates on the committed depth-100 scored
dumps (reports/extensions/topk/<category>/marius_seed42_topk100.npz, key 'scores' =
joint log-prob over the 100 beam candidates). It computes, for MARIUS alone:
  - the risk-coverage / selective-Hit@10 (+ selective-NDCG@10) curve,
  - AURC (area under the miss-at-10 risk-coverage curve) vs the random-ordering baseline,
  - the confound battery that is the load-bearing novelty defense:
      (1) corr(confidence, history length) and a history-length router for comparison,
      (2) within-history-bucket selective lift (a per-instance difficulty signal, not a
          cold/warm sort),
      (3) target-popularity confound: corr(confidence, log target-popularity) and the
          tail-lift vs head-lift split (tail-lift > head-lift => NOT a popularity proxy),
      (4) top-1 log-prob vs the margin (top1 - top2) as the confidence signal.

Honest scope: the cross-paradigm headline (does the frozen likelihood beat SASRec at
matched answered-coverage) is NOT computed here because SASRec scores are not dumped;
that needs a cheap SASRec --with-scores re-dump. Numbers here are single-seed (seed 42),
the only seed with scored depth-100 dumps.

  python scripts/extensions/selective_prediction.py --category Beauty
  python scripts/extensions/selective_prediction.py            # both datasets
  python scripts/extensions/selective_prediction.py --selftest # synthetic checks, no data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOPK_DIR = REPO / "reports" / "extensions" / "topk"
OUT_DIR = REPO / "reports" / "extensions" / "selective_prediction"

CATEGORIES = ["Beauty", "Sports_and_Outdoors"]
COVERAGES = [1.0, 0.5, 0.25, 0.10, 0.05]
K = 10
K_CB = 256
N_SPECIAL = 2  # len(SpecialTokens): PAD, BOS
LEVEL_OFFSET = np.arange(4, dtype=np.int64) * K_CB + N_SPECIAL  # [2, 258, 514, 770]


# ---------------------------------------------------------------------------
# Core ranking + selective metrics (pure numpy).
# ---------------------------------------------------------------------------

def rank_and_hits(topk_codes, target_codes, scores, k=K):
    """Return (hit_at_k[U] bool, ndcg_at_k[U] float, confidence[U], margin[U]).

    Candidates are ranked by descending score (the joint log-prob). hit_at_k is whether
    the target code tuple is among the top-k candidates by that ranking; ndcg_at_k is
    1/log2(rank+2) for a single relevant item (0 if outside top-k). confidence is the
    rank-1 (max) score; margin is rank-1 minus rank-2 score.
    """
    U, C, _L = topk_codes.shape
    matches = (topk_codes == target_codes[:, None, :]).all(axis=2)  # (U, C)
    order = np.argsort(-scores, axis=1)  # descending score; (U, C)
    matches_ranked = np.take_along_axis(matches, order, axis=1)  # (U, C)
    scores_sorted = np.take_along_axis(scores, order, axis=1)  # (U, C)

    topk_match = matches_ranked[:, :k]  # (U, k)
    hit = topk_match.any(axis=1)
    # rank of the (first) match within top-k; -1 if none.
    rank = np.where(topk_match.any(axis=1), topk_match.argmax(axis=1), -1)
    safe_rank = np.maximum(rank, 0)  # avoid log2(1)=0 in the masked-out branch
    ndcg = np.where(rank >= 0, 1.0 / np.log2(safe_rank.astype(np.float64) + 2.0), 0.0)

    confidence = scores_sorted[:, 0]
    margin = scores_sorted[:, 0] - scores_sorted[:, 1] if C >= 2 else np.zeros(U)
    return hit, ndcg, confidence, margin


def selective_curve(metric, signal, coverages=COVERAGES):
    """Mean of `metric` over the top-`c` fraction of instances by descending `signal`."""
    U = len(metric)
    order = np.argsort(-signal)  # high signal first
    metric_ord = metric[order]
    out = {}
    for c in coverages:
        n = max(1, int(round(c * U)))
        out[c] = float(metric_ord[:n].mean())
    return out


def aurc(risk, signal):
    """Area under the risk-coverage curve (risk averaged over coverage in (0,1]).

    Lower is better. A random selector has expected risk == overall risk at every
    coverage, so its AURC equals the overall risk (returned as the baseline).
    """
    U = len(risk)
    order = np.argsort(-signal)  # most-confident first
    risk_ord = risk[order]
    cum = np.cumsum(risk_ord) / np.arange(1, U + 1)  # risk over the top-n subset
    coverage = np.arange(1, U + 1) / U
    trapz = getattr(np, "trapezoid", np.trapz)  # np>=2 renamed trapz -> trapezoid
    area = float(trapz(cum, coverage))  # integral over coverage in (0,1]
    return area, float(risk.mean())


# ---------------------------------------------------------------------------
# Popularity confound helpers.
# ---------------------------------------------------------------------------

def target_to_catalog_idx(target_codes, tuple_to_item):
    """Map each target code tuple to its catalog index via tuple_to_item.json.

    Handles both token-space (offset) and raw-code dumps by de-offsetting when the
    values exceed the per-level codebook range. Returns int array (-1 where unmapped).
    """
    tc = target_codes.copy()
    if tc.max() >= K_CB:  # token space -> de-offset to raw [0,255]
        tc = tc - LEVEL_OFFSET
    out = np.full(tc.shape[0], -1, dtype=np.int64)
    for i in range(tc.shape[0]):
        key = ",".join(str(int(v)) for v in tc[i])
        j = tuple_to_item.get(key)
        if j is not None:
            out[i] = int(j)
    return out


# ---------------------------------------------------------------------------
# Per-dataset analysis.
# ---------------------------------------------------------------------------

def analyse(category, out_dir=OUT_DIR):
    ddir = TOPK_DIR / category
    dump = np.load(ddir / "marius_seed42_topk100.npz")
    topk_codes = dump["topk_codes"]
    target_codes = dump["target_codes"]
    scores = dump["scores"].astype(np.float64)
    hist_len = dump["hist_len"]
    U = topk_codes.shape[0]

    hit, ndcg, conf, margin = rank_and_hits(topk_codes, target_codes, scores, k=K)
    miss = 1.0 - hit.astype(np.float64)

    # Risk-coverage by confidence.
    hit_cov = selective_curve(hit.astype(np.float64), conf)
    ndcg_cov = selective_curve(ndcg, conf)
    aurc_conf, aurc_random = aurc(miss, conf)

    # Confound 1: history-length router + correlation.
    corr_conf_hist = float(np.corrcoef(conf, hist_len.astype(np.float64))[0, 1])
    hit_cov_hist = selective_curve(hit.astype(np.float64), hist_len.astype(np.float64))

    # Confound 1b: margin router (top1 - top2).
    hit_cov_margin = selective_curve(hit.astype(np.float64), margin)

    # Confound 2: within-history-bucket selective lift at 25% coverage.
    buckets = {"cold(hist<=5)": hist_len <= 5,
               "mid(6-15)": (hist_len >= 6) & (hist_len <= 15),
               "warm(hist>=16)": hist_len >= 16}
    within_bucket = {}
    for name, mask in buckets.items():
        if mask.sum() < 20:
            within_bucket[name] = None
            continue
        base = float(hit[mask].mean())
        sel = selective_curve(hit[mask].astype(np.float64), conf[mask], [0.25])[0.25]
        within_bucket[name] = {"n": int(mask.sum()), "base_hit@10": base,
                               "sel@25%": sel, "lift": (sel / base) if base > 0 else None}

    # Confound 3: target-popularity. tail-lift vs head-lift.
    pop_info = {"available": False}
    pop_path = ddir / "popularity.npy"
    t2i_path = ddir / "tuple_to_item.json"
    if pop_path.exists() and t2i_path.exists():
        popularity = np.load(pop_path)
        tuple_to_item = json.loads(t2i_path.read_text())
        cat_idx = target_to_catalog_idx(target_codes, tuple_to_item)
        mapped = cat_idx >= 0
        if mapped.sum() > 100 and int(cat_idx[mapped].max()) < len(popularity):
            tgt_pop = np.full(U, np.nan)
            tgt_pop[mapped] = popularity[cat_idx[mapped]].astype(np.float64)
            valid = mapped & np.isfinite(tgt_pop) & (tgt_pop > 0)
            corr_conf_logpop = float(
                np.corrcoef(conf[valid], np.log(tgt_pop[valid]))[0, 1]
            )
            med = np.median(tgt_pop[valid])
            head = valid & (tgt_pop >= med)
            tail = valid & (tgt_pop < med)
            def _lift(mask):
                base = float(hit[mask].mean())
                sel = selective_curve(hit[mask].astype(np.float64), conf[mask], [0.10])[0.10]
                return base, sel, (sel / base if base > 0 else None)
            hb, hs, hl = _lift(head)
            tb, ts, tl = _lift(tail)
            pop_info = {"available": True, "mapped_frac": float(mapped.mean()),
                        "corr_conf_logpop": corr_conf_logpop,
                        "head_lift@10%": {"base": hb, "sel": hs, "lift": hl},
                        "tail_lift@10%": {"base": tb, "sel": ts, "lift": tl},
                        "tail_lift_exceeds_head": (tl is not None and hl is not None and tl > hl)}

    summary = {
        "category": category, "seed": 42, "n_instances": int(U), "k": K,
        "base_hit@10": float(hit.mean()), "base_ndcg@10": float(ndcg.mean()),
        "selective_hit@10_by_coverage": {f"{c:.2f}": hit_cov[c] for c in COVERAGES},
        "selective_ndcg@10_by_coverage": {f"{c:.2f}": ndcg_cov[c] for c in COVERAGES},
        "aurc_miss@10": aurc_conf, "aurc_random": aurc_random,
        "aurc_beats_random": aurc_conf < aurc_random,
        "confounds": {
            "corr_conf_histlen": corr_conf_hist,
            "histlen_router_hit@10_by_coverage": {f"{c:.2f}": hit_cov_hist[c] for c in COVERAGES},
            "conf_dominates_histlen_at_all_coverages": all(
                hit_cov[c] >= hit_cov_hist[c] for c in COVERAGES),
            "margin_router_hit@10_by_coverage": {f"{c:.2f}": hit_cov_margin[c] for c in COVERAGES},
            "conf_beats_margin_at_all_coverages": all(
                hit_cov[c] >= hit_cov_margin[c] for c in COVERAGES),
            "within_history_bucket": within_bucket,
            "target_popularity": pop_info,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{category}_summary.json").write_text(json.dumps(summary, indent=2))
    # Flat risk-coverage CSV.
    lines = ["coverage,n,selective_hit@10,selective_ndcg@10,histlen_router_hit@10"]
    for c in COVERAGES:
        n = max(1, int(round(c * U)))
        lines.append(f"{c},{n},{hit_cov[c]:.6f},{ndcg_cov[c]:.6f},{hit_cov_hist[c]:.6f}")
    (out_dir / f"{category}_risk_coverage.csv").write_text("\n".join(lines) + "\n")
    return summary


def _print_summary(s):
    print(f"\n=== {s['category']} (seed {s['seed']}, n={s['n_instances']}) ===")
    print(f"  base Hit@10={s['base_hit@10']:.4f}  NDCG@10={s['base_ndcg@10']:.4f}")
    sc = s["selective_hit@10_by_coverage"]
    print("  selective Hit@10:  " + "  ".join(f"{c}->{sc[c]:.4f}" for c in ["1.00", "0.50", "0.25", "0.10", "0.05"]))
    lift5 = sc["0.05"] / s["base_hit@10"] if s["base_hit@10"] > 0 else float("nan")
    print(f"  lift at 5% coverage = {lift5:.2f}x")
    print(f"  AURC(miss@10)={s['aurc_miss@10']:.4f} vs random {s['aurc_random']:.4f} "
          f"(beats_random={s['aurc_beats_random']})")
    cf = s["confounds"]
    print(f"  corr(conf,histlen)={cf['corr_conf_histlen']:.3f}  "
          f"conf>histlen all cov={cf['conf_dominates_histlen_at_all_coverages']}  "
          f"conf>margin all cov={cf['conf_beats_margin_at_all_coverages']}")
    pop = cf["target_popularity"]
    if pop.get("available"):
        print(f"  corr(conf,log pop)={pop['corr_conf_logpop']:.3f}  "
              f"tail_lift={pop['tail_lift@10%']['lift']}  head_lift={pop['head_lift@10%']['lift']}  "
              f"tail>head={pop['tail_lift_exceeds_head']}")


# ---------------------------------------------------------------------------
# Self-test (synthetic, no data / no GPU).
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.RandomState(0)
    U, C, L = 400, 20, 4
    # Construct: high-confidence instances are hits, low-confidence are misses.
    target = rng.randint(0, K_CB, size=(U, L))
    topk = rng.randint(0, K_CB, size=(U, C, L))
    scores = rng.randn(U, C)
    conf_rank = np.argsort(-scores.max(axis=1))  # most confident first
    # Plant the target at rank 0 for the top-half-confidence instances only.
    for idx in conf_rank[: U // 2]:
        top_cand = np.argmax(scores[idx])
        topk[idx, top_cand] = target[idx]
    hit, ndcg, conf, margin = rank_and_hits(topk, target, scores, k=K)
    cov = selective_curve(hit.astype(float), conf)
    assert cov[0.05] > cov[1.0] > 0, f"selective curve must rise: {cov}"
    assert cov[0.10] >= cov[0.50], "more abstention should not lower Hit@10 here"
    a_conf, a_rand = aurc(1.0 - hit.astype(float), conf)
    assert a_conf < a_rand, f"AURC must beat random: {a_conf} vs {a_rand}"
    # ndcg at full coverage must be in (0,1].
    assert 0 < ndcg.mean() <= 1.0
    # de-offset round trip
    tok = np.array([[2, 258, 514, 770], [3, 259, 515, 771]], dtype=np.int64)
    deoff = tok - LEVEL_OFFSET
    assert (deoff == np.array([[0, 0, 0, 0], [1, 1, 1, 1]])).all(), "de-offset wrong"
    print("  selective_prediction selftest: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", choices=CATEGORIES, default=None,
                    help="dataset; default runs both")
    ap.add_argument("--selftest", action="store_true", help="synthetic checks, no data")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    cats = [args.category] if args.category else CATEGORIES
    for cat in cats:
        s = analyse(cat)
        _print_summary(s)
    print(f"\nWrote summaries + risk-coverage CSVs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
