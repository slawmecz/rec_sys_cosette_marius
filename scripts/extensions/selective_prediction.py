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
# Cross-paradigm comparison (MARIUS vs SASRec) + multi-seed aggregation.
# Requires the SCORED depth-100 dumps produced by jobs/28_rescore_for_selective_*.sbatch:
#   sasrec_seed<s>_topk100.npz (keys: topk_items, target_item, hist_len, scores)
#   marius_seed<s>_topk100.npz (keys: topk_codes, target_codes, hist_len, scores)
# ---------------------------------------------------------------------------

SEEDS_TRY = [42, 43, 44]


def rank_and_hits_sasrec(topk_items, target_item, scores, k=K):
    """SASRec hit@k / ndcg@k / confidence from a dense scored dump.

    topk_items (U, C) item ids, target_item (U,), scores (U, C) dot-product logits.
    Candidates are ranked by descending score; confidence is the rank-1 (max) score.
    Note SASRec's max dot-product is NOT a normalized likelihood (unlike MARIUS), which
    is exactly the asymmetry this comparison probes.
    """
    order = np.argsort(-scores, axis=1)
    items_ranked = np.take_along_axis(topk_items, order, axis=1)
    scores_sorted = np.take_along_axis(scores, order, axis=1)
    match = items_ranked == target_item[:, None]  # (U, C)
    topk_match = match[:, :k]
    hit = topk_match.any(axis=1)
    rank = np.where(hit, topk_match.argmax(axis=1), -1)
    safe_rank = np.maximum(rank, 0)
    ndcg = np.where(rank >= 0, 1.0 / np.log2(safe_rank.astype(np.float64) + 2.0), 0.0)
    conf = scores_sorted[:, 0]
    return hit, ndcg, conf


def _marius_metrics(category, seed):
    p = TOPK_DIR / category / f"marius_seed{seed}_topk100.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if "scores" not in d.files:
        return None
    hit, ndcg, conf, _m = rank_and_hits(
        d["topk_codes"], d["target_codes"], d["scores"].astype(np.float64), k=K)
    return {"hit": hit, "ndcg": ndcg, "conf": conf}


def _sasrec_metrics(category, seed):
    p = TOPK_DIR / category / f"sasrec_seed{seed}_topk100.npz"
    if not p.exists():
        return None
    d = np.load(p)
    if "scores" not in d.files:
        return None
    hit, ndcg, conf = rank_and_hits_sasrec(
        d["topk_items"], d["target_item"], d["scores"].astype(np.float64), k=K)
    return {"hit": hit, "ndcg": ndcg, "conf": conf}


def _agg_curves(per_seed_curves):
    """per_seed_curves: list of selective_curve dicts (keyed by coverage float).
    Returns {"<cov>": {mean, std, n_seeds}} aggregated across seeds."""
    out = {}
    for c in COVERAGES:
        vals = np.array([d[c] for d in per_seed_curves], dtype=np.float64)
        out[f"{c:.2f}"] = {"mean": float(vals.mean()),
                           "std": float(vals.std(ddof=0)),
                           "n_seeds": int(vals.size)}
    return out


def analyse_cross(category, out_dir=OUT_DIR):
    """Cross-paradigm selective comparison + multi-seed aggregation.

    Each model abstains on its OWN low-confidence users, so the curves are compared
    head-to-head at matched answered-coverage (no per-user alignment needed). Degrades
    gracefully: if a model's scored dumps are absent, it is skipped and the head-to-head
    is reported as unavailable.
    """
    methods = {"marius": _marius_metrics, "sasrec": _sasrec_metrics}
    found = {}
    per_method = {}
    for name, fn in methods.items():
        seeds, curves_hit, aurcs, bases = [], [], [], []
        for s in SEEDS_TRY:
            m = fn(category, s)
            if m is None:
                continue
            seeds.append(s)
            hit = m["hit"].astype(np.float64)
            curves_hit.append(selective_curve(hit, m["conf"]))
            a_conf, a_rand = aurc(1.0 - hit, m["conf"])
            aurcs.append((a_conf, a_rand))
            bases.append(float(hit.mean()))
        found[name] = seeds
        if seeds:
            per_method[name] = {
                "seeds": seeds,
                "base_hit@10_mean": float(np.mean(bases)),
                "selective_hit@10": _agg_curves(curves_hit),
                "aurc_mean": float(np.mean([a for a, _ in aurcs])),
                "aurc_random_mean": float(np.mean([r for _, r in aurcs])),
            }

    summary = {"category": category, "k": K, "found_seeds": found, "per_method": per_method}
    if "marius" in per_method and "sasrec" in per_method:
        h2h = {}
        for c in COVERAGES:
            mk = f"{c:.2f}"
            mm = per_method["marius"]["selective_hit@10"][mk]["mean"]
            ss = per_method["sasrec"]["selective_hit@10"][mk]["mean"]
            h2h[mk] = {"marius": mm, "sasrec": ss, "marius_minus_sasrec": mm - ss}
        summary["head_to_head_selective_hit@10"] = h2h
        summary["marius_aurc_beats_sasrec"] = bool(
            per_method["marius"]["aurc_mean"] < per_method["sasrec"]["aurc_mean"])

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{category}_cross_paradigm.json").write_text(json.dumps(summary, indent=2))
    return summary


def _print_cross(s):
    print(f"\n=== CROSS-PARADIGM {s['category']} (seeds found: {s['found_seeds']}) ===")
    for name, pm in s.get("per_method", {}).items():
        sc = pm["selective_hit@10"]
        line = "  ".join(f"{c}->{sc[c]['mean']:.4f}+-{sc[c]['std']:.4f}"
                          for c in ["1.00", "0.10", "0.05"])
        print(f"  {name}: base={pm['base_hit@10_mean']:.4f}  sel {line}  "
              f"AURC={pm['aurc_mean']:.4f}")
    if "head_to_head_selective_hit@10" in s:
        h = s["head_to_head_selective_hit@10"]
        print("  head-to-head (MARIUS - SASRec) selective-Hit@10: "
              + "  ".join(f"{c}:{h[c]['marius_minus_sasrec']:+.4f}" for c in ["1.00", "0.10", "0.05"]))
        print(f"  MARIUS AURC beats SASRec: {s['marius_aurc_beats_sasrec']} "
              f"(honest: may be flat/negative since MARIUS absolute recall is lower)")
    else:
        missing = [m for m in ["marius", "sasrec"] if m not in s.get("per_method", {})]
        print(f"  head-to-head UNAVAILABLE; missing scored depth-100 dumps for: {missing}. "
              f"Run jobs/28_rescore_for_selective_*.sbatch on Snellius first.")


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

    # SASRec hit@k by item-id match, ranked by descending score.
    topk_items = np.array([[5, 2, 7, 1], [9, 3, 4, 8]])
    target_item = np.array([7, 9])
    sc = np.array([[0.9, 0.5, 0.8, 0.1], [0.2, 0.3, 0.1, 0.7]], dtype=np.float64)
    hit_s, ndcg_s, conf_s = rank_and_hits_sasrec(topk_items, target_item, sc, k=4)
    # row0 ranked [5,7,2,1] -> target 7 at rank 1 -> ndcg=1/log2(3); row1 ranked [8,3,9,4]
    # -> target 9 at rank 2 -> ndcg=1/log2(4)=0.5.
    assert hit_s.all(), "sasrec hits wrong"
    assert np.allclose(conf_s, [0.9, 0.7]), f"sasrec confidence wrong: {conf_s}"
    assert abs(ndcg_s[0] - 1.0 / np.log2(3.0)) < 1e-9 and abs(ndcg_s[1] - 0.5) < 1e-9, "sasrec ndcg wrong"
    miss_target = np.array([99, 99])  # target not in candidates -> miss
    hit_m, ndcg_m, _ = rank_and_hits_sasrec(topk_items, miss_target, sc, k=4)
    assert (~hit_m).all() and (ndcg_m == 0).all(), "sasrec miss must be 0"

    # Multi-seed curve aggregation: three seeds with per-coverage values 0.1/0.2/0.3.
    per_seed = [dict((c, 0.1 * i) for c in COVERAGES) for i in (1, 2, 3)]
    agg = _agg_curves(per_seed)
    assert abs(agg["1.00"]["mean"] - 0.2) < 1e-9 and agg["1.00"]["n_seeds"] == 3, "agg mean/n wrong"
    assert abs(agg["1.00"]["std"] - np.std([0.1, 0.2, 0.3])) < 1e-9, "agg std wrong"

    print("  selective_prediction selftest: PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", choices=CATEGORIES, default=None,
                    help="dataset; default runs both")
    ap.add_argument("--selftest", action="store_true", help="synthetic checks, no data")
    ap.add_argument("--cross-paradigm", action="store_true",
                    help="MARIUS-vs-SASRec selective comparison + multi-seed aggregation "
                         "(needs the scored depth-100 dumps from jobs/28)")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return 0
    cats = [args.category] if args.category else CATEGORIES
    if args.cross_paradigm:
        for cat in cats:
            _print_cross(analyse_cross(cat))
        print(f"\nWrote cross-paradigm summaries to {OUT_DIR}")
        return 0
    for cat in cats:
        s = analyse(cat)
        _print_summary(s)
    print(f"\nWrote summaries + risk-coverage CSVs to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
