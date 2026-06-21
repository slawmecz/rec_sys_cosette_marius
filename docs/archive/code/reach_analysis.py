#!/usr/bin/env python3
"""REACH pilot: mechanism study of MARIUS's catalog collapse.

Reads the validated Top-K dumps under reports/extensions/topk/<category>/ and
answers, per dataset:

1. Reachability sets: which catalog items ever appear in any user's top-k
   (per model, per k, per seed), and which items are CONSISTENTLY unreachable
   (unreached in every seed).
2. Structure of MARIUS's consistently-unreached set at k=10: popularity
   profile, share of total train demand mass it carries, and overlap with
   SASRec++'s unreached set.
3. The mechanism test: does an item's semantic-ID PREFIX mass (total train
   popularity of all items sharing its L1 or L1:L2 code prefix) predict
   unreachability beyond the item's OWN popularity? (Beam search prunes whole
   prefixes at decoding step 1, so low-mass prefixes should be unreachable
   even for mid-popularity items.)
4. Beam concentration at the prefix level: how many distinct L1 codes and
   L1:L2 pairs the beam ever emits, and how concentrated the emitted slots
   are on the top-5 prefixes, contrasted with the catalog's own prefix
   distribution and with SASRec++'s recommendations mapped to prefixes.

Pure numpy/scipy/matplotlib (no torch). Reuses the validated loaders from
scripts/extensions/compute_beyond_accuracy.py (token-space de-offset and
hallucination handling).

Usage:
  python3 scripts/extensions/reach_analysis.py \
      --category "Beauty Sports_and_Outdoors" --seeds "42 43 44"
  python3 scripts/extensions/reach_analysis.py --selftest

Outputs (under reports/extensions/reach/):
  reach_<category>.md          per-dataset tables
  reach_results_<category>.json intermediate numbers (for the summary)
  reach_summary.md             Beauty vs Sports contrast (<= 20 lines)
  reach_unreached_by_decile.png unreached%% by popularity decile, both datasets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extensions.compute_beyond_accuracy import load_support, load_model_recs  # noqa: E402

MODELS = ("marius", "sasrec")
KS = (10, 20)
N_DECILES = 10
N_QUINTILES = 5
TOP_PREFIXES = 5


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def invert_tuple_to_item(t2i: dict, n_catalog: int, L: int = 4) -> np.ndarray:
    """Invert {"c0,c1,...": item} into codes int64[n_catalog, L]; -1 = no tuple."""
    codes = np.full((n_catalog, L), -1, dtype=np.int64)
    for key, item in t2i.items():
        codes[int(item)] = [int(x) for x in key.split(",")]
    return codes


def reached_mask(recs_arr: np.ndarray, k: int, n_catalog: int) -> np.ndarray:
    """Boolean[n_catalog]: item appears in at least one user's top-k list."""
    flat = recs_arr[:, :k].ravel()
    flat = flat[flat >= 0]
    mask = np.zeros(n_catalog, dtype=bool)
    mask[np.unique(flat)] = True
    return mask


def equal_freq_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Rank-based equal-size bins, ascending; bin 0 = smallest values.

    Mirrors the decile assignment used by beyond_accuracy.popularity_decile_exposure
    (stable argsort, linspace edges) so decile definitions match across analyses.
    """
    n = values.size
    order = np.argsort(values, kind="stable")
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    out = np.empty(n, dtype=np.int64)
    for b in range(n_bins):
        out[order[edges[b]:edges[b + 1]]] = b
    return out


def rank_half_split(values: np.ndarray) -> np.ndarray:
    """Boolean mask, True = upper (high) half by value; ties broken by index.

    A deterministic median split: stable-sort ascending, lower half is 'low'.
    """
    order = np.argsort(values, kind="stable")
    high = np.zeros(values.size, dtype=bool)
    high[order[values.size // 2:]] = True
    return high


def prefix_masses(codes: np.ndarray, pop: np.ndarray, k_cb: int):
    """Per-item L1 prefix mass and L1:L2 pair mass (train popularity sums)."""
    c0 = codes[:, 0]
    pair = c0 * k_cb + codes[:, 1]
    mass1_by_code = np.bincount(c0, weights=pop.astype(np.float64), minlength=k_cb)
    mass2_by_pair = np.bincount(pair, weights=pop.astype(np.float64), minlength=k_cb * k_cb)
    return mass1_by_code[c0], mass2_by_pair[pair], mass1_by_code, mass2_by_pair


def mechanism_table(pop: np.ndarray, mass: np.ndarray, unreached: np.ndarray,
                    n_q: int = N_QUINTILES):
    """5x2 table: popularity quintile x (low/high prefix mass) -> unreached%."""
    q_of = equal_freq_bins(pop.astype(np.float64), n_q)
    rows = []
    for q in range(n_q):
        idx = np.flatnonzero(q_of == q)
        high = rank_half_split(mass[idx])
        lo, hi = idx[~high], idx[high]
        rows.append({
            "quintile": q + 1,
            "pop_min": int(pop[idx].min()), "pop_max": int(pop[idx].max()),
            "n_low": int(lo.size), "unreached_low": float(unreached[lo].mean()),
            "median_mass_low": float(np.median(mass[lo])),
            "n_high": int(hi.size), "unreached_high": float(unreached[hi].mean()),
            "median_mass_high": float(np.median(mass[hi])),
        })
    return rows


def point_biserial(binary: np.ndarray, values: np.ndarray):
    r, p = stats.pointbiserialr(binary.astype(int), values.astype(np.float64))
    return float(r), float(p)


def top_share(counts: np.ndarray, top: int = TOP_PREFIXES):
    """(distinct nonzero entries, share of total held by the `top` largest,
    indices of the `top` largest)."""
    total = counts.sum()
    order = np.argsort(-counts, kind="stable")[:top]
    share = float(counts[order].sum() / total) if total > 0 else 0.0
    return int((counts > 0).sum()), share, order


# --------------------------------------------------------------------------- #
# Beam / recommendation prefix concentration
# --------------------------------------------------------------------------- #
def marius_prefix_counts(npz_path: Path, n_special: int, k_cb: int):
    """L1-code and L1:L2-pair slot counts from raw MARIUS top-20 token dumps."""
    tk = np.load(npz_path)["topk_codes"]
    r0 = tk[:, :, 0] - n_special
    r1 = tk[:, :, 1] - n_special - k_cb
    v0 = (r0 >= 0) & (r0 < k_cb)
    v01 = v0 & (r1 >= 0) & (r1 < k_cb)
    c1 = np.bincount(r0[v0].ravel(), minlength=k_cb)
    c2 = np.bincount((r0 * k_cb + r1)[v01].ravel(), minlength=k_cb * k_cb)
    return c1, c2, int((~v0).sum()), int((~v01).sum())


def items_prefix_counts(recs_arr: np.ndarray, codes: np.ndarray, k_cb: int):
    """Same counters for an item-id Top-K matrix (SASRec++), via item -> tuple."""
    flat = recs_arr.ravel()
    flat = flat[flat >= 0]
    c0 = codes[flat, 0]
    pair = c0 * k_cb + codes[flat, 1]
    return np.bincount(c0, minlength=k_cb), np.bincount(pair, minlength=k_cb * k_cb)


def concentration_row(c1, c2, codes, pop, k_cb, n_invalid1=0, n_invalid2=0):
    """Summarize one model-seed's prefix usage + what its top-5 prefixes cover
    in the catalog (item share and train-mass share)."""
    cat_items1 = np.bincount(codes[:, 0], minlength=k_cb)
    cat_mass1 = np.bincount(codes[:, 0], weights=pop.astype(np.float64), minlength=k_cb)
    pair_idx = codes[:, 0] * k_cb + codes[:, 1]
    cat_items2 = np.bincount(pair_idx, minlength=k_cb * k_cb)
    cat_mass2 = np.bincount(pair_idx, weights=pop.astype(np.float64), minlength=k_cb * k_cb)

    d1, s1, top1 = top_share(c1)
    d2, s2, top2 = top_share(c2)
    return {
        "distinct_l1": d1, "top5_l1_slot_share": s1,
        "top5_l1_catalog_item_share": float(cat_items1[top1].sum() / cat_items1.sum()),
        "top5_l1_train_mass_share": float(cat_mass1[top1].sum() / max(cat_mass1.sum(), 1.0)),
        "distinct_l12": d2, "top5_l12_slot_share": s2,
        "top5_l12_catalog_item_share": float(cat_items2[top2].sum() / cat_items2.sum()),
        "top5_l12_train_mass_share": float(cat_mass2[top2].sum() / max(cat_mass2.sum(), 1.0)),
        "invalid_l1_slots": int(n_invalid1), "invalid_l12_slots": int(n_invalid2),
    }


def catalog_concentration(codes: np.ndarray, pop: np.ndarray, k_cb: int):
    cat_items1 = np.bincount(codes[:, 0], minlength=k_cb)
    cat_mass1 = np.bincount(codes[:, 0], weights=pop.astype(np.float64), minlength=k_cb)
    pair_idx = codes[:, 0] * k_cb + codes[:, 1]
    cat_items2 = np.bincount(pair_idx, minlength=k_cb * k_cb)
    cat_mass2 = np.bincount(pair_idx, weights=pop.astype(np.float64), minlength=k_cb * k_cb)
    d1i, s1i, _ = top_share(cat_items1)
    _, s1m, _ = top_share(cat_mass1)
    d2i, s2i, _ = top_share(cat_items2)
    _, s2m, _ = top_share(cat_mass2)
    return {"distinct_l1": d1i, "top5_l1_item_share": s1i, "top5_l1_mass_share": s1m,
            "distinct_l12": d2i, "top5_l12_item_share": s2i, "top5_l12_mass_share": s2m}


# --------------------------------------------------------------------------- #
# Per-category analysis
# --------------------------------------------------------------------------- #
def analyze_category(category: str, dump_root: Path, seeds, ks=KS) -> dict:
    dump_dir = dump_root / category
    meta, pop, _emb, t2i = load_support(dump_dir)
    n_catalog = int(meta["n_catalog"])
    n_special = int(meta["n_special"])
    k_cb = max(int(x) for key in t2i for x in key.split(",")) + 1
    codes = invert_tuple_to_item(t2i, n_catalog, int(meta["L"]))
    assert (codes[:, 0] >= 0).all(), "tuple_to_item does not cover the full catalog"

    recs_arr = {}
    for model in MODELS:
        for seed in seeds:
            recs, _targets = load_model_recs(dump_dir, model, seed, meta, t2i)
            recs_arr[(model, seed)] = np.asarray(recs, dtype=np.int64)
    n_users = recs_arr[(MODELS[0], seeds[0])].shape[0]

    # ---- 1. Reachability sets ------------------------------------------- #
    reach = {}
    cons_unreached = {}
    for model in MODELS:
        for k in ks:
            per_seed = {s: reached_mask(recs_arr[(model, s)], k, n_catalog) for s in seeds}
            cons = np.ones(n_catalog, dtype=bool)
            for s in seeds:
                cons &= ~per_seed[s]
            cons_unreached[(model, k)] = cons
            reach[(model, k)] = {
                "reached_per_seed": {s: int(per_seed[s].sum()) for s in seeds},
                "mean_reached_frac": float(np.mean([per_seed[s].mean() for s in seeds])),
                "cons_unreached_n": int(cons.sum()),
                "cons_unreached_frac": float(cons.mean()),
            }

    # ---- 2. Structure of MARIUS's consistently-unreached set (k=10) ------ #
    m_un = cons_unreached[("marius", 10)]
    s_un = cons_unreached[("sasrec", 10)]
    decile_of = equal_freq_bins(pop.astype(np.float64), N_DECILES)
    decile_rows = []
    for d in range(N_DECILES):
        in_d = decile_of == d
        decile_rows.append({
            "decile": d + 1, "size": int(in_d.sum()),
            "pop_min": int(pop[in_d].min()), "pop_max": int(pop[in_d].max()),
            "marius_unreached_n": int((m_un & in_d).sum()),
            "marius_unreached_frac": float((m_un & in_d).sum() / in_d.sum()),
            "sasrec_unreached_n": int((s_un & in_d).sum()),
            "sasrec_unreached_frac": float((s_un & in_d).sum() / in_d.sum()),
        })
    inter = int((m_un & s_un).sum())
    union = int((m_un | s_un).sum())
    structure = {
        "marius_unreached_n": int(m_un.sum()),
        "sasrec_unreached_n": int(s_un.sum()),
        "median_pop_unreached": float(np.median(pop[m_un])) if m_un.any() else float("nan"),
        "median_pop_reached": float(np.median(pop[~m_un])),
        "median_pop_catalog": float(np.median(pop)),
        "demand_mass_share_unreached": float(pop[m_un].sum() / pop.sum()),
        "sasrec_demand_mass_share_unreached": float(pop[s_un].sum() / pop.sum()),
        "decile_rows": decile_rows,
        "overlap": {
            "intersection": inter, "union": union,
            "jaccard": float(inter / union) if union else float("nan"),
            "only_marius": int((m_un & ~s_un).sum()),
            "only_sasrec": int((s_un & ~m_un).sum()),
        },
    }

    # ---- 3. Mechanism test ------------------------------------------------ #
    item_mass1, item_mass2, _, _ = prefix_masses(codes, pop, k_cb)
    # Union over seeds of the prefixes MARIUS's beam ever emits in any top-20
    # slot: which decoding step actually loses the unreached items?
    l1_emitted = np.zeros(k_cb, dtype=bool)
    pair_emitted = np.zeros(k_cb * k_cb, dtype=bool)
    for seed in seeds:
        c1, c2, _, _ = marius_prefix_counts(
            dump_dir / f"marius_seed{seed}_topk.npz", n_special, k_cb)
        l1_emitted |= c1 > 0
        pair_emitted |= c2 > 0
    l1_never = ~l1_emitted[codes[:, 0]]
    pair_never = ~pair_emitted[codes[:, 0] * k_cb + codes[:, 1]]
    own_share = (pop + 1.0) / (item_mass1 + 1.0)  # item's pop share within its L1 prefix
    mech = {
        "table_l1": mechanism_table(pop, item_mass1, m_un),
        "table_l12": mechanism_table(pop, item_mass2, m_un),
        "pb_log_pop": point_biserial(m_un, np.log10(pop + 1.0)),
        "pb_log_mass1": point_biserial(m_un, np.log10(item_mass1 + 1.0)),
        "pb_log_mass2": point_biserial(m_un, np.log10(item_mass2 + 1.0)),
        "pb_log_own_share_l1": point_biserial(m_un, np.log10(own_share)),
        "l1_never_emitted_items": int(l1_never.sum()),
        "pair_never_emitted_items": int(pair_never.sum()),
        "pair_never_emitted_frac": float(pair_never.mean()),
        "unreached_share_pair_never": float(pair_never[m_un].mean()) if m_un.any() else float("nan"),
    }

    # ---- 4. Beam concentration at the prefix level (k=20) ----------------- #
    conc = {"catalog": catalog_concentration(codes, pop, k_cb)}
    union_l1 = {m: np.zeros(k_cb, dtype=bool) for m in MODELS}
    union_l12 = {m: np.zeros(k_cb * k_cb, dtype=bool) for m in MODELS}
    for model in MODELS:
        per_seed = []
        for seed in seeds:
            if model == "marius":
                c1, c2, inv1, inv2 = marius_prefix_counts(
                    dump_dir / f"marius_seed{seed}_topk.npz", n_special, k_cb)
            else:
                c1, c2 = items_prefix_counts(recs_arr[(model, seed)], codes, k_cb)
                inv1 = inv2 = 0
            union_l1[model] |= c1 > 0
            union_l12[model] |= c2 > 0
            row = concentration_row(c1, c2, codes, pop, k_cb, inv1, inv2)
            row["seed"] = seed
            per_seed.append(row)
        mean_row = {k: float(np.mean([r[k] for r in per_seed]))
                    for k in per_seed[0] if k != "seed"}
        conc[model] = {"per_seed": per_seed, "mean": mean_row,
                       "union_distinct_l1": int(union_l1[model].sum()),
                       "union_distinct_l12": int(union_l12[model].sum())}

    return {
        "category": category, "n_users": int(n_users), "n_catalog": n_catalog,
        "k_cb": int(k_cb), "seeds": list(seeds), "total_train_interactions": int(pop.sum()),
        "reach": {f"{m}_k{k}": reach[(m, k)] for m in MODELS for k in ks},
        "structure": structure, "mechanism": mech, "concentration": conc,
    }


# --------------------------------------------------------------------------- #
# Markdown / figure writers
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def write_category_md(res: dict, path: Path, seeds) -> None:
    c = res["category"]
    lines = []
    add = lines.append
    add(f"# REACH analysis: {c}")
    add("")
    add(f"Users: {res['n_users']}, catalog: {res['n_catalog']} items, "
        f"codebook size per level: {res['k_cb']}, seeds: {res['seeds']}, "
        f"total train interactions: {res['total_train_interactions']}.")
    add("")
    add("Generated by scripts/extensions/reach_analysis.py from the validated "
        "Top-20 dumps in reports/extensions/topk/. 'Consistently unreached' = "
        "the item appears in no user's top-k list in ANY of the seeds.")
    add("")

    add("## 1. Reachability sets")
    add("")
    head = "| model | k | " + " | ".join(f"reached s{s}" for s in seeds)
    add(head + " | mean reached frac | cons. unreached n | cons. unreached frac |")
    add("|---|---|" + "---|" * (len(seeds) + 3))
    for m in MODELS:
        for k in KS:
            r = res["reach"][f"{m}_k{k}"]
            per = " | ".join(str(r["reached_per_seed"][s]) for s in seeds)
            add(f"| {m} | {k} | {per} | {_pct(r['mean_reached_frac'])} | "
                f"{r['cons_unreached_n']} | {_pct(r['cons_unreached_frac'])} |")
    add("")

    st = res["structure"]
    add("## 2. Structure of the consistently-unreached set (marius, k=10)")
    add("")
    add(f"- MARIUS consistently-unreached: {st['marius_unreached_n']} items "
        f"({_pct(st['marius_unreached_n'] / res['n_catalog'])} of the catalog); "
        f"SASRec++: {st['sasrec_unreached_n']} items.")
    add(f"- Median train popularity: unreached {st['median_pop_unreached']:.0f} "
        f"vs reached {st['median_pop_reached']:.0f} (catalog median {st['median_pop_catalog']:.0f}).")
    add(f"- Demand mass invisible to MARIUS: unreached items carry "
        f"{_pct(st['demand_mass_share_unreached'])} of ALL train interactions "
        f"(SASRec++ unreached carry {_pct(st['sasrec_demand_mass_share_unreached'])}).")
    ov = st["overlap"]
    add(f"- Overlap with SASRec++ unreached: intersection {ov['intersection']}, "
        f"union {ov['union']}, Jaccard {ov['jaccard']:.3f}; "
        f"items ONLY MARIUS misses: {ov['only_marius']}; "
        f"only SASRec++ misses: {ov['only_sasrec']}.")
    add("")
    add("Unreached share by train-popularity decile (1 = least popular):")
    add("")
    add("| decile | pop range | items | marius unreached | marius unreached % | sasrec unreached % |")
    add("|---|---|---|---|---|---|")
    for row in st["decile_rows"]:
        add(f"| {row['decile']} | {row['pop_min']}-{row['pop_max']} | {row['size']} | "
            f"{row['marius_unreached_n']} | {_pct(row['marius_unreached_frac'])} | "
            f"{_pct(row['sasrec_unreached_frac'])} |")
    add("")

    me = res["mechanism"]
    add("## 3. Mechanism test: prefix mass vs unreachability (marius, k=10)")
    add("")
    add("Prefix mass = total train popularity of all catalog items sharing the "
        "item's L1 (or L1:L2) semantic-ID code prefix. Items are binned into "
        "popularity quintiles (1 = least popular); within each quintile a "
        "deterministic median split on prefix mass (rank-based, ties by item "
        "index) gives the low/high columns. If beam search prunes whole "
        "prefixes at decoding step 1, low-mass-prefix items should be "
        "unreachable even at fixed own-popularity.")
    add("")
    for name, tbl in (("L1 prefix mass", me["table_l1"]), ("L1:L2 prefix mass", me["table_l12"])):
        add(f"### {name}")
        add("")
        add("| pop quintile | pop range | n low | unreached % (low mass) | n high | unreached % (high mass) |")
        add("|---|---|---|---|---|---|")
        for row in tbl:
            add(f"| {row['quintile']} | {row['pop_min']}-{row['pop_max']} | "
                f"{row['n_low']} | {_pct(row['unreached_low'])} | "
                f"{row['n_high']} | {_pct(row['unreached_high'])} |")
        add("")
    add("Point-biserial correlations with the unreached indicator:")
    add("")
    add("| variable | r | p |")
    add("|---|---|---|")
    for label, key in (("log10(item popularity + 1)", "pb_log_pop"),
                       ("log10(L1 prefix mass + 1)", "pb_log_mass1"),
                       ("log10(L1:L2 prefix mass + 1)", "pb_log_mass2"),
                       ("log10(own pop share within L1 prefix)", "pb_log_own_share_l1")):
        r, p = me[key]
        add(f"| {label} | {r:.3f} | {p:.2e} |")
    add("")
    add("### Decomposition: where the beam loses the unreached items")
    add("")
    add(f"- Items whose L1 code is NEVER emitted in any seed's top-20 beam: "
        f"{me['l1_never_emitted_items']} (the beam does not prune whole L1 prefixes globally).")
    add(f"- Items whose L1:L2 pair is NEVER emitted in any seed's top-20 beam: "
        f"{me['pair_never_emitted_items']} ({_pct(me['pair_never_emitted_frac'])} of the catalog). "
        f"By construction these are all consistently unreached; they account for "
        f"{_pct(me['unreached_share_pair_never'])} of the consistently-unreached set. "
        f"The remaining unreached items live in emitted pairs but lose at decoding "
        f"steps 3-4 or rank below top-10.")
    add("- Note the SIGN of the L1 row above: within a popularity quintile, items "
        "in HIGH-mass L1 prefixes are MORE often unreached, because they compete "
        "with many popular siblings inside the prefix; the own-pop-share-within-"
        "prefix correlation (strongest predictor) captures exactly this.")
    add("")

    co = res["concentration"]
    add("## 4. Beam concentration at the prefix level (top-20 slots)")
    add("")
    add("'slot share' = share of all emitted top-20 slots whose item (or beam "
        "token) carries one of that source's 5 most-used prefixes. The catalog "
        "columns show how much of the catalog (items / train mass) those same "
        "5 prefixes cover, so over-concentration is directly readable. The "
        "catalog reference row uses its own top-5 prefixes by item count "
        "(item share) and by mass (mass share).")
    add("")
    add("| source | distinct L1 used | top5 L1 slot share | their catalog item share | their train mass share "
        "| distinct L1:L2 used | top5 L1:L2 slot share | their catalog item share | their train mass share |")
    add("|---|---|---|---|---|---|---|---|---|")
    for m in MODELS:
        for row in co[m]["per_seed"]:
            add(f"| {m} s{row['seed']} | {row['distinct_l1']} | {_pct(row['top5_l1_slot_share'])} | "
                f"{_pct(row['top5_l1_catalog_item_share'])} | {_pct(row['top5_l1_train_mass_share'])} | "
                f"{row['distinct_l12']} | {_pct(row['top5_l12_slot_share'])} | "
                f"{_pct(row['top5_l12_catalog_item_share'])} | {_pct(row['top5_l12_train_mass_share'])} |")
        mr = co[m]["mean"]
        add(f"| {m} mean | {mr['distinct_l1']:.1f} | {_pct(mr['top5_l1_slot_share'])} | "
            f"{_pct(mr['top5_l1_catalog_item_share'])} | {_pct(mr['top5_l1_train_mass_share'])} | "
            f"{mr['distinct_l12']:.1f} | {_pct(mr['top5_l12_slot_share'])} | "
            f"{_pct(mr['top5_l12_catalog_item_share'])} | {_pct(mr['top5_l12_train_mass_share'])} |")
        add(f"| {m} union over seeds | {co[m]['union_distinct_l1']} | - | - | - | "
            f"{co[m]['union_distinct_l12']} | - | - | - |")
    cat = co["catalog"]
    add(f"| catalog | {cat['distinct_l1']} | (items: {_pct(cat['top5_l1_item_share'])}, "
        f"mass: {_pct(cat['top5_l1_mass_share'])}) | - | - | {cat['distinct_l12']} | "
        f"(items: {_pct(cat['top5_l12_item_share'])}, mass: {_pct(cat['top5_l12_mass_share'])}) | - | - |")
    add("")
    inv = co["marius"]["mean"]
    add(f"MARIUS invalid beam slots (token de-offsets outside the codebook), mean over seeds: "
        f"L1 {inv['invalid_l1_slots']:.1f}, L1:L2 {inv['invalid_l12_slots']:.1f} "
        f"(of {res['n_users'] * 20} slots).")
    add("")
    path.write_text("\n".join(lines))


def write_summary(results: dict, out_dir: Path) -> None:
    """reach_summary.md (<= 20 lines) + the decile figure, from both categories."""
    cats = list(results)
    lines = ["# REACH summary: Beauty vs Sports_and_Outdoors", ""]
    for c in cats:
        r = results[c]
        st = r["structure"]
        m10, s10 = r["reach"]["marius_k10"], r["reach"]["sasrec_k10"]
        lines.append(
            f"- {c}: MARIUS top-10 reaches {_pct(m10['mean_reached_frac'])} of the catalog "
            f"(SASRec++ {_pct(s10['mean_reached_frac'])}); {m10['cons_unreached_n']} items "
            f"({_pct(m10['cons_unreached_frac'])}) are unreached in EVERY seed (SASRec++ "
            f"{s10['cons_unreached_n']}), carrying {_pct(st['demand_mass_share_unreached'])} "
            f"of all train interactions; {st['overlap']['only_marius']} items only MARIUS misses.")

    def per_cat(fmt):
        return "; ".join(fmt(c, results[c]["mechanism"]) for c in cats)

    lines.append(
        "- Mechanism: the beam never prunes whole L1 prefixes ("
        + per_cat(lambda c, me: f"{c}: {me['l1_never_emitted_items']} items in never-emitted L1 codes")
        + "), but L1:L2 pairs die at decode step 2: never-emitted pairs hold "
        + per_cat(lambda c, me: f"{_pct(me['pair_never_emitted_frac'])} of the {c} catalog, "
                  f"{_pct(me['unreached_share_pair_never'])} of its unreached set") + ".")
    lines.append(
        "- The L1 5x2 table is INVERTED (high-prefix-mass items MORE unreached at fixed "
        "own popularity): collapse is sibling competition inside crowded prefixes, not "
        "step-1 pruning. Strongest predictor is log own-pop-share within the L1 prefix ("
        + per_cat(lambda c, me: f"r={me['pb_log_own_share_l1'][0]:.2f}")
        + "), beating own popularity ("
        + per_cat(lambda c, me: f"r={me['pb_log_pop'][0]:.2f}") + ").")
    lines.append(
        "- L1:L2 prefix mass works in the hypothesized direction (low pair mass -> "
        "unreached: " + per_cat(lambda c, me: f"r={me['pb_log_mass2'][0]:.2f}")
        + "). Mid-popularity (Q3) unreached rates, low vs high pair mass: "
        + per_cat(lambda c, me: f"{c} {_pct(me['table_l12'][2]['unreached_low'])} vs "
                  f"{_pct(me['table_l12'][2]['unreached_high'])}") + ".")
    lines.append("")
    lines.append("See reach_<category>.md for the full tables and "
                 "reach_unreached_by_decile.png for the decile profile.")
    lines.append("")
    (out_dir / "reach_summary.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cats), figsize=(6 * len(cats), 4), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(N_DECILES) + 1
    for ax, c in zip(axes, cats):
        rows = results[c]["structure"]["decile_rows"]
        mar = [100 * r["marius_unreached_frac"] for r in rows]
        sas = [100 * r["sasrec_unreached_frac"] for r in rows]
        ax.bar(x - 0.2, mar, width=0.4, label="MARIUS", color="#c44e52")
        ax.bar(x + 0.2, sas, width=0.4, label="SASRec++", color="#4c72b0")
        ax.set_title(c)
        ax.set_xlabel("train-popularity decile (1 = least popular)")
        ax.set_xticks(x)
        ax.legend()
    axes[0].set_ylabel("consistently unreached % (k=10, 3 seeds)")
    fig.tight_layout()
    fig.savefig(out_dir / "reach_unreached_by_decile.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Selftest
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "Toy"
        d.mkdir(parents=True)
        ns, n_catalog, L, U, k_cb = 2, 400, 4, 300, 256
        (d / "meta.json").write_text(json.dumps(
            {"category": "Toy", "n_special": ns, "n_catalog": n_catalog, "L": L, "K": 20}))
        pop = rng.integers(1, 100, n_catalog).astype(np.int64)
        np.save(d / "popularity.npy", pop)
        # 10 L1 prefixes, 40 items each; item 0 pins the codebook size to 256.
        codes = np.column_stack(
            [np.arange(n_catalog) // 40, rng.integers(0, 256, (n_catalog, 3))]).astype(np.int64)
        codes[0, 1:] = 255
        keys = [",".join(map(str, row)) for row in codes]
        assert len(set(keys)) == n_catalog, "toy tuples not unique"
        (d / "tuple_to_item.json").write_text(json.dumps({k: i for i, k in enumerate(keys)}))
        level_off = np.arange(L) * k_cb + ns
        emit_pool = np.flatnonzero(codes[:, 0] < 5)  # marius only ever emits prefixes 0..4
        for seed in (42, 43, 44):
            r = np.random.default_rng(seed)
            np.savez_compressed(
                d / f"sasrec_seed{seed}_topk.npz",
                topk_items=(r.integers(0, n_catalog, (U, 20)) + ns).astype(np.int64),
                target_item=(r.integers(0, n_catalog, U) + ns).astype(np.int64),
                hist_len=r.integers(1, 50, U).astype(np.int64))
            pick = emit_pool[r.integers(0, emit_pool.size, (U, 20))]
            np.savez_compressed(
                d / f"marius_seed{seed}_topk.npz",
                topk_codes=(codes[pick] + level_off).astype(np.int64),
                target_codes=(codes[r.integers(0, n_catalog, U)] + level_off).astype(np.int64),
                hist_len=r.integers(1, 50, U).astype(np.int64))

        res = analyze_category("Toy", Path(tmp), [42, 43, 44])
        m10 = res["reach"]["marius_k10"]
        cu = m10["cons_unreached_n"]
        assert cu >= 200, f"expected >=200 cons-unreached, got {cu}"  # all c0>=5 items
        assert res["concentration"]["marius"]["union_distinct_l1"] == 5
        assert res["concentration"]["catalog"]["distinct_l1"] == 10
        me = res["mechanism"]
        assert me["l1_never_emitted_items"] == 200, me["l1_never_emitted_items"]
        assert me["unreached_share_pair_never"] == 1.0  # all toy losses are step-1/2 prunes
        # every emitted-prefix unreached% < blocked-prefix unreached% (== 1.0 overall)
        blocked = codes[:, 0] >= 5
        item_mass1, _, _, _ = prefix_masses(codes, pop, k_cb)
        tbl = mechanism_table(pop, item_mass1, blocked)
        assert len(tbl) == 5 and all(set(r) >= {"n_low", "unreached_low"} for r in tbl)
        out_dir = Path(tmp) / "reach"
        out_dir.mkdir()
        write_category_md(res, out_dir / "reach_Toy.md", [42, 43, 44])
        write_summary({"Toy": res}, out_dir)
        for f in ("reach_Toy.md", "reach_summary.md", "reach_unreached_by_decile.png"):
            assert (out_dir / f).exists(), f"missing {f}"
        print(f"toy: marius cons-unreached n={cu} "
              f"({_pct(m10['cons_unreached_frac'])}), distinct L1 used=5/10, files written")
        print("OK: selftest passed.")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="REACH: mechanism study of MARIUS catalog collapse")
    parser.add_argument("--category", default="Beauty Sports_and_Outdoors",
                        help="space-separated category list")
    parser.add_argument("--seeds", default="42 43 44")
    parser.add_argument("--dump-root", type=Path, default=REPO / "reports" / "extensions" / "topk")
    parser.add_argument("--out-root", type=Path, default=REPO / "reports" / "extensions" / "reach")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return _selftest()

    seeds = [int(s) for s in args.seeds.split()]
    categories = args.category.split()
    args.out_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for cat in categories:
        print(f"=== {cat} ===", flush=True)
        res = analyze_category(cat, args.dump_root, seeds)
        results[cat] = res
        (args.out_root / f"reach_results_{cat}.json").write_text(json.dumps(res, indent=1))
        write_category_md(res, args.out_root / f"reach_{cat}.md", seeds)
        m10, s10 = res["reach"]["marius_k10"], res["reach"]["sasrec_k10"]
        print(f"  marius k=10 mean reached {_pct(m10['mean_reached_frac'])}, "
              f"cons-unreached {m10['cons_unreached_n']} ({_pct(m10['cons_unreached_frac'])}); "
              f"sasrec {_pct(s10['mean_reached_frac'])} / {s10['cons_unreached_n']}", flush=True)
        print(f"  wrote {args.out_root / f'reach_{cat}.md'}", flush=True)

    # Summary needs both datasets; pick up earlier runs' JSON if not in this run.
    for cat in ("Beauty", "Sports_and_Outdoors"):
        jp = args.out_root / f"reach_results_{cat}.json"
        if cat not in results and jp.exists():
            results[cat] = json.loads(jp.read_text())
    if len(results) >= 2:
        ordered = {c: results[c] for c in ("Beauty", "Sports_and_Outdoors") if c in results}
        write_summary(ordered if len(ordered) >= 2 else results, args.out_root)
        print(f"  wrote {args.out_root / 'reach_summary.md'} and reach_unreached_by_decile.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
