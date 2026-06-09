#!/usr/bin/env python3
"""Beyond-accuracy metrics for the COSETTE/MARIUS extension.

Pure-numpy implementations (no torch, no GPU) of the item-side distribution
metrics that the base paper only shows informally (Fig 4 popularity decile plot,
Fig 8 collision plot) but never tabulates: catalog coverage, Gini, Shannon
entropy, average recommendation popularity, average percentage of long-tail
items (APLT), novelty / mean self-information, intra-list diversity (ILD), the
MARIUS-specific hallucination rate, and the popularity-decile exposure profile.

These operate on the ranked Top-K lists that both models already produce in
evaluation (sasrec.search / marius.search), so they require no change to the
authors' model or eval code. The Top-K lists are produced once on Snellius by
scripts/extensions/dump_topk.py; this module turns them into numbers.

Contract
--------
recs:        sequence (length U) of per-user ranked item lists. Each entry is a
             list of integer catalog item indices in [0, n_catalog). The sentinel
             HALLUCINATION (-1) marks a generated tuple that maps to no real item
             (MARIUS only); sentinels are excluded from all metrics except
             hallucination_rate.
item_pop:    1D array of length n_catalog, train interaction count per item.
item_emb:    optional 2D array (n_catalog x d), item content embeddings for ILD.
targets:     optional length-U array of the ground-truth held-out item index per
             user (-1 if unknown), for recall / tail-recall convenience helpers.

All "@K" metrics truncate each list to its first K entries before counting.
"""

from __future__ import annotations

import numpy as np

HALLUCINATION = -1


# --------------------------------------------------------------------------- #
# Item-exposure primitives
# --------------------------------------------------------------------------- #
def exposure_vector(recs, n_catalog, k=None):
    """Count, over all users' Top-k lists, how often each catalog item appears."""
    counts = np.zeros(int(n_catalog), dtype=np.int64)
    for row in recs:
        items = row if k is None else row[:k]
        for it in items:
            if it is not None and it >= 0:
                counts[it] += 1
    return counts


def catalog_coverage(recs, n_catalog, k=None):
    """Fraction of the catalog that appears in at least one Top-k list (aggregate diversity)."""
    counts = exposure_vector(recs, n_catalog, k)
    return float((counts > 0).sum() / float(n_catalog))


def gini(counts):
    """Gini index of an exposure distribution. 0 = perfectly equal, ->1 = concentrated.

    Computed over the full catalog (zero-exposure items included), so it captures
    both how unequal and how narrow the recommended set is.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    s = x.sum()
    if n == 0 or s == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * s) - (n + 1.0) / n)


def shannon_entropy(counts, normalize=True):
    """Shannon entropy (bits) of the exposure distribution.

    With normalize=True, divide by log2(n_catalog) to land in [0, 1]; higher means
    exposure is spread more evenly across more items.
    """
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum()
    if total == 0:
        return 0.0
    p = c[c > 0] / total
    h = float(-(p * np.log2(p)).sum())
    if normalize and c.size > 1:
        return h / float(np.log2(c.size))
    return h


# --------------------------------------------------------------------------- #
# Popularity / long-tail
# --------------------------------------------------------------------------- #
def head_mask_by_item_fraction(item_pop, head_frac=0.2):
    """Head = the most popular `head_frac` fraction of items. Returns boolean head mask."""
    pop = np.asarray(item_pop, dtype=np.float64)
    n = pop.size
    n_head = max(1, int(round(head_frac * n)))
    order = np.argsort(-pop, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_head]] = True
    return mask


def head_mask_by_interaction_share(item_pop, head_share=0.8):
    """Head = fewest popular items that together account for `head_share` of all
    interactions (the Pareto / long-tail split). Returns boolean head mask."""
    pop = np.asarray(item_pop, dtype=np.float64)
    order = np.argsort(-pop, kind="stable")
    cum = np.cumsum(pop[order])
    total = pop.sum()
    mask = np.zeros(pop.size, dtype=bool)
    if total == 0:
        return mask
    n_head = int(np.searchsorted(cum, head_share * total) + 1)
    mask[order[:n_head]] = True
    return mask


def average_recommendation_popularity(recs, item_pop, k=None):
    """Mean train popularity (raw interaction count) of recommended real items."""
    pop = np.asarray(item_pop, dtype=np.float64)
    vals = [pop[it] for row in recs for it in (row if k is None else row[:k]) if it is not None and it >= 0]
    return float(np.mean(vals)) if vals else 0.0


def aplt(recs, head_mask, k=None):
    """Average Percentage of Long-Tail items: per user the share of recommended
    items that are NOT in the head set, then averaged over users."""
    head = np.asarray(head_mask, dtype=bool)
    per_user = []
    for row in recs:
        items = [it for it in (row if k is None else row[:k]) if it is not None and it >= 0]
        if not items:
            continue
        per_user.append(float(np.mean([not head[it] for it in items])))
    return float(np.mean(per_user)) if per_user else 0.0


def novelty(recs, item_pop, k=None):
    """Mean self-information -log2(p(item)), p = pop / total interactions. Higher = more novel."""
    pop = np.asarray(item_pop, dtype=np.float64)
    total = pop.sum()
    if total == 0:
        return 0.0
    vals = [
        -np.log2(pop[it] / total)
        for row in recs
        for it in (row if k is None else row[:k])
        if it is not None and it >= 0 and pop[it] > 0
    ]
    return float(np.mean(vals)) if vals else 0.0


def popularity_decile_exposure(recs, item_pop, n_deciles=10, k=None):
    """Share of total recommendation exposure that falls in each popularity decile.

    Decile 1 = least popular items, decile `n_deciles` = most popular. Mirrors the
    base paper's Figure 4 (which only plotted a MARIUS-minus-SASRec difference) but
    as an absolute, tabulated exposure profile. Returns an array of length n_deciles
    summing to 1.
    """
    pop = np.asarray(item_pop, dtype=np.float64)
    n = pop.size
    order = np.argsort(pop, kind="stable")  # ascending popularity
    decile_of = np.empty(n, dtype=np.int64)
    edges = np.linspace(0, n, n_deciles + 1).astype(int)
    for d in range(n_deciles):
        decile_of[order[edges[d]:edges[d + 1]]] = d
    counts = exposure_vector(recs, n, k)
    share = np.zeros(n_deciles, dtype=np.float64)
    for d in range(n_deciles):
        share[d] = counts[decile_of == d].sum()
    tot = share.sum()
    return share / tot if tot > 0 else share


# --------------------------------------------------------------------------- #
# Content diversity (needs embeddings)
# --------------------------------------------------------------------------- #
def intra_list_diversity(recs, item_emb, k=None):
    """Mean over users of the mean pairwise cosine distance (1 - cos) within the Top-k list."""
    emb = np.asarray(item_emb, dtype=np.float64)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    embn = emb / norm
    per_user = []
    for row in recs:
        items = [it for it in (row if k is None else row[:k]) if it is not None and it >= 0]
        if len(items) < 2:
            continue
        v = embn[items]
        sims = v @ v.T
        iu = np.triu_indices(len(items), k=1)
        per_user.append(float(np.mean(1.0 - sims[iu])))
    return float(np.mean(per_user)) if per_user else 0.0


# --------------------------------------------------------------------------- #
# Generative-specific + accuracy convenience
# --------------------------------------------------------------------------- #
def hallucination_rate(recs, k=None):
    """Fraction of Top-k slots filled by a sentinel (generated tuple with no real item)."""
    tot = hall = 0
    for row in recs:
        for it in (row if k is None else row[:k]):
            tot += 1
            if it is None or it < 0:
                hall += 1
    return float(hall / tot) if tot else 0.0


def recall_at_k(recs, targets, k=None):
    """Leave-one-out Recall@k = HitRate@k with a single held-out positive per user."""
    hits = 0
    n = 0
    for row, tgt in zip(recs, targets):
        if tgt is None or tgt < 0:
            continue
        n += 1
        if tgt in (row if k is None else row[:k]):
            hits += 1
    return float(hits / n) if n else 0.0


def tail_recall_at_k(recs, targets, head_mask, k=None):
    """Recall@k restricted to users whose held-out target is a long-tail (non-head) item.

    This is the accuracy-meets-beyond-accuracy quantity: it says whether a model
    actually surfaces correct tail items, not just whether it spreads exposure.
    """
    head = np.asarray(head_mask, dtype=bool)
    sub_r, sub_t = [], []
    for row, tgt in zip(recs, targets):
        if tgt is not None and tgt >= 0 and not head[tgt]:
            sub_r.append(row)
            sub_t.append(tgt)
    if not sub_t:
        return float("nan")
    return recall_at_k(sub_r, sub_t, k)


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
def compute_all(recs, item_pop, n_catalog, item_emb=None, targets=None, k=10, head_frac=0.2):
    """Compute the full beyond-accuracy suite at cutoff k. Returns a flat dict."""
    head = head_mask_by_item_fraction(item_pop, head_frac=head_frac)
    counts = exposure_vector(recs, n_catalog, k)
    out = {
        "k": int(k),
        "coverage": catalog_coverage(recs, n_catalog, k),
        "gini": gini(counts),
        "entropy_norm": shannon_entropy(counts, normalize=True),
        "arp": average_recommendation_popularity(recs, item_pop, k),
        "aplt": aplt(recs, head, k),
        "novelty": novelty(recs, item_pop, k),
        "hallucination_rate": hallucination_rate(recs, k),
    }
    if item_emb is not None:
        out["ild"] = intra_list_diversity(recs, item_emb, k)
    if targets is not None:
        out["recall"] = recall_at_k(recs, targets, k)
        out["tail_recall"] = tail_recall_at_k(recs, targets, head, k)
    return out


# --------------------------------------------------------------------------- #
# Self-test: synthetic sanity check (run: python scripts/extensions/beyond_accuracy.py)
# --------------------------------------------------------------------------- #
def _demo():
    rng = np.random.default_rng(0)
    n_catalog, n_users, k = 1000, 2000, 10

    # Zipfian popularity (a few items dominate).
    ranks = np.arange(1, n_catalog + 1)
    item_pop = (1.0 / ranks) * 1e5
    item_pop = item_pop.astype(np.int64) + 1
    pop_p = item_pop / item_pop.sum()

    item_emb = rng.normal(size=(n_catalog, 32))

    # A popularity-chasing recommender: almost always recommends the head.
    head_items = np.argsort(-item_pop)[:50]
    recs_pop = [list(rng.choice(head_items, size=k, replace=False)) for _ in range(n_users)]

    # A diverse recommender: samples broadly across the catalog.
    recs_div = [list(rng.choice(n_catalog, size=k, replace=False, p=pop_p / pop_p.sum())) for _ in range(n_users)]
    recs_uni = [list(rng.choice(n_catalog, size=k, replace=False)) for _ in range(n_users)]

    # A generative recommender with 8% hallucinations.
    recs_gen = []
    for _ in range(n_users):
        row = list(rng.choice(n_catalog, size=k, replace=False, p=pop_p / pop_p.sum()))
        for i in range(k):
            if rng.random() < 0.08:
                row[i] = HALLUCINATION
        recs_gen.append(row)

    head = head_mask_by_item_fraction(item_pop, 0.2)
    tag = lambda name, r: (name, compute_all(r, item_pop, n_catalog, item_emb=item_emb, k=k))
    rows = [tag("popularity-chaser", recs_pop), tag("popularity-weighted", recs_div),
            tag("uniform", recs_uni), tag("generative(+halluc)", recs_gen)]

    cols = ["coverage", "gini", "entropy_norm", "arp", "aplt", "novelty", "ild", "hallucination_rate"]
    print(f"{'recommender':<22}" + "".join(f"{c:>12}" for c in cols))
    for name, m in rows:
        print(f"{name:<22}" + "".join(f"{m[c]:>12.4f}" for c in cols))

    mp = dict(rows)["popularity-chaser"]
    md = dict(rows)["uniform"]
    # Sanity: the popularity-chaser must be more concentrated and less novel than uniform.
    assert mp["gini"] > md["gini"], "Gini sanity failed"
    assert mp["coverage"] < md["coverage"], "coverage sanity failed"
    assert mp["novelty"] < md["novelty"], "novelty sanity failed"
    assert mp["aplt"] < md["aplt"], "APLT sanity failed"
    assert abs(dict(rows)["generative(+halluc)"]["hallucination_rate"] - 0.08) < 0.02, "halluc sanity failed"
    print("\nOK: all sanity assertions passed.")


if __name__ == "__main__":
    _demo()
