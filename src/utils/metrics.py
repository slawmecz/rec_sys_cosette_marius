from collections import defaultdict

import numpy as np


def gini(y, n_total=None):
    """Lorenz-curve Gini coefficient over category counts in `y`.

    `n_total` is the full catalog size; categories absent from `y` are
    zero-padded so unrecommended items count toward inequality too.
    """
    _, counts = np.unique(y, return_counts=True)
    if n_total is not None:
        counts = np.concatenate([counts, np.zeros(n_total - len(counts), dtype=counts.dtype)])
    counts = np.sort(counts)
    n = len(counts)
    cum = np.cumsum(counts)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n

def conditional_gini(codes, level, n_total=None):
    """Gini coefficient of token values at `level`, conditioned on levels `0..level-1`.

    Groups codes by their prefix (the preceding levels) and computes a
    per-group Gini, then averages across groups weighted by group size.
    `level=0` is unconditioned (single group: the whole dataset).
    """
    groups = defaultdict(list)
    for code in codes:
        groups[code[:level]].append(code[level])

    ginis, weights = [], []
    for vals in groups.values():
        ginis.append(gini(vals, n_total=n_total))
        weights.append(len(vals))
    return np.average(ginis, weights=weights)


def summarize_generative(gen, n_total_per_level):
    """Per-RVQ-level conditional Gini for MARIUS-style output.

    `gen` is recommended semantic IDs, shape (B, n_results, L) (raw token ids,
    pre- or post-Remap offset - only relative spacing within a level matters
    since `conditional_gini` groups by prefix). Returns one Gini per level.
    """
    codes = [tuple(rec) for user in gen for rec in user]
    L = len(codes[0])
    return [conditional_gini(codes, level, n_total=n_total_per_level) for level in range(L)]


def summarize_generative_support(gen):
    """Per-RVQ-level support diagnostics for interpreting conditional Gini/entropy.

    Deeper levels condition on more preceding tokens, so each prefix-group holds
    fewer recommendations and can mechanically use fewer codes - which shifts
    Gini/entropy regardless of true concentration. For each level this returns
    the number of conditioning groups, the mean recommendations per group, the
    mean distinct codes used per group, and the total distinct codes used at the
    level, so a per-level Gini change can be checked against shrinking support.

    `gen` shape (B, K, L). Returns one dict per level (mirrors the grouping in
    `conditional_gini`).
    """
    codes = [tuple(rec) for user in gen for rec in user]
    L = len(codes[0])
    out = []
    for level in range(L):
        groups = defaultdict(list)
        for code in codes:
            groups[code[:level]].append(code[level])
        sizes = np.array([len(v) for v in groups.values()])
        distinct = np.array([len(set(v)) for v in groups.values()])
        out.append(
            {
                "n_groups": int(len(groups)),
                "mean_group_size": float(sizes.mean()),
                "mean_distinct_codes": float(distinct.mean()),
                "total_distinct_codes": int(len({c[level] for c in codes})),
            }
        )
    return out


def summarize_dense(gen, n_items):
    """Flat Gini over recommended item ids for SASRec-style output.

    `gen` is recommended item ids, shape (B, n_results).
    """
    return gini(gen.reshape(-1).tolist(), n_total=n_items)


def _item_ids(gen):
    """Map each recommended semantic-ID tuple to a unique integer item id.

    `gen` shape (B, K, L); the full L-tuple identifies one catalog item (codes
    are unique per item after collision removal). Returns a flat (B*K,) array of
    item identities, suitable for the flat `gini`/`entropy` helpers.
    """
    gen = np.asarray(gen)
    flat = gen.reshape(-1, gen.shape[-1])
    _, item_ids = np.unique(flat, axis=0, return_inverse=True)
    return item_ids.ravel()


def summarize_generative_item(gen, n_items):
    """Flat item-level Gini for MARIUS-style output.

    Treats each full semantic-ID tuple as one item, so the result is directly
    comparable to `summarize_dense` (SASRec). `n_items` is the full catalog size,
    so unrecommended items count toward inequality just like the dense case.
    """
    return gini(_item_ids(gen), n_total=n_items)


def entropy(y, n_total=None):
    """Shannon entropy (nats) over category counts in `y`.

    `n_total` is the full catalog size; categories absent from `y` contribute
    zero probability and thus zero entropy mass (0 * log 0 = 0 by convention).
    """
    _, counts = np.unique(y, return_counts=True)
    if n_total is not None:
        counts = np.concatenate([counts, np.zeros(n_total - len(counts), dtype=counts.dtype)])
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def conditional_entropy(codes, level, n_total=None):
    """Shannon entropy of token values at `level`, conditioned on levels `0..level-1`.

    Mirrors the structure of `conditional_gini`: groups by prefix and returns
    a weighted average of per-group entropies.
    `level=0` is unconditioned (single group: the whole dataset).
    """
    groups = defaultdict(list)
    for code in codes:
        groups[code[:level]].append(code[level])

    entropies, weights = [], []
    for vals in groups.values():
        entropies.append(entropy(vals, n_total=n_total))
        weights.append(len(vals))
    return np.average(entropies, weights=weights)


def summarize_generative_entropy(gen, n_total_per_level):
    """Per-RVQ-level conditional entropy for MARIUS-style output.

    `gen` shape (B, n_results, L). Returns one entropy value per level.
    """
    codes = [tuple(rec) for user in gen for rec in user]
    L = len(codes[0])
    return [conditional_entropy(codes, level, n_total=n_total_per_level) for level in range(L)]


def summarize_dense_entropy(gen, n_items):
    """Flat entropy over recommended item ids for SASRec-style output.

    `gen` shape (B, n_results).
    """
    return entropy(gen.reshape(-1).tolist(), n_total=n_items)


def summarize_generative_item_entropy(gen, n_items):
    """Flat item-level entropy for MARIUS-style output.

    Treats each full semantic-ID tuple as one item (comparable to SASRec's
    `summarize_dense_entropy`). `n_items` is the full catalog size.
    """
    return entropy(_item_ids(gen), n_total=n_items)


def summarize_dense_ild(gen):
    """Intra-List Diversity for SASRec-style output using binary item distance.

    `gen` shape (B, K). For each user, computes the fraction of item pairs that
    are distinct, then averages across users. Range [0, 1].
    """
    gen = np.asarray(gen)
    B, K = gen.shape
    if K < 2:
        return 0.0
    n_pairs = K * (K - 1) / 2
    diff = gen[:, :, None] != gen[:, None, :]  # B x K x K
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    return float(diff[:, mask].sum(axis=1).mean() / n_pairs)


def summarize_generative_ild(gen):
    """Intra-List Diversity for MARIUS-style output using normalized Hamming distance.

    `gen` shape (B, K, L). For each user, computes the mean pairwise normalized
    Hamming distance (fraction of levels that differ) across all item pairs, then
    averages across users. Range [0, 1].

    This scores *code-level* diversity (how much the RVQ tokens differ); for a
    binary item-level ILD directly comparable to SASRec, see
    `summarize_generative_item_ild`.
    """
    gen = np.asarray(gen)
    B, K, L = gen.shape
    if K < 2:
        return 0.0
    n_pairs = K * (K - 1) / 2
    diff = gen[:, :, None, :] != gen[:, None, :, :]  # B x K x K x L
    hamming = diff.sum(axis=-1) / L  # B x K x K
    mask = np.triu(np.ones((K, K), dtype=bool), k=1)
    return float(hamming[:, mask].sum(axis=1).mean() / n_pairs)


def summarize_generative_item_ild(gen):
    """Binary item-level Intra-List Diversity for MARIUS-style output.

    Maps each recommended semantic-ID tuple to a unique item id (codes are unique
    per item after collision removal) and applies the same binary distance as
    `summarize_dense_ild` - the fraction of distinct item pairs per user. Unlike
    `summarize_generative_ild` (normalized Hamming over RVQ codes), this measures
    diversity from item identity, so it is directly comparable to SASRec's ILD.

    `gen` shape (B, K, L). Range [0, 1].
    """
    gen = np.asarray(gen)
    B, K = gen.shape[0], gen.shape[1]
    item_ids = _item_ids(gen).reshape(B, K)
    return summarize_dense_ild(item_ids)


def category_diversity(rec_categories):
    """Mean per-user category diversity: distinct categories / K, averaged over users.

    Unlike ILD (which scores diversity from item identity / RVQ codes), this uses
    an external product taxonomy, and is computed identically for SASRec and
    MARIUS - so the two are directly comparable. Map each recommended item to its
    category label upstream (use a shared sentinel for items with no category).

    `rec_categories` shape (B, K): the category label of each recommended item
    (any hashable). Range (0, 1]: 1.0 = every recommended item a distinct category,
    1/K = all K share one category.
    """
    rec_categories = np.asarray(rec_categories, dtype=object)
    B, K = rec_categories.shape
    if K == 0:
        return 0.0
    return float(np.mean([len(set(row)) / K for row in rec_categories]))

