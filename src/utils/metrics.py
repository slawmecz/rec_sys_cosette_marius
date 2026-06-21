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


def summarize_dense(gen, n_items):
    """Flat Gini over recommended item ids for SASRec-style output.

    `gen` is recommended item ids, shape (B, n_results).
    """
    return gini(gen.reshape(-1).tolist(), n_total=n_items)


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

