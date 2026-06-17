"""Utility functions for the MARIUS distillation pipeline.

Two capabilities are provided:

  1. build_item_to_codes_np / build_item_to_codes
     Maps each catalog item to its 4 COSETTE codes in MARIUS TOKEN space.
     Token formula (from src/data/marius.py Remap):
       token[l] = raw_code[l] + l * 256 + 2
     where 256 is the codebook size per level and 2 = len(SpecialTokens)
     (PAD=0, BOS=1).

  2. make_candidates_np / make_candidates
     For each user in a batch, selects n_cand candidate catalog ids = the
     teacher's top-n_cand by score, with the TRUE item forced into column 0
     so it is always present in the candidate set.

Design: the numpy cores (_np functions) are the logic source of truth and are
importable without torch.  The torch wrappers lazily import torch and mirror
the numpy cores exactly.  This lets the local selftest (no torch) run the
numpy cores directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. item->code-tuple table
# ---------------------------------------------------------------------------

_CODEBOOK_SIZE = 256  # codes per level
_SPECIAL_TOKENS = 2   # len(SpecialTokens): PAD=0, BOS=1


def build_item_to_codes_np(
    quant_df: pd.DataFrame,
    items: list[str],
) -> np.ndarray:
    """Return an int64 array of shape (n_catalog, 4) giving MARIUS token ids.

    Parameters
    ----------
    quant_df:
        DataFrame indexed by product_id with integer columns L0, L1, L2, L3
        (raw codes in [0, 255]).
    items:
        Catalog-ordered list of product_ids.  Position i in this list
        corresponds to catalog id i (= SASRec item index - 2).

    Returns
    -------
    np.ndarray with dtype int64, shape (len(items), 4).
    Token formula: token[l] = raw_code[l] + l * 256 + 2, for l in 0..3.
    """
    # Reindex quant_df to the catalog order and extract columns L0..L3.
    raw = quant_df.loc[items, ["L0", "L1", "L2", "L3"]].values.astype(np.int64)
    # offsets: [0, 256, 512, 768] + 2 = [2, 258, 514, 770]
    offsets = np.arange(4, dtype=np.int64) * _CODEBOOK_SIZE + _SPECIAL_TOKENS
    return raw + offsets  # broadcast over rows


def build_item_to_codes(
    quant_df: pd.DataFrame,
    items: list[str],
):
    """Torch wrapper around build_item_to_codes_np.

    Returns a torch.LongTensor of shape (n_catalog, 4).
    Lazily imports torch so this module is importable without it.
    """
    import torch  # noqa: PLC0415 (lazy import intentional)

    return torch.from_numpy(build_item_to_codes_np(quant_df, items))


# ---------------------------------------------------------------------------
# 2. candidate builder
# ---------------------------------------------------------------------------

def make_candidates_np(
    scores_2d: np.ndarray,
    target_idx_1d: np.ndarray,
    n_cand: int,
) -> np.ndarray:
    """Return candidate catalog ids for each user, shape (B, n_cand).

    Column 0 is always the TRUE item (target_idx_1d[b]).  The remaining
    n_cand-1 columns are filled with the teacher's highest-scoring catalog ids
    (other than any slot already occupied by the true item in the top-n_cand).

    Parameters
    ----------
    scores_2d:
        Float array of shape (B, n_catalog) with per-user catalog scores.
    target_idx_1d:
        Int array of shape (B,) with the true item catalog id for each user.
    n_cand:
        Number of candidates to return per user (including the true item).

    Returns
    -------
    np.ndarray with dtype int64, shape (B, n_cand).
    Column 0 = true item; columns 1..n_cand-1 = the teacher's top items
    other than the true item.  The true item appears exactly once per row
    (in col 0): the swap path moves it out of its original top-n slot, and
    the replace path only triggers when it was not in the top-n at all.
    """
    B, n_catalog = scores_2d.shape
    result = np.empty((B, n_cand), dtype=np.int64)

    # For each row: take top n_cand from the teacher, then force true item
    # into column 0 by swapping if needed.
    # We fetch n_cand scores; if the true item is already in them, no slot
    # is wasted.  If not, the n_cand-th top item is dropped to make room.
    top_n = np.argsort(-scores_2d, axis=1)[:, :n_cand]  # (B, n_cand) desc

    for b in range(B):
        true_item = int(target_idx_1d[b])
        row = top_n[b].tolist()  # list of length n_cand

        # Find where the true item sits in the top-n list.
        if true_item in row:
            idx = row.index(true_item)
            # Swap true item to position 0.
            row[0], row[idx] = row[idx], row[0]
        else:
            # True item is not in the top-n list.  Drop the lowest-scoring
            # candidate (last position) and put the true item at col 0.
            row = [true_item] + top_n[b, :-1].tolist()

        result[b] = row

    return result


def make_candidates(
    teacher_cat_scores,
    target_catalog_idx,
    n_cand: int,
):
    """Torch wrapper around make_candidates_np.

    Parameters
    ----------
    teacher_cat_scores:
        torch.Tensor of shape (B, n_catalog).
    target_catalog_idx:
        torch.Tensor of shape (B,) with true item catalog ids.
    n_cand:
        Number of candidates per user.

    Returns
    -------
    torch.LongTensor of shape (B, n_cand).
    Column 0 = true item.  Logic mirrors make_candidates_np exactly.
    Lazily imports torch so this module is importable without it.
    """
    import torch  # noqa: PLC0415 (lazy import intentional)

    scores_np = teacher_cat_scores.detach().cpu().numpy().astype(np.float32)
    target_np = target_catalog_idx.detach().cpu().numpy().astype(np.int64)
    result_np = make_candidates_np(scores_np, target_np, n_cand)
    return torch.from_numpy(result_np)
