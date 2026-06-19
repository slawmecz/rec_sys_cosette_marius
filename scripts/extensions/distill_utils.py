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

  3. dump_item_to_codes (offline CLI)
     Builds the (n_catalog, 4) token table once from the COSETTE -col parquet
     and the unique_items pickle and torch.save()s it so MARIUSDistill can
     load it at train time (item_to_codes_path). Lazy torch import; run with:
       python scripts/extensions/distill_utils.py \
         --quant_parquet <parquet> --items_pickle <pkl> --out <table.pt>

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

    scores_np = teacher_cat_scores.detach().float().cpu().numpy().astype(np.float32)
    target_np = target_catalog_idx.detach().cpu().numpy().astype(np.int64)
    result_np = make_candidates_np(scores_np, target_np, n_cand)
    return torch.from_numpy(result_np)


# ---------------------------------------------------------------------------
# 3. offline dumper (build the item->code-tuple table once, save to disk)
# ---------------------------------------------------------------------------

def dump_item_to_codes(quant_parquet: str, items_pickle: str, out_path: str) -> None:
    """Build the catalog item->code-tuple table offline and save it to disk.

    Loads the COSETTE -col tokenizer parquet and the unique_items pickle that
    the MARIUS training pipeline uses, builds the (n_catalog, 4) MARIUS token
    table with build_item_to_codes_np, and torch.save()s it as a LongTensor so
    MARIUSDistill can torch.load() it at train time (item_to_codes_path).

    Loading mirrors src/data/ray_data.py exactly:
      - get_quantized: read_parquet, set_index("product_id"), keep sorted L*
        columns (L0..L3).
      - get_items_map: pickle.load gives the catalog-ordered list of
        product_ids; catalog id i corresponds to position i in this list
        (= SASRec item index - len(SpecialTokens)).

    Parameters
    ----------
    quant_parquet:
        Path to the COSETTE -col tokenizer parquet (a "product_id" column plus
        per-level code columns L0, L1, L2, L3).
    items_pickle:
        Path to the unique_items pickle (a list of product_ids in catalog
        order).
    out_path:
        Destination path for torch.save of the LongTensor table.

    Notes
    -----
    Torch is imported LAZILY so this module stays importable without torch
    (the numpy cores build_item_to_codes_np / make_candidates_np are used by
    the torch-free local selftest).
    """
    import pickle  # noqa: PLC0415 (stdlib; kept local for symmetry)

    import torch  # noqa: PLC0415 (lazy import intentional)

    # Mirror get_quantized: index by product_id, keep sorted L* columns.
    quant_df = pd.read_parquet(quant_parquet)
    quant_df = quant_df.set_index("product_id")
    sorted_cols = sorted(col for col in quant_df.columns if col.startswith("L"))
    quant_df = quant_df[sorted_cols]

    # Mirror get_items_map: the pickle is the catalog-ordered list of items.
    with open(items_pickle, "rb") as f:
        items = pickle.load(f)

    table = build_item_to_codes_np(quant_df, list(items))
    torch.save(torch.from_numpy(table), out_path)
    print(
        f"dump_item_to_codes: wrote {table.shape[0]} items x {table.shape[1]} "
        f"codes to {out_path}"
    )


def _main() -> None:
    import argparse  # noqa: PLC0415 (CLI entry; kept local)

    parser = argparse.ArgumentParser(
        description="Build and save the catalog item->code-tuple table for "
        "MARIUSDistill (offline, run once per dataset/quant_id)."
    )
    parser.add_argument(
        "--quant_parquet",
        required=True,
        help="Path to the COSETTE -col tokenizer parquet (product_id + L0..L3).",
    )
    parser.add_argument(
        "--items_pickle",
        required=True,
        help="Path to the unique_items pickle (catalog-ordered product_ids).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Destination path for the saved LongTensor table (torch.save).",
    )
    args = parser.parse_args()
    dump_item_to_codes(args.quant_parquet, args.items_pickle, args.out)


if __name__ == "__main__":
    _main()
