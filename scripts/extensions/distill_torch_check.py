#!/usr/bin/env python3
"""Torch-dependent numerical verification of the grad-enabled tuple scorer.

This script CANNOT run on the local machine (torch not installed).
Run it on Snellius after installing the project env.

  python scripts/extensions/distill_torch_check.py

Checks:
  (i)  score_marius_tuples_train(...) matches score_marius_tuples(...) within
       atol=1e-5 (same computation, different grad mode).
  (ii) score_marius_tuples_train(...).sum().backward() populates a nonzero
       net.depth_emb.weight.grad (gradient flows through the depth decoder).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("torch unavailable -> SKIP (run on Snellius)")
    raise SystemExit(0)

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from scripts.extensions.scoring import score_marius_tuples, score_marius_tuples_train
from scripts.extensions.distill_utils import (
    build_item_to_codes,
    build_item_to_codes_np,
    make_candidates,
    make_candidates_np,
)


# ---------------------------------------------------------------------------
# Minimal stub net that exposes the interface used by the scorer.
# ---------------------------------------------------------------------------

class _StubNet(nn.Module):
    """Tiny MARIUS-shaped stub: temporal_forward, mid_proj, depth_emb, depth_forward."""

    def __init__(self, d_temp: int = 16, d: int = 8, V: int = 32, L: int = 4) -> None:
        super().__init__()
        # temporal encoder: produces (B, T, d_temp)
        self._enc = nn.Linear(d_temp, d_temp)
        # mid projection: d_temp -> d
        self.mid_proj = nn.Linear(d_temp, d)
        # depth token embeddings: V tokens -> d
        self.depth_emb = nn.Embedding(V, d)
        # depth forward: maps (N, L, d) -> (N, L, V)
        self._depth_fc = nn.Linear(d, V)

    def temporal_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is int64[B, T, L] in the real model; stub just returns a float
        # tensor of shape (B, T, d_temp) derived from the shape of x.
        B, T, _L = x.shape
        return self._enc(torch.ones(B, T, self._enc.in_features))

    def depth_forward(self, e: torch.Tensor) -> torch.Tensor:
        # e: (N, L, d) -> (N, L, V)
        return self._depth_fc(e)


def check_numeric() -> None:
    torch.manual_seed(0)
    B, T, L, C = 2, 5, 4, 6
    V = 32
    net = _StubNet(d_temp=16, d=8, V=V, L=L)
    net.eval()

    batch_input = torch.randint(0, V, (B, T, L))
    code_tokens = torch.randint(0, V, (B, C, L))

    with torch.no_grad():
        out_eval = score_marius_tuples(net, batch_input, code_tokens)

    out_train = score_marius_tuples_train(net, batch_input, code_tokens)

    # (i) Numerical equivalence.
    assert out_eval.shape == (B, C), f"shape mismatch: {out_eval.shape}"
    assert out_train.shape == (B, C), f"shape mismatch: {out_train.shape}"
    assert torch.allclose(out_eval, out_train.detach(), atol=1e-5), (
        f"eval vs train mismatch: max abs diff = {(out_eval - out_train.detach()).abs().max()}"
    )
    print("  (i) numerical equivalence: PASS")

    # (ii) Gradient flows through depth_emb.
    net.zero_grad()
    score_marius_tuples_train(net, batch_input, code_tokens).sum().backward()
    grad = net.depth_emb.weight.grad
    assert grad is not None, "depth_emb.weight.grad is None after backward"
    assert grad.abs().sum() > 0, "depth_emb.weight.grad is all zeros"
    print("  (ii) gradient through depth_emb: PASS")


def check_distill_utils_torch() -> None:
    """Verify that the torch wrappers agree with the numpy cores."""
    # --- build_item_to_codes ---
    quant_df = pd.DataFrame(
        {"L0": [0, 3], "L1": [5, 10], "L2": [0, 127], "L3": [255, 0]},
        index=["pA", "pB"],
    )
    quant_df.index.name = "product_id"
    items = ["pA", "pB"]

    torch_result = build_item_to_codes(quant_df, items)
    np_result = build_item_to_codes_np(quant_df, items)
    assert torch.equal(torch_result, torch.from_numpy(np_result)), (
        "build_item_to_codes torch vs numpy mismatch"
    )
    print("  (iii) build_item_to_codes torch==numpy: PASS")

    # --- make_candidates ---
    torch.manual_seed(0)
    B, n_catalog, n_cand = 4, 10, 5
    scores_t = torch.randn(B, n_catalog)
    target_t = torch.randint(0, n_catalog, (B,))

    torch_cands = make_candidates(scores_t, target_t, n_cand)
    np_cands = make_candidates_np(
        scores_t.numpy().astype("float32"),
        target_t.numpy().astype("int64"),
        n_cand,
    )
    assert torch.equal(torch_cands, torch.from_numpy(np_cands)), (
        "make_candidates torch vs numpy mismatch"
    )
    # Column 0 must equal the true target for every row.
    assert torch.equal(torch_cands[:, 0], target_t.to(torch.int64)), (
        "make_candidates: column 0 does not match target"
    )
    print("  (iv) make_candidates torch==numpy + col0==target: PASS")


def check_distill_prepro_live() -> None:
    """Behavioral check for MARIUSDistillPrePro.__call__.

    Constructs a MARIUSDistillPrePro via __new__ (bypassing ray-based __init__),
    injects a minimal stub crop_and_augment, quant_df, remap, and item_to_id,
    then calls it on a synthetic row and asserts:
      (a) teacher_query has correct ids and left-padding
      (b) target_catalog_idx == item_to_id[last_item] - 2
      (c) input and target shapes match those produced by MARIUSPrePro.__call__
          on the SAME row (same crop, identical construction)
    """
    try:
        from src.data.marius import MARIUSDistillPrePro, MARIUSPrePro, Remap
        from src.models import SpecialTokens
    except ImportError as exc:
        print(f"  (v) check_distill_prepro_live: SKIP (import failed: {exc})")
        return

    PAD = SpecialTokens.PAD.value  # 0
    BOS = SpecialTokens.BOS.value  # 1
    n_special = len(SpecialTokens)   # 2

    # --- stub crop_and_augment: identity (no crop, no augment) ---
    class _StubCrop:
        crop_length = 6

        def __call__(self, tl, ts):
            return tl, ts

    # Synthetic quantized data: 6 items, 4 levels, codes in 0..255
    items = ["a", "b", "c", "d", "e", "f"]
    raw_codes = np.array(
        [[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11],
         [12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]],
        dtype=np.int64,
    )
    quant_df = pd.DataFrame(
        raw_codes, index=items, columns=["L0", "L1", "L2", "L3"]
    )
    L, K = 4, 256
    remap = Remap(L=L, K=K)

    item_to_id = {it: i + n_special for i, it in enumerate(items)}
    # {"a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7}

    # Build MARIUSDistillPrePro via __new__ to skip ray-based __init__.
    obj = MARIUSDistillPrePro.__new__(MARIUSDistillPrePro)
    obj.crop_and_augment = _StubCrop()
    obj.quant_df = quant_df
    obj.L = L
    obj.K = K
    obj.remap = remap
    obj.item_to_id = item_to_id
    obj.split = "train"

    # Synthetic row: timeline = all 6 items; timestamps = ones.
    # np already imported at module level.
    timeline = np.array(items)
    timestamps = np.ones(len(items), dtype=np.int64)
    row = {"timeline": timeline, "timestamp": timestamps}

    out = obj(row)

    # All four keys must be present.
    for key in ("input", "target", "teacher_query", "target_catalog_idx"):
        assert key in out, f"missing key in output: {key}"

    crop_length = obj.crop_and_augment.crop_length  # 6

    # input shape: (crop_length, L)
    assert out["input"].shape == (crop_length, L), (
        f"input shape {out['input'].shape} != ({crop_length}, {L})"
    )
    # target shape: (crop_length, L)
    assert out["target"].shape == (crop_length, L), (
        f"target shape {out['target'].shape} != ({crop_length}, {L})"
    )
    # teacher_query shape: (crop_length,) int64
    assert out["teacher_query"].shape == (crop_length,), (
        f"teacher_query shape {out['teacher_query'].shape} != ({crop_length},)"
    )
    assert out["teacher_query"].dtype == np.int64, (
        f"teacher_query dtype {out['teacher_query'].dtype} != int64"
    )
    # teacher_query values: tl = [a,b,c,d,e,f], tl[:-1] = [a,b,c,d,e]
    # item_to_id: a->2, b->3, c->4, d->5, e->6; crop_length=6, len(tq_ids)=5
    # => left pad 1 zero, then [2,3,4,5,6]
    expected_tq = np.array([PAD, 2, 3, 4, 5, 6], dtype=np.int64)
    assert np.array_equal(out["teacher_query"], expected_tq), (
        f"teacher_query mismatch: got {out['teacher_query']}, expected {expected_tq}"
    )
    # target_catalog_idx: item_to_id["f"] - 2 = 7 - 2 = 5
    assert out["target_catalog_idx"] == np.int64(5), (
        f"target_catalog_idx mismatch: got {out['target_catalog_idx']}, expected 5"
    )
    assert out["target_catalog_idx"].dtype == np.int64, (
        f"target_catalog_idx dtype {out['target_catalog_idx'].dtype} != int64"
    )

    # Cross-check input/target against MARIUSPrePro on the SAME row to verify
    # the construction is byte-identical.
    parent_obj = MARIUSPrePro.__new__(MARIUSPrePro)
    parent_obj.crop_and_augment = _StubCrop()
    parent_obj.quant_df = quant_df
    parent_obj.L = L
    parent_obj.K = K
    parent_obj.remap = remap
    parent_obj.split = "train"

    parent_out = parent_obj(row)
    assert np.array_equal(out["input"], parent_out["input"]), (
        "input differs from MARIUSPrePro output"
    )
    assert np.array_equal(out["target"], parent_out["target"]), (
        "target differs from MARIUSPrePro output"
    )

    print("  (v) check_distill_prepro_live: PASS")


def main() -> int:
    check_numeric()
    check_distill_utils_torch()
    check_distill_prepro_live()
    print("OK: distill_torch_check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
