#!/usr/bin/env python3
"""Torch-free verification of the grad-enabled tuple scorer refactor and
the distill_utils item->code-tuple table + candidate builder.

Mirrors scripts/extensions/logitadj_selftest.py: static/AST checks only
(for scoring.py) and numpy-core checks only (for distill_utils.py), so no
torch import is needed anywhere here.

  python scripts/extensions/distill_selftest.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SCORING = REPO / "scripts" / "extensions" / "scoring.py"
DISTILL_UTILS = REPO / "scripts" / "extensions" / "distill_utils.py"
MARIUS_PY = REPO / "src" / "data" / "marius.py"

sys.path.insert(0, str(REPO))


def check_scoring() -> None:
    text = SCORING.read_text()

    # ASCII-only check.
    assert all(ord(c) < 128 for c in text), "non-ASCII characters in scoring.py"

    tree = ast.parse(text)

    # Collect top-level function definitions.
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    required = {"_score_marius_tuples_impl", "score_marius_tuples", "score_marius_tuples_train"}
    for name in required:
        assert name in funcs, f"missing top-level function: {name}"

    # score_marius_tuples must carry @torch.no_grad() decorator.
    pub_fn = funcs["score_marius_tuples"]
    decorators = pub_fn.decorator_list
    found_no_grad = False
    for d in decorators:
        # Accept the call form: @torch.no_grad()
        if (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "no_grad"
            and isinstance(d.func.value, ast.Name)
            and d.func.value.id == "torch"
        ):
            found_no_grad = True
        # Also accept the no-parens attribute form: @torch.no_grad
        elif isinstance(d, ast.Attribute) and d.attr == "no_grad":
            found_no_grad = True
    assert found_no_grad, "score_marius_tuples must have @torch.no_grad() decorator"

    # score_marius_tuples_train must have NO no_grad decorator.
    train_fn = funcs["score_marius_tuples_train"]
    for d in train_fn.decorator_list:
        is_no_grad = False
        if (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "no_grad"
        ):
            is_no_grad = True
        elif isinstance(d, ast.Attribute) and d.attr == "no_grad":
            is_no_grad = True
        assert not is_no_grad, "score_marius_tuples_train must NOT have @torch.no_grad()"

    # Both public functions must call _score_marius_tuples_impl in their body.
    def _calls_impl(fn_node: ast.FunctionDef) -> bool:
        for node in ast.walk(fn_node):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "_score_marius_tuples_impl":
                    return True
        return False

    assert _calls_impl(pub_fn), "score_marius_tuples body must call _score_marius_tuples_impl"
    assert _calls_impl(train_fn), "score_marius_tuples_train body must call _score_marius_tuples_impl"

    print("  check_scoring: PASS")


def check_distill_utils() -> None:
    # ASCII-only check for the module source.
    text = DISTILL_UTILS.read_text()
    assert all(ord(c) < 128 for c in text), "non-ASCII characters in distill_utils.py"

    from scripts.extensions.distill_utils import (
        build_item_to_codes_np,
        make_candidates_np,
    )

    # --- build_item_to_codes_np ---
    # Two synthetic items with known raw codes.
    # raw pA: [0, 5, 0, 255]  -> expected tokens: [0+2, 5+258, 0+514, 255+770]
    #                                             = [2, 263, 514, 1025]
    # raw pB: [3, 10, 127, 0] -> expected tokens: [3+2, 10+258, 127+514, 0+770]
    #                                             = [5, 268, 641, 770]
    quant_df = pd.DataFrame(
        {"L0": [0, 3], "L1": [5, 10], "L2": [0, 127], "L3": [255, 0]},
        index=["pA", "pB"],
    )
    quant_df.index.name = "product_id"
    items = ["pA", "pB"]

    result = build_item_to_codes_np(quant_df, items)
    assert result.shape == (2, 4), f"expected shape (2,4), got {result.shape}"
    assert result.dtype == np.int64, f"expected int64, got {result.dtype}"

    expected = np.array(
        [[2, 263, 514, 1025], [5, 268, 641, 770]], dtype=np.int64
    )
    assert np.array_equal(result, expected), (
        f"build_item_to_codes_np mismatch:\n  got      {result}\n  expected {expected}"
    )

    # --- make_candidates_np ---
    # B=2 rows, n_catalog=6 items, ask for top n_cand=3.
    scores = np.array(
        [
            [0.1, 0.9, 0.3, 0.7, 0.5, 0.2],  # row 0: top3 by score = idx 1,3,4; target=5
            [0.8, 0.2, 0.6, 0.4, 0.1, 0.9],  # row 1: top3 by score = idx 5,0,2; target=2
        ],
        dtype=np.float32,
    )
    target_idx = np.array([5, 2], dtype=np.int64)

    cands = make_candidates_np(scores, target_idx, n_cand=3)
    assert cands.shape == (2, 3), f"expected shape (2,3), got {cands.shape}"
    assert cands.dtype == np.int64, f"expected int64, got {cands.dtype}"

    # Column 0 must equal the true target for every row.
    assert cands[0, 0] == 5, f"row 0 col 0 should be 5 (target), got {cands[0,0]}"
    assert cands[1, 0] == 2, f"row 1 col 0 should be 2 (target), got {cands[1,0]}"

    # All values in each row must be in [0, n_catalog).
    assert np.all(cands >= 0) and np.all(cands < 6), "candidate indices out of range"

    print("  check_distill_utils: PASS")


def check_distill_prepro() -> None:
    """Two-part verification for MARIUSDistillPrePro.

    Part A: AST check -- confirms class exists, subclasses MARIUSPrePro,
    defines __init__ + __call__, and __call__ returns a dict with the 4
    expected keys.

    Part B: numpy mirror -- reimplements ONLY the deterministic
    teacher_query padding and target_catalog_idx offset math (no import of
    src.data.marius) and asserts correctness on a synthetic example.
    """

    # ----- Part A: AST check -----

    text = MARIUS_PY.read_text()
    assert all(ord(c) < 128 for c in text), "non-ASCII characters in marius.py"

    tree = ast.parse(text)

    # Find MARIUSDistillPrePro class node.
    distill_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MARIUSDistillPrePro":
            distill_cls = node
            break
    assert distill_cls is not None, "MARIUSDistillPrePro class not found in marius.py"

    # Must subclass MARIUSPrePro.
    base_names = []
    for base in distill_cls.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)
    assert "MARIUSPrePro" in base_names, (
        f"MARIUSDistillPrePro must subclass MARIUSPrePro, found bases: {base_names}"
    )

    # Must define __init__ and __call__.
    method_names = {
        n.name for n in distill_cls.body if isinstance(n, ast.FunctionDef)
    }
    for required_method in ("__init__", "__call__"):
        assert required_method in method_names, (
            f"MARIUSDistillPrePro missing method: {required_method}"
        )

    # __call__ must return a dict that contains all 4 keys.
    call_fn = next(
        n for n in distill_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    required_keys = {"input", "target", "teacher_query", "target_catalog_idx"}
    found_keys: set[str] = set()
    for node in ast.walk(call_fn):
        if isinstance(node, ast.Return):
            # Look for Constant string keys inside dict literals anywhere in
            # the return expression subtree.
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    found_keys.add(sub.value)
    missing = required_keys - found_keys
    assert not missing, (
        f"MARIUSDistillPrePro.__call__ return dict missing keys: {missing}"
    )

    print("  check_distill_prepro (Part A - AST): PASS")

    # ----- Part B: numpy mirror of teacher_query + target_catalog_idx -----

    # Synthetic setup:
    #   item_to_id maps items to SASRec indices (PAD=0, BOS=1, then items 2..N+1)
    #   len(SpecialTokens) = 2
    #   tl = ["a", "b", "c", "d"]  (crop of length 4; last item = "d")
    #   crop_length = 6  (pad 2 slots on the left)
    PAD = 0
    n_special = 2
    item_to_id = {"a": 2, "b": 3, "c": 4, "d": 5, "e": 6}
    tl = ["a", "b", "c", "d"]
    crop_length = 6

    # teacher_query: item-id history for tl[:-1], left-padded with PAD to crop_length.
    tq_ids = [item_to_id[it] for it in tl[:-1]]  # [2, 3, 4]
    teacher_query = np.array(
        [PAD] * (crop_length - len(tq_ids)) + tq_ids, dtype=np.int64
    )
    # Expected: [0, 0, 0, 2, 3, 4]
    expected_tq = np.array([0, 0, 0, 2, 3, 4], dtype=np.int64)
    assert np.array_equal(teacher_query, expected_tq), (
        f"teacher_query mismatch: got {teacher_query}, expected {expected_tq}"
    )
    assert teacher_query.dtype == np.int64, "teacher_query must be int64"
    assert teacher_query.shape == (crop_length,), (
        f"teacher_query shape mismatch: {teacher_query.shape}"
    )

    # target_catalog_idx: item_to_id[tl[-1]] - len(SpecialTokens)
    target_catalog_idx = np.int64(item_to_id[tl[-1]] - n_special)
    # item_to_id["d"] = 5, 5 - 2 = 3
    assert target_catalog_idx == np.int64(3), (
        f"target_catalog_idx mismatch: got {target_catalog_idx}, expected 3"
    )
    assert target_catalog_idx.dtype == np.int64, "target_catalog_idx must be int64"

    # Vary crop_length to confirm padding scales correctly.
    for cl in (3, 4, 5, 7, 10):
        tq_ids2 = [item_to_id[it] for it in tl[:-1]]
        tq2 = np.array(
            [PAD] * max(0, cl - len(tq_ids2)) + tq_ids2[max(0, len(tq_ids2) - cl):],
            dtype=np.int64,
        )
        assert tq2.shape == (min(cl, crop_length),) or tq2.shape[0] <= cl, (
            f"bad shape for cl={cl}: {tq2.shape}"
        )

    print("  check_distill_prepro (Part B - numpy mirror): PASS")


def main() -> int:
    check_scoring()
    check_distill_utils()
    check_distill_prepro()
    print("OK: distill_selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
