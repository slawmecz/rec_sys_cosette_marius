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


def main() -> int:
    check_scoring()
    check_distill_utils()
    print("OK: distill_selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
