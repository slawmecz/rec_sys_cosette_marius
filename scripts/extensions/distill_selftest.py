#!/usr/bin/env python3
"""Torch-free verification of the grad-enabled tuple scorer refactor.

Mirrors scripts/extensions/logitadj_selftest.py: static/AST checks only,
no GPU, no torch import.

  python scripts/extensions/distill_selftest.py
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORING = REPO / "scripts" / "extensions" / "scoring.py"


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


def main() -> int:
    check_scoring()
    print("OK: distill_selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
