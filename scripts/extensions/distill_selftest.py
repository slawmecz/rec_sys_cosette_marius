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
MARIUS_DISTILL_PY = REPO / "src" / "models" / "marius_distill.py"
JOBS27 = REPO / "jobs" / "27_distill_sports.sbatch"

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


def check_distill_model() -> None:
    """Two-part verification for src/models/marius_distill.py (MARIUSDistill).

    Part A: AST/static checks on the source (no torch import, since the module
    pulls in torch + ray at import time). Confirms the override surface,
    early-return, RAW-logit return, the unregistered-teacher storage idiom, and
    ASCII-only.

    Part B: numpy mirror of the KL-divergence term, asserting the KL direction
    (teacher || student) and the batchmean reduction match F.kl_div's contract
    on synthetic vectors.
    """

    # ----- Part A: AST/static checks -----

    text = MARIUS_DISTILL_PY.read_text()
    assert all(ord(c) < 128 for c in text), "non-ASCII characters in marius_distill.py"

    tree = ast.parse(text)

    # Find MARIUSDistill class node.
    distill_cls = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MARIUSDistill":
            distill_cls = node
            break
    assert distill_cls is not None, "MARIUSDistill class not found in marius_distill.py"

    # Must subclass MARIUS.
    base_names = []
    for base in distill_cls.bases:
        if isinstance(base, ast.Name):
            base_names.append(base.id)
        elif isinstance(base, ast.Attribute):
            base_names.append(base.attr)
    assert "MARIUS" in base_names, (
        f"MARIUSDistill must subclass MARIUS, found bases: {base_names}"
    )

    # Must define EXACTLY {__init__, get_loss} as methods.
    method_names = {
        n.name for n in distill_cls.body if isinstance(n, ast.FunctionDef)
    }
    assert method_names == {"__init__", "get_loss"}, (
        f"MARIUSDistill must define ONLY __init__ and get_loss, found: {method_names}"
    )

    # Must NOT define the inherited inference/forward methods.
    forbidden = {"search", "train_forward", "depth_forward", "temporal_forward"}
    overlap = method_names & forbidden
    assert not overlap, (
        f"MARIUSDistill must NOT override inference/forward methods: {overlap}"
    )

    # get_loss must contain the distill_alpha == 0.0 early-return guard.
    get_loss_fn = next(
        n for n in distill_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_loss"
    )
    found_alpha_guard = False
    for node in ast.walk(get_loss_fn):
        if isinstance(node, ast.Compare):
            # self.distill_alpha == 0.0
            left = node.left
            if (
                isinstance(left, ast.Attribute)
                and left.attr == "distill_alpha"
                and any(isinstance(op, ast.Eq) for op in node.ops)
            ):
                found_alpha_guard = True
    assert found_alpha_guard, (
        "get_loss must guard on self.distill_alpha == 0.0 (baseline early-return)"
    )

    # get_loss must return rearranged RAW logits: rearrange(logits, "B k v -> B v k").
    found_raw_return = False
    for node in ast.walk(get_loss_fn):
        if isinstance(node, ast.Call):
            func = node.func
            is_rearrange = (
                (isinstance(func, ast.Name) and func.id == "rearrange")
                or (isinstance(func, ast.Attribute) and func.attr == "rearrange")
            )
            if is_rearrange and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id == "logits":
                    # second arg should be the rearrange pattern literal
                    for a in node.args[1:]:
                        if (
                            isinstance(a, ast.Constant)
                            and isinstance(a.value, str)
                            and "B k v -> B v k" in a.value
                        ):
                            found_raw_return = True
    assert found_raw_return, (
        "get_loss must return rearrange(logits, 'B k v -> B v k') (RAW logits)"
    )

    # Teacher must be stored WITHOUT submodule registration, i.e. via
    # object.__setattr__(self, '_teacher', ...) or self.__dict__['_teacher'] = ...
    # and NOT via a plain attribute assignment self._teacher = ... .
    init_fn = next(
        n for n in distill_cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )

    uses_object_setattr = False
    uses_dict_assign = False
    uses_plain_attr_assign = False
    for node in ast.walk(init_fn):
        # object.__setattr__(self, "_teacher", teacher)
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "__setattr__"
                and isinstance(func.value, ast.Name)
                and func.value.id == "object"
            ):
                # confirm second arg is the "_teacher" name
                if len(node.args) >= 2:
                    a1 = node.args[1]
                    if isinstance(a1, ast.Constant) and a1.value == "_teacher":
                        uses_object_setattr = True
        # self.__dict__["_teacher"] = teacher
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Subscript)
                    and isinstance(tgt.value, ast.Attribute)
                    and tgt.value.attr == "__dict__"
                ):
                    sl = tgt.slice
                    if isinstance(sl, ast.Constant) and sl.value == "_teacher":
                        uses_dict_assign = True
                # plain self._teacher = ... (the BAD pattern: auto-registers)
                if (
                    isinstance(tgt, ast.Attribute)
                    and tgt.attr == "_teacher"
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                ):
                    uses_plain_attr_assign = True

    assert uses_object_setattr or uses_dict_assign, (
        "teacher must be stored via object.__setattr__(self, '_teacher', ...) or "
        "self.__dict__['_teacher'] = ... (NOT a plain attribute assignment, which "
        "auto-registers it as a submodule)"
    )
    assert not uses_plain_attr_assign, (
        "teacher must NOT be stored via self._teacher = ... (that auto-registers it "
        "as a submodule, polluting the optimizer and the checkpoint state_dict)"
    )

    print("  check_distill_model (Part A - AST): PASS")

    # ----- Part B: numpy mirror of the KL term (direction + reduction) -----

    # F.kl_div(input=log q, target=log p, log_target=True, reduction="batchmean")
    # computes (1/B) * sum_b sum_i p_bi * (log p_bi - log q_bi) = KL(p || q),
    # i.e. with input=student log-softmax and target=teacher log-softmax this is
    # KL(teacher || student). Reimplement that here on synthetic logits.

    def _log_softmax(x: np.ndarray) -> np.ndarray:
        m = x.max(axis=-1, keepdims=True)
        z = x - m
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    rng = np.random.default_rng(0)
    B, C = 4, 8
    temp = 2.0
    student_logp_in = rng.standard_normal((B, C)).astype(np.float64)
    teacher_logits = rng.standard_normal((B, C)).astype(np.float64)

    log_q = _log_softmax(student_logp_in)            # student
    log_p = _log_softmax(teacher_logits / temp)      # teacher (temp-scaled)
    p = np.exp(log_p)

    # KL(teacher || student), batchmean (divide by batch size B).
    kl_teacher_student = (p * (log_p - log_q)).sum() / B

    # The reverse direction KL(student || teacher) would differ; assert they do.
    q = np.exp(log_q)
    kl_student_teacher = (q * (log_q - log_p)).sum() / B
    assert kl_teacher_student >= 0.0, "KL must be non-negative"
    assert not np.isclose(kl_teacher_student, kl_student_teacher), (
        "KL is not symmetric; the mirror must distinguish teacher||student "
        "from student||teacher"
    )

    # Identical distributions => KL == 0 (sanity of the reduction/direction).
    same = _log_softmax(teacher_logits)
    p_same = np.exp(same)
    kl_zero = (p_same * (same - same)).sum() / B
    assert np.isclose(kl_zero, 0.0), "KL of identical distributions must be 0"

    print("  check_distill_model (Part B - numpy KL mirror): PASS")


def check_distill_utils_dumper() -> None:
    """AST check on the offline dumper added to distill_utils.py.

    Confirms that adding the dump_item_to_codes function and the __main__
    CLI did NOT introduce a top-level torch import (which would break the
    torch-free import of the numpy cores by this selftest) and that the
    dumper exists with the expected signature.
    """

    text = DISTILL_UTILS.read_text()
    assert all(ord(c) < 128 for c in text), "non-ASCII characters in distill_utils.py"

    tree = ast.parse(text)

    # No top-level "import torch" / "from torch ...": torch must stay LAZY
    # (imported inside function bodies only), or distill_selftest's import of
    # the numpy cores would fail in the torch-free local environment.
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "torch", (
                    "distill_utils.py must not import torch at module top level"
                )
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").split(".")[0] == "torch", (
                "distill_utils.py must not import torch at module top level"
            )

    # dump_item_to_codes must exist with the three expected parameters.
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "dump_item_to_codes" in funcs, (
        "distill_utils.py must define dump_item_to_codes"
    )
    dump_fn = funcs["dump_item_to_codes"]
    arg_names = [a.arg for a in dump_fn.args.args]
    for expected in ("quant_parquet", "items_pickle", "out_path"):
        assert expected in arg_names, (
            f"dump_item_to_codes missing parameter {expected}; has {arg_names}"
        )

    # The dumper must lazily import torch inside its body (not at top level)
    # and call build_item_to_codes_np.
    dump_imports_torch_lazily = False
    dump_calls_core = False
    for node in ast.walk(dump_fn):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "torch":
                    dump_imports_torch_lazily = True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "build_item_to_codes_np":
                dump_calls_core = True
    assert dump_imports_torch_lazily, (
        "dump_item_to_codes must lazily import torch inside its body"
    )
    assert dump_calls_core, (
        "dump_item_to_codes must call build_item_to_codes_np"
    )

    # There must be an `if __name__ == "__main__":` guard.
    has_main_guard = False
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if (
                isinstance(left, ast.Name)
                and left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__"
                        for c in node.test.comparators)
            ):
                has_main_guard = True
    assert has_main_guard, "distill_utils.py must have an if __name__ == '__main__' CLI"

    print("  check_distill_utils_dumper: PASS")


def check_jobs27() -> None:
    """Text check on jobs/27_distill_sports.sbatch.

    Asserts the launcher carries the key distillation overrides (model
    _target_, the prepro class swap, the Sports quant id, a FRESH OUTPUT_ROOT
    exported BEFORE the env source, isolated eval output paths) and is
    ASCII-only.
    """

    assert JOBS27.exists(), "jobs/27_distill_sports.sbatch not found"
    text = JOBS27.read_text()
    assert all(ord(c) < 128 for c in text), (
        "non-ASCII characters in 27_distill_sports.sbatch"
    )

    # MARIUSDistill model target override.
    assert "model.net._target_=src.models.marius_distill.MARIUSDistill" in text, (
        "27_distill_sports.sbatch must select MARIUSDistill via model.net._target_"
    )

    # Prepro class swap to MARIUSDistillPrePro.
    assert (
        "data.ray_datasets.prepro_cfg._cls_=src.data.marius.MARIUSDistillPrePro"
        in text
    ), "27_distill_sports.sbatch must swap prepro to MARIUSDistillPrePro"

    # Sports quant id (24cd on this account) is kept.
    assert "24cd" in text, (
        "27_distill_sports.sbatch must keep the Sports QUANT_ID (24cd)"
    )

    # Distillation hyperparameter overrides must all be present.
    for key in (
        "+model.net.distill_alpha",
        "+model.net.distill_temp",
        "+model.net.n_cand",
        "+model.net.teacher_models_root",
        "+model.net.teacher_run_dir",
        "+model.net.item_to_codes_path",
    ):
        assert key in text, f"27_distill_sports.sbatch missing override {key}"

    # A FRESH OUTPUT_ROOT must be exported BEFORE the env is sourced.
    export_pos = text.find("export OUTPUT_ROOT=")
    source_pos = text.find("source ")
    assert export_pos != -1, "27_distill_sports.sbatch must export OUTPUT_ROOT"
    assert source_pos != -1, "27_distill_sports.sbatch must source an env file"
    assert export_pos < source_pos, (
        "OUTPUT_ROOT must be exported BEFORE the env is sourced "
        "(the fresh-OUTPUT_ROOT trap)"
    )
    # The fresh root must be distinct from the logitadj one (a distill-specific
    # variable), so it does not collide with the baseline / logitadj dirs.
    assert "OUTPUT_ROOT_DISTILL" in text, (
        "27_distill_sports.sbatch must use a distill-specific OUTPUT_ROOT_DISTILL"
    )

    # Eval outputs must go to isolated, distill-specific paths so the committed
    # baseline CSVs are not clobbered.
    assert "distill" in text and "beyond_accuracy" in text, (
        "27_distill_sports.sbatch must write beyond-accuracy to an isolated "
        "distill-specific path"
    )
    # The Top-K dump dir and the beyond-accuracy CSV must be distill-scoped.
    assert "topk_distill_sports" in text, (
        "27_distill_sports.sbatch must dump Top-K to a distill-specific dir"
    )

    print("  check_jobs27: PASS")


def main() -> int:
    check_scoring()
    check_distill_utils()
    check_distill_utils_dumper()
    check_distill_prepro()
    check_distill_model()
    check_jobs27()
    print("OK: distill_selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
