#!/usr/bin/env python3
"""Torch-free verification of src/models/marius_logitadj.py (no GPU, no torch import).

Mirrors scripts/extensions/fuse_selftest.py: static/AST checks that the subclass is
additive and well-formed, plus a numpy mirror of the logit-adjustment arithmetic.
Also validates the committed prior npy lines up with the model vocab (V = 1026).

  python scripts/extensions/logitadj_selftest.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "models" / "marius_logitadj.py"
PRIORS = REPO / "reports" / "extensions" / "logitadj"


def _static_checks() -> None:
    text = SRC.read_text()
    assert all(ord(c) < 128 for c in text), "non-ASCII characters in marius_logitadj.py"
    tree = ast.parse(text)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert len(classes) == 1 and classes[0].name == "MARIUSLogitAdj", "expected one class MARIUSLogitAdj"
    cls = classes[0]
    bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", None) for b in cls.bases]
    assert "MARIUS" in bases, f"must subclass MARIUS, bases={bases}"
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert methods == {"__init__", "get_loss"}, f"must override ONLY __init__ and get_loss, got {methods}"
    # The base is not edited (inference path inherited): the subclass must NOT define
    # any of the forward/search methods.
    for forbidden in ("search", "train_forward", "depth_forward", "temporal_forward"):
        assert forbidden not in methods, f"{forbidden} must stay inherited (inference unchanged)"
    # get_loss must return the RAW logits (not the adjusted ones).
    assert "return loss, rearrange(logits," in text, "get_loss must return RAW logits"
    assert "self.register_buffer(\"logit_adj_log_prior\"" in text, "prior must be a buffer, not a Parameter"
    print("  static/AST checks: PASS")


def _math_mirror() -> None:
    rng = np.random.default_rng(0)
    V = 8
    log_prior = np.log(np.array([8, 4, 2, 1, 1, 1, 1, 1], dtype=np.float64) / 19.0)  # token0 frequent, token3.. rare
    logits = rng.standard_normal((5, 3, V))  # N x K x V

    def adjust(tau):
        return logits + tau * log_prior

    # tau = 0 is an exact no-op.
    assert np.allclose(adjust(0.0), logits), "tau=0 must reproduce raw logits"
    # The adjustment subtracts more from rarer tokens (lower log_prior): at equal raw
    # logit, the frequent token keeps a higher ADJUSTED logit, so cross-entropy on a
    # rare target forces the model to raise its RAW logit to compensate (-> tail boost
    # at inference, which uses raw logits).
    flat = np.zeros(V)
    adj = flat + 1.0 * log_prior
    assert adj[0] > adj[3], "frequent token must have higher adjusted logit than rare at equal raw logit"
    assert np.argmax(adj) == 0 and np.argmin(adj) == np.argmin(log_prior)
    print("  numpy math mirror: PASS")


def _prior_files() -> None:
    seen = 0
    for f in sorted(PRIORS.glob("prior_*.npy")):
        arr = np.load(f)
        assert arr.shape == (1026,), f"{f.name}: shape {arr.shape} != (1026,)"
        assert np.isfinite(arr).all(), f"{f.name}: non-finite log-prior"
        # The two special tokens (PAD, BOS) have zero count -> lowest prior.
        assert int(np.argmin(arr)) in (0, 1), f"{f.name}: argmin {int(np.argmin(arr))} not a special token"
        seen += 1
        print(f"  prior {f.name}: V={arr.shape[0]} range [{arr.min():.2f}, {arr.max():.2f}] PASS")
    if seen == 0:
        print("  (no prior_*.npy yet; run compute_token_prior.py)")


def main() -> int:
    _static_checks()
    _math_mirror()
    _prior_files()
    print("OK: logitadj_selftest passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
