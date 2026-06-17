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

from scripts.extensions.scoring import score_marius_tuples, score_marius_tuples_train


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


def main() -> int:
    check_numeric()
    print("OK: distill_torch_check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
