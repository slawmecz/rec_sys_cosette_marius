#!/usr/bin/env python3
"""Torch-free selftest for src/models/marius_fuse.py (FUSE pilot).

torch is broken on the local machine, so this script never imports torch or
the model module. It validates the new file in two ways:

1. Static checks (ast + py_compile on the real file):
   - src/models/marius_fuse.py compiles;
   - it is pure ASCII;
   - class MARIUSFuse subclasses MARIUS and overrides ONLY __init__ and
     temporal_forward (everything else must be inherited);
   - the new parameter names avoid the "temp_emb"/"depth_emb" substrings that
     get_param_groups uses to select the no-weight-decay group;
   - none of the protected files (authors'/TA's code) are touched by import:
     marius_fuse only imports from src.models and src.models.marius.

2. Numerical checks (numpy mirror of the exact fusion math):
   - level_bias with b = 0 equals the sum baseline exactly (init equivalence);
   - attn with q = 0, scale = 4 equals the sum baseline (init equivalence);
   - padded positions (all-PAD codes -> zero embedding rows) produce EXACTLY
     zero fused vectors in all three variants (the pad contract);
   - without the pad mask, level_bias with b != 0 would corrupt padded
     positions by sum_l b_l (i.e. the mask is load-bearing);
   - DEGENERACY (an actual finding of this selftest): level_bias is ALSO
     permutation-invariant; the biases sum out, so on real positions
     level_bias == sum baseline + the constant sum_l b_l. It is a control
     arm, not an order intervention (see the module docstring);
   - attn is content-adaptive (non-uniform weights for q != 0, output differs
     from the sum) but, as documented, still permutation-invariant.

Run: python3 scripts/extensions/fuse_selftest.py
"""

import ast
import pathlib
import py_compile
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
FUSE_PATH = REPO / "src" / "models" / "marius_fuse.py"

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    status = "PASS" if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    line = f"[{status}] {name}"
    if detail:
        line += f" ({detail})"
    print(line)


# ---------------------------------------------------------------------------
# 1. Static checks on the real file
# ---------------------------------------------------------------------------

def static_checks():
    py_compile.compile(str(FUSE_PATH), doraise=True)
    check("py_compile src/models/marius_fuse.py", True)

    raw = FUSE_PATH.read_bytes()
    check("file is pure ASCII", all(b < 128 for b in raw))

    tree = ast.parse(raw.decode("ascii"))
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    check("exactly one class defined", len(classes) == 1,
          f"found {[c.name for c in classes]}")
    cls = classes[0]
    check("class is MARIUSFuse", cls.name == "MARIUSFuse")

    bases = [ast.unparse(b) for b in cls.bases]
    check("subclasses MARIUS", bases == ["MARIUS"], f"bases={bases}")

    methods = sorted(
        n.name for n in cls.body if isinstance(n, ast.FunctionDef)
    )
    check("overrides ONLY __init__ and temporal_forward",
          methods == ["__init__", "temporal_forward"],
          f"methods={methods}")

    imports = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports.extend(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            imports.append(n.module or "")
    allowed = {"torch", "src.models", "src.models.marius"}
    check("imports only torch and src.models[.marius]",
          set(imports) <= allowed, f"imports={sorted(set(imports))}")

    # New params must fall in the weight-decay group of get_param_groups
    # (which exempts names containing "temp_emb" or "depth_emb").
    src = raw.decode("ascii")
    new_params = ["fuse_level_bias", "fuse_level_gain", "fuse_query", "fuse_scale"]
    for p in new_params:
        check(f"param {p} defined and decay-group-safe",
              p in src and "temp_emb" not in p and "depth_emb" not in p)

    # The fusion output must be masked before the positional embedding is
    # added: assert the pad mask multiplication appears for all three variants.
    n_masked = src.count("pad_mask.unsqueeze(-1).to(fused.dtype)")
    check("pad mask applied in all three new variants", n_masked == 3,
          f"count={n_masked}")


# ---------------------------------------------------------------------------
# 2. Numpy mirror of the fusion math
# ---------------------------------------------------------------------------
# Mirrors temporal_forward's fusion stage only (embedding lookup -> fused
# B x L x D), which is the only part MARIUSFuse changes.

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def fuse_sum(code_embs):
    # base: self.temp_emb(input).sum(dim=-2)
    return code_embs.sum(axis=-2)


def fuse_level_bias(code_embs, bias, pad_mask, apply_mask=True):
    K = code_embs.shape[-2]
    fused = (code_embs + bias[None, None, :K, :]).sum(axis=-2)
    if apply_mask:
        fused = fused * pad_mask[:, :, None].astype(fused.dtype)
    return fused


def fuse_level_gain(code_embs, gain, pad_mask, apply_mask=True):
    K = code_embs.shape[-2]
    fused = (code_embs * gain[None, None, :K, :]).sum(axis=-2)
    if apply_mask:
        fused = fused * pad_mask[:, :, None].astype(fused.dtype)
    return fused


def fuse_attn(code_embs, query, scale, pad_mask, apply_mask=True):
    D = code_embs.shape[-1]
    scores = np.einsum("blkd,d->blk", code_embs, query) / np.sqrt(D)
    w = softmax(scores, axis=-1)
    fused = scale * np.einsum("blk,blkd->bld", w, code_embs)
    if apply_mask:
        fused = fused * pad_mask[:, :, None].astype(fused.dtype)
    return fused, w


def numeric_checks():
    rng = np.random.default_rng(0)
    B, L, K, D = 3, 7, 4, 256

    # Simulated embedding lookup output: real positions random, the first
    # 3 positions of sequence 0 and first 5 of sequence 1 padded (all-zero
    # rows, exactly what padding_idx=0 produces).
    code_embs = rng.normal(size=(B, L, K, D)).astype(np.float32)
    pad_mask = np.ones((B, L), dtype=bool)
    pad_mask[0, :3] = False
    pad_mask[1, :5] = False
    code_embs[~pad_mask] = 0.0

    base = fuse_sum(code_embs)

    # --- init equivalence -------------------------------------------------
    lb0 = fuse_level_bias(code_embs, np.zeros((K, D), np.float32), pad_mask)
    check("level_bias(b=0) == sum baseline (exact)",
          np.array_equal(lb0, base * pad_mask[:, :, None]),
          f"max|diff|={np.abs(lb0 - base).max():.2e}")
    # note: base already has exact zeros at pads, so masking is a no-op there
    check("level_bias(b=0) == sum baseline incl. pads",
          np.array_equal(lb0, base))

    at0, w0 = fuse_attn(code_embs, np.zeros(D, np.float32), 4.0, pad_mask)
    check("attn(q=0, scale=4) == sum baseline (allclose 1e-5)",
          np.allclose(at0, base, atol=1e-5),
          f"max|diff|={np.abs(at0 - base).max():.2e}")
    check("attn(q=0) weights exactly uniform 1/4",
          np.array_equal(w0, np.full_like(w0, 0.25)))

    # --- pad contract -----------------------------------------------------
    bias = rng.normal(size=(K, D)).astype(np.float32)
    query = rng.normal(size=D).astype(np.float32)
    lb = fuse_level_bias(code_embs, bias, pad_mask)
    at, w = fuse_attn(code_embs, query, 4.0, pad_mask)
    check("sum: padded positions exactly zero",
          np.all(base[~pad_mask] == 0.0))
    check("level_bias(b!=0): padded positions exactly zero",
          np.all(lb[~pad_mask] == 0.0))
    check("attn(q!=0): padded positions exactly zero",
          np.all(at[~pad_mask] == 0.0))

    lb_unmasked = fuse_level_bias(code_embs, bias, pad_mask, apply_mask=False)
    leak = np.abs(lb_unmasked[~pad_mask] - bias.sum(axis=0)[None, :]).max()
    check("mask is load-bearing: unmasked level_bias leaks sum_l b_l to pads",
          np.all(lb_unmasked[~pad_mask] != 0.0) and leak < 1e-5,
          f"pad rows == sum_l b_l (max|diff|={leak:.2e})")

    # --- order (in)variance and the level_bias degeneracy -------------------
    perm = np.array([3, 1, 0, 2])
    code_perm = code_embs[:, :, perm, :]
    base_perm = fuse_sum(code_perm)
    check("base sum is permutation-INVARIANT (the deficiency)",
          np.allclose(base_perm, base, atol=1e-5))
    lb_perm = fuse_level_bias(code_perm, bias, pad_mask)
    diff = np.abs(lb_perm - lb)[pad_mask].max()
    check("DEGENERACY: level_bias is ALSO permutation-invariant (biases sum out)",
          diff < 1e-4, f"max|diff| on real positions={diff:.2e}")
    const_diff = np.abs(
        lb[pad_mask] - (base[pad_mask] + bias.sum(axis=0)[None, :])
    ).max()
    check("DEGENERACY: level_bias == sum + constant sum_l b_l on real positions",
          const_diff < 1e-4, f"max|diff|={const_diff:.2e}")

    # --- level_gain: the NON-degenerate cheap arm ---------------------------
    lg1 = fuse_level_gain(code_embs, np.ones((K, D), np.float32), pad_mask)
    check("level_gain(g=1) == sum baseline (exact)",
          np.array_equal(lg1, base),
          f"max|diff|={np.abs(lg1 - base).max():.2e}")
    gain = (1.0 + rng.normal(size=(K, D)) * 0.5).astype(np.float32)
    lg = fuse_level_gain(code_embs, gain, pad_mask)
    check("level_gain(g!=1): padded positions exactly zero",
          np.all(lg[~pad_mask] == 0.0))
    lg_perm = fuse_level_gain(code_perm, gain, pad_mask)
    perm_diff = np.abs(lg_perm - lg)[pad_mask].max()
    check("NON-degenerate: level_gain is permutation-SENSITIVE across levels",
          perm_diff > 1.0, f"max|diff| under level perm={perm_diff:.3f}")
    # Not expressible as sum + constant: the residual (lg - base) varies
    # across real positions (unlike level_bias, where it is the constant B).
    resid = (lg - base)[pad_mask]
    resid_spread = np.abs(resid - resid[0][None, :]).max()
    check("NON-degenerate: level_gain != sum + constant",
          resid_spread > 1.0, f"residual spread across positions={resid_spread:.3f}")

    # --- attn adaptivity (and honest invariance) ----------------------------
    real_w = w[pad_mask]
    check("attn(q!=0) weights non-uniform on real items",
          np.abs(real_w - 0.25).max() > 0.05,
          f"max|w-0.25|={np.abs(real_w - 0.25).max():.3f}")
    check("attn(q!=0) output differs from sum baseline",
          np.abs(at - base)[pad_mask].max() > 1.0,
          f"max|diff|={np.abs(at - base)[pad_mask].max():.3f}")
    at_perm, _ = fuse_attn(code_perm, query, 4.0, pad_mask)
    check("attn is (as documented) still permutation-invariant",
          np.allclose(at_perm, at, atol=1e-4))

    # --- parameter accounting (small config) -------------------------------
    d_model, vocab, crop, levels = 256, 1026, 50, 4
    per_layer = (3 * d_model * d_model + 3 * d_model
                 + d_model * d_model + d_model
                 + d_model * 4 * d_model + 4 * d_model
                 + 4 * d_model * d_model + d_model
                 + 4 * d_model)
    tf_total = 2 * per_layer + 2 * d_model  # 2 layers + final LayerNorm
    base_params = (2 * vocab * d_model            # temp_emb + depth_emb
                   + crop * d_model + levels * d_model  # pos embs
                   + 2 * tf_total                 # temporal + depth TF
                   + d_model * d_model + d_model)  # mid_proj
    lb_extra = levels * d_model
    lg_extra = levels * d_model
    at_extra = d_model + 1
    print(f"[info] base MARIUS_small params ~{base_params:,}; "
          f"level_bias +{lb_extra} (+{100*lb_extra/base_params:.3f}%); "
          f"level_gain +{lg_extra} (+{100*lg_extra/base_params:.3f}%); "
          f"attn +{at_extra} (+{100*at_extra/base_params:.3f}%)")
    check("level_bias adds 1024 params", lb_extra == 1024)
    check("level_gain adds 1024 params", lg_extra == 1024)
    check("attn adds 257 params", at_extra == 257)


def main():
    print(f"FUSE selftest on {FUSE_PATH}")
    static_checks()
    numeric_checks()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
