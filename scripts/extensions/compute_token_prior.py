#!/usr/bin/env python3
"""Per-token training-prior for logit-adjusted MARIUS training (Tier "arm").

Logit adjustment (Menon et al. 2021, "Long-tail learning via logit adjustment")
trains with CE(logits + tau * log pi_y, y), where pi_y is the class (label) prior,
and predicts at inference with the RAW logits. It is the model-side analogue of
our decode-time PMI re-rank: it pushes the model to not collapse onto frequent
classes during training. Here a "class" at depth position l is the next item's
level-l code TOKEN, so we need the marginal frequency of each code token among
training targets.

This script estimates that marginal as the popularity-weighted token frequency:
for each catalog item j with codes (c0..c_{L-1}) and train popularity pop[j], each
of its L tokens (token[l] = c_l + l*K_cb + n_special) accrues pop[j]. (A next-item
target's frequency is well-approximated by the item's train interaction count.)
It writes log pi over the full V = n_special + L*K_cb token vocab.

CRITICAL: the prior is QUANT-SPECIFIC (it depends on which items map to which
codes). It is computed from the committed tuple_to_item.json under
reports/extensions/topk/<category>/, which matches the existing baseline
checkpoints' COSETTE -col tokens. The logit-adjusted retrain MUST reuse the SAME
quant_id, or the prior will not line up with the model's vocabulary.

  python scripts/extensions/compute_token_prior.py --category Sports_and_Outdoors
  python scripts/extensions/compute_token_prior.py --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from scripts.extensions.compute_beyond_accuracy import load_support  # noqa: E402


def token_log_prior(t2i: dict, pop, n_special: int, L: int, k_cb: int, eps: float = 1.0):
    """log pi over V = n_special + L*k_cb tokens (popularity-weighted token marginal)."""
    V = n_special + L * k_cb
    count = np.zeros(V, dtype=np.float64)
    pop = np.asarray(pop, dtype=np.float64)
    for key, j in t2i.items():
        raw = [int(x) for x in key.split(",")]
        w = float(pop[int(j)])
        for l, c in enumerate(raw):
            count[c + l * k_cb + n_special] += w
    smoothed = count + eps
    log_prior = np.log(smoothed / smoothed.sum())
    return log_prior.astype(np.float32), count


def _selftest() -> int:
    # 2 items, L=2, k_cb=2, n_special=2 -> V=6. item0 codes (0,0) pop 9; item1 (1,1) pop 1.
    t2i = {"0,0": 0, "1,1": 1}
    pop = np.array([9.0, 1.0])
    lp, count = token_log_prior(t2i, pop, n_special=2, L=2, k_cb=2, eps=0.0)
    # tokens: item0 -> [0+0+2=2, 0+2+2=4]; item1 -> [1+0+2=3, 1+2+2=5]
    assert count.tolist() == [0, 0, 9, 1, 9, 1], count.tolist()
    # special tokens 0,1 have count 0 (eps=0 -> -inf); the popular token 2 > rare token 3
    assert lp[2] > lp[3] and lp[4] > lp[5]
    # normalization: exp(log_prior) sums to 1 over nonzero-count tokens
    finite = np.isfinite(lp)
    assert abs(np.exp(lp[finite]).sum() - 1.0) < 1e-6
    # with eps>0 all finite and still monotone in popularity
    lp2, _ = token_log_prior(t2i, pop, 2, 2, 2, eps=1.0)
    assert np.isfinite(lp2).all() and lp2[2] > lp2[3]
    print("OK: compute_token_prior selftest passed.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Per-token training prior for logit-adjusted MARIUS")
    p.add_argument("--dump-root", type=Path, default=REPO / "reports" / "extensions" / "topk")
    p.add_argument("--category", default="Sports_and_Outdoors")
    p.add_argument("--eps", type=float, default=1.0, help="Laplace smoothing on token counts")
    p.add_argument("--out-dir", type=Path, default=REPO / "reports" / "extensions" / "logitadj")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return _selftest()

    dump_dir = args.dump_root / args.category
    meta, pop, _emb, t2i = load_support(dump_dir)
    L, n_special = meta["L"], meta["n_special"]
    k_cb = max(int(x) for key in t2i for x in key.split(",")) + 1
    log_prior, count = token_log_prior(t2i, pop, n_special, L, k_cb, args.eps)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"prior_{args.category}.npy"
    np.save(out, log_prior)
    meta_out = args.out_dir / f"prior_{args.category}.json"
    meta_out.write_text(json.dumps({
        "category": args.category, "V": int(len(log_prior)), "n_special": n_special,
        "L": L, "k_cb": k_cb, "eps": args.eps,
        "n_tokens_zero_count": int((count == 0).sum()),
        "log_prior_min": float(log_prior.min()), "log_prior_max": float(log_prior.max()),
    }, indent=2))
    print(f"V={len(log_prior)} k_cb={k_cb} L={L} n_special={n_special} "
          f"zero-count tokens={int((count == 0).sum())} "
          f"log_prior range [{log_prior.min():.3f}, {log_prior.max():.3f}]")
    print(f"Wrote {out}\nWrote {meta_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
