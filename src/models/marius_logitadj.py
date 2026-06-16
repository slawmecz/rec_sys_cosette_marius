"""MARIUSLogitAdj: MARIUS subclass with logit-adjusted training (no inference change).

OWNERSHIP: this file is OURS (course-project extension code). It is NOT part of the
original authors' release. The base class in src/models/marius.py is the authors'
code, byte-identical and unmodified; everything here is additive (a subclass in a
new file), exactly like src/models/marius_fuse.py.

Hypothesis / role
-----------------
The exact full-catalog scoring oracle (scripts/extensions/exact_catalog.py) showed
the catalog-reachability collapse is MODEL-bound: the beam is a near-optimal search
over the model, but the model itself ranks tail items deep (median exact rank
~1300). Decode-time repairs (MBR, conformal, the PMI re-rank) only trade accuracy
for reach -- they are bounded dials. This arm asks the model-side question: can a
TRAINING-time correction make the model rank tail items higher without retraining
the tokenizer?

Method: logit adjustment (Menon et al. 2021, "Long-tail learning via logit
adjustment"). During training we minimise

    CE( logits + tau * log pi , target )

where pi is the per-token class prior (the popularity-weighted marginal of each
code token among training targets, precomputed by
scripts/extensions/compute_token_prior.py). At INFERENCE the model uses the RAW
logits (search() is inherited untouched), which is the Bayes-consistent balanced
predictor. tau scales the correction; tau = 0 reproduces the baseline exactly.

This is the model-side analogue of the decode-time PMI re-rank
(scripts/extensions/pmi_rerank.py). The forked outcome is informative either way:
if it recovers tail recall without hurting head accuracy it is a genuine fix (and
the course's strongest popularity-bias mitigation); if it behaves like the
decode-time dials (a bounded accuracy-vs-reach trade), that is strong evidence the
collapse is a fundamental representation / expressiveness limit, not a trainable
ranking artefact.

What is overridden
------------------
ONLY __init__ (loads the fixed prior buffer + tau) and get_loss (adds the prior to
the logits before the cross-entropy). train_forward, depth_forward, temporal_forward
and the beam search ``search`` are inherited UNCHANGED, so inference is identical to
the baseline. The prior is a registered BUFFER, not a Parameter, so it is excluded
from optimisation and from get_param_groups' weight-decay split (no name-collision
concern). tau is a plain float.

CRITICAL: the prior is QUANT-SPECIFIC. The npy passed via logit_adj_prior_path must
be computed (compute_token_prior.py) from the SAME COSETTE -col tokens (quant_id)
this model trains on, or it will not line up with the token vocabulary.

How to run WITHOUT touching the harness (and the fresh-OUTPUT_ROOT trap)
-----------------------------------------------------------------------
scripts/marius_5seed.py appends EXTRA_TRAIN_OVERRIDES (space-separated hydra
overrides) and hydra can swap the instantiated class:

  EXTRA_TRAIN_OVERRIDES="model.net._target_=src.models.marius_logitadj.MARIUSLogitAdj \
      +model.net.logit_adj_tau=1.0 \
      +model.net.logit_adj_prior_path=reports/extensions/logitadj/prior_<Category>.npy"

As with MARIUSFuse, the harness records existing (category, seed) runs instead of
training, so every arm MUST use a FRESH OUTPUT_ROOT (DATA_ROOT stays shared).
"""

import numpy as np
import torch
from einops import rearrange

from src.models.marius import MARIUS


class MARIUSLogitAdj(MARIUS):
    """MARIUS with logit-adjusted training; inference (search) is unchanged.

    Overrides ONLY __init__ (adds the fixed log-prior buffer and tau) and get_loss
    (adds tau * log pi to the logits before cross-entropy). See the module docstring.
    """

    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
        logit_adj_tau=1.0,
        logit_adj_prior_path=None,
    ):
        super().__init__(
            temporal_cfg,
            depth_cfg,
            tie_embeddings=tie_embeddings,
            filter_preds=filter_preds,
        )

        self.logit_adj_tau = float(logit_adj_tau)
        V = self.temporal_cfg.vocab_size
        if logit_adj_prior_path is None:
            # No prior supplied -> zero log-prior == baseline (sanity/ablation arm).
            log_prior = torch.zeros(V, dtype=torch.float32)
        else:
            arr = np.load(logit_adj_prior_path).astype("float32")
            if arr.shape != (V,):
                raise ValueError(
                    f"logit_adj prior has shape {arr.shape}, expected ({V},); "
                    f"the prior must be computed from the same quant_id this model uses"
                )
            log_prior = torch.from_numpy(arr)
        # Buffer (not a Parameter): fixed, moves with .to(device)/dtype, and is
        # invisible to get_param_groups' weight-decay split.
        self.register_buffer("logit_adj_log_prior", log_prior)

    def get_loss(self, batch):
        # Mirrors the base get_loss, but adds tau * log pi to the per-token logits
        # BEFORE the cross-entropy. The returned logits are the RAW (unadjusted)
        # ones, so any downstream consumer and inference see the base behaviour.
        input, target = batch["input"], batch["target"]
        logits, m_target = self.train_forward(input, target)  # N x K x V

        adjusted = logits + self.logit_adj_tau * self.logit_adj_log_prior.to(logits.dtype)
        loss = self.criterion(rearrange(adjusted, "B k v -> B v k"), m_target)

        return loss, rearrange(logits, "B k v -> B v k")
