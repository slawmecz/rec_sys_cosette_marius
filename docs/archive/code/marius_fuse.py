"""MARIUSFuse: MARIUS subclass that replaces the temporal token fusion sum.

OWNERSHIP: this file is OURS (course-project extension code). It is NOT part
of the original authors' release (github.com/Simon-Lepage/cosette_and_marius).
The base class in src/models/marius.py is the authors' code, byte-identical
and unmodified; everything here is additive (a subclass in a new file).

Hypothesis
----------
MARIUS fuses each history item's L=4 COSETTE residual-quantization (RQ) code
embeddings into a single temporal token with a plain sum
(src/models/marius.py, temporal_forward: ``self.temp_emb(input).sum(dim=-2)``).
That sum is permutation-invariant and magnitude-blind: it discards which RQ
level each code came from, even though RQ levels form a coarse-to-fine
hierarchy (level 0 carries most of the semantics, level 3 is a fine residual).
Two items that differ only in their fine code collapse to nearly the same
temporal token, so the temporal transformer cannot distinguish them in the
history. This is a carryover from audio/image RQ-Transformers where the
fused frame is one of thousands; here the fused token IS the item identity.
We test whether a level-aware (level_bias) or content-adaptive (attn) fusion
recovers accuracy that the sum throws away.

Honest qualification (found during selftest, scripts/extensions/
fuse_selftest.py): MARIUS token ids already encode the level via disjoint
vocab ranges (token = raw_code + 256*level + 2), so each summand e_l is
level-disambiguated before fusion; the live deficiency of the sum is its
UNIFORM, content-blind weighting of the four levels, which the attn arm
targets directly. The level_bias arm is degenerate; see its bullet below.

Variants (constructor arg ``fusion``)
-------------------------------------
- "sum":        identical to the base class (sanity / ablation arm).
- "level_bias": fused = sum_l (e_l + b_l), with b_l in R^D a learned per-level
                bias, zero-initialized, applied only on non-pad positions.
                DEGENERACY NOTE (verified numerically in the selftest):
                because the pooling is an unweighted sum, the biases sum out,
                fused = sum_l e_l + B with B = sum_l b_l a learned CONSTANT
                on every real position. The arm is therefore still
                permutation-invariant and its function class equals the
                baseline's (B is absorbable into temp_pos_emb on non-pad
                positions). Keep it as an optimization-null / control arm
                (what does a vacuous +1024-param intervention do across
                seeds?), not as a true order intervention. Adds
                num_levels * d_model = 4 * 256 = 1024 params.
- "level_gain": fused = sum_l (g_l * e_l), with g_l in R^D a learned per-level
                GAIN, ones-initialized (so init output == the sum baseline
                exactly). Unlike level_bias, the gains do NOT sum out: they
                reweight each level's contribution multiplicatively, so this
                is the cheapest NON-degenerate level-sensitive arm (it breaks
                permutation invariance across levels and can amplify or mute
                the fine residual globally). Adds num_levels * d_model =
                4 * 256 = 1024 params. Added after the selftest exposed the
                level_bias degeneracy; this is the arm level_bias was meant
                to be.
- "attn":       learned-query attention pool over the 4 code embeddings:
                scores_l = (e_l . q) / sqrt(D), w = softmax_l(scores),
                fused = scale * sum_l w_l * e_l, with q in R^D zero-initialized
                and scale a learned scalar initialized to num_levels (= 4.0).
                At init q = 0 gives uniform w = 1/4, so fused = 4 * mean = the
                exact sum baseline; training can then learn to up- or
                down-weight levels by content (e.g. amplify the fine residual).
                Note this variant is content-adaptive but still permutation-
                invariant across levels (a shared query scores each code
                independently); only level_bias injects level identity.
                Adds d_model + 1 = 257 params.

Exact-zero pad handling (CRITICAL)
----------------------------------
Histories are left-padded; a padded position has all-PAD codes, and
temp_emb has padding_idx=PAD(=0), so its code embeddings are exactly zero and
the base sum is exactly zero there. Both new variants multiply the fused
output by (input[:, :, 0] != SpecialTokens.PAD.value) so padded positions are
exactly zero BEFORE the positional embedding is added, byte-for-byte matching
the base behavior. Without the mask, level_bias would add sum_l b_l to padded
positions. (For attn, zero rows give uniform w over zero vectors, hence zero
output even unmasked, but we mask anyway for cleanliness.) Downstream masking
semantics (causal mask, src_key_padding_mask) are EXACTLY the base's.

Shape walk-through (small config: D=256, K=4 levels, crop_length=50)
--------------------------------------------------------------------
input:      B x L x K   int64 token ids (token = raw_code + 256*level + 2;
                        PAD=0, BOS=1), left-padded along L (L <= 50)
code_embs:  B x L x K x D   temp_emb lookup (PAD rows are exact zeros)
pad_mask:   B x L           bool, True where the position is a real item
fusion:     B x L x K x D -> B x L x D
            sum:        sum over dim=-2
            level_bias: (code_embs + b[None, None, :K, :]).sum(dim=-2),
                        then * pad_mask[..., None]
            attn:       scores B x L x K = einsum(code_embs, q)/sqrt(D);
                        w = softmax(scores, dim=-1);
                        fused = scale * einsum(w, code_embs),
                        then * pad_mask[..., None]
+ pos emb:  B x L x D   (temp_pos_emb[:, :L, :], same as base)
dropout, temp_tf with causal_mask[:L, :L] and src_key_padding_mask: unchanged.

Everything else (depth transformer, train_forward, get_loss, beam search
``search``, filter_preds) is inherited untouched. get_param_groups exempts
parameters whose names contain "temp_emb"/"depth_emb" from weight decay; the
new parameters (fuse_level_bias, fuse_query, fuse_scale) match neither, so
they correctly fall into the decay group. Left as is on purpose.

How to run WITHOUT touching the harness
---------------------------------------
scripts/marius_5seed.py hardcodes experiment=marius_small but appends
EXTRA_TRAIN_OVERRIDES (env, space-separated hydra overrides), and hydra can
swap the instantiated class:

  EXTRA_TRAIN_OVERRIDES="model.net._target_=src.models.marius_fuse.MARIUSFuse +model.net.fusion=attn"

OPERATIONAL TRAP (will silently produce WRONG results if ignored): the
harness's process_seed/find_run_for_seed discovers existing runs only by
(category, seed) under OUTPUT_ROOT/models with prefix MARIUS_small, and
"records existing metrics instead of training". A FUSE run pointed at the
same OUTPUT_ROOT as the baseline would silently RECORD THE BASELINE RUN
instead of training the FUSE model. Every FUSE variant MUST use a fresh
OUTPUT_ROOT (e.g. /scratch-shared/$USER/cosette_marius/outputs_fuse_attn);
DATA_ROOT stays shared (jobs/env_common.sh honors a pre-set OUTPUT_ROOT).
"""

import torch

from src.models import SpecialTokens
from src.models.marius import MARIUS

_FUSION_CHOICES = ("sum", "level_bias", "level_gain", "attn")


class MARIUSFuse(MARIUS):
    """MARIUS with a configurable fusion of the per-item RQ code embeddings.

    Overrides ONLY __init__ (adds the fusion parameters) and temporal_forward
    (replaces the fusion sum). See the module docstring for the hypothesis,
    the variants, and the exact-zero pad-handling contract.
    """

    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
        fusion="sum",
    ):
        super().__init__(
            temporal_cfg,
            depth_cfg,
            tie_embeddings=tie_embeddings,
            filter_preds=filter_preds,
        )

        if fusion not in _FUSION_CHOICES:
            raise ValueError(
                f"fusion must be one of {_FUSION_CHOICES}, got {fusion!r}"
            )
        self.fusion = fusion

        # Number of RQ levels per item (= depth sequence length = 4).
        self.num_levels = int(self.depth_cfg.seq_len)
        d_model = self.temporal_cfg.d_model

        # NOTE: parameter names deliberately avoid the "temp_emb"/"depth_emb"
        # substrings so get_param_groups keeps them in the weight-decay group.
        if fusion == "level_bias":
            # Zero init: the model starts exactly at the sum baseline.
            self.fuse_level_bias = torch.nn.Parameter(
                torch.zeros(self.num_levels, d_model)
            )
        elif fusion == "level_gain":
            # Ones init: the model starts exactly at the sum baseline.
            self.fuse_level_gain = torch.nn.Parameter(
                torch.ones(self.num_levels, d_model)
            )
        elif fusion == "attn":
            # Zero-init query -> uniform softmax weights at init; scale = K
            # makes scale * mean == sum, so init output == sum baseline.
            self.fuse_query = torch.nn.Parameter(torch.zeros(d_model))
            self.fuse_scale = torch.nn.Parameter(
                torch.tensor(float(self.num_levels))
            )

    def temporal_forward(self, input):
        # Assuming input has already been left-padded (same contract as base).
        # Shape: B x L x K -> B x L x D
        B, L, K = input.shape

        if self.fusion == "sum":
            # Sanity/ablation arm: byte-identical to the authors' forward.
            return super().temporal_forward(input)

        # B x L x K x D; padded positions are exact zero rows (padding_idx).
        code_embs = self.temp_emb(input)

        # B x L bool, True where the position holds a real item. Same
        # position-level criterion the base uses for src_key_padding_mask;
        # real tokens are >= 2 (token = raw + 256*level + 2), so no real
        # code can equal PAD.
        pad_mask = input[:, :, 0] != SpecialTokens.PAD.value

        if self.fusion == "level_bias":
            # fused = sum_l (e_l + b_l); B x L x K x D -> B x L x D
            fused = (
                code_embs + self.fuse_level_bias[None, None, :K, :]
            ).sum(dim=-2)
            # Exact zero on padded positions (else they would get sum_l b_l).
            input_embs = fused * pad_mask.unsqueeze(-1).to(fused.dtype)
        elif self.fusion == "level_gain":
            # fused = sum_l (g_l * e_l); B x L x K x D -> B x L x D
            fused = (
                code_embs * self.fuse_level_gain[None, None, :K, :]
            ).sum(dim=-2)
            # Pads are already exact zeros (gains multiply zero rows), but
            # mask anyway so the contract is explicit.
            input_embs = fused * pad_mask.unsqueeze(-1).to(fused.dtype)
        else:  # "attn"
            # scores_l = (e_l . q) / sqrt(D); B x L x K
            scores = torch.einsum(
                "blkd,d->blk", code_embs, self.fuse_query
            ) / (code_embs.shape[-1] ** 0.5)
            # Softmax over the K levels; B x L x K
            weights = torch.softmax(scores, dim=-1)
            # fused = scale * sum_l w_l e_l; B x L x D
            fused = self.fuse_scale * torch.einsum(
                "blk,blkd->bld", weights, code_embs
            )
            # Already exactly zero on pads (uniform w over zero vectors),
            # but mask anyway so the contract is explicit.
            input_embs = fused * pad_mask.unsqueeze(-1).to(fused.dtype)

        # From here on: EXACTLY the base temporal_forward.
        input_embs = input_embs + self.temp_pos_emb[:, : input.shape[1], :]

        input_embs = self.temp_dropout(input_embs)

        out = self.temp_tf(
            input_embs,
            mask=self.causal_mask[:L, :L],
            src_key_padding_mask=input[:, :, 0] == SpecialTokens.PAD.value,
        )

        return out
