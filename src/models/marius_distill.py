"""MARIUSDistill: MARIUS subclass with cross-paradigm distillation training.

OWNERSHIP: this file is OURS (course-project extension code). It is NOT part of
the original authors' release. The base class in src/models/marius.py is the
authors' code, byte-identical and unmodified; everything here is additive (a
subclass in a new file), exactly like src/models/marius_logitadj.py.

Hypothesis / role
-----------------
The exact full-catalog scoring oracle (scripts/extensions/exact_catalog.py)
showed the catalog-reachability collapse is MODEL-bound: the beam is a
near-optimal search over the model, but the model itself ranks tail items deep
(median exact rank ~1300). The logit-adjusted arm (MARIUSLogitAdj) corrected
the per-token marginal and behaved like a bounded dial.

This arm asks a different model-side question: can we teach the generative
MARIUS to RANK the catalog the way the strong autoregressive SASRec++ teacher
does, at the final decode position? SASRec++ scores the entire catalog densely
(one dot product per item) and reaches the tail far better. We distill that
ranking into MARIUS without retraining the COSETTE tokenizer: during training we
add a KL term between the teacher's catalog scores and MARIUS's joint code-tuple
log-probabilities over a shared candidate set (the teacher's top-n_cand items
plus the true item). At INFERENCE, search() is inherited UNCHANGED and uses the
RAW logits, so the decoder is unchanged at eval time.

Method
------
For each user in the batch:
  1. Score the catalog with the frozen SASRec++ teacher:
       seq = teacher(teacher_query)[:, -1, :]           # B x d (last position)
       voc = teacher.get_embs()                         # V_sasrec x d (l2-norm)
       cat_scores = (seq @ voc.T)[:, len(SpecialTokens):]  # B x n_catalog
     The index spaces coincide: SASRec item index = catalog index + n_special,
     so dropping the first n_special columns yields catalog-aligned scores.
  2. Build a shared candidate set (the teacher's top-n_cand catalog ids with the
     TRUE item forced into column 0): cand_cat = make_candidates(...).
  3. Map candidates to MARIUS TOKEN space via the precomputed item_to_codes
     table: cand_tokens = item_to_codes[cand_cat]   # B x n_cand x 4 (tokens).
  4. Score the candidates under MARIUS with the grad-enabled tuple scorer
     (teacher-forced joint code-tuple log-prob, conditioned on the same crop's
     history): student_logp = score_marius_tuples_train(self, input, cand_tokens).
  5. Distillation loss = KL(teacher || student) over the candidate set:
       l_distill = F.kl_div(
           F.log_softmax(student_logp, dim=-1),
           F.log_softmax(teacher_cand / distill_temp, dim=-1),
           log_target=True, reduction="batchmean")
  6. loss = base_cross_entropy + distill_alpha * l_distill.

What is overridden
------------------
ONLY __init__ (loads the frozen teacher + the item_to_codes table + the three
distillation hyper-parameters) and get_loss (adds the KL term). train_forward,
depth_forward, temporal_forward and the beam search ``search`` are inherited
UNCHANGED, so inference is identical to the baseline. distill_alpha = 0.0
reproduces the baseline get_loss exactly (early return), and the returned logits
are always the RAW (unadjusted) ones so any downstream consumer and inference
see the base behaviour.

CRITICAL: the item_to_codes table and the teacher checkpoint are QUANT-SPECIFIC
and DATASET-SPECIFIC. The table (built offline by the Task-5 dumper from the
SAME COSETTE -col tokens / quant_id this model trains on) and the teacher run
directory must both correspond to the dataset this MARIUS trains on, or the
token vocabulary and the catalog ordering will not line up.

How to run WITHOUT touching the harness (and the fresh-OUTPUT_ROOT trap)
-----------------------------------------------------------------------
scripts/marius_5seed.py appends EXTRA_TRAIN_OVERRIDES (space-separated hydra
overrides) and hydra can swap the instantiated class:

  EXTRA_TRAIN_OVERRIDES="model.net._target_=src.models.marius_distill.MARIUSDistill \
      +model.net.distill_alpha=0.5 +model.net.distill_temp=1.0 +model.net.n_cand=128 \
      +model.net.teacher_models_root=<output_root>/models \
      +model.net.teacher_run_dir=<sasrec_run_dir> \
      +model.net.item_to_codes_path=reports/extensions/distill/item_to_codes_<Category>.pt"

As with MARIUSFuse / MARIUSLogitAdj, the harness records existing
(category, seed) runs instead of training, so every arm MUST use a FRESH
OUTPUT_ROOT (DATA_ROOT stays shared).
"""

import pathlib

import hydra
import torch
import torch.nn.functional as F
from einops import rearrange

from src.models import SpecialTokens
from src.models.marius import MARIUS
from scripts.benchmark_extensions import get_best_checkpoint
from scripts.extensions.distill_utils import make_candidates
from scripts.extensions.scoring import score_marius_tuples_train


class MARIUSDistill(MARIUS):
    """MARIUS with SASRec++ -> MARIUS catalog-ranking distillation at training.

    Overrides ONLY __init__ (loads the frozen teacher, the item_to_codes table
    and the distillation hyper-parameters) and get_loss (adds the KL term).
    Inference (search) is inherited UNCHANGED. See the module docstring.
    """

    def __init__(
        self,
        temporal_cfg,
        depth_cfg,
        tie_embeddings=False,
        filter_preds=False,
        distill_alpha=0.5,
        distill_temp=1.0,
        n_cand=128,
        teacher_models_root=None,
        teacher_run_dir=None,
        item_to_codes_path=None,
    ):
        super().__init__(
            temporal_cfg,
            depth_cfg,
            tie_embeddings=tie_embeddings,
            filter_preds=filter_preds,
        )

        self.distill_alpha = float(distill_alpha)
        self.distill_temp = float(distill_temp)
        self.n_cand = int(n_cand)

        # item -> 4 COSETTE code tokens (TOKEN space) table, shape (n_catalog, 4).
        # Built offline by the Task-5 dumper. Registered as a NON-persistent
        # buffer so it moves with .to(device) but is excluded from the saved
        # checkpoint (it is a fixed lookup table, reconstructible offline).
        if item_to_codes_path is None:
            raise ValueError(
                "item_to_codes_path is required: pass the path to the offline "
                "item_to_codes table (built from the same quant_id this model uses)"
            )
        table = torch.load(item_to_codes_path)
        if not torch.is_tensor(table):
            table = torch.as_tensor(table)
        self.register_buffer("item_to_codes", table.long(), persistent=False)

        # Load the FROZEN SASRec++ teacher.
        if teacher_models_root is None or teacher_run_dir is None:
            raise ValueError(
                "teacher_models_root and teacher_run_dir are required to load the "
                "frozen SASRec++ teacher"
            )
        cfg, ckpt_path = get_best_checkpoint(
            pathlib.Path(teacher_models_root), teacher_run_dir
        )
        teacher = hydra.utils.instantiate(cfg.model.net)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        net_sd = {
            k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")
        }
        teacher.load_state_dict(net_sd)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # CRITICAL teacher-storage gotcha: assigning an nn.Module to a normal
        # attribute (self._teacher = teacher) auto-registers it as a SUBMODULE
        # via nn.Module.__setattr__. It would then enter self.parameters() (so
        # the optimizer would try to update the frozen teacher and the
        # weight-decay split in get_param_groups would see its tensors) and
        # self.state_dict() (bloating every checkpoint with a duplicate teacher).
        # Storing it directly in __dict__ via object.__setattr__ bypasses
        # nn.Module.__setattr__, so the teacher is held as a plain reference and
        # is invisible to parameters()/state_dict(). The trade-off: it does NOT
        # move with self.to()/self.cuda() (it is unregistered), so we move it to
        # the batch device lazily inside get_loss.
        object.__setattr__(self, "_teacher", teacher)

    def get_loss(self, batch):
        # Base cross-entropy term, identical to MARIUS.get_loss.
        logits, m_target = self.train_forward(batch["input"], batch["target"])
        base = self.criterion(rearrange(logits, "B k v -> B v k"), m_target)

        # distill_alpha == 0.0 reproduces the baseline exactly (no teacher use).
        if self.distill_alpha == 0.0:
            return base, rearrange(logits, "B k v -> B v k")

        device = batch["input"].device

        # The teacher is unregistered, so it does not follow self.to(); move it
        # to the batch device lazily (cheap once it is already there).
        teacher = self._teacher
        if next(teacher.parameters()).device != device:
            teacher.to(device)

        # Teacher catalog scores (no grad: the teacher is frozen).
        with torch.no_grad():
            seq = teacher(batch["teacher_query"])[:, -1, :]  # B x d (last position)
            voc = teacher.get_embs()                          # V_sasrec x d (l2-norm)
            # Drop the special-token columns so scores are catalog-aligned:
            # SASRec item index = catalog index + len(SpecialTokens).
            cat_scores = (seq.float() @ voc.float().T)[:, len(SpecialTokens):]  # B x n_catalog

        # Shared candidate set: teacher's top-n_cand with the TRUE item in col 0.
        cand_cat = make_candidates(
            cat_scores, batch["target_catalog_idx"].long(), self.n_cand
        ).to(cat_scores.device)  # B x n_cand catalog ids (on the score device)

        # Teacher scores at the candidates (gather along the catalog axis).
        teacher_cand = torch.gather(cat_scores, 1, cand_cat)  # B x n_cand

        # Map candidates to MARIUS TOKEN space: B x n_cand x 4.
        cand_tokens = self.item_to_codes.to(cand_cat.device)[cand_cat]

        # Student joint code-tuple log-probs over the same candidates (grad-on).
        student_logp = score_marius_tuples_train(
            self, batch["input"], cand_tokens
        )  # B x n_cand

        # KL(teacher || student): F.kl_div(input=log q, target=log p,
        # log_target=True, reduction="batchmean") = (1/B) sum_b KL(p_b || q_b),
        # with input = student log-softmax and target = teacher log-softmax.
        l_distill = F.kl_div(
            F.log_softmax(student_logp, dim=-1),
            F.log_softmax(teacher_cand / self.distill_temp, dim=-1),
            log_target=True,
            reduction="batchmean",
        )

        loss = base + self.distill_alpha * l_distill

        # Always return the RAW (unadjusted) logits, so inference and any
        # downstream consumer see the base behaviour.
        return loss, rearrange(logits, "B k v -> B v k")
