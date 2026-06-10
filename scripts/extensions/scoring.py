#!/usr/bin/env python3
"""Candidate scoring for the dumped Top-K lists (torch; GPU-side only).

Used by scripts/extensions/dump_topk.py --with-scores to attach per-candidate
model scores to the npz dumps, which scripts/extensions/mbr.py then consumes
as MBR weights. This module imports torch and therefore must NEVER be imported
by the locally-run numpy CLIs (mbr.py, compute_beyond_accuracy.py); on the
local machine it is only syntax-checked with python3 -m py_compile.

Both functions are read-only with respect to the models: they call the
authors' PUBLIC modules (MARIUS.temporal_forward / mid_proj / depth_emb /
depth_forward, SASRec.forward / get_embs) exactly as train_forward / search
do, without touching any model code.

Autocast: neither function opens an autocast context; the CALLER is expected
to invoke them inside the same torch.autocast(bfloat16) scope it used for
search() (see TopKCollector in dump_topk.py), so the scores match the
arithmetic that produced the dumped rankings. The final log_softmax / gather
is computed in float32 for numerical stability, which is autocast-safe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def score_marius_tuples(net, batch_input, code_tokens):
    """Joint log-probability of given TOKEN-space code tuples under MARIUS.

    Teacher-forces the depth decoder on each candidate tuple, conditioned on
    the last temporal position of the user's history, replicating exactly the
    per-step quantity the beam search accumulates in net.search (sum over
    levels of log_softmax of the emitted token).

    Args:
      net:         the MARIUS module (eval mode; caller handles autocast).
      batch_input: int64[B, T, L] left-padded history in TOKEN space (the
                   "input" tensor of a MARIUS eval batch).
      code_tokens: int64[B, C, L] candidate code tuples in TOKEN space, i.e.
                   token[l] = raw_code[l] + l*K_cb + n_special, exactly as
                   returned by net.search (the dumped topk_codes).

    Returns:
      float32[B, C]: sum over the L levels of the log-prob of the emitted
      token, i.e. log p(candidate | history). Higher = more confident.

    Shapes inside: temporal_forward(B,T,L) -> (B,T,D_temp); mid_proj last
    position -> (B,d); per candidate dec_embs = cat([mid (1,d),
    depth_emb(tokens[:-1]) (L-1,d)]) -> (B*C, L, d); depth_forward ->
    (B*C, L, V); gather emitted tokens -> (B*C, L) -> sum -> (B, C).
    """
    B, C, L = code_tokens.shape

    temporal_tokens = net.temporal_forward(batch_input)  # B x T x D_temp
    mid = net.mid_proj(temporal_tokens)[:, -1, :]  # B x d (same slice as search)
    d = mid.shape[-1]

    # One decoder row per (user, candidate).
    mid_rep = mid[:, None, None, :].expand(B, C, 1, d).reshape(B * C, 1, d)
    flat = code_tokens.reshape(B * C, L)

    # Teacher forcing mirrors train_forward: feed [mid, emb(c0..c_{L-2})],
    # so position l of the decoder output predicts token c_l.
    dec_embs = torch.cat([mid_rep, net.depth_emb(flat[:, :-1])], dim=1)  # B*C x L x d
    logits = net.depth_forward(dec_embs)  # B*C x L x V

    log_probs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = log_probs.gather(2, flat[:, :, None]).squeeze(-1)  # B*C x L
    return tok_lp.sum(dim=-1).view(B, C).float()


@torch.no_grad()
def score_sasrec_topk(net, batch, topk_items):
    """Logit values of SASRec++ at the already-dumped Top-K items.

    Recomputes the dense scores exactly as net.search does (last-position user
    embedding against the full vocabulary table) and gathers them at the given
    item ids. It does NOT re-apply filter_preds: the dumped topk_items were
    already selected after the -inf masking of seen items in search(), so
    their logits are the true unmasked values and re-filtering would only risk
    reintroducing -inf at masked-but-dumped positions.

    Args:
      net:        the SASRec module (eval mode; caller handles autocast).
      batch:      eval batch dict; only batch["query"] int64[B, T] is used.
      topk_items: int64[B, K] RAW VOCAB ids (special-token offset included),
                  exactly as returned by net.search (the dumped topk_items).

    Returns:
      float32[B, K]: logits embs @ voc.T at the requested items (monotone in
      the dumped ranking; pass through softmax/temperature downstream if a
      normalized weight is needed).
    """
    query = batch["query"]
    embs = net(query)[:, -1, :]  # B x D, normalized per net config
    voc = net.get_embs()  # V x D
    logits = embs.float() @ voc.float().T  # B x V
    return logits.gather(1, topk_items).float()
