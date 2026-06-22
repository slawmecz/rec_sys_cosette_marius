# Extension investigation log (condensed)

Canonical results: [`../EXTENSION_RESULTS.md`](../EXTENSION_RESULTS.md) (verified RQ1-4
numbers) and the paper. This file is the condensed "what we tried and why" record, including
the dead ends. Where it differs from `EXTENSION_RESULTS.md`, trust that file.

## Question

MARIUS generates a recommendation by autoregressively decoding a four-code semantic ID (256
codes per level, RQ-VAE codebooks via COSETTE) with beam search of width 20. We observed a
catalogue-reachability collapse: on Sports, MARIUS reaches about 39 percent of the catalogue
at top-10 against about 74 percent for SASRec++ (milder on Beauty, 71 vs 81 percent). The
question: where does the collapse originate (the tokenizer, the data, the sum-fusion, the
beam search, or the model's learned ranking), and can it be fixed?

## Bottom line

The collapse is a model-ranking property, not a decoding, fusion, tokenizer, or
popularity-reweighting problem. An exact full-catalogue teacher-forced oracle shows the beam
is near-optimal over the model (exact Top-K equals beam Top-K at matched k), while the model
itself ranks the missed items deep (median exact rank about 1300 to 1400, 0 percent in the
exact top-20). This holds at the 90,000-item scale (Arts-2023, five seeds). Every
popularity-correction lever we tried, at the decode stage and at the train stage, cuts
popularity bias but is a bounded dial: it cannot pull buried items into the top-K, and it is
decoupled from catalogue coverage (on Beauty it can even shrink coverage). Cross-paradigm
distillation spreads exposure the most (coverage +123 percent) yet ranks the true targets
deeper still.

## Causal decomposition

| Candidate cause | Test | Verdict |
|---|---|---|
| Search / beam pruning | exact full-catalogue oracle | not it (exact equals beam at matched k) |
| Fusion of code embeddings | un-sum FUSE ablation | not it (three-seed Beauty null) |
| Codebook bug / collisions | occupancy + collision audit | not it (256/256 used, near-uniform) |
| Tokenizer symmetry (order) | directional-structure gate | not it (directional signal too sparse on small data) |
| Code exhaustion | tuple occupancy (~4e-6 of 256^4) | not it |
| Decode-time popularity | PMI / logit re-rank | bounded dial |
| Train-time popularity | logit-adjusted loss; distillation | bounded dial |
| Model's learned ranking | by elimination + oracle | the locus: tail items sit below the top-K exposure cutoff |

## What we tried

Kept:

- Reachability estimator (Chao1 / Good-Turing): the ceiling is structural, not a sampling
  artifact (Sports asymptotic coverage@10 about 46 percent).
- Exact full-catalogue scoring oracle: the centrepiece; decomposes beam misses into search
  error versus model error and shows model error dominates.
- Decode-time PMI / logit re-rank and train-time logit-adjusted loss: bounded dials that trade
  recall for exposure. Honest finding: a generic popularity prior matches the mechanism-matched
  within-prefix prior, so the bias is global-popularity-shaped.
- MBR re-rank: an earlier baseline dial, superseded by the PMI re-rank.
- Cross-paradigm distillation: the most aggressive dial; spreads exposure widely but buries
  targets deeper, confirming model-bound from a third angle.

Dead ends (dropped; code and reports preserved in git history):

- Order-aware / directional COSETTE: a pre-registered controlled null (directional structure
  is too data-starved on the small datasets to beat COSETTE's symmetric signal). Saved a
  multi-day retrain.
- FUSE (un-sum fusion): null, exonerates the architecture.
- Conformal sets: a depth-100 crossover (MARIUS's deeper pool beats SASRec's, but its top
  ordering wastes it), which foreshadowed the top-K-exposure-ceiling reading.
- Targeted step-2 "exploration" decoder: killed before building once the oracle showed the
  miss is not a search error.
- Reach / sibling-competition analysis: describes the pattern (within-prefix popularity), but
  the oracle later located the cause in model ranking rather than beam pruning.

The archived pilot code and reports are removed from the working tree; see git history and
[`archive/README.md`](archive/README.md).

## Positioning and citations

The phenomenon (catalogue collapse in generative recommenders) is not itself novel; the
contribution is the exact-versus-beam oracle instrument, the systematic exoneration of the
other loci, and the negative mitigation result. Concurrent work: Ghost (arXiv:2605.16825)
cures the bias on different data with an asymmetric-unlikelihood objective and a skeleton
tokenizer, so we read as its negative counterweight (untested here). Latte (arXiv:2605.06331)
argues a model-level expressiveness limit theoretically, so we lead with the empirical oracle
rather than claiming the model-level locus as ours. SimGR (arXiv:2602.07847) attributes the
problem to a token-level versus item-level modelling mismatch and ranks items directly; the
oracle independently localizes the cause in the model ranking, consistent with it. The
debias-versus-coverage decoupling is the narrow novel slice (Abdollahpouri et al. document the
general Gini-versus-coverage divergence).
