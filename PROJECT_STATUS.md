# Project Status After Midterm Presentation

Date: 2026-06-16

This is the current handoff state after the midterm presentation. It summarizes
what we established in the reproduction and extension work, and what should be
treated as open.

## Reproduction State

- The mandatory small-dataset reproduction is complete for Amazon 2014 Beauty
  and Sports & Outdoors.
- We ran SASRec++ and MARIUS+COSETTE with five seeds, 42-46, reporting Recall
  and NDCG at 5 and 10.
- The authors' model, loss, and evaluation code are unmodified. The TA fork
  mainly added Amazon-2014 ingestion, small-dataset configs, and infrastructure.
- The senior TA's `REPRODUCIBILITY.md` is an earlier single-seed MARIUS-focused
  study. It did not run SASRec++; our later reproduction did.

## Confirmed Reproduction Findings

- Repository-default SASRec++ is not the paper's selected small-dataset config.
  The repo default uses `d=128`, L2 normalization, and sampled-softmax
  temperature `0.05`.
- The paper-faithful Beauty SASRec++ configuration is `d=32`, two layers, no
  normalization, and temperature `1.0`. We used `d_head=16` to keep two heads.
- Correcting this raises Beauty SASRec++ R@10 from `8.24 +/- 0.14` to
  `9.06 +/- 0.06`, recovering about 55% of the gap to the paper's `9.73`.
- Sports was not part of the paper's SASRec++ sizing study. We bracketed it
  with `d=32` and `d=64`; they are statistically tied around R@10 5.1, still
  about 20-21% below the paper.
- MARIUS already used the paper's small-dataset configuration:
  `d=256`, 2 temporal layers, 2 depth layers, dropout `0.4/0.1`.
- The paper's small-dataset claim that MARIUS+COSETTE is competitive with or
  above SASRec++ does not reproduce in our Beauty setup:
  faithful SASRec++ R@10 is 9.06 while MARIUS+COSETTE is 8.17.

## Ruled-Out Explanations

- Undertraining: the confusing W&B `_step` field is a logging counter, not the
  optimizer step. Runs trained for roughly 80k optimizer steps and converged.
- Data split: processed user/item counts match the paper exactly.
- Evaluation: full-catalog ranking, history filtering, and single-positive
  Recall/NDCG match the authors' code.
- Batch size: global batch 256 matches the paper.
- COSETTE health: code usage, entropy, collision rate, and target-code
  alignment do not support the hypothesis of a broken tokenizer.

## Remaining Reproduction Uncertainty

- The residual Beauty and Sports gaps are not fully explained.
- The leading hypothesis is validation-to-test generalization and checkpoint
  selection granularity, since validation is checked only every 10k steps.
- The authors do not publish their exact random seeds or all preprocessing
  tie-breaking details, so an exact numerical reproduction is impossible to
  certify from public artifacts alone.

## Extension State (updated 2026-06-17)

The extension is a diagnosis of catalog-reachability collapse in semantic-ID
generative recommendation. The full investigation log, with every idea, result,
and hypothesis, is in EXTENSION_SUMMARY.md. The headline below replaces the older
"decoding is the locus" framing.

Established (this session resolved the central question):

- The decisive experiment (exact full-catalog teacher-forced scoring) is DONE.
  Verdict: the collapse is MODEL-bound, not a search/beam error. The beam is a
  near-optimal search over the model; the model itself ranks the missed items deep
  (median exact rank about 1300-1400, 0% in the exact top-20). With history
  filtering, exact top-K equals beam top-K.
- The reachability ceiling is structural at top-K (Good-Turing/Chao1: Sports
  asymptotic coverage about 46%, unseen-mass 0.34%), but the model can score the
  buried items into a deeper frontier (exact top-100 reaches about 61% Sports / 83%
  Beauty), so it is a top-K exposure-ranking ceiling, not an absolute representational
  impossibility.
- Popularity correction is a bounded dial at BOTH stages: decode-time (PMI /
  logit-adjusted re-rank, beats MBR) and train-time (MARIUSLogitAdj). Each cuts ARP/
  Gini (train-time: ARP Sports 101->55, Beauty 56->33) but trades accuracy and does
  not recover buried items as hits. Popularity de-biasing is DECOUPLED from catalog
  reach (it can even shrink coverage: Beauty -1.7pp).
- The order-aware / directional COSETTE idea was tested with a pre-registered gate
  and SHELVED: a directed grouping predicts the next item worse than COSETTE's
  existing codes (adjacency transitions are too sparse on small data).
- Systematic exoneration: not the search (oracle), not the fusion (FUSE null), not
  the codebook (audit + occupancy about 4e-6), not the tokenizer symmetry (gate), not
  code-assignment (misses spread across 240/256 L1 codes). By elimination the locus
  is the model's learned ranking.

## Current Extension Conclusion

Catalog collapse in COSETTE+MARIUS is a model-ranking phenomenon: the model ranks
about half the catalog's relevant items below the top-K exposure cutoff for their
target users, the beam faithfully reflects this, and popularity re-weighting at
either stage is a bounded dial. This is novel as a package (exact-scoring oracle +
structural ceiling + two-lever bounded dial + debias/coverage decoupling + density
inversion) and not scooped, with three required framing fixes (cite/contrast Ghost
2605.16825 and Latte 2605.06331; down-scope "expressiveness limit"; claim the narrow
version of the decoupling, citing Abdollahpouri). See EXTENSION_SUMMARY.md sections
7-8 for novelty and the reviewer-objection defenses.

The only remaining experiment that needs new GPU is the large-dataset scale test
(Tier E, Amazon-2023 Office Products ~77k), which is also the course rubric's
large-dataset prong. Cheap consolidation (accuracy-vs-coverage correlation, per-seed
std, oracle on the logit-adj checkpoint) defends the remaining reviewer objections
with artifacts we already have.

## Large-Dataset Status

Teammates are currently running larger-dataset reproductions. These results
should be integrated into:

- reproduction tables, to check whether paper-level accuracy is recovered at
  larger scale;
- reachability metrics, to test whether catalog collapse disappears, persists,
  or changes regime with more data, larger models, and different code-space load.

Do not claim yet that large datasets solve or worsen the reachability problem.
Accuracy alone is insufficient; we need coverage, unreachable demand mass,
prefix-family survival, and exact-versus-beam decomposition.

## Key Files

- `EXTENSION_SUMMARY.md`: AUTHORITATIVE extension log (every idea, result, hypothesis,
  novelty, reviewer defenses, next steps). Read this first for the extension.
- `REPLICATION_REPORT.md`: authoritative small-dataset reproduction report.
- `FAITHFUL_RERUN.md`: SASRec++ paper-faithful rerun rationale and outcome.
- `reports/paper_faithful_rerun/FINDINGS.md`: Snellius rerun results.
- `reports/extensions/PILOTS.md`: the four early pilots (REACH/MBR/conformal/FUSE).
  PARTLY SUPERSEDED: its "locus is decoding, build a targeted decoder" verdict was
  refuted by the exact-scoring oracle; trust EXTENSION_SUMMARY.md where they differ.
- `scripts/extensions/`: exact_catalog.py, pmi_rerank.py, direction_gate.py,
  reach_estimator.py, compute_token_prior.py (+ the earlier pilot scripts).
- `src/models/marius_logitadj.py`: train-time logit-adjustment subclass.
- `notebooks/project_story.ipynb`: end-to-end story notebook.
- `notebooks/extension_pilots.ipynb`: focused extension-pilot notebook.
- `reports/figures/`: presentation-ready figures exported after the midterm.
