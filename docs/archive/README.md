# Archived pilots (index)

Exploratory extension pilots that were tried and then superseded or abandoned. None is part
of the final RQ1-4 findings. Their code, reports, and jobs have been removed from the working
tree to keep the repository minimal; the full pre-cleanup material is preserved in git history
(and on the `pre-cleanup-backup` branch). This page is the record of what was tried and why it
was retired. For the canonical results see [`../../EXTENSION_RESULTS.md`](../../EXTENSION_RESULTS.md),
[`../EXTENSION_SUMMARY.md`](../EXTENSION_SUMMARY.md), and `reports/figures/fig_rq{1,2,3,4}_*`.

| Pilot | What it was | Why retired |
|---|---|---|
| REACH | "collapse at decode step 2" mechanism analysis (sibling competition within an L1 prefix); reproduced on a second Sports codebook | superseded by the validated Chao1/Good-Turing estimator and by the exact oracle, which located the cause in the model ranking, not a decode-step artifact |
| Conformal sets | split conformal prediction at depth 20 and 100 | part of the earlier "depth-100 crossover / decoding" framing the exact oracle superseded; kept conceptually as the top-K-exposure-ceiling hint |
| MBR re-rank | minimum-Bayes-risk accuracy-vs-reach dial (tau sweep) | a bounded dial, superseded as the best dial by the PMI re-rank (which is in the main tree) |
| FUSE | replacing MARIUS's sum-fusion with level-gain / attention | null result (accuracy unchanged); exonerated the fusion as the cause |
| Selective prediction | abstention via MARIUS joint log-probability | a standalone capability, not part of the RQ1-4 story; the cross-paradigm head-to-head was negative (SASRec selects at least as well) |
| Directional gate | order-aware-tokenizer probe (directed grouping vs COSETTE's codes) | shelved clean null: the directed grouping predicted the next item worse than COSETTE's symmetric codes; an order-aware retrain was not justified |

The validated descendants of these pilots that remain in the main tree are
`scripts/extensions/reach_estimator.py` (structural ceiling), `scripts/extensions/exact_catalog.py`
(the oracle), `scripts/extensions/pmi_rerank.py` and `scripts/extensions/mbr.py` (the mitigation
dials), and the train-time `src/models/marius_logitadj.py`.
