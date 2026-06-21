# Extension Results (RQ1-4), verified

Paper-facing consolidation of the final extension results, all numbers verified
against the committed CSVs/JSONs (harvest 2026-06-20). Small datasets = Amazon-2014
Beauty/Sports; large = Amazon-2023 Arts_Crafts_and_Sewing (~90k items). Plain ASCII.

Provenance: Arts 5-seed (42,44,46,48,50) from reports/extensions/topk_arts/seed*/;
oracle from reports/extensions/{topk_arts/seed42/exact_catalog, exact_catalog,
exact_catalog_distill}/; 2014 logit-adj from reports/extensions/topk_logitadj_*;
distillation from reports/extensions/distill/FINDINGS.md.

## RQ1: Is COSETTE/MARIUS reproducible vs SASRec++?

Yes, with a scale-dependent verdict. Test R@10 (mean over seeds, %):

| Dataset (scale) | SASRec++ | MARIUS (COSETTE) | verdict |
|---|---|---|---|
| Beauty 2014 (12,101 items), faithful 5-seed | 9.06 | 8.17 | MARIUS below SASRec |
| Sports 2014 (18,357 items), 5-seed | ~5.1 | 4.87 | MARIUS ~ SASRec |
| Arts 2023 (89,958 items), 5-seed | 4.86 | 5.01 | MARIUS > SASRec (Holm p=0.032) |

Headline: the paper's "COSETTE competitive with / beats SASRec++" claim does NOT
reproduce on small 2014 data but DOES reproduce at large 2023 scale (the small-to-large
flip). See REPLICATION_REPORT.md (2014) and the repro/arts-crafts-2023 report (Arts).

### Arts-2023 reproduction detail (the large-scale prong)

Dataset (Amazon Reviews 2023, Arts_Crafts_and_Sewing, 5-core): 197,286 users,
89,958 items. Seeds {42,44,46,48,50} (spaced by 2; COSETTE retrained per seed).
Both models evaluated under the SAME protocol (filter_preds=True: already-seen items
removed from candidates), so the MARIUS-vs-SASRec comparison is apples-to-apples.
SASRec++ on Arts is a NEW baseline (the reference repro never ran it).

Test metrics, 5-seed mean with 95% Student-t CI (df=4), % (recomputed from the
committed per-seed scores, reports/results/{marius,sasrec}_arts_5seed_full_scores.jsonl):

| Metric | MARIUS | SASRec++ | paper (MARIUS / SASRec) |
|---|---|---|---|
| R@5 | 3.32 [3.27, 3.36] | 3.31 [3.25, 3.37] | 3.49 / 3.51 |
| NDCG@5 | 2.22 [2.19, 2.26] | 2.28 [2.23, 2.33] | 2.37 / 2.42 |
| R@10 | 5.01 [4.97, 5.05] | 4.86 [4.79, 4.93] | 5.30 / 5.09 |
| NDCG@10 | 2.77 [2.74, 2.80] | 2.78 [2.73, 2.83] | 2.95 / 2.93 |

Significance (MARIUS vs SASRec++), exact two-sample permutation test over all
C(10,5)=252 relabelings, two-sided, Holm-corrected across the 4 metrics:

| Metric | diff (M-S) | Welch t | perm p | Holm p | significant |
|---|---|---|---|---|---|
| R@10 | +0.151 | +5.00 | 0.0079 | 0.032 | yes (MARIUS) |
| NDCG@5 | -0.056 | -2.74 | 0.032 | 0.095 | no |
| NDCG@10 | -0.009 | -0.43 | 0.683 | 1.000 | no |
| R@5 | +0.005 | +0.17 | 0.865 | 1.000 | no |

So R@10 is the ONLY metric where MARIUS significantly beats SASRec++ after correction
(the 5+5 seeds are perfectly separated, so perm p hits the n=5 exact floor 2/252=0.0079;
it survives Holm at 0.032). NDCG@5 favors SASRec uncorrected but does not survive Holm.
Quote both p-values; the raw 0.008 alone reads as underpowered.

Caveats (n=5): a paired across-seed test cannot reach alpha=0.05 (sign-flip floor
2/32=0.0625), so the two-sample permutation test is the one with power; resolving the
sub-threshold metrics would need a well-powered per-user paired test (not run). Versus
the paper, we use CI-containment + effect size (no paper per-seed variance is published):
the paper point estimate lies outside our 95% CI on every metric, at 3.8 to 8.1 std from
our mean -- a systematic shortfall (validation-to-test generalization), not seed noise;
we do NOT claim "significantly worse than the paper." Reproducibility note: SASRec on
Arts needs vocab_size = n_items + 2 special = 89,960 (a +2 off-by-one vs the config
default 89,959; PAD + BOS); fixed by a per-run override.

## RQ2: Diversity and popularity bias (Arts 2023, 90k, 5-seed mean +/- std)

### k = 10
| Metric | MARIUS | SASRec++ | who is "better" |
|---|---|---|---|
| coverage (catalog reach) | 0.397 +/- 0.016 | 0.593 +/- 0.025 | SASRec (reaches ~50% more) |
| Chao1 coverage ceiling | 0.467 +/- 0.019 | 0.693 +/- 0.022 | SASRec |
| Gini (lower=less concentrated) | 0.930 +/- 0.003 | 0.924 +/- 0.007 | tied (both very concentrated) |
| entropy_norm (higher=better) | 0.768 | 0.749 | MARIUS |
| ILD (intra-list diversity) | 0.1447 +/- 0.0003 | 0.1494 +/- 0.0008 | tied (SASRec a hair higher) |
| APLT (avg % long-tail in list) | 0.119 +/- 0.002 | 0.101 +/- 0.012 | MARIUS (more tail per list) |
| ARP (avg rec popularity, lower=less pop-biased) | 171.5 | 207.8 | MARIUS |
| recall | 0.0501 +/- 0.0004 | 0.0486 +/- 0.0006 | MARIUS (tied within noise) |
| tail_recall | 0.0082 | 0.0090 | SASRec (slightly) |
| hallucination_rate | 1.8e-5 | 0.0 | SASRec (MARIUS tiny nonzero) |

### k = 20 (for completeness)
MARIUS: coverage 0.518, gini 0.909, ILD 0.150, APLT 0.142, recall 0.0734, tail_recall 0.0146.
SASRec: coverage 0.681, gini 0.921, ILD 0.154, APLT 0.097, recall 0.0698, tail_recall 0.0133.
(At k=20 MARIUS edges SASRec on recall and tail_recall but is still far behind on coverage.)

READING (the precise characterization, NOT "MARIUS is less diverse"): MARIUS produces
INDIVIDUAL lists of comparable diversity (ILD tied) that lean MORE to the long tail
(APLT higher, ARP lower), yet it reaches far FEWER distinct catalog items in AGGREGATE
(coverage 0.40 vs 0.59) with the same concentration (Gini ~0.93). This is the classic
aggregate-diversity vs individual-diversity distinction (Adomavicius and Kwon 2012):
MARIUS recommends a reasonably varied but largely OVERLAPPING popular slice across users.
The 2014 Sports collapse (coverage 0.39 vs 0.74) PERSISTS at 90k scale.

## RQ3: What factors contribute to the popularity bias? -> the MODEL, not search

Exact full-catalog teacher-forced oracle (scores every catalog item per user, 1000-user
subsample). filter-seen = apples-to-apples beam-vs-exact.

| checkpoint | filter | exact R@10 | beam R@10 | median target rank | beam-missed median | verdict |
|---|---|---|---|---|---|---|
| Arts MARIUS seed42 | yes | 0.047 | 0.047 | 1956 / ~90k | 2579 | MODEL-bound |
| Arts MARIUS seed42 | no | 0.044 | 0.047 | 1963 | 2583 | MODEL-bound |
| Sports MARIUS baseline seed42 | yes | 0.049 | 0.049 | 996 / 18357 | 1324 | MODEL-bound |
| Sports MARIUS baseline seed42 | no | 0.046 | 0.049 | 1000 | 1328 | MODEL-bound |
| Sports MARIUS DISTILLED seed42 | no | 0.052 | 0.058 | 1405 | 1762 | MODEL-bound |

READING: exact scoring recovers the SAME recall as beam (delta ~0), so beam search is
near-optimal over the model -- the collapse is NOT a search/decoding artifact. The true
held-out targets sit at median exact rank ~1956/90k (Arts), 0% in the exact top-20. The
model's learned ranking buries the tail. This 2014 centerpiece REPLICATES at 90k scale.
The 4 candidate sources resolve as: data (long-tailed, contributory) / semantic-ID
(codebook audit refutes it as cause) / MODEL (the locus, by the oracle) / beam (NOT it:
exact == beam at matched K, so the beam is near-optimal over the model).
Relates to SimGR (arXiv:2602.07847), which critiques a token-level vs item-level generation
mismatch and sidesteps generation by ranking items directly: our exact oracle independently
localizes the cause in the trained model's ranking (the beam is near-optimal), a model-level
rather than search-level cause, in line with Latte's expressiveness-limit view (2605.06331).
(Verified 2026-06-21: SimGR's actual claim is the modeling mismatch, NOT "premature beam
pruning"; do not frame our result as refuting a beam-pruning claim.)

Semantic-ID / codebook exoneration (rules out the tokenizer as the cause): COSETTE uses
all 256 codes at every one of the 4 RVQ levels (usage ratio 1.0), with near-uniform
per-level usage (perplexity ~238-253 of 256; normalized usage entropy ~0.99). Code
collisions before the de-duplication (-col) token are low and fully resolved after it:
Beauty 1.48% -> 0.0% (max 14 items sharing a semantic ID), Sports 0.62% -> 0.0%. So the
collapse is not a code-utilization failure or an unresolved-collision artifact.

Where in the RVQ hierarchy the collapse sits (per-RVQ-level CONDITIONAL Gini/entropy of
MARIUS's recommended semantic IDs, Arts-2023, 5-seed mean; each level conditioned on the
preceding codes; reports/diversity/):

| RVQ level | conditional Gini | entropy (nats) | effective codes (e^H) |
|---|---|---|---|
| 0 (coarsest) | 0.994 | 0.062 | ~1.1 |
| 1 | 0.996 | 0.015 | ~1.0 |
| 2 | 0.996 | 0.005 | ~1.0 |
| 3 (leaf) | 0.922 | 2.980 | ~19.7 |

READING: the MARGINAL per-level codebook usage is near-uniform (above), but CONDITIONALLY
the model is near-degenerate at the 3 coarse levels (~1 effective code each) and branches
only at the leaf (~20). MARIUS commits to essentially ONE coarse semantic bucket and varies
only the residual code -- so the model-bound concentration is produced at the COARSE codes,
not the leaves, and it reflects which items the model recommends rather than an underused
codebook (consistent with the marginal-uniformity exoneration). CAVEAT: this is descriptive
(5-seed std <= 0.0006, no sig test); entropy support is 256 codes/level for MARIUS vs 89,958
items for SASRec (flat Gini 0.9998, ~20 head items), so the leaf entropy is NOT a like-for-like
MARIUS-vs-SASRec comparison. Figure: reports/figures/fig_rq3_conditional_diversity. Metric is
the maks_new_metrics conditional_gini/conditional_entropy (committed in scripts/, src/utils/metrics.py).

## RQ4: Mitigation across pipeline stages, and the accuracy-bias trade-off

The unified result: EVERY intervention -- at all three FairDiverse pipeline stages -- is
a BOUNDED DIAL that trades recall for distributional spread. NONE moves the model-bound
ceiling: under each, the exact oracle's true-target ranks do not improve (under the
strongest, distillation, they get WORSE). The dials differ only in how aggressively they
spread and at what recall cost.

### (a) In-processing: train-time logit-adjusted loss (MARIUSLogitAdj), 2014, k=10
| dataset | tau | dR@10 | dCoverage | dAPLT |
|---|---|---|---|---|
| Beauty | 0.5 | -5.5% | -6.0% | +8.1% |
| Beauty | 1.5 | -26.0% | -7.7% | +47.4% |
| Sports | 0.5 | -5.5% | +6.8% | +23.5% |
| Sports | 1.5 | -20.1% | +2.1% | +73.1% |
Raises per-list tail exposure (APLT) strongly with tau; coverage barely moves and is
non-monotone (Beauty coverage DROPS). Popularity reweighting at train time is decoupled
from catalog reach.

### (b) Post-processing: PMI re-rank (Arts, 5-seed, vs alpha=0)
| prior | alpha | dR@10 | dCoverage |
|---|---|---|---|
| cond2 | 0.5 | -15.5% | +18.2% |
| pair | 0.5 | -17.2% | +19.0% |
| item (global pop) | 0.5 | -25.6% | +55.7% |
| item (global pop) | 1.0 | -56.6% | +80.1% |
Coverage gain is non-monotone for the mechanism-matched priors (peaks ~alpha 0.5); the
generic item-popularity prior is the strongest coverage lever at the steepest recall cost.

### (c) Beam stage: MBR re-rank (Arts, 5-seed, vs original)
| tau | dR@10 | dCoverage |
|---|---|---|
| 0.25 | -20.2% | +34.1% |
| 1.0 | -25.9% | +27.6% |
| 2.0 | -38.3% | +27.4% |

### (d) Cross-paradigm distillation (SASRec++ d64 -> MARIUS KL, Sports, 2-seed, k=10)
| metric | baseline | distilled | delta |
|---|---|---|---|
| R@10 | 4.71 | 3.69 | -21.7% |
| NDCG@10 | 2.48 | 2.07 | -16.5% |
| coverage | 0.392 | 0.874 | +123% |
| Chao1 ceiling | 0.456 | 0.907 | +99% |
| Gini | 0.913 | 0.618 | -32% |
| ARP | 101.4 | 45.5 | -55% |
| APLT | 0.110 | 0.586 | +433% |
| tail_recall | 0.0112 | 0.0218 | +95% |
Config: alpha=0.5, temp=1.0, n_cand=128; KL(teacher||student) + base CE; inference unchanged.

DISTILLATION READING (the key cautionary result): this is the MOST aggressive de-concentration
of any arm (coverage 0.39->0.87, almost the teacher's spread; tail_recall +95%), but it is
NOT a fix. Its pre-registered success criterion (reach lifts AT held recall, with beam-missed
targets moving UP in exact rank) FAILED on both counts: held recall fell -22%, and the
distilled-checkpoint oracle shows the true targets ranked DEEPER (median 1000->1405;
beam-missed 1328->1762; top-100 recovery 11.7%->6.7%), verdict still MODEL-bound. So the
huge coverage gain is SPREADING, not genuine reach: distillation makes MARIUS smear mass
across the catalog like the teacher without learning to rank the correct items higher. It
is a (strong) third bounded dial. Methodological point: aggregate coverage can be inflated
2x while the model's ability to serve the tail does not improve -- only the exact oracle
separates real reach from cosmetic spreading.

## Figures to produce

- F1: accuracy(recall)-vs-coverage Pareto, all RQ4 arms on one plot (logit-adj, PMI x3
  priors, MBR, distillation) + the SASRec++ reference point. This single figure IS RQ4.
- F2: coverage and Gini, MARIUS vs SASRec, small (Beauty/Sports) vs large (Arts) bars
  (shows the collapse persists at scale).
- F3: exact-oracle target-rank histogram / CDF (Arts + Sports baseline + Sports distilled),
  showing targets buried at median ~1000-1960 and pushed deeper by distillation.

## Cross-references for the writeup

FairDiverse (SIGIR'25) 3-stage taxonomy + Gini/entropy metrics; Adomavicius and Kwon 2012
(aggregate vs individual diversity); Abdollahpouri (popularity-bias, coverage-vs-Gini
divergence); SimGR (2602.07847, token-vs-item modeling mismatch; model-level, our oracle
localizes it); Ghost 2605.16825 and Latte 2605.06331 (contrast / concurrent). All references
web-verified 2026-06-21. See EXTENSION_SUMMARY.md for the full framing + citations.
