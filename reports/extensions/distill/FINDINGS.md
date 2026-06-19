# Cross-paradigm distillation arm (Sports): findings

Distill a FROZEN SASRec++ teacher's catalog ranking into MARIUS at the last decode
position (KL on a shared candidate set + base cross-entropy; inference unchanged).
Question: is MARIUS's catalog-reachability collapse "supervision starvation" (the model
CAN spread across the catalog if taught to) or a deeper model-ranking limit?

## Setup

- Teacher: frozen SASRec++ d64 seed42 (outputs_paper/sports_d64). The canonical d128
  SASRec baseline checkpoint had been purged from scratch; d64 was the strongest
  surviving teacher and was used for both student seeds.
- Student: MARIUS (COSETTE 24cd -col tokens), trained FROM SCRATCH, 80k steps.
- Distill config: alpha=0.5, temp=1.0, n_cand=128. Seeds 42, 43.
- Two code fixes were required for the teacher path on this environment (torch 2.6+):
  load the teacher with weights_only=False, and cast teacher catalog scores to float32
  before numpy under bf16-mixed autocast.

## Results (Sports, k=10)

Accuracy (full test, mean over seeds 42-43; baseline = reports/extensions/baseline_scur1217):
- R@10   distilled 3.69  vs baseline 4.71   (DOWN)
- NDCG@10 distilled 2.07 vs baseline 2.48   (DOWN)
- per seed: s42 R@10 3.96 / NDCG 2.19 ; s43 R@10 3.41 / NDCG 1.95

Beyond-accuracy / reach (2-seed mean; baseline = committed CSVs):
- coverage@10        0.874  vs 0.392   (UP, 2.2x)
- chao1_coverage@10  0.907  vs 0.456   (UP)
- tail_recall@10     0.0218 vs 0.0112  (UP, ~2x)
- gini@10            0.618  vs 0.913    (DOWN, more equal)
- arp@10             45.5   vs 101.4    (DOWN, less popularity-peaked)
- aplt@10            0.586  vs 0.110    (UP, 5x)

Exact full-catalog oracle (seed 42, 1000 users, same protocol as the committed
baseline oracle exact_catalog_Sports_and_Outdoors_seed42.json):
- median exact rank of beam-missed targets   1762  vs baseline 1328  (ROSE, deeper)
- beam-missed top-100 per-user recovery       6.7% vs baseline 11.7% (DROPPED)
- target_exact_rank_median_all                1405 vs baseline 1000.5
- verdict                                     MODEL-bound (unchanged)

## Decision: NOT Fork A -> bounded coverage/diversity dial (Fork B/C family)

Pre-registered Fork A required reach to lift AT HELD RECALL and the median exact rank
of beam-missed targets to DROP. Neither held: held recall fell (4.71 -> 3.69 R@10) and
the beam-missed targets are ranked even DEEPER under the distilled model (1328 -> 1762),
with top-100 recovery dropping (11.7% -> 6.7%). The oracle verdict stays MODEL-bound.

Read: distilling SASRec's dense catalog ranking into MARIUS does make MARIUS spread
across the catalog like the teacher (coverage 0.39 -> 0.87, tail and ARP/Gini all move
strongly), but it does NOT recover the true held-out targets -- it trades top-of-list
accuracy for catalog reach. This is the same accuracy-vs-coverage trade seen in the PMI
re-rank and train-time logit-adjustment arms, achieved here through teacher supervision.
It is therefore a (strong) third bounded dial, NOT evidence of supervision starvation:
forcing teacher-like spreading did not fix the underlying ranking of the true items.

Because the result is not Fork A, the Beauty replication was not run (the pre-registered
trigger for Beauty was a positive Fork A headline on Sports).
