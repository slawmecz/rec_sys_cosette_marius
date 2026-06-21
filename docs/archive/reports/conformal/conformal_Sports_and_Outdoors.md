# Conformal recommendation sets: Sports_and_Outdoors

Split conformal on rank nonconformity over the frozen depth-20 Top-K
dumps (reports/extensions/topk/Sports_and_Outdoors). Users split 50/50 into
calibration (n=17799) / evaluation (n=17799) with numpy rng seed 0;
seeds 42, 43, 44; values below are seed means
(per-seed k* in parentheses).

## Feasibility ceiling

With a single held-out positive and a depth-20 list, the largest coverage
any conformal wrapper can guarantee is the model's Recall@20:

| model | Recall@20 (eval half) | Recall@20 (all users) |
|---|---|---|
| sasrec | 0.0735 | 0.0747 |
| marius | 0.0713 | 0.0739 |

Targets above this ceiling are marked unachievable (>20). Conventional
90% coverage guarantees are impossible for next-item hit at this depth.

## Split conformal: target coverage -> set size k* -> empirical coverage

| target c | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|
| 0.04 | 6.33 (6/6/7) | 0.0376 +/- 0.0012 | 8.00 (8/8/8) | 0.0393 +/- 0.0010 |
| 0.06 | 13.33 (13/13/14) | 0.0583 +/- 0.0014 | 14.00 (15/14/13) | 0.0560 +/- 0.0019 |
| 0.08 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |
| 0.10 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |
| 0.12 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |

Empirical coverage is computed on the held-out evaluation half; the
conformal guarantee is empirical coverage >= target c (rank ties make it
conservative, so overshoot is expected). The guarantee is marginal over
the calibration draw: with a single fixed split the eval estimate has a
binomial s.e. of about 0.0020 at c=0.08,
so undershoot of a few thousandths is within noise.

MARIUS effective set sizes (real items only; hallucinated
slots in the top-k* are wasted):

| target c | k* (mean) | real items in set (mean) |
|---|---|---|
| 0.04 | 8.00 | 8.00 |
| 0.06 | 14.00 | 14.00 |

## Mondrian conformal at target c = 0.08 (history-length buckets)

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9238 | sas 0.075, mar 0.075 | >20 (0/3 seeds achievable) | n/a | >20 (1/3 seeds achievable) | 0.0744 |
| hist6-15 | 7480 | sas 0.072, mar 0.069 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |
| hist>=16 | 1080 | sas 0.069, mar 0.056 | >20 (1/3 seeds achievable) | 0.0728 | >20 (0/3 seeds achievable) | n/a |

## Mondrian conformal at target c = 0.06 (history-length buckets)

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9238 | sas 0.075, mar 0.075 | 13.00 (13/13/13) | 0.0590 +/- 0.0015 | 12.67 (13/13/12) | 0.0564 +/- 0.0013 |
| hist6-15 | 7480 | sas 0.072, mar 0.069 | 14.00 (13/14/15) | 0.0586 +/- 0.0030 | 15.67 (17/15/15) | 0.0573 +/- 0.0027 |
| hist>=16 | 1080 | sas 0.069, mar 0.056 | 12.33 (10/14/13) | 0.0506 +/- 0.0037 | 14.67 (16/14/14) | 0.0482 +/- 0.0064 |

Mondrian k* is calibrated inside each bucket, so each bucket carries its
own >= c guarantee and the per-user set size varies with history length.

CSV: split_conformal_Sports_and_Outdoors.csv, mondrian_Sports_and_Outdoors.csv.
