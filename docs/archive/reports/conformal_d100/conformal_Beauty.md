# Conformal recommendation sets: Beauty

Split conformal on rank nonconformity over the frozen depth-100 Top-K
dumps (reports/extensions/topk/Beauty). Users split 50/50 into
calibration (n=11181) / evaluation (n=11182) with numpy rng seed 0;
seeds 42, 43, 44; values below are seed means
(per-seed k* in parentheses).

## Feasibility ceiling

With a single held-out positive and a depth-100 list, the largest coverage
any conformal wrapper can guarantee is the model's Recall@100:

| model | Recall@100 (eval half) | Recall@100 (all users) |
|---|---|---|
| sasrec | 0.2450 | 0.2480 |
| marius | 0.2579 | 0.2598 |

Targets above this ceiling are marked unachievable (>100). Conventional
90% coverage guarantees are impossible for next-item hit at this depth.

## Split conformal: target coverage -> set size k* -> empirical coverage

| target c | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|
| 0.04 | 3.00 (3/3/3) | 0.0464 +/- 0.0010 | 3.67 (3/4/4) | 0.0413 +/- 0.0044 |
| 0.06 | 5.33 (5/5/6) | 0.0646 +/- 0.0055 | 6.00 (6/6/6) | 0.0572 +/- 0.0006 |
| 0.08 | 8.33 (8/8/9) | 0.0820 +/- 0.0038 | 9.67 (9/10/10) | 0.0784 +/- 0.0016 |
| 0.10 | 12.67 (12/13/13) | 0.1004 +/- 0.0016 | 13.67 (13/14/14) | 0.0979 +/- 0.0012 |
| 0.12 | 18.00 (18/18/18) | 0.1188 +/- 0.0003 | 18.67 (18/19/19) | 0.1163 +/- 0.0016 |
| 0.15 | 29.67 (30/30/29) | 0.1488 +/- 0.0012 | 28.67 (28/29/29) | 0.1460 +/- 0.0014 |
| 0.20 | 57.67 (57/58/58) | 0.1964 +/- 0.0024 | 53.67 (54/53/54) | 0.1977 +/- 0.0023 |
| 0.25 | >100 (2/3 seeds achievable) | 0.2444 +/- 0.0056 | 90.00 (90/90/90) | 0.2462 +/- 0.0019 |
| 0.30 | >100 (0/3 seeds achievable) | n/a | >100 (0/3 seeds achievable) | n/a |

Empirical coverage is computed on the held-out evaluation half; the
conformal guarantee is empirical coverage >= target c (rank ties make it
conservative, so overshoot is expected). The guarantee is marginal over
the calibration draw: with a single fixed split the eval estimate has a
binomial s.e. of about 0.0026 at c=0.08,
so undershoot of a few thousandths is within noise.

MARIUS effective set sizes (real items only; hallucinated
slots in the top-k* are wasted):

| target c | k* (mean) | real items in set (mean) |
|---|---|---|
| 0.04 | 3.67 | 3.66 |
| 0.06 | 6.00 | 5.99 |
| 0.08 | 9.67 | 9.64 |
| 0.10 | 13.67 | 13.63 |
| 0.12 | 18.67 | 18.60 |
| 0.15 | 28.67 | 28.55 |
| 0.20 | 53.67 | 53.31 |
| 0.25 | 90.00 | 88.87 |

## Mondrian conformal at target c = 0.08 (history-length buckets)

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.215, mar 0.230 | 10.00 (10/10/10) | 0.0815 +/- 0.0013 | 11.33 (11/12/11) | 0.0769 +/- 0.0018 |
| hist6-15 | 4677 | sas 0.250, mar 0.264 | 8.67 (8/9/9) | 0.0812 +/- 0.0032 | 10.00 (10/10/10) | 0.0812 +/- 0.0022 |
| hist>=16 | 882 | sas 0.423, mar 0.415 | 3.00 (3/3/3) | 0.0931 +/- 0.0063 | 3.00 (3/3/3) | 0.0744 +/- 0.0007 |

## Mondrian conformal at target c = 0.12 (history-length buckets)

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.215, mar 0.230 | 23.00 (23/23/23) | 0.1188 +/- 0.0007 | 22.67 (22/23/23) | 0.1151 +/- 0.0024 |
| hist6-15 | 4677 | sas 0.250, mar 0.264 | 18.67 (18/19/19) | 0.1168 +/- 0.0020 | 19.33 (19/19/20) | 0.1180 +/- 0.0044 |
| hist>=16 | 882 | sas 0.423, mar 0.415 | 5.67 (5/6/6) | 0.1321 +/- 0.0073 | 6.67 (7/6/7) | 0.1126 +/- 0.0084 |

Mondrian k* is calibrated inside each bucket, so each bucket carries its
own >= c guarantee and the per-user set size varies with history length.

CSV: split_conformal_Beauty.csv, mondrian_Beauty.csv.
