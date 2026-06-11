# Conformal recommendation sets: Sports_and_Outdoors

Split conformal on rank nonconformity over the frozen depth-100 Top-K
dumps (reports/extensions/topk/Sports_and_Outdoors). Users split 50/50 into
calibration (n=17799) / evaluation (n=17799) with numpy rng seed 0;
seeds 42, 43, 44; values below are seed means
(per-seed k* in parentheses).

## Feasibility ceiling

With a single held-out positive and a depth-100 list, the largest coverage
any conformal wrapper can guarantee is the model's Recall@100:

| model | Recall@100 (eval half) | Recall@100 (all users) |
|---|---|---|
| sasrec | 0.1644 | 0.1654 |
| marius | 0.1742 | 0.1775 |

Targets above this ceiling are marked unachievable (>100). Conventional
90% coverage guarantees are impossible for next-item hit at this depth.

## Split conformal: target coverage -> set size k* -> empirical coverage

| target c | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|
| 0.04 | 6.33 (6/6/7) | 0.0375 +/- 0.0012 | 8.00 (8/8/8) | 0.0392 +/- 0.0011 |
| 0.06 | 13.33 (13/13/14) | 0.0581 +/- 0.0013 | 14.00 (15/14/13) | 0.0560 +/- 0.0020 |
| 0.08 | 22.33 (22/22/23) | 0.0776 +/- 0.0008 | 22.33 (23/22/22) | 0.0761 +/- 0.0009 |
| 0.10 | 34.00 (34/33/35) | 0.0962 +/- 0.0009 | 32.00 (32/32/32) | 0.0942 +/- 0.0008 |
| 0.12 | 49.33 (49/49/50) | 0.1158 +/- 0.0016 | 44.33 (45/44/44) | 0.1132 +/- 0.0002 |
| 0.15 | 80.67 (80/80/82) | 0.1483 +/- 0.0019 | 68.33 (69/68/68) | 0.1421 +/- 0.0024 |
| 0.20 | >100 (0/3 seeds achievable) | n/a | >100 (0/3 seeds achievable) | n/a |
| 0.25 | >100 (0/3 seeds achievable) | n/a | >100 (0/3 seeds achievable) | n/a |
| 0.30 | >100 (0/3 seeds achievable) | n/a | >100 (0/3 seeds achievable) | n/a |

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
| 0.08 | 22.33 | 22.33 |
| 0.10 | 32.00 | 31.99 |
| 0.12 | 44.33 | 44.31 |
| 0.15 | 68.33 | 68.23 |

## Mondrian conformal at target c = 0.08 (history-length buckets)

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9239 | sas 0.167, mar 0.181 | 21.67 (21/22/22) | 0.0783 +/- 0.0018 | 20.67 (21/21/20) | 0.0764 +/- 0.0017 |
| hist6-15 | 7479 | sas 0.163, mar 0.170 | 23.67 (23/23/25) | 0.0789 +/- 0.0021 | 23.33 (25/22/23) | 0.0756 +/- 0.0024 |
| hist>=16 | 1080 | sas 0.153, mar 0.150 | 22.00 (20/23/23) | 0.0721 +/- 0.0018 | 27.00 (28/26/27) | 0.0676 +/- 0.0083 |

## Mondrian conformal at target c = 0.12 (history-length buckets)

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9239 | sas 0.167, mar 0.181 | 48.33 (49/49/47) | 0.1176 +/- 0.0042 | 42.00 (43/41/42) | 0.1153 +/- 0.0011 |
| hist6-15 | 7479 | sas 0.163, mar 0.170 | 50.67 (48/50/54) | 0.1149 +/- 0.0030 | 45.67 (47/45/45) | 0.1123 +/- 0.0012 |
| hist>=16 | 1080 | sas 0.153, mar 0.150 | 56.00 (60/48/60) | 0.1154 +/- 0.0125 | 58.33 (57/58/60) | 0.1110 +/- 0.0079 |

Mondrian k* is calibrated inside each bucket, so each bucket carries its
own >= c guarantee and the per-user set size varies with history length.

CSV: split_conformal_Sports_and_Outdoors.csv, mondrian_Sports_and_Outdoors.csv.
