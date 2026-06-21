# Conformal recommendation sets: Beauty

Split conformal on rank nonconformity over the frozen depth-20 Top-K
dumps (reports/extensions/topk/Beauty). Users split 50/50 into
calibration (n=11181) / evaluation (n=11182) with numpy rng seed 0;
seeds 42, 43, 44; values below are seed means
(per-seed k* in parentheses).

## Feasibility ceiling

With a single held-out positive and a depth-20 list, the largest coverage
any conformal wrapper can guarantee is the model's Recall@20:

| model | Recall@20 (eval half) | Recall@20 (all users) |
|---|---|---|
| sasrec | 0.1240 | 0.1259 |
| marius | 0.1202 | 0.1227 |

Targets above this ceiling are marked unachievable (>20). Conventional
90% coverage guarantees are impossible for next-item hit at this depth.

## Split conformal: target coverage -> set size k* -> empirical coverage

| target c | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|
| 0.04 | 3.00 (3/3/3) | 0.0464 +/- 0.0015 | 4.00 (4/4/4) | 0.0442 +/- 0.0007 |
| 0.06 | 5.33 (5/5/6) | 0.0644 +/- 0.0057 | 6.00 (6/6/6) | 0.0571 +/- 0.0005 |
| 0.08 | 8.67 (8/9/9) | 0.0839 +/- 0.0040 | 9.67 (9/10/10) | 0.0785 +/- 0.0017 |
| 0.10 | 13.00 (13/13/13) | 0.1016 +/- 0.0012 | 13.67 (13/14/14) | 0.0978 +/- 0.0010 |
| 0.12 | 18.00 (18/18/18) | 0.1187 +/- 0.0003 | 18.67 (18/19/19) | 0.1162 +/- 0.0016 |

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
| 0.04 | 4.00 | 3.99 |
| 0.06 | 6.00 | 5.99 |
| 0.08 | 9.67 | 9.64 |
| 0.10 | 13.67 | 13.63 |
| 0.12 | 18.67 | 18.60 |

## Mondrian conformal at target c = 0.08 (history-length buckets)

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.112, mar 0.108 | 10.33 (10/10/11) | 0.0822 +/- 0.0038 | 11.67 (11/12/12) | 0.0787 +/- 0.0014 |
| hist6-15 | 4677 | sas 0.121, mar 0.120 | 9.00 (8/9/10) | 0.0828 +/- 0.0048 | 10.00 (10/10/10) | 0.0813 +/- 0.0018 |
| hist>=16 | 882 | sas 0.222, mar 0.207 | 3.00 (3/3/3) | 0.0927 +/- 0.0072 | 3.00 (3/3/3) | 0.0748 +/- 0.0014 |

## Mondrian conformal at target c = 0.06 (history-length buckets)

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.112, mar 0.108 | 6.33 (6/6/7) | 0.0626 +/- 0.0044 | 7.67 (7/8/8) | 0.0590 +/- 0.0020 |
| hist6-15 | 4677 | sas 0.121, mar 0.120 | 5.33 (5/5/6) | 0.0621 +/- 0.0058 | 6.67 (6/7/7) | 0.0618 +/- 0.0045 |
| hist>=16 | 882 | sas 0.222, mar 0.207 | 2.00 (2/2/2) | 0.0708 +/- 0.0110 | 2.33 (2/3/2) | 0.0625 +/- 0.0121 |

Mondrian k* is calibrated inside each bucket, so each bucket carries its
own >= c guarantee and the per-user set size varies with history length.

CSV: split_conformal_Beauty.csv, mondrian_Beauty.csv.
