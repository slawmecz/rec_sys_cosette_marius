# Conformal recommendation sets: feasibility pilot (summary)

Split conformal prediction on rank nonconformity, wrapped around the
frozen MARIUS and SASRec++ depth-100 Top-K dumps (3 seeds, 50/50
calibration/evaluation split, numpy rng seed 0). Full method and
per-category detail in conformal_<category>.md.

## Honest feasibility verdict

With a single held-out positive per user and lists truncated at depth 100,
the largest guaranteeable coverage equals the model's Recall@100:

| category | model | coverage ceiling (Recall@100, eval half) |
|---|---|---|
| Beauty | sasrec | 0.2450 |
| Beauty | marius | 0.2579 |
| Sports_and_Outdoors | sasrec | 0.1644 |
| Sports_and_Outdoors | marius | 0.1742 |

So conventional 90% (or even 20%) next-item coverage guarantees are
impossible at this depth. The viable framings are:
(a) low-coverage guarantees with tiny sets (quantified below);
(b) Mondrian per-bucket set size as a calibrated difficulty signal
(quantified below);
(c) deeper beams raise the ceiling, with sublinearly diminishing returns
(needs the deeper dumps; see the deeper-beam dependency note).

## Headline

- Beauty: sasrec needs a set no larger than marius at 5/7 jointly achievable targets (largest, c=0.20: mean k* 57.67 vs 53.67).
- Sports_and_Outdoors: sasrec needs a set no larger than marius at 3/6 jointly achievable targets (largest, c=0.15: mean k* 80.67 vs 68.33).
- Beauty Mondrian at c=0.12: sasrec k* 23.00 (hist<=5) vs 5.67 (hist>=16); cold users need the larger sets.
- Sports_and_Outdoors Mondrian at c=0.12: sasrec k* 48.33 (hist<=5) vs 56.00 (hist>=16); set size does NOT shrink with history.

## (a) Matched guaranteed coverage: which model needs the smaller set?

### Beauty

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

### Sports_and_Outdoors

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

## (b) Mondrian set size by history length (target c = 0.08)

### Beauty

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.215, mar 0.230 | 10.00 (10/10/10) | 0.0815 +/- 0.0013 | 11.33 (11/12/11) | 0.0769 +/- 0.0018 |
| hist6-15 | 4677 | sas 0.250, mar 0.264 | 8.67 (8/9/9) | 0.0812 +/- 0.0032 | 10.00 (10/10/10) | 0.0812 +/- 0.0022 |
| hist>=16 | 882 | sas 0.423, mar 0.415 | 3.00 (3/3/3) | 0.0931 +/- 0.0063 | 3.00 (3/3/3) | 0.0744 +/- 0.0007 |

### Sports_and_Outdoors

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9239 | sas 0.167, mar 0.181 | 21.67 (21/22/22) | 0.0783 +/- 0.0018 | 20.67 (21/21/20) | 0.0764 +/- 0.0017 |
| hist6-15 | 7479 | sas 0.163, mar 0.170 | 23.67 (23/23/25) | 0.0789 +/- 0.0021 | 23.33 (25/22/23) | 0.0756 +/- 0.0024 |
| hist>=16 | 1080 | sas 0.153, mar 0.150 | 22.00 (20/23/23) | 0.0721 +/- 0.0018 | 27.00 (28/26/27) | 0.0676 +/- 0.0083 |

## Mondrian at target c = 0.12 (secondary)

### Beauty

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.215, mar 0.230 | 23.00 (23/23/23) | 0.1188 +/- 0.0007 | 22.67 (22/23/23) | 0.1151 +/- 0.0024 |
| hist6-15 | 4677 | sas 0.250, mar 0.264 | 18.67 (18/19/19) | 0.1168 +/- 0.0020 | 19.33 (19/19/20) | 0.1180 +/- 0.0044 |
| hist>=16 | 882 | sas 0.423, mar 0.415 | 5.67 (5/6/6) | 0.1321 +/- 0.0073 | 6.67 (7/6/7) | 0.1126 +/- 0.0084 |

### Sports_and_Outdoors

| bucket | n_cal | bucket Recall@100 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9239 | sas 0.167, mar 0.181 | 48.33 (49/49/47) | 0.1176 +/- 0.0042 | 42.00 (43/41/42) | 0.1153 +/- 0.0011 |
| hist6-15 | 7479 | sas 0.163, mar 0.170 | 50.67 (48/50/54) | 0.1149 +/- 0.0030 | 45.67 (47/45/45) | 0.1123 +/- 0.0012 |
| hist>=16 | 1080 | sas 0.153, mar 0.150 | 56.00 (60/48/60) | 0.1154 +/- 0.0125 | 58.33 (57/58/60) | 0.1110 +/- 0.0079 |

## Deeper beam (this run)

These sets are computed on the depth-100 dumps (dump_topk.py
--n-results 100); the ceiling is Recall@100, well above the
depth-20 numbers, so larger coverage targets are now achievable (see the
split-conformal tables above for where k* lands).
