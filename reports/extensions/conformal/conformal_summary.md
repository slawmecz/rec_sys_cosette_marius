# Conformal recommendation sets: feasibility pilot (summary)

Split conformal prediction on rank nonconformity, wrapped around the
frozen MARIUS and SASRec++ depth-20 Top-K dumps (3 seeds, 50/50
calibration/evaluation split, numpy rng seed 0). Full method and
per-category detail in conformal_<category>.md.

## Honest feasibility verdict

With a single held-out positive per user and lists truncated at depth 20,
the largest guaranteeable coverage equals the model's Recall@20:

| category | model | coverage ceiling (Recall@20, eval half) |
|---|---|---|
| Beauty | sasrec | 0.1240 |
| Beauty | marius | 0.1202 |
| Sports_and_Outdoors | sasrec | 0.0735 |
| Sports_and_Outdoors | marius | 0.0713 |

So conventional 90% (or even 20%) next-item coverage guarantees are
impossible at this depth. The viable framings are:
(a) low-coverage guarantees with tiny sets (quantified below);
(b) Mondrian per-bucket set size as a calibrated difficulty signal
(quantified below);
(c) deeper beams raise the ceiling, with sublinearly diminishing returns
(needs the deeper dumps; see the deeper-beam dependency note).

## Headline

- Beauty: sasrec needs a set no larger than marius at 5/5 jointly achievable targets (largest, c=0.12: mean k* 18.00 vs 18.67).
- Sports_and_Outdoors: sasrec needs a set no larger than marius at 2/2 jointly achievable targets (largest, c=0.06: mean k* 13.33 vs 14.00).
- Beauty Mondrian at c=0.08: sasrec k* 10.33 (hist<=5) vs 3.00 (hist>=16); cold users need the larger sets.
- Sports_and_Outdoors Mondrian at c=0.06: sasrec k* 13.00 (hist<=5) vs 12.33 (hist>=16); approximately flat across history (c=0.08 exceeds at least one bucket ceiling).

## (a) Matched guaranteed coverage: which model needs the smaller set?

### Beauty

| target c | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|
| 0.04 | 3.00 (3/3/3) | 0.0464 +/- 0.0015 | 4.00 (4/4/4) | 0.0442 +/- 0.0007 |
| 0.06 | 5.33 (5/5/6) | 0.0644 +/- 0.0057 | 6.00 (6/6/6) | 0.0571 +/- 0.0005 |
| 0.08 | 8.67 (8/9/9) | 0.0839 +/- 0.0040 | 9.67 (9/10/10) | 0.0785 +/- 0.0017 |
| 0.10 | 13.00 (13/13/13) | 0.1016 +/- 0.0012 | 13.67 (13/14/14) | 0.0978 +/- 0.0010 |
| 0.12 | 18.00 (18/18/18) | 0.1187 +/- 0.0003 | 18.67 (18/19/19) | 0.1162 +/- 0.0016 |

### Sports_and_Outdoors

| target c | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|
| 0.04 | 6.33 (6/6/7) | 0.0376 +/- 0.0012 | 8.00 (8/8/8) | 0.0393 +/- 0.0010 |
| 0.06 | 13.33 (13/13/14) | 0.0583 +/- 0.0014 | 14.00 (15/14/13) | 0.0560 +/- 0.0019 |
| 0.08 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |
| 0.10 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |
| 0.12 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |

## (b) Mondrian set size by history length (target c = 0.08)

### Beauty

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.112, mar 0.108 | 10.33 (10/10/11) | 0.0822 +/- 0.0038 | 11.67 (11/12/12) | 0.0787 +/- 0.0014 |
| hist6-15 | 4677 | sas 0.121, mar 0.120 | 9.00 (8/9/10) | 0.0828 +/- 0.0048 | 10.00 (10/10/10) | 0.0813 +/- 0.0018 |
| hist>=16 | 882 | sas 0.222, mar 0.207 | 3.00 (3/3/3) | 0.0927 +/- 0.0072 | 3.00 (3/3/3) | 0.0748 +/- 0.0014 |

### Sports_and_Outdoors

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9238 | sas 0.075, mar 0.075 | >20 (0/3 seeds achievable) | n/a | >20 (1/3 seeds achievable) | 0.0744 |
| hist6-15 | 7480 | sas 0.072, mar 0.069 | >20 (0/3 seeds achievable) | n/a | >20 (0/3 seeds achievable) | n/a |
| hist>=16 | 1080 | sas 0.069, mar 0.056 | >20 (1/3 seeds achievable) | 0.0728 | >20 (0/3 seeds achievable) | n/a |

## Mondrian at target c = 0.06 (secondary)

### Beauty

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 5622 | sas 0.112, mar 0.108 | 6.33 (6/6/7) | 0.0626 +/- 0.0044 | 7.67 (7/8/8) | 0.0590 +/- 0.0020 |
| hist6-15 | 4677 | sas 0.121, mar 0.120 | 5.33 (5/5/6) | 0.0621 +/- 0.0058 | 6.67 (6/7/7) | 0.0618 +/- 0.0045 |
| hist>=16 | 882 | sas 0.222, mar 0.207 | 2.00 (2/2/2) | 0.0708 +/- 0.0110 | 2.33 (2/3/2) | 0.0625 +/- 0.0121 |

### Sports_and_Outdoors

| bucket | n_cal | bucket Recall@20 (eval) | sasrec: k* | sasrec: emp. cov. | marius: k* | marius: emp. cov. |
|---|---|---|---|---|---|---|
| hist<=5 | 9238 | sas 0.075, mar 0.075 | 13.00 (13/13/13) | 0.0590 +/- 0.0015 | 12.67 (13/13/12) | 0.0564 +/- 0.0013 |
| hist6-15 | 7480 | sas 0.072, mar 0.069 | 14.00 (13/14/15) | 0.0586 +/- 0.0030 | 15.67 (17/15/15) | 0.0573 +/- 0.0027 |
| hist>=16 | 1080 | sas 0.069, mar 0.056 | 12.33 (10/14/13) | 0.0506 +/- 0.0037 | 14.67 (16/14/14) | 0.0482 +/- 0.0064 |

## Deeper-beam dependency

Raising the coverage ceiling requires deeper ranked lists than the local
depth-20 dumps. The extended dump_topk.py run planned by the MBR agent
(--n-results 100, both marius and sasrec) produces exactly the artifacts
needed; re-running this script on those dumps is the only follow-up.
