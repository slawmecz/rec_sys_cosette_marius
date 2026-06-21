# Paper-faithful re-run: closing the reproduction ambiguity

**Branch:** `stanislaw-finish-reproduction` (from `slawek`, keeps all of slawek's + our work)
**Status:** completed on Snellius. The results are in `reports/paper_faithful_rerun/`
and summarized below.

This document explains (1) what we fixed, (2) what we diagnosed about the
reproduction gap, (3) the one factual dispute about the paper and how it was
resolved, and (4) the paper-faithful re-run that settled the largest SASRec++
configuration ambiguity.

---

## 1. Reporting bug: fixed on this branch

The paper baseline for **MARIUS (COSETTE) on Sports & Outdoors** was wrong in the
reporting layer. Verified against the paper's Table 5 (Amazon-2014) via the PDF
text layer:

| | was (wrong) | now (paper Table 5) |
|---|---|---|
| Sports MARIUS-COSETTE R@5 / NDCG@5 / R@10 / NDCG@10 | 4.29 / 2.91 / 6.31 / 3.55 | **4.31 / 2.83 / 6.72 / 3.62** |

The R@10 error (6.31 to **6.72**) hid that COSETTE actually *beats* SASRec++ on
Sports R@10 in the paper (6.72 > 6.44). Fixed in **three** places, then
re-rendered: `scripts/reporting/constants.py`, `scripts/marius_5seed.py` (its own copy),
`reports/results/table5_sports_marius_5seed_summary_full_latest.txt`, and
`notebooks/replication_report.ipynb` (re-executed; the Beauty/SASRec/Sports-SASRec
baselines were already correct).

---

## 2. Verified diagnosis of the gap

slawek's 5-seed reproduction (Beauty + Sports, SASRec++ and MARIUS-COSETTE,
seeds 42-46) lands about 15-28% below the paper on **test**, e.g. Beauty SASRec++
R@10 = 8.24 vs 9.73; MARIUS R@10 = 8.17 vs 10.02. We forensically checked every
pipeline stage. What is **ruled out** (verified):

- **Undertraining** was the leading auto-generated hypothesis; it is FALSE. The
  metric CSVs log W&B `_step` (a throttled logging counter, about 1 per 25 optimiser
  steps), not optimiser steps. Wall-clock (about 42 min/seed SASRec, about 64 min/seed
  MARIUS), the 7-8 validation events spaced at `val_check_interval=10000`, and
  SASRec's validation **peaking at the first checkpoint then declining** all show
  the models trained the full ~80k steps and converged (SASRec even mildly
  overfits). Do **not** repeat the "3,200 steps" claim.
- **Data split / preprocessing**: the Amazon-2014 LOOCV (`_loocv_split`) is
  provably equivalent to the canonical 5-core last-item split (item counts match
  the paper exactly: 12,101 Beauty / 18,357 Sports).
- **Eval code**: byte-identical to upstream; full-corpus ranking; correct
  single-positive NDCG; history filtering on.
- **Global batch** = 256 (correct). **COSETTE** converged (collisions <1.5%,
  config matches section 4.3.2). **MARIUS architecture** matches the paper (see section 3).

What remains, the actual contributors:

- **SASRec++ used the repo's *default* config, not the paper's per-dataset Beauty
  config.** This is the main, testable lever (see section 3).
- A **validation-to-test generalisation gap** (our validation reaches paper-test
  level; test lags), larger on Sports, consistent with overfitting from an
  over-wide SASRec++ and/or coarse checkpoint selection.

---

## 3. The one dispute: SASRec++ model size, and its resolution

**Claim under review (from a teammate's agent):** "d_model = 128 + L2 is the
paper's config; d=32 is just the smallest axis value in Figure 11."

**Resolution: partly right, but the conclusion is wrong.**

- Correct: the repo's `configs/experiment/sasrec.yaml` (d=128, `normalization: l2`,
  `SampledSoftmax` temp 0.05 / 30k negs) is **byte-identical to the authors'
  released config** (verified by diff). So the *config file* is faithful, but it
  is the **default**, which the paper **overrides per dataset**.
- Incorrect: the "d=32 is just the axis minimum" reading mis-reads Figure 11. **Each
  subplot has a different y-axis:**
  - (a) **Beauty, 12k**: axis = `16, 32, 64, 128, 256`, with the **star (best) at d=32, L=2**.
    The d=128/256 rows are the **dark/worst** region (about 8 R@10).
  - (b) Office (77k) and (c) Beauty&Personal-Care (207k): axis = `32 to 512`, with the star at d=128.

  The `{32 to 512}` axis and the `d in {128,256}` search belong to the **large 2023**
  datasets (section 4.4.3), **not** to Beauty. Appendix C.1 states plainly:
  *"larger datasets benefit from slightly larger models"* and *"for the smallest
  dataset, better results are obtained using **no normalization**."*

So the paper's **reported Beauty SASRec++ (9.73)** used **d=32, L=2, no
normalization**, not the released default d=128 + L2 that slawek ran. Tellingly,
**d=128 lands in the about-8 R@10 dark region of the paper's own Figure 11a, right
where slawek's run landed (8.24).**

**We do not need to win this on figure-reading: the re-run decided it.**

### Faithful SASRec++ config used by the re-run

| param | repo default (slawek ran) | paper Beauty (faithful) |
|---|---|---|
| d_model | 128 | **32** (Fig 11a star) |
| d_head | 64 | **16** (keeps 2 heads; paper doesn't specify head dim for small d) |
| layers L | 2 | 2 |
| normalization | l2 | **None** (App C.1, smallest dataset) |
| loss / temperature | SampledSoftmax / 0.05 | SampledSoftmax / **1.0** (unnormalised logits; n_items=30k about full over a 12k vocab) |
| dropout / lr / steps / batch | 0.4 / 5e-4 / 80k / 256 | unchanged (already match) |

Sports (18k) is **not** in the paper's sizing study; "slightly larger models for
larger datasets" puts it at d about 32-64. We bracket it.

### MARIUS: already faithful, no dispute

MARIUS uses `experiment=marius_small`: d=256, 2 temporal + 2 depth layers,
dropout 0.4/0.1, 80k steps, batch 256, which matches Table 9 / Figure 9a (Beauty star)
/ Appendix C.2 exactly. So the MARIUS re-run (job 21) is **optional** and only
tests reproducibility + finer checkpoint selection, not a config fix.

---

## 4. The experiment

`paper_train_args()` in `scripts/{sasrec,marius}_5seed.py` now appends
`EXTRA_TRAIN_OVERRIDES` (empty by default, so existing jobs are unaffected) and honours
`TRAIN_VAL_CHECK_INTERVAL`. The jobs set a **fresh `OUTPUT_ROOT`
(`…/cosette_marius/outputs_paper`)** so the 5-seed harness trains from scratch
instead of re-recording slawek's d=128 runs, while **reusing `DATA_ROOT`** (the
existing parquet; SASRec needs no embeddings/COSETTE).

### Observed outcomes

All 20 SASRec++ faithful rerun seeds completed (5 seeds for each of Beauty d=32,
Beauty d=64, Sports d=32, Sports d=64).

| Dataset | Config | R@10 | Interpretation |
|---|---:|---:|---|
| Beauty | repo default d=128 + L2 | 8.24 | original team run |
| Beauty | faithful d=32, no-norm | **9.06 +/- 0.06** | closes about 55% of the paper gap |
| Beauty | faithful d=64, no-norm | 8.88 +/- 0.09 | lower than d=32 |
| Sports | faithful d=32, no-norm | 5.08 +/- 0.10 | still far below paper |
| Sports | faithful d=64, no-norm | **5.15 +/- 0.11** | statistically tied with d=32 |

Conclusion: the default-vs-paper SASRec++ configuration mismatch is a real and
large contributor on Beauty, but it does not fully reproduce the paper. Sports
retains a roughly 20-21% residual gap that is not explained by model size.

### Data dependency

SASRec needs only the **parquet timelines** in `DATA_ROOT`
(`data/timelines/{Beauty,Sports_and_Outdoors}.{train,valid,test}.parquet`). If
they survive on scratch from slawek's runs, nothing to regenerate; otherwise run
`jobs/01_*` (download) + `jobs/02_*` (parquet) first, both CPU-only. MARIUS (job
21) additionally needs the COSETTE `-col` tokens (jobs 04 to 05 to 06).

### How to re-run

```bash
cd ~/rec_sys_cosette_marius
git fetch origin && git checkout stanislaw-finish-reproduction && git pull

# 0) sanity: parquet timelines exist?
ls "$SCRATCH"/cosette_marius/data/data/timelines/Beauty.*.parquet

# 1) SMOKE TEST (~2 min) - catches config errors (e.g. d_head divisibility) cheaply
sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,VOCAB=12103,SAS_D=32,SAS_DH=16,MODE=smoke jobs/20_sasrec_paper_faithful.sbatch
#   -> check the .out log shows it starts training and writes a smoke summary.

# 2) FULL faithful runs (5 seeds each)
# PRIMARY - Beauty, paper config:
sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,VOCAB=12103,SAS_D=32,SAS_DH=16 jobs/20_sasrec_paper_faithful.sbatch
# HEDGE - Beauty d=64 (in case the Fig-11a star reads one tick off):
sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,VOCAB=12103,SAS_D=64,SAS_DH=32 jobs/20_sasrec_paper_faithful.sbatch
# Sports (interpolated):
sbatch --export=ALL,CATEGORY=Sports_and_Outdoors,CATEGORY_SLUG=sports,VOCAB=18359,SAS_D=64,SAS_DH=32 jobs/20_sasrec_paper_faithful.sbatch
sbatch --export=ALL,CATEGORY=Sports_and_Outdoors,CATEGORY_SLUG=sports,VOCAB=18359,SAS_D=32,SAS_DH=16 jobs/20_sasrec_paper_faithful.sbatch

# 3) OPTIONAL - MARIUS reproducibility + finer checkpointing (needs COSETTE -col tokens)
sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,QUANT_ID=COSETTE_128d_256x4_f958-col jobs/21_marius_paper_fineckpt.sbatch
sbatch --export=ALL,CATEGORY=Sports_and_Outdoors,CATEGORY_SLUG=sports,QUANT_ID=COSETTE_128d_256x4_8ed1-col jobs/21_marius_paper_fineckpt.sbatch
```

### What a rerun should report

For each run, the per-seed scores and the paper-vs-ours table:
```
$OUTPUT_ROOT/results/sasrec_<slug>_5seed_full_scores.jsonl
$OUTPUT_ROOT/results/table5_<slug>_5seed_summary_full_latest.txt
```
(`OUTPUT_ROOT = $SCRATCH/cosette_marius/outputs_paper`). Report mean±std R@5,
NDCG@5, R@10, NDCG@10 per config, plus any failures from the `.err` logs.
