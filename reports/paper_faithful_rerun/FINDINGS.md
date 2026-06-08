# Paper-faithful SASRec++ re-run — RESULTS & FINDINGS

Executed 2026-06-08 on Snellius (account `scur1217`, `gpu_h100`), 5 seeds (42–46)
per config, 80k steps, global batch 256. Settles the SASRec++ config dispute in
[`FAITHFUL_RERUN.md`](../../FAITHFUL_RERUN.md). Raw artifacts (summary / per-seed
individual / scores jsonl) live next to this file, one set per config.

## Headline

**The paper-faithful config (d=32, L=2, no normalization, SampledSoftmax temp=1.0)
recovers most — but not all — of the Beauty gap, and confirms that smaller is
better on Beauty. It does not close the Sports gap.**

| Config | R@5 | NDCG@5 | **R@10** | NDCG@10 |
|---|---|---|---|---|
| **Beauty d=32, no-norm** (PRIMARY, faithful) | 6.20 ±0.10 | 4.31 ±0.08 | **9.06 ±0.06** | 5.22 ±0.06 |
| Beauty d=64, no-norm (hedge) | 6.15 ±0.10 | 4.24 ±0.07 | 8.88 ±0.09 | 5.12 ±0.08 |
| **Beauty — paper SASRec++** | 6.66 ±0.08 | 4.58 ±0.08 | **9.73 ±0.10** | 5.57 ±0.04 |
| *Beauty — prior committed run (d=128 + L2)* | — | — | *8.24* | — |
| | | | | |
| Sports d=64, no-norm | 3.50 ±0.07 | 2.38 ±0.04 | 5.15 ±0.11 | 2.91 ±0.05 |
| Sports d=32, no-norm | 3.44 ±0.04 | 2.35 ±0.03 | 5.08 ±0.10 | 2.87 ±0.05 |
| **Sports — paper SASRec++** | 4.37 ±0.09 | 2.96 ±0.05 | **6.44 ±0.10** | 3.62 ±0.04 |

## Verdict on the FAITHFUL_RERUN.md hypotheses

1. **Config mismatch was a real, major contributor — CONFIRMED.** Beauty R@10 went
   from **8.24** (repo default d=128 + L2) → **9.06** (paper-faithful d=32 + no-norm).
   That closes **~55%** of the gap to the paper's 9.73 ((9.06−8.24)/(9.73−8.24)).
2. **Smaller is better on Beauty — CONFIRMED.** d=32 (9.06) > d=64 (8.88), by more
   than the combined seed std. This matches the paper's Figure 11a (star at d=32 for
   the 12k-item Beauty) and contradicts the "d=128 + L2 is the real config" reading.
3. **A residual ~0.67-point gap remains (≈6.9% rel below paper).** Config alone does
   NOT fully reproduce 9.73. This points back at the *secondary* hypotheses in
   FAITHFUL_RERUN.md §2 — validation→test generalization gap and/or coarse checkpoint
   selection — as the remaining contributors. (Not undertraining: §2 ruled that out.)
4. **Sports gap is NOT model size.** d=32 (5.08) ≈ d=64 (5.15), both ~20% below the
   paper's 6.44. The faithful config does not help Sports; its gap is elsewhere
   (consistent with the sparser-dataset generalization story).

## Suggested next steps (for the next session)
- Test the checkpoint-selection / generalization hypothesis on Beauty d=32: finer
  `val_check_interval`, select best-val checkpoint, inspect val→test drop per seed.
- The MARIUS finer-checkpoint probe (job 21) is still optional and needs COSETTE
  `-col` tokens (jobs 04→05→06) which are NOT regenerated on this account yet.

## Reproduction environment (this account was bootstrapped from scratch)
`scur1217` had no conda env, no data, and the scaffolding hardcoded the original
author's (`scur1266`) home/scratch. Bootstrapped here:
- `recsys` conda env from `requirements.txt` (torch 2.6+cu124, ray 2.44.1,
  lightning 2.5, wandb 0.19.7); `PYTHONNOUSERSITE=1` to isolate from `~/.local`.
- Data regenerated via download + `0_raw_to_parquet` — **item counts match the paper
  exactly** (Beauty 12,101 / Sports 18,357), i.e. the LOOCV split is faithful.
- SLURM log paths generalized to `/scratch-shared/%u/...`; data/outputs under
  `$SCRATCH=/scratch-shared/scur1217/cosette_marius`.
- Ran on `gpu_h100` (a100 partition was saturated); ~34–36 min/seed, all 20 seeds
  COMPLETED with 0 failures.
- W&B logged online to `wasilewski-sf-vrije-universiteit-amsterdam/RecSys`.

To re-run a config (example, Beauty d=32):
```bash
cd ~/rec_sys_cosette_marius
sbatch --account=gpuuva080 --partition=gpu_h100 --cpus-per-task=16 \
  --export=ALL,SCRATCH=/scratch-shared/$USER,PROJECT_ROOT=$HOME/rec_sys_cosette_marius,\
OUTPUT_ROOT=/scratch-shared/$USER/cosette_marius/outputs_paper/beauty_d32,\
CATEGORY=Beauty,CATEGORY_SLUG=beauty,VOCAB=12103,SAS_D=32,SAS_DH=16 \
  jobs/20_sasrec_paper_faithful.sbatch
```
(`--cpus-per-task=16` keeps H100 billing at 1 GPU; each config needs its own
`OUTPUT_ROOT` so same-category/different-d runs don't clobber each other's
`table5_<slug>_*` files.)
