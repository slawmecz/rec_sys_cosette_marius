# Arts, Crafts & Sewing (Amazon Reviews 2023) — 5-seed reproduction (Snellius)

Full write-up, deviations, and SBU estimates: [`../../REPRO_ARTS_2023.md`](../../REPRO_ARTS_2023.md).

Submit **from the repo root** (`/gpfs/home2/scur1250/rec_sys_cosette_marius`).
Shared config: [`env_arts.sh`](env_arts.sh) (venv, scratch `DATA_ROOT`, offline flags, `paths.root` override).

## Run order

| # | Step | Command | Where / GPUs | Notes |
|---|------|---------|--------------|-------|
| 0 | Verify GPU stack | `sbatch jobs/arts2023/00_verify_gpu.sbatch` | gpu_h100 ×1 | no data needed; safe anytime |
| A1 | Download (data + T5-XL) | `bash jobs/arts2023/A1_download.sh` | **login node** | once; needs internet |
| A2 | Parquet | `sbatch jobs/arts2023/A2_parquet.sbatch` | genoa (CPU) | once |
| A3 | Embeddings | `sbatch jobs/arts2023/A3_embeddings.sbatch` | gpu_h100 ×2 | once; offline |
| — | **Smoke test** | `sbatch jobs/arts2023/smoke_sasrec.sbatch` | gpu_h100 ×2 | after A2; confirm steps/epoch |
| B | COSETTE ×5 seeds | `sbatch jobs/arts2023/B_cosette.sbatch` | gpu_h100 ×1 (array 42–46) | per-seed tokenizer |
| C | Collisions + MARIUS ×5 | `sbatch jobs/arts2023/C_marius.sbatch` | gpu_h100 ×2 (array 42–46) | needs B per seed |
| D | SASRec++ ×5 seeds | `sbatch jobs/arts2023/D_sasrec.sbatch` | gpu_h100 ×2 (array 42–46) | independent of B/C |
| E | Aggregate vs paper | `python scripts/aggregate_arts.py` | login node | reads `runs/seed_*.{marius,sasrec}` |

Recommended order: **0 → A1 → A2 → smoke → (one full seed first) → A3 → B → C, with D in parallel → E.**

Chain C after B (per-seed): `sbatch --dependency=aftercorr:<B_jobid> jobs/arts2023/C_marius.sbatch`

## Reproduction overrides (reference recipe, applied per-run only)

- COSETTE: `optim.batch_size=1024 optim.epochs=3000`
- MARIUS & SASRec: `ray.scaling_config.num_workers=2  trainer.max_steps=81000  trainer.limit_val_batches=1000`
- Partition `gpu_h100` (94 GB) so reference batch sizes fit without an OOM-driven deviation.
- Seeds 42–46 (full pipeline: COSETTE retrained per seed).

Committed defaults (`num_workers=1`, `max_steps=80000`, `limit_val_batches=null`) are the
repo's gap-study config and are **left untouched** — overrides live in these scripts.

## Provenance (written at run time, committable)

- `runs/seed_<seed>.quant` — COSETTE quant id for the seed (B)
- `runs/seed_<seed>.marius` / `.sasrec` — Ray run-directory name (C / D)
