#!/usr/bin/env bash
#SBATCH --job-name=arts_smoke
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%j.out
#
# STEP 8 SMOKE TEST. Tiny end-to-end SASRec++ run to confirm env, Ray, data loading,
# 2-GPU DDP, validation and checkpointing on a real Snellius node. Needs A2 parquet
# only (no embeddings / COSETTE). Run BEFORE the full sweep.
#
# Uses the SAME 2-worker setup as the real runs so the log reveals steps-per-epoch:
# with num_workers=2 and total_len = max_steps*batch, expect ~max_steps/2 steps per
# epoch (here ~300) and max_steps reached over 2 epochs. CONFIRM THIS in the log.
#   sbatch jobs/arts2023/smoke_sasrec.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"
echo "[smoke] SASRec++ 600-step / 2-worker run on $(hostname)"

python src/train.py experiment=sasrec \
  seed=42 \
  data.ray_datasets.category="${CATEGORY}" \
  ray.scaling_config.num_workers=2 \
  trainer.max_steps=600 \
  trainer.val_check_interval=300 \
  trainer.limit_val_batches=50 \
  ${PATHS_OVERRIDES}

echo "SMOKE_DONE"
