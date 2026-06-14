#!/usr/bin/env bash
#SBATCH --job-name=arts_sasrec
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --array=42-46
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%A_%a.out
#
# PHASE D (per-seed, 2 GPUs). SASRec++ baseline (dense, no COSETTE) -> test.
# Independent of Phases B/C; only needs A2 parquet (no embeddings).
#   sbatch jobs/arts2023/D_sasrec.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"
SEED="${SLURM_ARRAY_TASK_ID}"
echo "[D] SASRec++ seed=${SEED} | GPUs visible: $(nvidia-smi -L | wc -l)"

# REPRODUCTION OVERRIDES (reference recipe; committed defaults left intact):
#   num_workers=2 (effective batch 512), max_steps=81000, limit_val_batches=1000
python src/train.py experiment=sasrec \
  seed="${SEED}" \
  data.ray_datasets.category="${CATEGORY}" \
  ray.scaling_config.num_workers=2 \
  trainer.max_steps=81000 \
  trainer.limit_val_batches=1000 \
  ${PATHS_OVERRIDES}

RUN_DIR="$(ls -t -d "${DATA_ROOT}"/models/SASRec_"${CATEGORY}"_seed"${SEED}"_* | head -1 | xargs -n1 basename)"
echo "${RUN_DIR}" > "${PROJECT_ROOT}/runs/seed_${SEED}.sasrec"
# SASRec is dense: enforce_filtering=false avoids setting filter_preds on a model that
# has no such argument (MARIUS keeps the default True).
python src/test.py run_directory="${RUN_DIR}" enforce_filtering=false ${PATHS_OVERRIDES}
echo "D_SASREC_DONE seed=${SEED} run_dir=${RUN_DIR}"
