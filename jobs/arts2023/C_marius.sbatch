#!/usr/bin/env bash
#SBATCH --job-name=arts_marius
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --array=42-46
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%A_%a.out
#
# PHASE C (per-seed, 2 GPUs). Collision removal -> MARIUS (full model) -> test.
# Requires Phase B for the SAME seed (reads runs/seed_${SEED}.quant).
#   sbatch jobs/arts2023/C_marius.sbatch
# Chain after B:  sbatch --dependency=aftercorr:<B_arrayjobid> jobs/arts2023/C_marius.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"
SEED="${SLURM_ARRAY_TASK_ID}"
echo "[C] seed=${SEED} | GPUs visible: $(nvidia-smi -L | wc -l)"

QUANT="$(cat "${PROJECT_ROOT}/runs/seed_${SEED}.quant")"
echo "[C] COSETTE quant=${QUANT}"

# 1) Distance-based collision removal -> writes ${QUANT}-col.parquet
python data_scripts/3_remove_colisions.py \
  data.category="${CATEGORY}" \
  data.quant_method="${QUANT}" \
  ${PATHS_OVERRIDES}
COLQUANT="${QUANT}-col"

# 2) Train MARIUS (full model, experiment=marius). REPRODUCTION OVERRIDES (the reference
#    recipe; our repo's committed defaults are the gap-study config and are left intact):
#      num_workers=2     -> 2 GPUs, effective batch 512 (this is the setting that gave
#                           the reference's R@10 5.04, NOT 256)
#      max_steps=81000   -> reference (repo default is 80000)
#      limit_val_batches=1000 -> reference (repo default is null/full)
python src/train.py experiment=marius \
  seed="${SEED}" \
  data.ray_datasets.category="${CATEGORY}" \
  data.ray_datasets.quant_id="${COLQUANT}" \
  ray.scaling_config.num_workers=2 \
  trainer.max_steps=81000 \
  trainer.limit_val_batches=1000 \
  ${PATHS_OVERRIDES}

# 3) Record run directory and compute validation + test metrics (writes metrics.pkl).
RUN_DIR="$(ls -t -d "${DATA_ROOT}"/models/MARIUS_"${CATEGORY}"_seed"${SEED}"_* | head -1 | xargs -n1 basename)"
echo "${RUN_DIR}" > "${PROJECT_ROOT}/runs/seed_${SEED}.marius"
python src/test.py run_directory="${RUN_DIR}" ${PATHS_OVERRIDES}
echo "C_MARIUS_DONE seed=${SEED} run_dir=${RUN_DIR}"
