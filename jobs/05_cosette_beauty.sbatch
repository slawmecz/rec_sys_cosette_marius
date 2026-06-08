#!/bin/bash
#SBATCH --job-name=cm_cosette_beauty
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Train COSETTE on Beauty (paper small-dataset settings: bs=256, 1000 epochs).
# Requires job 04 (embeddings). Output quant id e.g. COSETTE_128d_256x4-xxxx - save for job 06.
# sbatch jobs/05_cosette_beauty.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_beauty.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/cosette"
cd "${PROJECT_ROOT}"

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export CUDA_VISIBLE_DEVICES=0

WANDB_ARGS=(paths.wandb_mode="${WANDB_MODE}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(paths.wandb_entity="${WANDB_ENTITY}")
fi

"${PYTHON_BIN}" data_scripts/2_train_cosette.py \
  data.category="${CATEGORY}" \
  num_gpus=1 \
  optim.batch_size=256 \
  optim.epochs=1000 \
  optim.eval_step=100 \
  optim.dropout_prob=0.1 \
  paths.root="${DATA_ROOT}" \
  ckpt_dir="${OUTPUT_ROOT}/cosette" \
  "${WANDB_ARGS[@]}"

echo "COSETTE outputs (note the COSETTE_128d_256x4-* name for collision removal):"
ls -la "${DATA_ROOT}/data/embeddings/sentence-t5-xl/Beauty/" || true
