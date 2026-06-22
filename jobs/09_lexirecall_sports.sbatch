#!/bin/bash
#SBATCH --job-name=cm_lexi_sports
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# TSE / lexirecall evaluation (Sports, seed 42). Needs SASRec + MARIUS checkpoints.
# sbatch jobs/09_lexirecall_sports.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/lexirecall_eval.py \
  --category Sports_and_Outdoors \
  --category-slug sports \
  --seed "${SEED}" \
  --topk 20 \
  --output-root "${OUTPUT_ROOT}" \
  --results-dir "${OUTPUT_ROOT}/results" \
  --data-root "${DATA_ROOT}"

echo "Lexirecall outputs:"
ls -la "${PROJECT_ROOT}/reports/extensions/lexirecall"* 2>/dev/null || true
ls -la "${PROJECT_ROOT}/reports/extensions/trec/sports/" 2>/dev/null || true
