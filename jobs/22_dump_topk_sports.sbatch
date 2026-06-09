#!/bin/bash
#SBATCH --job-name=cm_topk_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Beyond-accuracy extension: dump ranked Top-K lists + support tables (Sports & Outdoors).
# Read-only eval pass over existing checkpoints; no retraining, no model changes.
# Default seed 42; override with: sbatch --export=ALL,SEED=43 jobs/22_dump_topk_sports.sbatch
# Smoke-test shapes first: sbatch --export=ALL,SMOKE=1 jobs/22_dump_topk_sports.sbatch
# sbatch jobs/22_dump_topk_sports.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions/topk"
cd "${PROJECT_ROOT}"

QUANT_METHOD="${QUANT_METHOD:-COSETTE_128d_256x4_8ed1-col}"
SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

"${PYTHON_BIN}" scripts/extensions/dump_topk.py \
  --category Sports_and_Outdoors \
  --category-slug sports \
  --seed "${SEED}" \
  --emb-method sentence-t5-xl \
  --quant-method "${QUANT_METHOD}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-dir "${OUTPUT_ROOT}/results" \
  ${SMOKE_FLAG}

echo "Top-K dumps:"
ls -la "${PROJECT_ROOT}/reports/extensions/topk/Sports_and_Outdoors/" || true
