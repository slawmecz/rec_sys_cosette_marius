#!/bin/bash
#SBATCH --job-name=cm_topk_beauty
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Beyond-accuracy extension: dump ranked Top-K lists + support tables (Beauty).
# Read-only eval pass over existing checkpoints; no retraining, no model changes.
# Default seed 42; override with: sbatch --export=ALL,SEED=43 jobs/22_dump_topk_beauty.sbatch
# Smoke-test shapes first: sbatch --export=ALL,SMOKE=1 jobs/22_dump_topk_beauty.sbatch
# sbatch jobs/22_dump_topk_beauty.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_beauty.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions/topk"
cd "${PROJECT_ROOT}"

QUANT_METHOD="${QUANT_METHOD:-COSETTE_128d_256x4_f958-col}"
SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

"${PYTHON_BIN}" scripts/extensions/dump_topk.py \
  --category Beauty \
  --category-slug beauty \
  --seed "${SEED}" \
  --emb-method sentence-t5-xl \
  --quant-method "${QUANT_METHOD}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-dir "${OUTPUT_ROOT}/results" \
  ${SMOKE_FLAG}

echo "Top-K dumps:"
ls -la "${PROJECT_ROOT}/reports/extensions/topk/Beauty/" || true
