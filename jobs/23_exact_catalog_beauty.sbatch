#!/bin/bash
#SBATCH --job-name=cm_exact_beauty
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Extension SPINE: exact full-catalog teacher-forced scoring for MARIUS (Beauty).
# Read-only eval pass over the existing COSETTE+MARIUS checkpoint; no retraining,
# no model changes (calls only public modules, like dump_topk.py/scoring.py).
# Decomposes catalog-reachability collapse into SEARCH error vs MODEL-ranking error.
#
#   sbatch jobs/23_exact_catalog_beauty.sbatch
#   sbatch --export=ALL,SEED=43 jobs/23_exact_catalog_beauty.sbatch
#   sbatch --export=ALL,MAX_USERS=99999999 jobs/23_exact_catalog_beauty.sbatch   # all users
#   sbatch --export=ALL,SMOKE=1 jobs/23_exact_catalog_beauty.sbatch              # shape check

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_beauty.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions/exact_catalog"
cd "${PROJECT_ROOT}"

QUANT_METHOD="${QUANT_METHOD:-COSETTE_128d_256x4_f958-col}"
MAX_USERS="${MAX_USERS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
BEAM_K="${BEAM_K:-20}"
[[ "${SMOKE:-0}" == "1" ]] && MAX_USERS=64 && EVAL_BATCH_SIZE=16

"${PYTHON_BIN}" scripts/extensions/exact_catalog.py run \
  --category Beauty \
  --category-slug beauty \
  --seed "${SEED}" \
  --quant-method "${QUANT_METHOD}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-dir "${OUTPUT_ROOT}/results" \
  --max-users "${MAX_USERS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --beam-k "${BEAM_K}"

"${PYTHON_BIN}" scripts/extensions/exact_catalog.py analyze \
  --category Beauty --seed "${SEED}"

echo "Exact-catalog artifacts:"
ls -la "${PROJECT_ROOT}/reports/extensions/exact_catalog/" || true
