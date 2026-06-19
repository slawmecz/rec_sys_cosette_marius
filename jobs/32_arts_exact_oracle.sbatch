#!/bin/bash
#SBATCH --job-name=cm_arts_oracle
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# RQ3 CENTERPIECE AT SCALE: exact full-catalog teacher-forced oracle on Arts MARIUS.
# Scores EVERY catalog item's code-tuple for a user subsample -> decomposes the
# catalog-reachability collapse into SEARCH error vs MODEL-ranking error at 90k items.
# Answers "does the model-bound collapse persist at scale?" (the 2014 headline).
# Read-only; DEPENDS ON jobs/30 for the same seed (reads its meta + tuple_to_item).
#
# 90k items is ~7.5x Beauty's per-user candidate work -> smaller chunk/batch, more time.
#   --max-users 1000 (first-N test users, not random), --chunk-size 384, --eval-batch-size 16
#
#   sbatch --export=ALL,SEED=42 jobs/32_arts_exact_oracle.sbatch                 # filtered (default)
#   sbatch --export=ALL,SEED=42,FILTER_SEEN=0 jobs/32_arts_exact_oracle.sbatch   # unfiltered
#   sbatch --export=ALL,SEED=42,MAX_USERS=500,CHUNK_SIZE=256 jobs/32_arts_exact_oracle.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_arts.sh"
cd "${PROJECT_ROOT}"

SEED="${SEED:-42}"
QUANT="$(arts_quant_for_seed "${SEED}")"
DUMP_ROOT="${PROJECT_ROOT}/reports/extensions/topk_arts/seed${SEED}"
OUT_DIR="${DUMP_ROOT}/exact_catalog"
MAX_USERS="${MAX_USERS:-1000}"
CHUNK_SIZE="${CHUNK_SIZE:-384}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
BEAM_K="${BEAM_K:-20}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUT_DIR}"

# Default to filtered (apples-to-apples beam-vs-exact); set FILTER_SEEN=0 for raw.
FILTER_FLAG="--filter-seen"
[[ "${FILTER_SEEN:-1}" == "0" ]] && FILTER_FLAG=""

if [[ ! -f "${DUMP_ROOT}/Arts_Crafts_and_Sewing/tuple_to_item.json" ]]; then
  echo "ERROR: ${DUMP_ROOT}/Arts_Crafts_and_Sewing/tuple_to_item.json missing. Run jobs/30 for seed ${SEED} first." >&2
  exit 1
fi

echo "=== exact oracle | seed=${SEED} max_users=${MAX_USERS} chunk=${CHUNK_SIZE} batch=${EVAL_BATCH_SIZE} filter='${FILTER_FLAG}' ==="

"${PYTHON_BIN}" scripts/extensions/exact_catalog.py run \
  --category Arts_Crafts_and_Sewing --category-slug arts --seed "${SEED}" \
  --quant-method "${QUANT}" \
  --output-root "${ARTS_MODELS_ROOT}" --results-dir "${ARTS_RESULTS_DIR}" \
  --dump-root "${DUMP_ROOT}" --out-dir "${OUT_DIR}" \
  --max-users "${MAX_USERS}" --chunk-size "${CHUNK_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" --beam-k "${BEAM_K}" ${FILTER_FLAG}

"${PYTHON_BIN}" scripts/extensions/exact_catalog.py analyze \
  --category Arts_Crafts_and_Sewing --seed "${SEED}" \
  --dump-root "${DUMP_ROOT}" --out-dir "${OUT_DIR}" ${FILTER_FLAG}

echo "=== exact-catalog verdict (seed ${SEED}) ==="
ls -la "${OUT_DIR}" || true
