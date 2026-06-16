#!/bin/bash
#SBATCH --job-name=cm_dirgate
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Tier D: directional-structure gate for order-aware COSETTE (CPU-only, quick).
# Tests whether a directed (next-item) signal carries next-item-predictive
# structure that COSETTE's symmetric co-occurrence codes discard, BEFORE any
# tokenizer retrain. Reads the train timelines parquet + items pickle from the
# dataset and the committed support tables / MARIUS dumps. No GPU is actually
# needed: override for a CPU partition with
#   sbatch --partition=<cpu_partition> --gpus=0 jobs/25_direction_gate.sbatch
#
#   sbatch jobs/25_direction_gate.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_common.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions/direction_gate"
cd "${PROJECT_ROOT}"

for CAT in Sports_and_Outdoors Beauty; do
  TIMELINES="$(find "${DATA_ROOT}" -name "${CAT}.train.parquet" 2>/dev/null | head -1)"
  ITEMS="$(find "${DATA_ROOT}" -name "${CAT}_items.pkl" 2>/dev/null | head -1)"
  if [[ -z "${TIMELINES}" || -z "${ITEMS}" ]]; then
    echo "MISSING data for ${CAT}: timelines='${TIMELINES}' items='${ITEMS}' (check DATA_ROOT=${DATA_ROOT})"
    continue
  fi
  echo "=== ${CAT}: timelines=${TIMELINES} items=${ITEMS} ==="
  "${PYTHON_BIN}" scripts/extensions/direction_gate.py \
    --category "${CAT}" --timelines "${TIMELINES}" --items "${ITEMS}" \
    --dim "${SVD_DIM:-32}" --groups "${N_GROUPS:-256}" --seed "${SEED}"
done

echo "Direction-gate artifacts:"
ls -la "${PROJECT_ROOT}/reports/extensions/direction_gate/" || true
