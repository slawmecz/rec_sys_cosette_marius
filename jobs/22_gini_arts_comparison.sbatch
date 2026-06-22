#!/bin/bash
#SBATCH --job-name=cm_diversity_arts
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Per-RVQ-level CONDITIONAL Gini index + Shannon entropy of the generated
# semantic IDs: MARIUS (per level, conditioned on the preceding levels) vs the
# SASRec flat baseline, across the 5 Arts_Crafts_and_Sewing seeds trained by
# scur1250 (/scratch-shared/scur1250/cosette_arts/models, read-only to us).
#
# Runs on OUR conda env (recsys) via jobs/env_arts.sh, and points the models
# root at Jan's read-only Arts dir. All outputs are isolated under
# reports/diversity/Arts_Crafts_and_Sewing/. This is the env-correct adaptation
# of Maks's original (which skipped conda for scur1250's pip --user python).
#
#   sbatch jobs/22_gini_arts_comparison.sbatch                                       # full: both methods, all seeds
#   sbatch --export=ALL,METHODS=MARIUS,SEEDS=42,LIMIT_BATCHES=4,SUMMARIZE=0 jobs/22_gini_arts_comparison.sbatch  # smoke
#   sbatch --export=ALL,METHODS=MARIUS,SUMMARIZE=0 jobs/22_gini_arts_comparison.sbatch   # MARIUS-only (priority)
#   sbatch --export=ALL,METHODS=SASRec,SUMMARIZE=0 jobs/22_gini_arts_comparison.sbatch   # SASRec-only baseline

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_arts.sh"
cd "${PROJECT_ROOT}"

METHODS="${METHODS:-MARIUS SASRec}"
SEEDS="${SEEDS:-42 44 46 48 50}"
SUMMARIZE="${SUMMARIZE:-1}"
LIMIT_FLAG=""
[[ -n "${LIMIT_BATCHES:-}" ]] && LIMIT_FLAG="--limit-batches ${LIMIT_BATCHES}"

echo "=== Arts conditional Gini/entropy | methods=[${METHODS}] seeds=[${SEEDS}] ${LIMIT_FLAG} ==="
echo "    models-root=${ARTS_MODELS_ROOT}/models  output=reports/diversity  python=${PYTHON_BIN}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"${PYTHON_BIN}" -m scripts.eval_diversity \
  --category "${CATEGORY}" \
  --models-root "${ARTS_MODELS_ROOT}/models" \
  --output-dir reports/diversity \
  --methods ${METHODS} \
  --seeds ${SEEDS} \
  --device cuda ${LIMIT_FLAG}

if [[ "${SUMMARIZE}" == "1" ]]; then
  "${PYTHON_BIN}" -m scripts.summarize_diversity \
    --category "${CATEGORY}" \
    --input-dir reports/diversity \
    --output-dir reports/diversity
  echo "Diversity summary:"
  cat "${PROJECT_ROOT}/reports/diversity/${CATEGORY}_diversity_summary.txt" 2>/dev/null || true
fi

echo "=== artifacts ==="
ls -la "${PROJECT_ROOT}/reports/diversity/${CATEGORY}" 2>/dev/null || true
