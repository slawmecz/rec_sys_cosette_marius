#!/bin/bash
#SBATCH --job-name=cm_marius_paper
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/scur1266/scratch/cosette_marius/outputs_paper/logs/%x_%j.out
#SBATCH --error=/home/scur1266/scratch/cosette_marius/outputs_paper/logs/%x_%j.err

# =============================================================================
# OPTIONAL / SECONDARY — MARIUS (COSETTE) re-run  (see FAITHFUL_RERUN.md)
# -----------------------------------------------------------------------------
# MARIUS's config is ALREADY paper-faithful (d=256, 2 temporal + 2 depth layers,
# dropout 0.4/0.1, 80k steps, batch 256 — Table 9 / Fig 9a / App C.2). So this is
# NOT a config-fix experiment. It does two things:
#   1. Confirms the faithful MARIUS run reproduces (~8.17 Beauty R@10) in a fresh
#      environment (rules out env confounds).
#   2. Tests the checkpoint-selection-granularity hypothesis by validating more
#      often (TRAIN_VAL_CHECK_INTERVAL=2000 -> ~40 candidate checkpoints instead
#      of ~8), in case coarse best-val selection cost test recall.
#
# Reuses the EXISTING COSETTE tokens on scratch (quant-id below). If those are
# gone, regenerate with jobs 04 -> 05 -> 06 first. Writes to a FRESH OUTPUT_ROOT.
#
# SUBMIT:
#   sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,QUANT_ID=COSETTE_128d_256x4_f958-col jobs/21_marius_paper_fineckpt.sbatch
#   sbatch --export=ALL,CATEGORY=Sports_and_Outdoors,CATEGORY_SLUG=sports,QUANT_ID=COSETTE_128d_256x4_8ed1-col jobs/21_marius_paper_fineckpt.sbatch
# =============================================================================

set -uo pipefail

export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/home/scur1266/scratch}/cosette_marius/outputs_paper}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Finer checkpoint selection (paper only says "best checkpoint on validation").
export TRAIN_VAL_CHECK_INTERVAL="${TRAIN_VAL_CHECK_INTERVAL:-2000}"

CATEGORY="${CATEGORY:-Beauty}"
CATEGORY_SLUG="${CATEGORY_SLUG:-beauty}"
QUANT_ID="${QUANT_ID:-COSETTE_128d_256x4_f958-col}"
MODE="${MODE:-full}"

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_common.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
cd "${PROJECT_ROOT}"

echo "==============================================================="
echo " MARIUS faithful re-run  |  category=${CATEGORY}  mode=${MODE}"
echo " OUTPUT_ROOT=${OUTPUT_ROOT}  quant_id=${QUANT_ID}"
echo " val_check_interval=${TRAIN_VAL_CHECK_INTERVAL} (finer than default 10000)"
echo "==============================================================="

"${PYTHON_BIN}" scripts/marius_5seed.py --mode "${MODE}" \
  --category "${CATEGORY}" \
  --category-slug "${CATEGORY_SLUG}" \
  --quant-id "${QUANT_ID}"

echo ""
echo "=== Paper-vs-faithful summary (${CATEGORY}, MARIUS) ==="
cat "${OUTPUT_ROOT}/results/table5_${CATEGORY_SLUG}_marius_5seed_summary_${MODE}_latest.txt" 2>/dev/null || true
