#!/bin/bash
#SBATCH --job-name=cm_sasrec_paper
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs_paper/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs_paper/logs/%x_%j.err

# =============================================================================
# Paper-faithful SASRec++ re-run  (see FAITHFUL_RERUN.md for the full rationale)
# -----------------------------------------------------------------------------
# Reproduces the paper's PER-DATASET selected SASRec++ config (Appendix C.1 +
# Figure 11a), which differs from the repo's released DEFAULT (d=128, L2 norm).
#   * Beauty (smallest, 12k items): paper selects d=32, L=2, NO normalization.
#   * The released default d=128 + L2 sits in the DARK (worst, ~8 R@10) region
#     of the paper's own Figure 11a for Beauty — i.e. it is NOT the config that
#     produced the paper's reported 9.73.
#
# This job keeps DATA_ROOT (reuses the existing parquet timelines — SASRec needs
# NO embeddings / COSETTE) and writes to a FRESH OUTPUT_ROOT so the original
# d=128 runs are NOT re-recorded by the resume logic.
#
# SUBMIT (smoke-test first, then full — see FAITHFUL_RERUN.md):
#   # Beauty, paper config (PRIMARY):
#   sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,VOCAB=12103,SAS_D=32,SAS_DH=16 jobs/20_sasrec_paper_faithful.sbatch
#   # Beauty, d=64 hedge (in case the Fig-11a star is one tick off):
#   sbatch --export=ALL,CATEGORY=Beauty,CATEGORY_SLUG=beauty,VOCAB=12103,SAS_D=64,SAS_DH=32 jobs/20_sasrec_paper_faithful.sbatch
#   # Sports (18k, interpolated "slightly larger"):
#   sbatch --export=ALL,CATEGORY=Sports_and_Outdoors,CATEGORY_SLUG=sports,VOCAB=18359,SAS_D=64,SAS_DH=32 jobs/20_sasrec_paper_faithful.sbatch
#   sbatch --export=ALL,CATEGORY=Sports_and_Outdoors,CATEGORY_SLUG=sports,VOCAB=18359,SAS_D=32,SAS_DH=16 jobs/20_sasrec_paper_faithful.sbatch
#
# For a ~2-min pipeline check before the full run, add MODE=smoke to --export.
# =============================================================================

set -uo pipefail

# --- Isolated output root: forces a fresh 5-seed train (no re-record of d=128) -
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/home/scur1266/scratch}/cosette_marius/outputs_paper}"
# Results are read from metrics.pkl (src/test.py), not W&B -> offline avoids auth stalls.
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- Faithful config knobs (defaults = paper's Beauty selection) --------------
CATEGORY="${CATEGORY:-Beauty}"
CATEGORY_SLUG="${CATEGORY_SLUG:-beauty}"
VOCAB="${VOCAB:-12103}"
SAS_D="${SAS_D:-32}"          # paper Fig 11a star for Beauty
SAS_DH="${SAS_DH:-16}"        # d_head: keep 2 heads (n_heads = d_model/d_head)
SAS_NORM="${SAS_NORM:-null}"  # null = NO normalization (paper, smallest dataset)
SAS_TEMP="${SAS_TEMP:-1.0}"   # temperature 1.0 for unnormalized logits
MODE="${MODE:-full}"          # full | smoke

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_common.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
cd "${PROJECT_ROOT}"

# Sanity guard: n_heads must be >= 1 and divide d_model.
if (( SAS_D % SAS_DH != 0 )); then
  echo "ERROR: d_model (${SAS_D}) must be divisible by d_head (${SAS_DH})." >&2
  exit 2
fi

export EXTRA_TRAIN_OVERRIDES="model.net.d_model=${SAS_D} model.net.d_head=${SAS_DH} model.net.normalization=${SAS_NORM} model.net.criterion.temperature=${SAS_TEMP}"

echo "==============================================================="
echo " Paper-faithful SASRec++  |  category=${CATEGORY}  mode=${MODE}"
echo " OUTPUT_ROOT = ${OUTPUT_ROOT}   (DATA_ROOT = ${DATA_ROOT})"
echo " EXTRA_TRAIN_OVERRIDES = ${EXTRA_TRAIN_OVERRIDES}"
echo "==============================================================="

"${PYTHON_BIN}" scripts/sasrec_5seed.py --mode "${MODE}" \
  --category "${CATEGORY}" \
  --category-slug "${CATEGORY_SLUG}" \
  --vocab-size "${VOCAB}"

echo ""
echo "=== Paper-vs-faithful summary (${CATEGORY}) ==="
cat "${OUTPUT_ROOT}/results/table5_${CATEGORY_SLUG}_5seed_summary_${MODE}_latest.txt" 2>/dev/null || true
echo ""
echo "Per-seed scores: ${OUTPUT_ROOT}/results/sasrec_${CATEGORY_SLUG}_5seed_${MODE}_scores.jsonl"
