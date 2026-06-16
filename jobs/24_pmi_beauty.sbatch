#!/bin/bash
#SBATCH --job-name=cm_pmi_beauty
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# B1: label-bias (PMI) re-ranker Pareto (Beauty). Two steps:
#   (1) GPU: produce a depth-100 SCORED MARIUS dump (read-only eval pass; the
#       committed depth-20 dumps carry no scores). Writes marius_seed<s>_topk100.npz.
#   (2) CPU: PMI logit-adjusted re-rank sweep (cond2 / pair / item priors).
# No retraining, no model-code changes. Reuses the committed support tables.
#
#   sbatch jobs/24_pmi_beauty.sbatch
#   sbatch --export=ALL,SEED=43 jobs/24_pmi_beauty.sbatch
#   sbatch --export=ALL,EVAL_BATCH_SIZE=16 jobs/24_pmi_beauty.sbatch   # if depth-100 OOMs

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_beauty.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions/pmi"
cd "${PROJECT_ROOT}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
ALPHAS="${ALPHAS:-0 0.25 0.5 0.75 1.0 1.25 1.5}"

# (1) depth-100 scored MARIUS dump (support tables already committed -> --no-support)
"${PYTHON_BIN}" scripts/extensions/dump_topk.py \
  --method marius --no-support \
  --n-results 100 --with-scores --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --category Beauty --category-slug beauty --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}" --results-dir "${OUTPUT_ROOT}/results"

# (2) PMI sweep for all three priors (demand-weighted)
for PRIOR in cond2 pair item; do
  "${PYTHON_BIN}" scripts/extensions/pmi_rerank.py \
    --category Beauty --seeds "${SEED}" \
    --n-results 100 --prior "${PRIOR}" --weighting demand --alphas "${ALPHAS}"
done

echo "PMI artifacts:"
ls -la "${PROJECT_ROOT}/reports/extensions/pmi/" || true
