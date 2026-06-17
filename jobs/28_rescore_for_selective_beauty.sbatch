#!/bin/bash
#SBATCH --job-name=cm_rescore_sel_beauty
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Re-dump SCORED depth-100 Top-K to enable the selective-prediction CROSS-PARADIGM
# headline + multi-seed error bars (Beauty). Read-only eval over the existing baseline
# checkpoints; NO retraining, NO model-code changes, --no-support reuses the committed
# support tables (so no quant-id is needed -- sidesteps the f958-vs-a434 issue).
#   - SASRec --with-scores (dense dot-product) for SEEDS        -> sasrec_seed<s>_topk100.npz
#   - MARIUS --with-scores (joint log-prob) for MARIUS_SEEDS    -> marius_seed<s>_topk100.npz
#     (MARIUS seed 42 depth-100 is already committed; default re-dumps only 43 44)
# Writes *_topk100.npz alongside the committed *_topk.npz (different filename, no clobber).
#
#   sbatch jobs/28_rescore_for_selective_beauty.sbatch
#   sbatch --export=ALL,SEEDS="42 43 44",MARIUS_SEEDS="43 44" jobs/28_rescore_for_selective_beauty.sbatch
#   sbatch --export=ALL,EVAL_BATCH_SIZE=16 jobs/28_rescore_for_selective_beauty.sbatch   # if depth-100 OOMs

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_beauty.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions/topk"
cd "${PROJECT_ROOT}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
SEEDS="${SEEDS:-42 43 44}"            # SASRec scored dumps (its committed dumps carry no scores)
MARIUS_SEEDS="${MARIUS_SEEDS:-43 44}" # MARIUS depth-100 scored dumps (42 already committed)

for s in ${SEEDS}; do
  echo "=== SASRec scored depth-100 dump, seed ${s} ==="
  "${PYTHON_BIN}" scripts/extensions/dump_topk.py \
    --method sasrec --no-support \
    --n-results 100 --with-scores --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --category Beauty --category-slug beauty --seed "${s}" \
    --output-root "${OUTPUT_ROOT}" --results-dir "${OUTPUT_ROOT}/results"
done

for s in ${MARIUS_SEEDS}; do
  echo "=== MARIUS scored depth-100 dump, seed ${s} ==="
  "${PYTHON_BIN}" scripts/extensions/dump_topk.py \
    --method marius --no-support \
    --n-results 100 --with-scores --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --category Beauty --category-slug beauty --seed "${s}" \
    --output-root "${OUTPUT_ROOT}" --results-dir "${OUTPUT_ROOT}/results"
done

echo "Scored depth-100 dumps:"
ls -la "${PROJECT_ROOT}/reports/extensions/topk/Beauty/"*topk100.npz || true
