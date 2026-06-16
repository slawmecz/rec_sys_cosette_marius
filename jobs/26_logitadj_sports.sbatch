#!/bin/bash
#SBATCH --job-name=cm_logitadj_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Train-time logit-adjustment arm (Sports): the model-side analogue of the PMI
# re-rank. Trains MARIUSLogitAdj (logit-adjusted CE, inference unchanged) for 2
# seeds, then dumps Top-K + beyond-accuracy + reachability on the NEW checkpoints
# so we can compare against the baseline. Reuses the SAME quant_id as the baseline
# so the precomputed per-token prior lines up.
#
# CRITICAL (find_run_for_seed trap): this MUST use a FRESH OUTPUT_ROOT or the
# harness records the baseline run instead of training. Handled below.
#
#   sbatch jobs/26_logitadj_sports.sbatch
#   sbatch --export=ALL,TAU=0.5 jobs/26_logitadj_sports.sbatch
#   sbatch --export=ALL,LOGITADJ_SEEDS="42" jobs/26_logitadj_sports.sbatch   # quick 1-seed

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
cd "${PROJECT_ROOT}"

TAU="${TAU:-1.0}"
SEEDS="${LOGITADJ_SEEDS:-42 43}"
QUANT_ID="${QUANT_ID:-COSETTE_128d_256x4_8ed1-col}"
PRIOR="${PROJECT_ROOT}/reports/extensions/logitadj/prior_Sports_and_Outdoors.npy"

# Fresh OUTPUT_ROOT so the harness trains (does not record the baseline).
export OUTPUT_ROOT="${OUTPUT_ROOT_LOGITADJ:-${SCRATCH:-${HOME}/scratch}/cosette_marius/outputs_logitadj_sports_tau${TAU}}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
export EXTRA_TRAIN_OVERRIDES="model.net._target_=src.models.marius_logitadj.MARIUSLogitAdj +model.net.logit_adj_tau=${TAU} +model.net.logit_adj_prior_path=${PRIOR}"

echo "=== logit-adjustment arm: tau=${TAU} seeds='${SEEDS}' quant=${QUANT_ID} OUTPUT_ROOT=${OUTPUT_ROOT} ==="
test -f "${PRIOR}" || { echo "MISSING prior ${PRIOR} (run compute_token_prior.py)"; exit 1; }

# (1) train + test
"${PYTHON_BIN}" scripts/marius_5seed.py --mode full \
  --category Sports_and_Outdoors --category-slug sports \
  --quant-id "${QUANT_ID}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

# (2) dump Top-K + beyond-accuracy + reachability on the NEW checkpoints
OUT_TOPK="${PROJECT_ROOT}/reports/extensions/topk_logitadj_sports_tau${TAU}/Sports_and_Outdoors"
for S in ${SEEDS}; do
  "${PYTHON_BIN}" scripts/extensions/dump_topk.py --method marius \
    --category Sports_and_Outdoors --category-slug sports --seed "${S}" \
    --quant-method "${QUANT_ID}" --output-root "${OUTPUT_ROOT}" \
    --results-dir "${OUTPUT_ROOT}/results" --out-dir "${OUT_TOPK}"
done
DUMP_ROOT="$(dirname "${OUT_TOPK}")"
"${PYTHON_BIN}" scripts/extensions/compute_beyond_accuracy.py \
  --dump-root "${DUMP_ROOT}" --category Sports_and_Outdoors --models marius --seeds "${SEEDS}" --ks "10 20"
"${PYTHON_BIN}" scripts/extensions/reach_estimator.py \
  --dump-root "${DUMP_ROOT}" --category Sports_and_Outdoors --models marius --seeds "${SEEDS}" --ks "10 20"

echo "=== results table (logit-adjusted MARIUS, Sports) ==="
cat "${OUTPUT_ROOT}/results/table5_sports_marius_5seed_summary_full_latest.txt" 2>/dev/null || true
echo "Compare R@10/NDCG against baseline reports/extensions/baseline_scur1217/, and"
echo "coverage/gini/tail vs reports/extensions/beyond_accuracy_Sports_and_Outdoors.csv"
