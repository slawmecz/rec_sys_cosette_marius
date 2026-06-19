#!/bin/bash
#SBATCH --job-name=cm_arts_rerank
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# RQ4 (post-processing mitigation) AT SCALE on Arts. Two steps per seed:
#   (1) GPU: scored depth-100 dumps (MARIUS + SASRec, --no-support, --with-scores).
#   (2) CPU: PMI logit-adjusted re-rank sweep (cond2/pair/item priors) + MBR tau sweep
#       -> accuracy-vs-bias Pareto (recall/ndcg vs coverage/Gini/ARP/tail).
# DEPENDS ON jobs/30 for the same seed (reuses its support tables -> --no-support).
#
#   sbatch --export=ALL,SEED=42 jobs/31_arts_scored_rerank.sbatch
#   sbatch --export=ALL,SEED=42,EVAL_BATCH_SIZE=16 jobs/31_arts_scored_rerank.sbatch  # if OOM

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_arts.sh"
cd "${PROJECT_ROOT}"

SEED="${SEED:-42}"
DUMP_ROOT="${PROJECT_ROOT}/reports/extensions/topk_arts/seed${SEED}"
OUT_DIR="${DUMP_ROOT}/Arts_Crafts_and_Sewing"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
ALPHAS="${ALPHAS:-0 0.25 0.5 0.75 1.0 1.25 1.5}"
MBR_TAUS="${MBR_TAUS:-0.25 0.5 1.0 2.0}"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUT_DIR}"

if [[ ! -f "${OUT_DIR}/tuple_to_item.json" ]]; then
  echo "ERROR: support tables missing in ${OUT_DIR}. Run jobs/30 for seed ${SEED} first." >&2
  exit 1
fi

# (1) scored depth-100 dumps (reuse support tables -> --no-support; quant-id not needed).
for M in marius sasrec; do
  echo "=== scored depth-100 dump: ${M} seed ${SEED} ==="
  "${PYTHON_BIN}" scripts/extensions/dump_topk.py \
    --method "${M}" --no-support \
    --n-results 100 --with-scores --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --category Arts_Crafts_and_Sewing --category-slug arts --seed "${SEED}" \
    --output-root "${ARTS_MODELS_ROOT}" --results-dir "${ARTS_RESULTS_DIR}" \
    --out-dir "${OUT_DIR}"
done

# (2a) PMI re-rank sweep (three priors), isolated per-seed outputs.
for PRIOR in cond2 pair item; do
  "${PYTHON_BIN}" scripts/extensions/pmi_rerank.py \
    --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
    --seeds "${SEED}" --n-results 100 --prior "${PRIOR}" --weighting demand \
    --alphas "${ALPHAS}" \
    --out "${DUMP_ROOT}/pmi_Arts_Crafts_and_Sewing_${PRIOR}_demand.md"
done

# (2b) MBR tau sweep (score-weighted), one report per tau.
for TAU in ${MBR_TAUS}; do
  "${PYTHON_BIN}" scripts/extensions/mbr.py \
    --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
    --seeds "${SEED}" --n-results 100 --weights scores --tau "${TAU}" \
    --out "${DUMP_ROOT}/mbr_Arts_Crafts_and_Sewing_tau${TAU}.md"
done

echo "=== re-rank artifacts (seed ${SEED}) ==="
ls -la "${DUMP_ROOT}"/pmi_*_pareto.csv "${DUMP_ROOT}"/mbr_*.csv 2>/dev/null || true
