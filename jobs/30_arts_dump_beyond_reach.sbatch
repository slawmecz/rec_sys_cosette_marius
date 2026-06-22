#!/bin/bash
#SBATCH --job-name=cm_arts_dump
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# RQ2 + RQ3 (cheap) AT SCALE on Arts_Crafts_and_Sewing (Amazon-2023, ~90k items).
# Read-only eval over Jan's frozen MARIUS+SASRec checkpoints; no retraining.
# Per seed: dump depth-20 Top-K (both models) WITH support tables into a PER-SEED
# dir, then beyond-accuracy (coverage/Gini/entropy/ILD/ARP/APLT/tail) + Chao1 reach.
#
# PER-SEED dirs are mandatory: each Arts seed has its OWN COSETTE tokenizer, so a
# single shared tuple_to_item.json would mis-decode the other seeds' MARIUS codes.
#
# NOTE: if your account needs an allocation, add e.g. #SBATCH --account=gpuuva080
#       (and switch partition to gpu_h100 if that is what your allocation grants).
#
#   sbatch --export=ALL,SEED=42 jobs/30_arts_dump_beyond_reach.sbatch
#   sbatch --export=ALL,SEED=44 jobs/30_arts_dump_beyond_reach.sbatch
#   sbatch --export=ALL,SEED=42,SMOKE=1 jobs/30_arts_dump_beyond_reach.sbatch   # shape check
#   sbatch --export=ALL,SEED=42,EVAL_BATCH_SIZE=16 jobs/30_arts_dump_beyond_reach.sbatch  # if OOM

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_arts.sh"
cd "${PROJECT_ROOT}"

SEED="${SEED:-42}"
QUANT="$(arts_quant_for_seed "${SEED}")"
DUMP_ROOT="${PROJECT_ROOT}/reports/extensions/topk_arts/seed${SEED}"
OUT_DIR="${DUMP_ROOT}/Arts_Crafts_and_Sewing"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUT_DIR}"

SMOKE_FLAG=""
[[ "${SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"
EVAL_BATCH_FLAG=""
[[ -n "${EVAL_BATCH_SIZE:-}" ]] && EVAL_BATCH_FLAG="--eval-batch-size ${EVAL_BATCH_SIZE}"

echo "=== Arts dump + beyond-accuracy + reach | seed=${SEED} quant=${QUANT} ==="
echo "    models-root=${ARTS_MODELS_ROOT}  results-dir=${ARTS_RESULTS_DIR}"
echo "    dump-root=${DUMP_ROOT}"

# (1) depth-20 Top-K dump (both models) WITH support tables, into the per-seed dir.
"${PYTHON_BIN}" scripts/extensions/dump_topk.py \
  --category Arts_Crafts_and_Sewing --category-slug arts --seed "${SEED}" \
  --method both --emb-method sentence-t5-xl --quant-method "${QUANT}" \
  --output-root "${ARTS_MODELS_ROOT}" --results-dir "${ARTS_RESULTS_DIR}" \
  --out-dir "${OUT_DIR}" ${EVAL_BATCH_FLAG} ${SMOKE_FLAG}

# (2) RQ2 beyond-accuracy (single seed -> isolated CSV; aggregate across seeds later).
"${PYTHON_BIN}" scripts/extensions/compute_beyond_accuracy.py \
  --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
  --models "sasrec marius" --seeds "${SEED}" --ks "10 20" \
  --out "${DUMP_ROOT}/beyond_accuracy_Arts_Crafts_and_Sewing_seed${SEED}.csv"

# (3) RQ3 reachability ceiling (Chao1 / Good-Turing).
"${PYTHON_BIN}" scripts/extensions/reach_estimator.py \
  --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
  --models "marius sasrec" --seeds "${SEED}" --ks "10 20" \
  --out-dir "${DUMP_ROOT}/reach_estimator"

echo "=== artifacts (seed ${SEED}) ==="
ls -la "${OUT_DIR}" || true
cat "${DUMP_ROOT}/beyond_accuracy_Arts_Crafts_and_Sewing_seed${SEED}.csv" 2>/dev/null || true
