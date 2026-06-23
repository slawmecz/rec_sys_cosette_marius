#!/bin/bash
#SBATCH --job-name=cm_distill_arts
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=42,44,46,48,50
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%A_%a.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%A_%a.err
#
# Cross-paradigm distillation arm AT SCALE (Arts, ~90k items). Distills a FROZEN
# per-seed SASRec++ teacher's catalog ranking into MARIUS (KL on a shared candidate
# set + base CE; inference unchanged). Large arch, single config (alpha=0.5, temp=1.0,
# n_cand=128), 5 seeds; matches how the small-data distill is a single point.
#
# PREREQUISITES (verified in Snellius STEP 0):
#   * Arts processed data + per-seed -col parquet readable under ARTS_MODELS_ROOT.
#   * A loadable per-seed SASRec++ teacher. Default points at Jan's frozen run via
#     runs/seed_<SEED>.sasrec; if that checkpoint was purged, retrain SASRec on Arts
#     (jobs/arts2023/D_sasrec.sbatch recipe) and override TEACHER_MODELS_ROOT +
#     TEACHER_RUN_DIR on the CLI.
#
#   sbatch jobs/34_distill_arts.sbatch
#   sbatch --export=ALL,DISTILL_ALPHA=0.5,DISTILL_TEMP=1.0,N_CAND=128 jobs/34_distill_arts.sbatch
#   sbatch --export=ALL,ARTS_SMOKE=1 --array=42 jobs/34_distill_arts.sbatch  # smoke
#   sbatch --export=ALL,TEACHER_MODELS_ROOT=/scratch-shared/$USER/.../models,TEACHER_RUN_DIR=SASRec_... jobs/34_distill_arts.sbatch

set -uo pipefail
SEED="${SLURM_ARRAY_TASK_ID:-42}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
DISTILL_TEMP="${DISTILL_TEMP:-1.0}"
N_CAND="${N_CAND:-128}"

export ARTS_MODELS_ROOT="${ARTS_MODELS_ROOT:-/scratch-shared/scur1250/cosette_arts}"
export DATA_ROOT="${ARTS_MODELS_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT_DISTILL:-/scratch-shared/${USER}/cosette_marius/outputs_distill_arts_a${DISTILL_ALPHA}_seed${SEED}}"
export RAY_NUM_WORKERS=2
export TRAIN_GLOBAL_BATCH=512
export TRAIN_MAX_STEPS=81000
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_arts.sh"
cd "${PROJECT_ROOT}"

QUANT="$(arts_quant_for_seed "${SEED}")"
mkdir -p "${OUTPUT_ROOT}/models" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/logs"

# Per-seed frozen SASRec++ teacher (default: Jan's frozen run named in runs/seed_<S>.sasrec).
TEACHER_MODELS_ROOT="${TEACHER_MODELS_ROOT:-${ARTS_MODELS_ROOT}/models}"
if [[ -z "${TEACHER_RUN_DIR:-}" ]]; then
  TEACHER_RUN_DIR="$(cat "${PROJECT_ROOT}/runs/seed_${SEED}.sasrec" 2>/dev/null || true)"
fi
test -n "${TEACHER_RUN_DIR}" || { echo "MISSING TEACHER_RUN_DIR (no runs/seed_${SEED}.sasrec); set it on the CLI"; exit 1; }
test -d "${TEACHER_MODELS_ROOT}/${TEACHER_RUN_DIR}" || { echo "MISSING teacher dir ${TEACHER_MODELS_ROOT}/${TEACHER_RUN_DIR} (purged? retrain SASRec)"; exit 1; }

# Per-seed offline item->code-tuple table (built once from this seed's -col tokens).
ITEM_TO_CODES_PATH="${PROJECT_ROOT}/reports/extensions/distill/item_to_codes_Arts_Crafts_and_Sewing_seed${SEED}.pt"
QUANT_PARQUET="${DATA_ROOT}/data/embeddings/sentence-t5-xl/Arts_Crafts_and_Sewing/${QUANT}.parquet"
ITEMS_PICKLE="${DATA_ROOT}/data/timelines/Arts_Crafts_and_Sewing_items.pkl"
mkdir -p "$(dirname "${ITEM_TO_CODES_PATH}")"

echo "=== distill Arts | seed=${SEED} alpha=${DISTILL_ALPHA} temp=${DISTILL_TEMP} n_cand=${N_CAND} quant=${QUANT} ==="
echo "    teacher: ${TEACHER_MODELS_ROOT}/${TEACHER_RUN_DIR}"

# (0) build the per-seed item->code-tuple table if missing.
if [[ ! -f "${ITEM_TO_CODES_PATH}" ]]; then
  echo "=== building item->code table -> ${ITEM_TO_CODES_PATH} ==="
  test -f "${QUANT_PARQUET}" || { echo "MISSING quant parquet ${QUANT_PARQUET}"; exit 1; }
  test -f "${ITEMS_PICKLE}" || { echo "MISSING items pickle ${ITEMS_PICKLE}"; exit 1; }
  "${PYTHON_BIN}" scripts/extensions/distill_utils.py \
    --quant_parquet "${QUANT_PARQUET}" --items_pickle "${ITEMS_PICKLE}" \
    --out "${ITEM_TO_CODES_PATH}"
fi
test -f "${ITEM_TO_CODES_PATH}" || { echo "MISSING item->code table ${ITEM_TO_CODES_PATH}"; exit 1; }

LV="trainer.limit_val_batches=1000"
[[ "${ARTS_SMOKE:-0}" == "1" ]] && { export TRAIN_MAX_STEPS=200; LV="trainer.limit_val_batches=20"; }
export EXTRA_TRAIN_OVERRIDES="${LV} model.net._target_=src.models.marius_distill.MARIUSDistill +model.net.distill_alpha=${DISTILL_ALPHA} +model.net.distill_temp=${DISTILL_TEMP} +model.net.n_cand=${N_CAND} +model.net.teacher_models_root=${TEACHER_MODELS_ROOT} +model.net.teacher_run_dir=${TEACHER_RUN_DIR} +model.net.item_to_codes_path=${ITEM_TO_CODES_PATH} data.ray_datasets.prepro_cfg._cls_=src.data.marius.MARIUSDistillPrePro"

# (1) train + test via the proven harness (writes the run manifest dump_topk needs).
"${PYTHON_BIN}" scripts/marius_5seed.py --mode full \
  --experiment marius \
  --category Arts_Crafts_and_Sewing --category-slug arts \
  --quant-id "${QUANT}" --seeds "${SEED}" --output-root "${OUTPUT_ROOT}"

# (2) dump Top-K + beyond-accuracy + reachability on the NEW checkpoint (per-seed dir).
DUMP_ROOT="${PROJECT_ROOT}/reports/extensions/topk_distill_arts/seed${SEED}"
OUT_DIR="${DUMP_ROOT}/Arts_Crafts_and_Sewing"
mkdir -p "${OUT_DIR}"
EVAL_BATCH_FLAG=""
[[ -n "${EVAL_BATCH_SIZE:-}" ]] && EVAL_BATCH_FLAG="--eval-batch-size ${EVAL_BATCH_SIZE}"
"${PYTHON_BIN}" scripts/extensions/dump_topk.py \
  --category Arts_Crafts_and_Sewing --category-slug arts --seed "${SEED}" \
  --method marius --emb-method sentence-t5-xl --quant-method "${QUANT}" \
  --output-root "${OUTPUT_ROOT}" --results-dir "${OUTPUT_ROOT}/results" \
  --out-dir "${OUT_DIR}" ${EVAL_BATCH_FLAG}
"${PYTHON_BIN}" scripts/extensions/compute_beyond_accuracy.py \
  --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
  --models marius --seeds "${SEED}" --ks "10 20" \
  --out "${DUMP_ROOT}/beyond_accuracy_Arts_distill_seed${SEED}.csv"
"${PYTHON_BIN}" scripts/extensions/reach_estimator.py \
  --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
  --models marius --seeds "${SEED}" --ks "10 20" \
  --out-dir "${DUMP_ROOT}/reach_estimator"

echo "DISTILL_ARTS_DONE seed=${SEED} alpha=${DISTILL_ALPHA}"
