#!/bin/bash
#SBATCH --job-name=cm_distill_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Cross-paradigm distillation arm (Sports): distills a FROZEN SASRec++ teacher's
# catalog ranking into MARIUS at the last decode position. Trains MARIUSDistill
# (KL on a shared candidate set + base cross-entropy; inference unchanged) for 2
# seeds, then dumps Top-K + beyond-accuracy + reachability on the NEW checkpoints
# so we can compare against the baseline. Reuses the SAME quant_id as the baseline
# so the COSETTE -col tokens / catalog ordering line up with the teacher.
#
# CRITICAL (find_run_for_seed trap): this MUST use a FRESH OUTPUT_ROOT or the
# harness records the baseline run instead of training. Handled below.
#
# PREREQUISITES (set before submitting):
#   - A trained SASRec++ teacher on Sports (run dir under <its models>/SASRec_*).
#     Point TEACHER_MODELS_ROOT at that run's models dir and TEACHER_RUN_DIR at
#     the SASRec_* run directory name. Both default below to the baseline
#     OUTPUT_ROOT models dir (set TEACHER_RUN_DIR explicitly).
#   - The offline item->code-tuple table, built once by the Task-5 dumper from
#     the SAME -col tokens this trains on (see ITEM_TO_CODES_PATH below; the
#     dumper is invoked automatically if the table is missing).
#
#   sbatch --export=ALL,TEACHER_RUN_DIR=SASRec_xxxx jobs/27_distill_sports.sbatch
#   sbatch --export=ALL,DISTILL_ALPHA=0.5,DISTILL_TEMP=1.0,N_CAND=128,TEACHER_RUN_DIR=SASRec_xxxx jobs/27_distill_sports.sbatch
#   sbatch --export=ALL,DISTILL_SEEDS="42",TEACHER_RUN_DIR=SASRec_xxxx jobs/27_distill_sports.sbatch   # quick 1-seed

set -uo pipefail

# Set the FRESH OUTPUT_ROOT *before* sourcing env so env_common builds
# PATHS_OVERRIDES (paths.model_folder_tplt) against it. Otherwise checkpoints are
# written into the baseline models dir and find_run_for_seed / dump_topk miss them
# (env_common honors a pre-set OUTPUT_ROOT; this is the documented marius_fuse pattern).
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
DISTILL_TEMP="${DISTILL_TEMP:-1.0}"
N_CAND="${N_CAND:-128}"
export OUTPUT_ROOT="${OUTPUT_ROOT_DISTILL:-${SCRATCH:-${HOME}/scratch}/cosette_marius/outputs_distill_sports_a${DISTILL_ALPHA}_t${DISTILL_TEMP}}"
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
cd "${PROJECT_ROOT}"

SEEDS="${DISTILL_SEEDS:-42 43}"
# Tokenizer (-col) id this account's baseline was trained on; the teacher and the
# code table must match it. Override QUANT_ID if your -col parquet has a different suffix.
QUANT_ID="${QUANT_ID:-COSETTE_128d_256x4_24cd-col}"

# Frozen SASRec++ teacher. TEACHER_MODELS_ROOT must be the models dir that CONTAINS
# the SASRec_* run; TEACHER_RUN_DIR is that run's folder name. Defaults point at the
# baseline OUTPUT_ROOT models dir; set TEACHER_RUN_DIR for your account.
TEACHER_MODELS_ROOT="${TEACHER_MODELS_ROOT:-${SCRATCH:-${HOME}/scratch}/cosette_marius/outputs/models}"
TEACHER_RUN_DIR="${TEACHER_RUN_DIR:-SASRec_sports_seed42}"

# Offline item->code-tuple table (built once from the SAME -col tokens; see Task 5).
ITEM_TO_CODES_PATH="${ITEM_TO_CODES_PATH:-${PROJECT_ROOT}/reports/extensions/distill/item_to_codes_Sports_and_Outdoors.pt}"
QUANT_PARQUET="${DATA_ROOT}/data/embeddings/sentence-t5-xl/Sports_and_Outdoors/${QUANT_ID}.parquet"
ITEMS_PICKLE="${DATA_ROOT}/data/timelines/Sports_and_Outdoors_items.pkl"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results" "$(dirname "${ITEM_TO_CODES_PATH}")"

# Swap the model to MARIUSDistill AND the prepro to MARIUSDistillPrePro (which emits
# teacher_query + target_catalog_idx). items_ref is supplied to every prepro by
# make_pipeline, so only the prepro _cls_ needs overriding here.
export EXTRA_TRAIN_OVERRIDES="model.net._target_=src.models.marius_distill.MARIUSDistill +model.net.distill_alpha=${DISTILL_ALPHA} +model.net.distill_temp=${DISTILL_TEMP} +model.net.n_cand=${N_CAND} +model.net.teacher_models_root=${TEACHER_MODELS_ROOT} +model.net.teacher_run_dir=${TEACHER_RUN_DIR} +model.net.item_to_codes_path=${ITEM_TO_CODES_PATH} data.ray_datasets.prepro_cfg._cls_=src.data.marius.MARIUSDistillPrePro"

echo "=== distill arm: alpha=${DISTILL_ALPHA} temp=${DISTILL_TEMP} n_cand=${N_CAND} seeds='${SEEDS}' quant=${QUANT_ID} OUTPUT_ROOT=${OUTPUT_ROOT} ==="
echo "    teacher: ${TEACHER_MODELS_ROOT}/${TEACHER_RUN_DIR}"

# (0) build the offline item->code-tuple table if it is missing.
if [[ ! -f "${ITEM_TO_CODES_PATH}" ]]; then
  echo "=== building item->code-tuple table -> ${ITEM_TO_CODES_PATH} ==="
  test -f "${QUANT_PARQUET}" || { echo "MISSING quant parquet ${QUANT_PARQUET}"; exit 1; }
  test -f "${ITEMS_PICKLE}" || { echo "MISSING items pickle ${ITEMS_PICKLE}"; exit 1; }
  "${PYTHON_BIN}" scripts/extensions/distill_utils.py \
    --quant_parquet "${QUANT_PARQUET}" --items_pickle "${ITEMS_PICKLE}" \
    --out "${ITEM_TO_CODES_PATH}"
fi
test -f "${ITEM_TO_CODES_PATH}" || { echo "MISSING item->code table ${ITEM_TO_CODES_PATH}"; exit 1; }

# (1) train + test
"${PYTHON_BIN}" scripts/marius_5seed.py --mode full \
  --category Sports_and_Outdoors --category-slug sports \
  --quant-id "${QUANT_ID}" --seeds "${SEEDS}" --output-root "${OUTPUT_ROOT}"

# (2) dump Top-K + beyond-accuracy + reachability on the NEW checkpoints.
# Isolated, distill-specific paths so the committed baseline CSVs are never overwritten.
OUT_TOPK="${PROJECT_ROOT}/reports/extensions/topk_distill_sports_a${DISTILL_ALPHA}_t${DISTILL_TEMP}/Sports_and_Outdoors"
for S in ${SEEDS}; do
  "${PYTHON_BIN}" scripts/extensions/dump_topk.py --method marius \
    --category Sports_and_Outdoors --category-slug sports --seed "${S}" \
    --quant-method "${QUANT_ID}" --output-root "${OUTPUT_ROOT}" \
    --results-dir "${OUTPUT_ROOT}/results" --out-dir "${OUT_TOPK}"
done
DUMP_ROOT="$(dirname "${OUT_TOPK}")"
"${PYTHON_BIN}" scripts/extensions/compute_beyond_accuracy.py \
  --dump-root "${DUMP_ROOT}" --category Sports_and_Outdoors --models marius --seeds "${SEEDS}" --ks "10 20" \
  --out "${DUMP_ROOT}/beyond_accuracy_Sports_and_Outdoors_distill.csv"
"${PYTHON_BIN}" scripts/extensions/reach_estimator.py \
  --dump-root "${DUMP_ROOT}" --category Sports_and_Outdoors --models marius --seeds "${SEEDS}" --ks "10 20" \
  --out-dir "${DUMP_ROOT}/reach_estimator"

echo "=== results table (distilled MARIUS, Sports) ==="
cat "${OUTPUT_ROOT}/results/table5_sports_marius_5seed_summary_full_latest.txt" 2>/dev/null || true
echo "Compare R@10/NDCG against baseline reports/extensions/baseline_scur1217/, and"
echo "coverage/gini/tail vs reports/extensions/beyond_accuracy_Sports_and_Outdoors.csv"
