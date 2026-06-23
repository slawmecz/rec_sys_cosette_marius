#!/bin/bash
#SBATCH --job-name=cm_logitadj_arts
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=42,44,46,48,50
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%A_%a.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%A_%a.err
#
# Train-time logit-adjustment arm AT SCALE (Arts_Crafts_and_Sewing, ~90k items).
# In-processing analogue of the PMI/MBR re-rank, on the large dataset so Table rq4 /
# Figure rq4 panel A match the small-data panels.
#
# Differences vs the small-data jobs/26_logitadj_*:
#   * Large arch: marius_5seed --experiment marius (d=512, L_T=4, L_D=6), 2 GPUs
#     (effective batch 512), max_steps=81000, limit_val_batches=1000 -- replicates
#     jobs/arts2023/C_marius.sbatch so baseline-vs-intervention is not arch-confounded.
#   * Per-seed COSETTE tokenizer (arts_quant_for_seed; already -col).
#   * Per-seed committed prior.
#   * Data READ from Jan's frozen read-only ARTS_MODELS_ROOT; checkpoints WRITTEN to a
#     fresh per-(seed,tau) OUTPUT_ROOT (find_run_for_seed trap).
#
# Submit one array per tau (3 submissions = 5 seeds x 3 tau = 15 trains):
#   sbatch --export=ALL,TAU=0.5 jobs/33_logitadj_arts.sbatch
#   sbatch --export=ALL,TAU=1.0 jobs/33_logitadj_arts.sbatch
#   sbatch --export=ALL,TAU=1.5 jobs/33_logitadj_arts.sbatch
#   sbatch --export=ALL,TAU=1.0,ARTS_SMOKE=1 --array=42 jobs/33_logitadj_arts.sbatch  # smoke
# Adjust to your allocation on the CLI: sbatch --account=... --partition=gpu_a100 ...

set -uo pipefail
SEED="${SLURM_ARRAY_TASK_ID:-42}"
TAU="${TAU:-1.0}"

# Read Arts data from Jan's frozen scratch; write OUR checkpoints to a fresh root.
export ARTS_MODELS_ROOT="${ARTS_MODELS_ROOT:-/scratch-shared/scur1250/cosette_arts}"
export DATA_ROOT="${ARTS_MODELS_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT_LOGITADJ:-/scratch-shared/${USER}/cosette_marius/outputs_logitadj_arts_tau${TAU}_seed${SEED}}"
# Large-arch recipe: paper_train_args() reads these envs (2 GPUs, effective batch 512).
export RAY_NUM_WORKERS=2
export TRAIN_GLOBAL_BATCH=512
export TRAIN_MAX_STEPS=81000
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_arts.sh"
cd "${PROJECT_ROOT}"

QUANT="$(arts_quant_for_seed "${SEED}")"
PRIOR="${PROJECT_ROOT}/reports/extensions/logitadj/arts_seed${SEED}/prior_Arts_Crafts_and_Sewing.npy"
mkdir -p "${OUTPUT_ROOT}/models" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/logs"
test -f "${PRIOR}" || { echo "MISSING prior ${PRIOR} (build with compute_token_prior.py)"; exit 1; }

# limit_val_batches is not emitted by paper_train_args(), so add it here (appended last).
LV="trainer.limit_val_batches=1000"
[[ "${ARTS_SMOKE:-0}" == "1" ]] && { export TRAIN_MAX_STEPS=200; LV="trainer.limit_val_batches=20"; }
export EXTRA_TRAIN_OVERRIDES="${LV} model.net._target_=src.models.marius_logitadj.MARIUSLogitAdj +model.net.logit_adj_tau=${TAU} +model.net.logit_adj_prior_path=${PRIOR}"

echo "=== logit-adj Arts | seed=${SEED} tau=${TAU} quant=${QUANT} ==="
echo "    DATA_ROOT=${DATA_ROOT}  OUTPUT_ROOT=${OUTPUT_ROOT}"

# (1) train + test via the proven harness (writes the run manifest dump_topk needs).
"${PYTHON_BIN}" scripts/marius_5seed.py --mode full \
  --experiment marius \
  --category Arts_Crafts_and_Sewing --category-slug arts \
  --quant-id "${QUANT}" --seeds "${SEED}" --output-root "${OUTPUT_ROOT}"

# (2) dump Top-K + beyond-accuracy + reachability on the NEW checkpoint (per-seed dir).
DUMP_ROOT="${PROJECT_ROOT}/reports/extensions/topk_logitadj_arts_tau${TAU}/seed${SEED}"
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
  --out "${DUMP_ROOT}/beyond_accuracy_Arts_logitadj_tau${TAU}_seed${SEED}.csv"
"${PYTHON_BIN}" scripts/extensions/reach_estimator.py \
  --dump-root "${DUMP_ROOT}" --category Arts_Crafts_and_Sewing \
  --models marius --seeds "${SEED}" --ks "10 20" \
  --out-dir "${DUMP_ROOT}/reach_estimator"

echo "LOGITADJ_ARTS_DONE seed=${SEED} tau=${TAU}"
