# Shared paths, conda, WandB, and training defaults.
# Dataset jobs source env_beauty.sh or env_sports.sh (not this file directly).

export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/scur1266/rec_sys_cosette_marius}}"

export DATA_ROOT="${DATA_ROOT:-${SCRATCH:-/home/scur1266/scratch}/cosette_marius/data}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:-/home/scur1266/scratch}/cosette_marius/outputs}"

export CONDA_ENV="${CONDA_ENV:-recsys}"

if [[ -n "${CONDA_ENV:-}" && -z "${PYTHON_BIN:-}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda not found. Use: source ~/miniconda3/etc/profile.d/conda.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
  conda activate "${CONDA_ENV}"
  export PYTHON_BIN="$(command -v python)"
fi
export PYTHON_BIN="${PYTHON_BIN:-python}"

# optional local secrets (WANDB_API_KEY, WANDB_ENTITY, ...)
if [[ -f "${PROJECT_ROOT}/jobs/wandb.env" ]]; then
  source "${PROJECT_ROOT}/jobs/wandb.env"
fi

export LOG_DIR="${LOG_DIR:-${SCRATCH:-/home/scur1266/scratch}/cosette_marius/outputs/logs}"
mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}/models" "${OUTPUT_ROOT}/wandb" "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}"
# Isolate the conda env from any ~/.local user-site packages (prevents version shadowing).
export PYTHONNOUSERSITE=1
export RAY_TRAIN_V2_ENABLED=1

# Weights & Biases
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-cosette-and-marius}"
# set WANDB_ENTITY in jobs/wandb.env or: export WANDB_ENTITY=your_username
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export BEAUTY_VOCAB_SIZE="${BEAUTY_VOCAB_SIZE:-12103}"
export SPORTS_VOCAB_SIZE="${SPORTS_VOCAB_SIZE:-18359}"
export SEED="${SEED:-42}"
# Paper reports mean ± std over 5 runs; exact seeds are not published.
export SASREC_5SEEDS="${SASREC_5SEEDS:-42 43 44 45 46}"

# Paper Appendix C (SASRec++ / MARIUS): 80k steps, global batch 256
export TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-80000}"
export TRAIN_GLOBAL_BATCH="${TRAIN_GLOBAL_BATCH:-256}"
export RAY_NUM_WORKERS="${RAY_NUM_WORKERS:-1}"
export TRAIN_BATCH_PER_GPU=$((TRAIN_GLOBAL_BATCH / RAY_NUM_WORKERS))

WANDB_ENTITY_OVERRIDE=""
if [[ -n "${WANDB_ENTITY}" ]]; then
  WANDB_ENTITY_OVERRIDE="paths.wandb_entity=${WANDB_ENTITY}"
fi

export PATHS_OVERRIDES="paths.root=${DATA_ROOT} paths.model_folder_tplt=${OUTPUT_ROOT}/models paths.wandb_mode=${WANDB_MODE} ${WANDB_ENTITY_OVERRIDE}"
export TRAIN_OVERRIDES="trainer.logger.project=${WANDB_PROJECT}"
