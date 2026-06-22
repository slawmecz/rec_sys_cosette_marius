# Shared environment for the Arts_Crafts_and_Sewing (Amazon Reviews 2023) 5-seed
# reproduction on Snellius. SOURCE this (do not execute) from every job below.
#
# Storage layout (project space /projects/prjs2120 is ACL-locked to another user,
# so it is NOT used here):
#   - code + venv + HF model cache   -> $HOME        (persistent, 200 GiB quota)
#   - bulk data / embeddings / ckpts -> /scratch-shared (8 TiB, periodically purged)

export PROJECT_ROOT="${PROJECT_ROOT:-/gpfs/home2/scur1250/rec_sys_cosette_marius}"
export VENV="${VENV:-/home/scur1250/cosette-venv}"
export DATA_ROOT="${DATA_ROOT:-/scratch-shared/scur1250/cosette_arts}"
export HF_HOME="${HF_HOME:-/home/scur1250/hf_cache}"

export CATEGORY="${CATEGORY:-Arts_Crafts_and_Sewing}"
export EMB_METHOD="${EMB_METHOD:-sentence-t5-xl}"

# Use the venv WITHOUT `activate` (robust under `set -e`; child `python` calls,
# e.g. the subprocess in 0_raw_to_parquet.py, inherit it through PATH).
export VIRTUAL_ENV="${VENV}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}"

# Ray Train v2 is required by this codebase.
export RAY_TRAIN_V2_ENABLED=1
export TOKENIZERS_PARALLELISM=false

# Compute nodes are offline: use the on-disk HF cache only.
# (jobs/arts2023/A1_download.sh unsets these because it runs on a login node.)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Weights & Biases offline (no account needed for the reproduction).
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${DATA_ROOT}/wandb}"

# Hydra overrides applied to every pipeline command: bulk artifacts -> scratch.
export PATHS_OVERRIDES="paths.root=${DATA_ROOT} paths.wandb_mode=${WANDB_MODE}"

mkdir -p "${DATA_ROOT}" "${HF_HOME}" "${WANDB_DIR}" \
         "${PROJECT_ROOT}/runs" "${PROJECT_ROOT}/jobs/arts2023/logs"

echo "[env_arts] python=$(command -v python) ($(python --version 2>&1))"
echo "[env_arts] DATA_ROOT=${DATA_ROOT}  HF_HOME=${HF_HOME}"
echo "[env_arts] CATEGORY=${CATEGORY}  WANDB_MODE=${WANDB_MODE}"
