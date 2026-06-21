#!/usr/bin/env bash
#SBATCH --job-name=arts_embeddings
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%j.out
#
# PHASE A3 (run ONCE, GPU, seed-independent). Sentence-T5-XL embeddings for all items.
# Requires A2 parquet + the T5-XL model cached in HF_HOME (A1). Runs OFFLINE.
# Reference command (verbatim): 1_make_embeddings.py category=Arts_Crafts_and_Sewing
#   sbatch jobs/arts2023/A3_embeddings.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"

N_GPUS="$(nvidia-smi -L | wc -l)"
echo "[A3] embeddings on ${N_GPUS} GPU(s), offline (HF_HOME=${HF_HOME})"

# model_folder override replaces the author's hardcoded path with the HF id, which
# SentenceTransformer loads from the local HF cache (offline). This is the only
# departure from the reference command and is required on any machine.
python data_scripts/1_make_embeddings.py \
  category="${CATEGORY}" \
  num_gpus="${N_GPUS}" \
  model_folder=sentence-transformers/sentence-t5-xl \
  ${PATHS_OVERRIDES}

echo "A3_EMBEDDINGS_DONE"
ls -la "${DATA_ROOT}/data/embeddings/${EMB_METHOD}/${CATEGORY}/" || true
