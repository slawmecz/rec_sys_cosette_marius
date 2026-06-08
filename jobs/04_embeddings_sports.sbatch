#!/bin/bash
#SBATCH --job-name=cm_emb_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Sentence-T5-XL embeddings for Sports_and_Outdoors (one-time; reused by COSETTE → MARIUS).
# Requires job 02 (parquet) done. Skips if output parquet already exists.
# sbatch jobs/04_embeddings_sports.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${DATA_ROOT}/data/embeddings"
cd "${PROJECT_ROOT}"

export RAY_DISABLE_DOCKER_CPU_WARNING=1

"${PYTHON_BIN}" data_scripts/1_make_embeddings.py \
  category="${CATEGORY}" \
  num_gpus=1 \
  batch_size=64 \
  paths.root="${DATA_ROOT}" \
  model_folder=sentence-transformers/sentence-t5-xl

echo "Embeddings output:"
ls -la "${DATA_ROOT}/data/embeddings/sentence-t5-xl/Sports_and_Outdoors/" || true
