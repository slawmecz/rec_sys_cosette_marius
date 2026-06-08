#!/bin/bash
#SBATCH --job-name=cm_dedupe_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Deduplicate COSETTE semantic IDs for Sports_and_Outdoors (job 05 output).
# Writes COSETTE_128d_256x4_8ed1-col.parquet for MARIUS training.
# sbatch jobs/06_remove_collisions_sports.sbatch

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0

QUANT_METHOD="COSETTE_128d_256x4_8ed1"
OUT="${DATA_ROOT}/data/embeddings/sentence-t5-xl/Sports_and_Outdoors/${QUANT_METHOD}-col.parquet"

if [[ -f "${OUT}" ]]; then
  echo "Already exists, skipping: ${OUT}"
  ls -la "${OUT}"
  exit 0
fi

PYTHONPATH=. "${PYTHON_BIN}" data_scripts/3_remove_colisions.py \
  data.category="${CATEGORY}" \
  data.emb_method=sentence-t5-xl \
  data.quant_method="${QUANT_METHOD}" \
  paths.root="${DATA_ROOT}"

echo "Deduped output:"
ls -la "${DATA_ROOT}/data/embeddings/sentence-t5-xl/Sports_and_Outdoors/${QUANT_METHOD}"* || true
