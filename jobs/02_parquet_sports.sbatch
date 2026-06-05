#!/bin/bash
#SBATCH --job-name=cm_pq_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.err

# Ray often fails on the login node (int5). Run this via sbatch, not interactively on int5.

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

export RAY_DISABLE_DOCKER_CPU_WARNING=1

"${PYTHON_BIN}" data_scripts/0_raw_to_parquet.py \
  --config-name 0_raw_to_parquet_2014 \
  categories="[Sports_and_Outdoors]" \
  paths.skip_download=true \
  paths.root="${DATA_ROOT}"

ls -la "${DATA_ROOT}/data/timelines/"*Sports* || true
