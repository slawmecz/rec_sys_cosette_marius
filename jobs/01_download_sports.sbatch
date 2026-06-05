#!/bin/bash
#SBATCH --job-name=cm_dl_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "DATA_ROOT=${DATA_ROOT}"
"${PYTHON_BIN}" scripts/download_data.py --year 2014 --base_dir "${DATA_ROOT}" --categories Sports_and_Outdoors
ls -la "${DATA_ROOT}/amazon-2014/"*Sports* || true
