#!/bin/bash
#SBATCH --job-name=cm_ext_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.err

# Extension benchmarks: inference time + GPU memory (Sports, seed 42).
# Writes reports/extensions/*.csv and *.json (no retraining).
# sbatch jobs/08_extensions_sports.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${PROJECT_ROOT}/reports/extensions"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/benchmark_extensions.py \
  --category Sports_and_Outdoors \
  --category-slug sports \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}" \
  --results-dir "${OUTPUT_ROOT}/results"

echo "Extension artifacts:"
ls -la "${PROJECT_ROOT}/reports/extensions/"
