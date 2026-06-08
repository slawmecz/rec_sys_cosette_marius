#!/bin/bash
#SBATCH --job-name=cm_marius_sports
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# MARIUS (COSETTE) on Sports: 5 seeds, train + test + tables.
# Requires job 06 (COSETTE_128d_256x4_8ed1-col). ~5–9 h for 5 seeds.
# sbatch jobs/07_marius_sports_5seed_full.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_sports.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/marius_5seed.py --mode full \
  --category Sports_and_Outdoors \
  --category-slug sports \
  --quant-id COSETTE_128d_256x4_8ed1-col

echo "Full-run scores:"
cat "${OUTPUT_ROOT}/results/marius_sports_5seed_full_scores.jsonl" 2>/dev/null || true
echo "Summary table:"
cat "${OUTPUT_ROOT}/results/table5_sports_marius_5seed_summary_full_latest.txt" 2>/dev/null || true
