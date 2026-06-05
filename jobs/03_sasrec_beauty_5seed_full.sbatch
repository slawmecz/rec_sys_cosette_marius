#!/bin/bash
#SBATCH --job-name=cm_sasrec_beauty_full
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/home/scur1266/scratch/cosette_marius/outputs/logs/%x_%j.err

# Beauty SASRec++: 5 seeds, train + test + tables (paper-aligned).
# sbatch jobs/03_sasrec_beauty_5seed_full.sbatch

set -uo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_beauty.sh"
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/results"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/sasrec_5seed.py --mode full \
  --category Beauty \
  --category-slug beauty \
  --vocab-size 12103

echo "Full-run scores:"
cat "${OUTPUT_ROOT}/results/sasrec_beauty_5seed_full_scores.jsonl" 2>/dev/null || true
echo "Individual runs table:"
cat "${OUTPUT_ROOT}/results/table5_beauty_5seed_individual_full_latest.txt" 2>/dev/null || true
echo "Paper-style summary table:"
cat "${OUTPUT_ROOT}/results/table5_beauty_5seed_summary_full_latest.txt" 2>/dev/null || true
