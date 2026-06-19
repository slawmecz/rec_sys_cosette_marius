#!/bin/bash
#SBATCH --job-name=cm_diversity_arts_h100
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.out
#SBATCH --error=/scratch-shared/%u/cosette_marius/outputs/logs/%x_%j.err

# Diversity / popularity-bias comparison across the 5 MARIUS and 5 SASRec seeds
# trained on Arts_Crafts_and_Sewing by scur1250
# (/scratch-shared/scur1250/cosette_arts/models, read-only to us).
# Evaluates each seed's best checkpoint on the real test set, computing:
#   - Gini + entropy per RVQ level (MARIUS) and item-level (both models)
#   - per-level support diagnostics (n_groups, mean group size / distinct codes)
#   - ILD and category diversity (distinct categories / K)
# then writes a cross-seed comparison table + plot under reports/diversity/.
# H100 variant of jobs/22_gini_arts_comparison.sbatch.
# sbatch jobs/23_diversity_arts_h100.sbatch

set -uo pipefail

# NOTE: this account has no conda env (deps are `pip install --user` into the
# system python3.9, i.e. ~/.local/lib/python3.9/site-packages) - skip
# jobs/env_common.sh's conda activation, since its PYTHONNOUSERSITE=1 would
# hide exactly the site-packages this account's torch/ray/etc. live in.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"
export RAY_TRAIN_V2_ENABLED=1
PYTHON_BIN="$(command -v python3)"

mkdir -p "/scratch-shared/${USER}/cosette_marius/outputs/logs"

"${PYTHON_BIN}" -m scripts.eval_diversity \
  --category Arts_Crafts_and_Sewing \
  --models-root /scratch-shared/scur1250/cosette_arts/models \
  --output-dir reports/diversity \
  --category-level leaf \
  --device cuda

"${PYTHON_BIN}" -m scripts.summarize_diversity \
  --category Arts_Crafts_and_Sewing \
  --input-dir reports/diversity \
  --output-dir reports/diversity

echo "Diversity summary:"
cat "${PROJECT_ROOT}/reports/diversity/Arts_Crafts_and_Sewing_diversity_summary.txt" 2>/dev/null || true
