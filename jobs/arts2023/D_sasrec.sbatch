#!/usr/bin/env bash
#SBATCH --job-name=arts_sasrec
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --time=10:00:00
#SBATCH --array=42,44,46,48,50
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%A_%a.out
#
# PHASE D (per-seed, 2 GPUs). SASRec++ baseline (dense, no COSETTE) -> test.
# Independent of Phases B/C; only needs A2 parquet (no embeddings).
#   sbatch jobs/arts2023/D_sasrec.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"
SEED="${SLURM_ARRAY_TASK_ID}"
echo "[D] SASRec++ seed=${SEED} | GPUs visible: $(nvidia-smi -L | wc -l)"

# REPRODUCTION OVERRIDES (reference recipe; committed defaults left intact):
#   num_workers=2 (effective batch 512), max_steps=81000, limit_val_batches=1000
# BUGFIX: configs/experiment/sasrec.yaml hardcodes vocab_size=89959, but item token
#   ids run 2..89959 (= num_items 89958 + len(SpecialTokens)=2 offset), so the last
#   item (token 89959) overflowed the 89959-row embedding -> CUDA indexSelectLargeIndex
#   device-side assert ("srcIndex < srcSelectDimSize"). Correct size = N + #special =
#   89958 + 2 = 89960 (same convention as MARIUS vocab_size=1026 = 4*256 + 2).
python src/train.py experiment=sasrec \
  seed="${SEED}" \
  data.ray_datasets.category="${CATEGORY}" \
  model.net.vocab_size=89960 \
  ray.scaling_config.num_workers=2 \
  trainer.max_steps=81000 \
  trainer.limit_val_batches=1000 \
  ${PATHS_OVERRIDES}

RUN_DIR="$(ls -t -d "${DATA_ROOT}"/models/SASRec_"${CATEGORY}"_seed"${SEED}"_* | head -1 | xargs -n1 basename)"
echo "${RUN_DIR}" > "${PROJECT_ROOT}/runs/seed_${SEED}.sasrec"
# enforce_filtering=false is a no-op here: SASRec.filter_preds already defaults to True
# (it is not set in configs/experiment/sasrec.yaml), so eval filters already-seen items
# either way -- matching the paper's SASRec++ and MARIUS's filtered eval. The flag only
# skips force-setting an already-True value (it does NOT disable filtering).
python src/test.py run_directory="${RUN_DIR}" enforce_filtering=false ${PATHS_OVERRIDES}
echo "D_SASREC_DONE seed=${SEED} run_dir=${RUN_DIR}"
