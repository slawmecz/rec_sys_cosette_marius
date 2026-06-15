#!/usr/bin/env bash
#SBATCH --job-name=arts_cosette
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --array=42,44,46,48,50
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%A_%a.out
#
# PHASE B (per-seed, 1 GPU each). Trains one COSETTE tokenizer per seed so each of the
# paper's 5 runs has its own semantic IDs (the full 5-seed pipeline; captures the
# COSETTE-draw variance). Requires A3 embeddings. COSETTE multi-GPU is not viable
# (full-batch SigLIP), so 1 GPU per seed.
#   sbatch jobs/arts2023/B_cosette.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"
SEED="${SLURM_ARRAY_TASK_ID}"
MARKER="seed${SEED}"
echo "[B] COSETTE seed=${SEED} on $(hostname)"

# Reference Arts recipe + intent: bs=1024 (-> ~192 steps/epoch on Arts; CONFIRM in the
# log), 3000 epochs (~576k steps). The literal reference command omits batch_size, which
# would default to 256 and run ~4x the steps. 'marker' makes the generated quant id
# unique per seed (race-safe across the array); 'seed' drives the seed plumbing.
python data_scripts/2_train_cosette.py \
  data.category="${CATEGORY}" \
  optim.batch_size=1024 \
  optim.epochs=3000 \
  seed="${SEED}" \
  marker="${MARKER}" \
  ckpt_dir="${DATA_ROOT}/cosette_ckpt/${MARKER}" \
  ${PATHS_OVERRIDES}

# Record this seed's generated COSETTE quant id (e.g. COSETTE_128d_256x4_<hex>_seed42).
EMB_DIR="${DATA_ROOT}/data/embeddings/${EMB_METHOD}/${CATEGORY}"
QUANT="$(ls -t "${EMB_DIR}"/COSETTE_128d_256x4_*_"${MARKER}".parquet | head -1 | xargs -n1 basename | sed 's/\.parquet$//')"
echo "${QUANT}" > "${PROJECT_ROOT}/runs/seed_${SEED}.quant"
echo "B_COSETTE_DONE seed=${SEED} quant=${QUANT}"
