#!/usr/bin/env bash
#SBATCH --job-name=arts_verify_gpu
#SBATCH --account=gpuuva080
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%j.out
#
# GPU-stack verification (Step 3). Needs NO dataset and NO model download, so it is
# safe to run before approving the pipeline.
#   sbatch jobs/arts2023/00_verify_gpu.sbatch
# Check the 40 GB A100 path instead:
#   sbatch --partition=gpu_a100 --cpus-per-task=18 jobs/arts2023/00_verify_gpu.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"

echo "=== node $(hostname) | partition ${SLURM_JOB_PARTITION} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python - <<'PY'
import torch, ray
import pytorch_lightning as L
print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu:", p.name, "| vram_GiB:", round(p.total_memory / 2**30, 1),
          "| bf16:", torch.cuda.is_bf16_supported())
    a = torch.randn(2048, 2048, device="cuda")
    print("cuda_matmul_finite:", bool((a @ a).sum().isfinite().item()))
print("ray:", ray.__version__, "| lightning:", L.__version__)
PY
echo "VERIFY_OK"
