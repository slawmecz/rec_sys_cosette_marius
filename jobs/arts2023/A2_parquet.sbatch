#!/usr/bin/env bash
#SBATCH --job-name=arts_parquet
#SBATCH --account=gpuuva080
#SBATCH --partition=genoa
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/logs/%x_%j.out
#
# PHASE A2 (run ONCE, CPU). Converts the raw Amazon-2023 files to parquet timelines +
# metadata. Requires A1 download present under DATA_ROOT/amazon-2023.
# Reference command (verbatim): 0_raw_to_parquet.py paths.skip_download=true
#   sbatch jobs/arts2023/A2_parquet.sbatch
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"

# 0_raw_to_parquet.yaml already has categories=[Arts_Crafts_and_Sewing].
python data_scripts/0_raw_to_parquet.py paths.skip_download=true ${PATHS_OVERRIDES}

echo "A2_PARQUET_DONE"
ls -la "${DATA_ROOT}/data/timelines/" || true
