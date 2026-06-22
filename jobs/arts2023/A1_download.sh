#!/usr/bin/env bash
# PHASE A1 (run ONCE, on a LOGIN NODE - compute nodes have no internet).
# Seed-independent. Downloads:
#   (1) the Arts_Crafts_and_Sewing Amazon-2023 files -> DATA_ROOT/amazon-2023 (scratch)
#   (2) the Sentence-T5-XL model -> HF_HOME (home, persistent) for OFFLINE compute use
# Reference command (verbatim): scripts/download_data.py --categories Arts_Crafts_and_Sewing
# Usage:  bash jobs/arts2023/A1_download.sh
set -eo pipefail
source /gpfs/home2/scur1250/rec_sys_cosette_marius/jobs/arts2023/env_arts.sh
cd "${PROJECT_ROOT}"

# We ARE online on the login node, so disable the offline guards for this script only.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

echo "[A1] (1/2) Amazon-2023 ${CATEGORY} -> ${DATA_ROOT}/amazon-2023"
python scripts/download_data.py --year 2023 --categories "${CATEGORY}" --base_dir "${DATA_ROOT}"

echo "[A1] (2/2) Pre-fetch Sentence-T5-XL -> ${HF_HOME}"
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("sentence-transformers/sentence-t5-xl")
print("T5-XL cached OK | embedding dim:", m.get_sentence_embedding_dimension())
PY
echo "A1_DOWNLOAD_DONE"
