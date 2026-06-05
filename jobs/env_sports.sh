# Sports_and_Outdoors dataset overrides (sources shared env_common.sh).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_common.sh"

export CATEGORY="Sports_and_Outdoors"
export CATEGORY_SLUG="sports"
export SASREC_VOCAB_SIZE="${SPORTS_VOCAB_SIZE}"
