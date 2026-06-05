# Beauty dataset overrides (sources shared env_common.sh).

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_common.sh"

export CATEGORY="Beauty"
export CATEGORY_SLUG="beauty"
export SASREC_VOCAB_SIZE="${BEAUTY_VOCAB_SIZE}"
