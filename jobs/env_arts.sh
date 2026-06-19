# Arts_Crafts_and_Sewing (Amazon Reviews 2023, ~90k items) dataset overrides.
# Sources shared env_common.sh.
#
# IMPORTANT: this evaluates Jan's (scur1250) FROZEN checkpoints + processed data
# READ-ONLY. We do not own those dirs, so env OUTPUT_ROOT stays on OUR writable
# scratch (env_common mkdir's logs/wandb under it). Jan's checkpoint dir is passed
# to each script explicitly via --output-root "${ARTS_MODELS_ROOT}". All instrument
# OUTPUTS are written into the repo tree (reports/extensions/topk_arts/...).
#
# Prereq (Jan ran): chmod -R a+rX /scratch-shared/scur1250/cosette_arts
# Verify before running (see .context/snellius-overnight-prompt.md STEP 0):
#   - ${ARTS_MODELS_ROOT}/models/<run_dir>/checkpoint_*/{config.yaml,checkpoint.ckpt} readable
#   - the baked config.yaml paths.root resolves to ${ARTS_MODELS_ROOT}
#   - ${ARTS_MODELS_ROOT}/data/timelines/Arts_Crafts_and_Sewing.{train,valid,test}.parquet,
#     Arts_Crafts_and_Sewing_items.pkl, and
#     data/embeddings/sentence-t5-xl/Arts_Crafts_and_Sewing/{embeddings.parquet,<seed>-col.parquet}

source "${SLURM_SUBMIT_DIR:-$(pwd)}/jobs/env_common.sh"

export CATEGORY="Arts_Crafts_and_Sewing"
export CATEGORY_SLUG="arts"

# Jan's read-only artifact root (override with --export=ALL,ARTS_MODELS_ROOT=... if moved).
export ARTS_MODELS_ROOT="${ARTS_MODELS_ROOT:-/scratch-shared/scur1250/cosette_arts}"

# The *_arts_5seed_full_scores.jsonl join keys are committed in the repo tree.
export ARTS_RESULTS_DIR="${ARTS_RESULTS_DIR:-${PROJECT_ROOT}/reports/results}"

# Arts repro seeds (NOTE: 42,44,46,48,50 -- not 43/45).
export ARTS_SEEDS="${ARTS_SEEDS:-42 44 46 48 50}"

# SASRec vocab is baked in the trained checkpoint (89960 = 89958 items + 2 special);
# instruments read it from the checkpoint config, so this is informational only.
export ARTS_VOCAB_SIZE="${ARTS_VOCAB_SIZE:-89960}"

# Per-seed COSETTE -col quant id. CRITICAL: each Arts seed has its OWN tokenizer
# (the repro used marker=seed$SEED), so the semantic-ID space differs per seed.
# This is why every Arts instrument runs in a PER-SEED dump dir.
arts_quant_for_seed() {
  case "$1" in
    42) echo "COSETTE_128d_256x4_2da9_seed42-col" ;;
    44) echo "COSETTE_128d_256x4_5693_seed44-col" ;;
    46) echo "COSETTE_128d_256x4_ee89_seed46-col" ;;
    48) echo "COSETTE_128d_256x4_c00a_seed48-col" ;;
    50) echo "COSETTE_128d_256x4_0a09_seed50-col" ;;
    *)  echo "UNKNOWN_ARTS_SEED_$1" ;;
  esac
}
