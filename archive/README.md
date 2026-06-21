# archive/

Superseded and dead-end extension pilots, moved here so the main tree shows only
the final RQ1-4 pipeline. Nothing here is part of the final findings. For the
canonical results see `EXTENSION_RESULTS.md`, `notebooks/extension_story.ipynb`,
and `reports/figures/fig_rq{1,2,3,4}_*`.

These are kept (not deleted) because several are honest negative results that the
writeup may cite as "we ruled this out." Full pre-cleanup history is also on the
`pre-cleanup-backup` branch. Note: file-path strings inside these archived scripts
may point at their old (pre-move) locations; they are frozen as-was and not run.

## What is here and why it was retired

- `code/marius_fuse.py`, `code/fuse_selftest.py`, `reports/fuse/` -- FUSE pilot:
  replacing MARIUS's sum-fusion with level-gain / attention. NULL result (accuracy
  unchanged); exonerated the fusion as the cause. Not used by any committed job.
- `code/conformal.py`, `reports/conformal/`, `reports/conformal_d100/` -- split
  conformal prediction sets. Part of the earlier "depth-100 crossover / decoding"
  framing that the exact oracle (RQ3) superseded.
- `code/selective_prediction.py`, `reports/selective_prediction/` -- abstention via
  MARIUS joint log-prob. A standalone capability, not part of the RQ1-4 story; the
  cross-paradigm head-to-head was a negative (SASRec selects at least as well).
- `code/reach_analysis.py`, `reports/reach/`, `reports/reach_cb2/` -- the earlier
  REACH mechanism pilot (collapse "at decode step 2"). Superseded by the validated
  `scripts/extensions/reach_estimator.py` (Chao1/Good-Turing) and by the RQ3 oracle
  verdict that the collapse is model-bound, not a decode-step artifact. `reach_cb2`
  is the second-codebook robustness check (collapse reproduces on a 2nd tokenizer).
- `code/direction_gate.py`, `jobs/25_direction_gate.sbatch`, `reports/direction_gate/`
  -- the order-aware-tokenizer "directional gate" probe. SHELVED clean null
  (directed grouping predicted the next item worse than COSETTE's codes).
- `jobs/28_rescore_for_selective_*.sbatch` -- launchers for the selective-prediction
  depth-100 rescore; orphaned once selective_prediction was retired.
