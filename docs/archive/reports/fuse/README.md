# FUSE (un-sum fusion) full-run results: Beauty

MARIUSFuse replaces MARIUS's permutation-invariant, content-blind SUM over the
4 RQ code embeddings in `temporal_forward` with a level-aware / content-adaptive
pool. Arms run to the full paper schedule (80k steps, 3 seeds 42/43/44, COSETTE
codebook `COSETTE_128d_256x4_a434-col`), each into a FRESH OUTPUT_ROOT
(`outputs_fuse_<arm>`) so the 5-seed harness trains rather than recording the
baseline (the documented OUTPUT_ROOT trap). `level_bias` is the proven-degenerate
control (biases sum out under the sum pool) and was smoked only, not run full.

## Smoke acceptance (all three arms, clean run)
- each arm issued a real train command (no "recording only" / "already recorded")
- model instantiated as `src.models.marius_fuse.MARIUSFuse`
- weight-decay split correct: `No decay : 2` (only the two embedding tables)

## Accuracy (test R@10, 3 seeds) vs the sum baseline 8.26 +/- 0.20

| arm        | R@5         | NDCG@5      | R@10         | NDCG@10     |
|------------|-------------|-------------|--------------|-------------|
| baseline (sum) | -       | -           | 8.26 +/- 0.20 | -          |
| level_gain | 5.38 +/-0.08 | 3.56 +/-0.03 | 8.29 +/-0.09 | 4.50 +/-0.04 |
| attn       | 5.35 +/-0.07 | 3.48 +/-0.06 | 8.23 +/-0.15 | 4.41 +/-0.06 |

Un-summing does not move accuracy on Beauty: both arms equal the sum baseline
within seed noise.

## Beyond-accuracy (k=10, seed means; full CSVs alongside)

| metric              | baseline (sum) | level_gain | attn   |
|---------------------|---------------:|-----------:|-------:|
| recall              | 0.0826         | 0.0829     | 0.0824 |
| coverage  (higher=) | 0.7073         | 0.7138     | 0.6929 |
| gini      (lower=)  | 0.7943         | 0.7902     | 0.8002 |
| ARP       (lower=)  | 55.90          | 55.38      | 57.09  |
| APLT      (higher=) | 0.2793         | 0.2841     | 0.2758 |
| tail_recall(higher=)| 0.0366         | 0.0387     | 0.0378 |
| entropy_norm        | 0.8472         | 0.8495     | 0.8433 |
| hallucination_rate  | 0.0023         | 0.0023     | 0.0023 |

`level_gain` moves every diversity/reach metric in the favorable direction
(coverage, gini, ARP, APLT, tail-recall, entropy) at no accuracy cost; `attn`
is neutral-to-slightly-more-popularity-peaked. All deltas are small and within
seed std, as expected: Beauty is not the collapse case (Sports is; see
reports/extensions/reach_cb2/). Hallucination is unchanged.

## Files
- `<arm>/marius_beauty_5seed_full_scores.jsonl` - per-seed test metrics
- `<arm>/table5_beauty_marius_5seed_summary_full_latest.txt` - paper-vs-run table
- `beyond_accuracy_Beauty_<arm>.csv` - full beyond-accuracy suite (k=10, 20)
- Top-K dumps (20-deep) live on scratch: `cosette_marius/fuse_dumps/<arm>/Beauty/` (npz not committed)
