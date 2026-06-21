# Sports collapse: second-codebook confirmation (cb2)

Purpose: confirm MARIUS's Sports catalog collapse is a property of the
model/decoding, NOT an artifact of the specific COSETTE codebook used for the
validated runs (`COSETTE_128d_256x4_24cd-col`). We retrained COSETTE on Sports
from scratch (job 05, fresh random id), deduped (job 06), and trained MARIUS for
3 seeds (42/43/44) on the NEW codebook `COSETTE_128d_256x4_67e0-col` into a
fresh OUTPUT_ROOT (`outputs_sports_cb2`), then re-ran reach_analysis and
compute_beyond_accuracy on its 20-deep Top-K dumps. SASRec++ is codebook-
independent (dense) and is the faithful d32 run, reused as the reference.

## Accuracy (test R@10, 3 seeds)
cb2 MARIUS R@10 = 4.97 / 5.14 / 4.94 (mean 5.01). Same low-accuracy regime as
the 24cd Sports runs - Sports is the hard/collapsed category for MARIUS.

## Collapse signatures: 67e0 (cb2) vs 24cd (original)

| signal                                        | 24cd (original) | 67e0 (cb2) |
|-----------------------------------------------|----------------:|-----------:|
| MARIUS coverage@10                            | ~0.39           | 0.375      |
| consistently-unreached share of catalog       | ~50%            | 52.2%      |
| L1:L2 pairs NEVER emitted (catalog share)     | 25.6%           | 27.8%      |
|   ... share of the unreached set they explain | 50.7%           | 53.3%      |
| train demand mass invisible to top-10         | 22.2%           | 23.3%      |
| L1 codes never emitted (step-1 pruning)       | 0               | 0          |
| own-pop-share-within-L1 corr (top predictor)  | r = -0.62       | r = -0.621 |
| ARP@10                                         | ~101            | 102.3      |

Every collapse signature reproduces within 1-2 points on an independently
trained codebook. Conclusion: the Sports collapse is NOT a 24cd artifact - it is
a decode-step-2 phenomenon (whole L1:L2 pairs pruned; items lose to popular
prefix siblings), invariant to the codebook draw.

## Files
- `reach_Sports_and_Outdoors.md` / `reach_results_Sports_and_Outdoors.json` - full reach tables
- `reach_summary.md`, `reach_unreached_by_decile.png`
- `beyond_accuracy_Sports_and_Outdoors_cb2.csv` - beyond-accuracy suite (marius + sasrec, k=10/20)
- 20-deep Top-K dumps (npz) on scratch: `cosette_marius/topk_cb2/Sports_and_Outdoors/` (not committed)
- New codebook: `COSETTE_128d_256x4_67e0-col`; checkpoints in `outputs_sports_cb2` (scratch)
