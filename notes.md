## Our reproduction

- we re-ran the mandatory small-dataset scope on Amazon 2014 Beauty and Sports

- for both the baseline (SASRec++) and MARIUS+COSETTE, 5 random seeds each

- running SASRec++ with the repository default was misleading. The repository
  default is d=128 + L2 normalization, but the paper's Beauty setting is d=32
  with no normalization and temperature 1.0. Switching to the paper-faithful
  SASRec++ config improved Beauty R@10 from 8.24 to 9.06, closing about 55% of
  the gap.

- Final small-dataset reproduction: faithful SASRec++ beats our MARIUS+COSETTE
  on Beauty, so the paper's "competitive on small data" claim did not reproduce
  under our setup.

- Current status: teammates are running larger-dataset reproductions. We should
  not claim yet whether large datasets fix or worsen the reachability issue;
  accuracy alone is not enough.


## Beyond R and NDCG

Beauty: MARIUS is quite diverse, pushes less popular items more than SASRec++;

Sports: It only recommends about 39% of the catalog at top 10 (SASRec++ does
about 74%), diversity collapses

Current mechanism: not a simple "Sports is sparser" story. The sharp failure is
at the second code digit. Weak L1:L2 prefix families are pruned by the beam, so
items lose to stronger siblings that share their first code token.


# What looks weird/wrong in the architecture of these models

## COSETTE

- The codes are trained separately and then frozen, instead of training for recommendation directly (there are some papers alraedy)

- popularity bias from collaborative signal (co-ccurence is biased towards popular items) (there are some papers alraedy)

- Might be worth looking at collision handling (there are some papers alraedy)

## MARIUS

- Adding the 4 code-digits together by just adding their vectors (it was inspired by the audio model but still wtf?)

- ranking the guesses only by probability which might reinforce the bias

- catalog reachability collapse / sibling competition during decoding


# Cool things we can do

- Diagnose and attempt to fix the collapse: exact catalog scoring should tell
  us whether unreachable items are search errors or model-ranking errors.

- Experiment with other prediction ranking: MBR gives a useful accuracy-vs-
  reach trade-off but is not an accuracy fix. A more targeted decoder should
  focus on depth-2 prefix-family exploration.

- We tried replacing the sum over code digits with level_gain and attention.
  It did not move accuracy on Beauty, so sum-fusion is probably not the main
  bottleneck.


# 2026-06-17 session resolution (see EXTENSION_SUMMARY.md for the full log)

The exact-catalog-scoring question above is now ANSWERED: the collapse is
MODEL-bound, not a search/beam error. We teacher-forced every catalog item through
MARIUS; the beam-missed targets sit at median exact rank ~1300-1400 (0% in the exact
top-20), and with history filtering exact top-K equals beam top-K. So the beam is
near-optimal over the model; the model itself buries the tail.

Consequences for the "cool things" above:

- The "more targeted depth-2 decoder" is KILLED: it is not a search error, so a
  branch-aware decoder cannot recover the items. Do not build it.
- We tested both popularity-correction levers. Decode-time PMI / logit-adjusted
  re-rank (beats MBR) and train-time logit-adjusted loss (MARIUSLogitAdj) are BOTH
  bounded dials: they cut ARP/Gini (train-time ARP Sports 101->55, Beauty 56->33) but
  trade accuracy and do NOT recover buried items. Popularity de-bias is decoupled
  from catalog reach (it even shrinks Beauty coverage). The generic item-popularity
  prior is as good as the prefix-aware one.
- The order-aware / asymmetric COSETTE idea was tested with a pre-registered gate
  and SHELVED (directional next-item signal is too sparse on small data to beat
  COSETTE's symmetric codes). No tokenizer retrain.

Net: the locus is the model's learned ranking (top-K exposure ceiling), not codes,
fusion, search, or re-weighting. The package is novel (exact-scoring oracle +
structural ceiling + two-lever bounded dial + decoupling) with framing fixes (cite
Ghost, Latte, Abdollahpouri; down-scope "expressiveness limit"). Remaining: Tier E
(Office Products, large-dataset scale test) then writeup.
