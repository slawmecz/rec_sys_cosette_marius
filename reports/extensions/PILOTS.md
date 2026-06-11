# Extension pilots: status and index

Four candidate extensions piloted on top of the validated Top-K dumps
(reports/extensions/topk/, 3 seeds, both datasets). No authors' model/eval
code modified anywhere; FUSE is a subclass in a new file. All locally-run
pilots reuse the validated loaders in scripts/extensions/compute_beyond_accuracy.py.

| Pilot | Code | Results | Status |
|---|---|---|---|
| REACH (collapse mechanism) | scripts/extensions/reach_analysis.py | reports/extensions/reach/ (+ reach_cb2/) | DONE; confirmed on a 2nd Sports codebook |
| Conformal sets | scripts/extensions/conformal.py | reports/extensions/conformal/ (+ conformal_d100/) | DONE at depth 20 and depth 100 |
| MBR re-ranking | scripts/extensions/mbr.py (+ scoring.py, dump_topk.py --n-results/--with-scores) | reports/extensions/mbr/ | DONE: uniform + scored tau sweep at depth 100 |
| FUSE (un-sum fusion) | src/models/marius_fuse.py (+ scripts/extensions/fuse_selftest.py) | reports/extensions/fuse/ | DONE: 3-seed Beauty runs for level_gain and attn |

## Final pilot outcomes (all Snellius runs complete)

1. REACH: the collapse is STRUCTURAL, not a codebook artifact. A second
   freshly-trained Sports codebook (67e0) reproduces it within 1-2 points:
   coverage 0.375 (vs 0.392), never-emitted L1:L2 pairs 27.8% (vs 25.6%),
   invisible demand mass 23.3% (vs 22.2%), top predictor own-share-within-L1
   r = -0.621 (identical), 0 L1 codes pruned.
2. MBR (scored, depth-100 candidates, k=10): NOT an accuracy fix, but a clean
   tunable accuracy-vs-reach dial. Sports: beam R@10 0.0471 -> tau0.5 0.0412
   with coverage 0.39 -> 0.52, ARP 101 -> 73, tail-recall +32% (0.0112 ->
   0.0148; improves at EVERY tau). Beauty: 0.0825 -> 0.0719 with coverage
   0.71 -> 0.81. Confidence weighting recovers most of what uniform MBR
   loses (uniform: 0.0266 / 0.0446). topm10 reproduces the beam exactly
   (sanity).
3. Conformal at depth 100: ceilings roughly double (Beauty marius 0.258 /
   sasrec 0.245; Sports 0.174 / 0.164). NOTE the crossover: at depth 20
   SASRec++ has the higher ceiling, at depth 100 MARIUS does, on BOTH
   datasets. MARIUS's deeper candidate pool is better than the baseline's;
   its likelihood ORDERING of the top of the list is what wastes it.
4. FUSE: the sum-fusion is NOT the bottleneck. 3-seed Beauty R@10:
   level_gain 8.29 +/- 0.09, attn 8.23 +/- 0.15 vs baseline 8.26 +/- 0.20
   (all within noise). level_gain mildly improves every beyond-accuracy
   metric at zero accuracy cost (coverage 0.714 vs 0.707, tail-recall 0.0387
   vs 0.0366); attn is neutral-to-slightly-worse. A clean, honest negative
   for the architectural hypothesis that strengthens the decoding-side story.

Combined verdict: the evidence points at ONE coherent paper, "catalog
collapse in semantic-ID generative recommenders": diagnosis (step-2 sibling
competition, robust across codebooks), stakes (22-23% of demand invisible),
locus (decoding, not architecture: FUSE null + the depth-100 crossover), and
decode-time levers (MBR Pareto, conformal reliability). The open method gap
is a mechanism-TARGETED decoder (pair-level exploration at depth step 2)
that should dominate the generic MBR Pareto.

## Key pilot results so far

REACH (the headline): MARIUS's Sports collapse is a decode-step-2 phenomenon.
The beam uses all 256 L1 codes, but never emits L1:L2 pairs covering 25.6% of
the Sports catalog (6.6% Beauty), which by construction accounts for 50.7%
(38.2%) of all consistently-unreachable items. The strongest single predictor
of unreachability is an item's popularity share WITHIN its L1 prefix
(r = -0.62 Sports, -0.48 Beauty), beating the item's own popularity
(-0.58 / -0.38): items lose to their prefix siblings (sibling competition),
and whole rare pairs are pruned at depth step 2. 22.2% of all Sports train
demand mass is invisible to MARIUS top-10 (Beauty 5.8%; SASRec++ 5.7% / 3.6%).
6968 Sports items are missed ONLY by MARIUS (not by SASRec++).

Conformal (depth-20): guaranteeable coverage is capped at Recall@20 (~0.12
Beauty / ~0.073 Sports), so the pitch is low-coverage guarantees + set size as
an uncertainty signal, not 90% guarantees. On Beauty the Mondrian signal is
strong and monotone: cold users (hist <= 5) need k* ~10-12 vs warm users
(hist >= 16) k* = 3 at c = 0.08, a 3.4x set-size ratio. On Sports the signal
is flat-to-inverted (warm-bucket Recall@20 is LOWEST: honest negative,
consistent with the collapse). At matched coverage SASRec++ needs sets no
larger than MARIUS everywhere.

MBR (uniform weights = the no-confidence lower bound): accuracy drops as
expected (Beauty R@10 8.26 -> 7.19), but ALL diversity metrics improve on both
datasets, and on Sports tail-recall improves on EVERY seed (+21% relative,
0.0112 -> 0.0136) with coverage 0.39 -> 0.48 and ARP 101 -> 79. A pure
re-ranking of the same 20 candidates partially counteracts the collapse.
Score-weighted MBR (the real accuracy test) needs the scored dumps.

FUSE: implementation verified by a 33-check torch-free selftest. One genuine
spec finding: a per-level additive bias is mathematically DEGENERATE under sum
pooling (biases sum out to a constant; function class equals the baseline), so
the arms are: sum (baseline), level_bias (optimization-null control),
level_gain (cheapest non-degenerate level reweighting, ones-init), attn
(content-adaptive learned-query pooling, scale-matched at init). All arms
equal the baseline exactly at initialization. CRITICAL operational note: every
FUSE variant must train into a FRESH OUTPUT_ROOT or the 5-seed harness will
silently record the baseline run instead of training (see the module
docstring of src/models/marius_fuse.py).

## How to re-run

Each script has --selftest (or is a selftest); see headers for CLI usage.
Local: reach_analysis.py, conformal.py, mbr.py run on the committed dumps
with no GPU and no torch. Snellius: scored 100-deep dumps via
dump_topk.py --n-results 100 --with-scores (commands in the mbr.py header
and the session notes); FUSE runs via EXTRA_TRAIN_OVERRIDES (commands in
src/models/marius_fuse.py docstring).
