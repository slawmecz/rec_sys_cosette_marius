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

Combined verdict (SUPERSEDED 2026-06-17 -- see below and EXTENSION_SUMMARY.md):
these four pilots established the diagnosis (step-2 sibling competition, robust
across codebooks), the stakes (22-23% of demand invisible), and decode-time levers
(MBR Pareto, conformal reliability). They pointed at decoding as the locus and
proposed a mechanism-targeted step-2 decoder. THE FOLLOW-UP EXACT-SCORING ORACLE
REFUTED THE "DECODING LOCUS" CONCLUSION: the collapse is MODEL-bound (the beam is
near-optimal over the model; the model itself buries the tail), so the targeted
decoder was not built. Trust EXTENSION_SUMMARY.md where it differs from this file.

## Session 2 pilots (2026-06-17) -- the decisive resolution

| Pilot | Code | Result | Status |
|---|---|---|---|
| Reachability estimator (Chao1/Good-Turing) | reach_estimator.py | Sports asymptotic coverage@10 ~46% (observed 39%), unseen-mass 0.34% -> structural ceiling | DONE (zero-GPU) |
| EXACT full-catalog scoring oracle | exact_catalog.py (+ jobs/23) | MODEL-bound: beam-missed targets at median exact rank ~1300-1400, 0% in exact top-20; filtered exact == beam | DONE (the centerpiece) |
| PMI / logit-adjusted decode re-rank | pmi_rerank.py (+ jobs/24) | bounded dial, beats MBR; item-popularity prior ~= prefix-aware cond2 | DONE |
| Directional-structure gate (order-aware COSETTE) | direction_gate.py (+ jobs/25) | SHELVE: directed grouping predicts next item WORSE than COSETTE's codes | DONE -> idea dropped |
| Train-time logit-adjustment | src/models/marius_logitadj.py (+ jobs/26) | DIAL not FIX: crushes ARP/Gini, accuracy drops, debias decoupled from reach | DONE (2 seeds) |
| Local robustness checks | inline (exact top-100 coverage, missed-prefix spread, occupancy) | top-K exposure ceiling (not absolute); misses spread 240/256 L1 codes; occupancy ~4e-6 | DONE (zero-GPU) |

Session-2 combined verdict: the locus is the model's learned ranking, not the
search/fusion/tokenizer-symmetry/codes/re-weighting. Popularity correction at both
decode and train time is a bounded dial decoupled from catalog reach. Novel as a
package (exact-scoring oracle + structural ceiling + two-lever bounded dial +
decoupling + density inversion); cite/contrast Ghost (2605.16825) and Latte
(2605.06331), and Abdollahpouri for the decoupling. Remaining: Tier E large-dataset
scale test, then writeup. Full log in EXTENSION_SUMMARY.md.

## Detailed pilot notes

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

MBR: the original no-confidence uniform run is a diversity-heavy lower bound.
The scored depth-100 sweep is now the main result. MBR is not an accuracy fix,
but it gives a tunable Pareto curve. On Sports, tau=0.5 changes R@10
0.0471 -> 0.0412 while improving coverage 0.392 -> 0.519, ARP 101 -> 73,
and tail recall 0.0112 -> 0.0148. Uniform MBR pushes coverage still higher
but pays a much larger accuracy cost. The topm10 sanity check reproduces the
beam exactly.

FUSE: implementation verified by a 33-check torch-free selftest. A per-level
additive bias is mathematically degenerate under sum pooling, so the full runs
focus on level_gain and attention. Both train correctly from fresh output roots
and both remain within baseline seed noise on Beauty accuracy. This is a useful
negative result: the inherited sum-fusion is not the bottleneck we should spend
the extension on.

## How to re-run

Each script has --selftest (or is a selftest); see headers for CLI usage.
Local: reach_analysis.py, conformal.py, mbr.py run on the committed dumps
with no GPU and no torch. Snellius: scored 100-deep dumps via
dump_topk.py --n-results 100 --with-scores (commands in the mbr.py header
and the session notes); FUSE runs via EXTRA_TRAIN_OVERRIDES (commands in
src/models/marius_fuse.py docstring).
