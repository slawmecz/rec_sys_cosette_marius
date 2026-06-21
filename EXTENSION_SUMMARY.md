# Extension investigation summary (dated log; as of 2026-06-17)

> CANONICAL RESULTS: see `EXTENSION_RESULTS.md` (verified RQ1-4 numbers) and
> `notebooks/extension_story.ipynb` + `reports/figures/fig_rq*`. This file is the
> dated "what we tried and why" investigation log, kept for the writeup's
> related-work / negative-results; where it differs from `EXTENSION_RESULTS.md`,
> trust `EXTENSION_RESULTS.md`.
>
> RESOLVED since this log was written: (1) the large Amazon-2023 prong is DONE
> (Arts_Crafts_and_Sewing, 90k items, 5 seeds): the collapse persists and the
> exact oracle stays MODEL-bound at scale; RQ1 shows MARIUS > SASRec there
> (p=0.008). (2) The cross-paradigm distillation arm is DONE: a strong bounded
> dial (coverage +123%, recall -22%) that buries true targets DEEPER, confirming
> model-bound from a third angle. Anything below marked PENDING/queued is resolved.
>
> ARCHIVED: the superseded/dead-end pilots referenced below (conformal,
> selective_prediction, FUSE/marius_fuse, direction_gate, reach_analysis/reach,
> reach_cb2) and their jobs were moved to `archive/` during the 2026-06-21 cleanup;
> paths in this log point at their pre-archive locations. See `archive/README.md`.

Comprehensive log of the extension half of the COSETTE+MARIUS project: every idea
we considered, whether we tested it, the result, our hypotheses, and where we stand.
This supersedes the older "decoding is the locus" framing in
reports/extensions/PILOTS.md.

--------------------------------------------------------------------------------
## 0. Bottom line (where we stand)

The catalog-reachability collapse in MARIUS is a MODEL-RANKING problem, not a
decoding, fusion, tokenizer-symmetry, or popularity-reweighting problem. We proved
this with an exact full-catalog teacher-forced scoring oracle: the beam is a
near-optimal search over the model, and the model itself ranks the missed items
deep (median exact rank ~1300-1400, 0% in the exact top-20). Every intervention we
tried at the decode stage and the train stage cuts popularity bias (ARP/Gini) but
is a bounded dial: it cannot recover the buried items as hits, and it is decoupled
from catalog coverage (it can even shrink coverage).

The diagnostic is essentially complete and, as a package, novel and not scooped
(adversarially verified). Three framing fixes are required for publication (cite
and contrast Ghost and Latte; down-scope "expressiveness limit"; claim the narrow
version of the debias/coverage decoupling). The one remaining experiment that needs
new GPU is the large-dataset scale test (Tier E, Office Products); everything else
is reviewer-hardened with artifacts we already have. The next-compute decision is
pending with Stanislaw.

--------------------------------------------------------------------------------
## 1. The question and the setup

MARIUS generates a recommendation by autoregressively decoding a 4-digit semantic
ID (256 codes per level, RQ-VAE codebooks via COSETTE) with beam search (width 20),
ranking the Top-K by joint log-probability. We observed a catalog-reachability
collapse:

- Sports: MARIUS reaches ~39.2% of the catalog at top-10 vs SASRec++ ~73.7%.
- ~22.2% of Sports training demand is never recommended by MARIUS (vs ~5.7% SASRec).
- Beauty is milder: MARIUS 70.7% vs SASRec 81.3% coverage; 5.8% demand invisible.

The extension question: WHERE does the collapse originate (codes / popularity in the
data / fusion / decoding), and can we fix it? Candidate loci, each testable:
the COSETTE tokenizer, the sum-fusion of the 4 code embeddings, the beam search /
decoding, and the model's learned probabilities.

Reproduction context (from REPLICATION_REPORT.md): our MARIUS underperforms the
paper (Beauty R@10 ~8.2 vs paper 10.02; faithful SASRec++ 9.06 vs paper 9.73), so
the paper's "COSETTE >= SASRec++ on small data" claim did not reproduce for us. This
matters because a reviewer can argue we are diagnosing an underperforming model (see
section 8, objection #2).

--------------------------------------------------------------------------------
## 2. What we built and ran this session (experiments)

All code is ours under scripts/extensions/ and src/models/*_*.py subclasses; the
authors' model/loss/eval code is byte-identical and untouched.

### 2.1 Reachability estimator (Good-Turing / Chao1 + rarefaction) -- DONE
- File: scripts/extensions/reach_estimator.py (zero-GPU, committed dumps).
- Question: is the unreached catalog a sampling artifact or a structural ceiling?
- Result: STRUCTURAL. Treating users as the sampling unit, the Sports MARIUS
  decoder's asymptotic (Chao1) coverage@10 is ~45.6% (observed 39.2%), Good-Turing
  unseen-mass 0.34% (the reached set has saturated). SASRec Sports chao1 0.829.
  Beauty MARIUS chao1 0.777 (observed 0.707). Upgrades the claim from "reaches 39%
  in our sample" to "structurally bounded," and is the validated instrument that
  resists SimGR's single coverage number.
- Refinement (see 2.6): the ceiling is a top-K exposure ceiling, not absolute.

### 2.2 Exact full-catalog teacher-forced scoring oracle -- DONE (the centerpiece)
- Files: scripts/extensions/exact_catalog.py (run + analyze; --filter-seen),
  jobs/23_exact_catalog_{beauty,sports}.sbatch. Read-only; reuses scoring.py.
- Method: for ~1000 sampled test users, teacher-force EVERY catalog item's code
  tuple through MARIUS to get the exact joint log-prob over the whole catalog, rank,
  and compare exact Top-K to the model's own beam on the same users. This decomposes
  beam misses into SEARCH error (beam drops items the model ranks high) vs MODEL
  error (the model ranks them low).
- Result: MODEL-bound on both datasets. Beam-missed targets sit at median EXACT rank
  ~1328 (Sports) / ~1393 (Beauty), with 0% in the exact top-20 and only ~12-14% in
  the exact top-100. With history-filtering (apples-to-apples), exact Top-K == beam
  Top-K (Sports R@10 0.049 == 0.049, coverage identical; Beauty exact 0.081 vs beam
  0.078, coverage +0.12pp). So the beam is a near-optimal search over the model; the
  model itself buries the tail.
- Novelty: no semantic-ID recsys paper runs this exact-vs-beam full-catalog
  decomposition (verified). It exists only as theory in doc IR (2504.09935) and
  qualitatively in APAO.

### 2.3 PMI / logit-adjusted decode-time re-rank ("label-bias decoder") -- DONE
- Files: scripts/extensions/pmi_rerank.py, jobs/24_pmi_{beauty,sports}.sbatch.
- Method: re-rank the depth-100 scored beam candidates by
  adjusted = joint_logp - alpha * log p_hat(prior), with three priors: cond2
  (within-prefix conditional p_hat(digit1|digit0), mechanism-matched), pair (joint
  prefix), item (global popularity). alpha sweep 0..1.5.
- Result: a bounded DIAL, but it beats the MBR baseline. e.g. Sports cond2 at
  alpha=0.75: R@10 0.0387, coverage 0.6456, tail 0.0152 (vs MBR tau1.0 0.0396 /
  0.5326 / 0.0149 -> cond2 gives more coverage AND tail at matched recall). IMPORTANT
  honesty point: the GENERIC item-popularity prior is as good or BETTER than the
  mechanism-matched cond2 (Sports item alpha0.75: cov 0.6937, tail 0.0202; Beauty
  item alpha0.75: R 0.0756 cov 0.867 tail 0.054 dominates cond2). Recall drops
  monotonically with alpha for every prior. So the bias is global-popularity-shaped,
  not a prefix-decode phenomenon you can cleverly target at step 2.

### 2.4 Directional-structure gate (order-aware COSETTE test) -- DONE -> SHELVE
- Files: scripts/extensions/direction_gate.py, jobs/25_direction_gate.sbatch.
- Hypothesis (Stanislaw's idea): COSETTE's symmetric co-occurrence discards the
  next-item / transition structure; an order-aware (directional) tokenizer would help.
- Method: at G=256 (the L1-family granularity the decoder conditions on for digit 2),
  compare how much next-item entropy each grouping explains (normalized mutual
  information): COSETTE's actual L1 codes vs a directed PPMI+SVD+kmeans clustering vs
  a symmetrized control vs random. Metric A over all transitions, Metric B over
  buried/unreached next-items.
- Result: SHELVE on both. The directed grouping explains LESS next-item entropy than
  COSETTE's L1 codes (delta -0.034 Sports / -0.024 Beauty) and less than the
  symmetric control (ratio 0.95/0.99); on buried items even random edges out
  direction. Mechanism of the null: on small data, adjacency transitions are far
  sparser than full-timeline co-occurrence, so directional structure is too
  data-starved to beat COSETTE's symmetric signal. Verdict: an order-aware tokenizer
  retrain is NOT justified; this is a clean, pre-registered controlled null (it saved
  a multi-day retrain).

### 2.5 Train-time logit-adjustment arm (MARIUSLogitAdj) -- DONE -> DIAL (not FIX)
- Files: src/models/marius_logitadj.py (subclass; overrides only __init__ + get_loss,
  inference unchanged), scripts/extensions/compute_token_prior.py,
  reports/extensions/logitadj/prior_*.npy, scripts/extensions/logitadj_selftest.py,
  jobs/26_logitadj_{beauty,sports}.sbatch.
- Method: train with CE(logits + tau*log pi, target) where pi is the per-token
  popularity prior (Menon et al. 2021 logit adjustment); predict with raw logits.
  The model-side analogue of the decode-time PMI. tau=1.0, 2 seeds.
- Result: DIAL, not FIX, confirming MODEL-bound from a second angle. It crushes
  popularity bias on both (ARP Sports 101->55, Beauty 56->33; Gini down) but accuracy
  drops (Sports R@10 4.71->4.09, Beauty 8.26->6.77) and catalog reach barely moves
  (Sports coverage +1.5pp, chao1 +1.9pp) or SHRINKS (Beauty coverage -1.7pp, chao1
  -1.9pp, tail -0.0009). KEY FINDING: popularity de-biasing is DECOUPLED from catalog
  reach -- the most aggressive train-time popularity correction redistributes exposure
  among already-reachable items without lifting the reachable set.
- Two job bugs found and fixed first (see section 9): wrong quant id (8ed1/f958 ->
  this account's 24cd/a434), and OUTPUT_ROOT must be exported BEFORE sourcing env so
  PATHS_OVERRIDES/model_folder_tplt point at the fresh dir.

### 2.6 Local robustness checks (objection-preempting) -- DONE (zero-GPU)
Run on the committed exact-scoring dumps (which stored the exact top-100 per user):
- Exact coverage@10/20/50/100 = Sports 0.150/0.244/0.430/0.609, Beauty
  0.280/0.424/0.668/0.828. At matched K, exact == beam (MODEL-bound robust). But the
  model's deeper frontier reaches far more (top-100 ~61% Sports / 83% Beauty over
  1000 users). REFINEMENT: the buried items are NOT absolutely unrepresentable; they
  sit BELOW the top-K exposure cutoff for their target users (echoes the conformal
  depth-100 crossover). We down-scope "structural/expressiveness limit" to a "top-K
  exposure-ranking ceiling."
- Beam-missed targets are SPREAD across 240/256 L1 codes (top-5 share 0.06), so the
  tokenizer assignment is not a localized dead-zone defect.
- Code-tuple occupancy ~4e-6 of 256^4: collapse is not code exhaustion.

### 2.7 Adversarial verification (novelty / positioning / reviewer-2) -- DONE
4-agent workflow over the final thesis (see sections 7 and 8).

--------------------------------------------------------------------------------
## 3. Prior pilots (earlier in the project, for completeness)

These were done before this session (see reports/extensions/PILOTS.md, now partly
superseded). Their verdicts still hold as evidence, but the OVERALL conclusion
flipped: PILOTS.md said "the locus is decoding, build a targeted decoder"; the
exact-scoring oracle refuted that (it is the model, not the search).

- REACH (reach_analysis.py): the collapse is at decode step 2; all 256 L1 codes are
  used, but never-emitted L1:L2 pairs cover 25.6% (Sports) / 6.6% (Beauty) of the
  catalog; strongest predictor of unreachability is popularity share WITHIN the L1
  prefix (r=-0.62 Sports / -0.48 Beauty), beating global popularity. Reproduced on a
  2nd Sports codebook (within 1-2 points). This "sibling competition" describes the
  PATTERN; the oracle later showed the CAUSE is model-ranking, not beam pruning.
- MBR re-ranking (mbr.py): a dial, not a fix. tau-sweep accuracy-vs-reach Pareto;
  superseded as the "best dial" by the PMI re-rank.
- Conformal sets (conformal.py): depth-100 crossover -- MARIUS's deeper candidate
  pool beats SASRec's, but its top-of-list ordering wastes it. This foreshadowed the
  MODEL-bound / top-K-exposure-ceiling finding.
- FUSE (marius_fuse.py): the sum-fusion is NOT the bottleneck. 3-seed Beauty null
  (level_gain 8.29, attn 8.23 vs baseline 8.26). Exonerates the architecture.

--------------------------------------------------------------------------------
## 4. The causal decomposition (what we ruled out, and the one survivor)

| Candidate cause | Test | Verdict |
|---|---|---|
| Search / beam pruning | exact full-catalog oracle (2.2) | NOT IT (exact == beam at matched K) |
| Fusion of code embeddings | un-sum FUSE ablation (3) | NOT IT (null) |
| Codebook bug / collisions | audit (256/256, near-uniform; prior) | NOT IT (refuted) |
| Tokenizer symmetry (order) | directional gate (2.4) | NOT IT (SHELVE) |
| Tokenizer code-assignment | missed-prefix spread (2.6) | NOT IT (misses diffuse) |
| Code exhaustion | occupancy (2.6) | NOT IT (~4e-6 used) |
| Decode-time popularity | PMI re-rank (2.3) | bounded DIAL |
| Train-time popularity | logit-adjusted CE (2.5) | bounded DIAL |
| Model's learned ranking | (by elimination + oracle) | THE LOCUS: the model ranks tail items below the top-K exposure cutoff; popularity re-weighting at either stage is a bounded dial |

--------------------------------------------------------------------------------
## 5. All ideas considered (tested and untested)

| Idea | Tested? | Result / verdict | Status |
|---|---|---|---|
| Reachability census + structural ceiling (Chao1) | YES | structural ceiling ~46% Sports | KEPT (C1 instrument) |
| Exact full-catalog scoring oracle | YES | MODEL-bound; beam near-optimal | KEPT (C2, the novel core) |
| Sibling-competition mechanism (step-2) | YES | describes the pattern; cause is model-ranking | KEPT as descriptive, recontextualized |
| MBR re-ranking | YES | dial, not fix | KEPT as baseline dial |
| Conformal sets | YES | depth-100 crossover | KEPT as reliability/ordering evidence |
| FUSE (un-sum fusion) | YES | null | KEPT as architecture exoneration |
| PMI / label-bias decode-time re-rank | YES | best dial, beats MBR; item ~= cond2 | KEPT (C4 decode lever) |
| Order-aware / directional COSETTE | YES (gate) | SHELVE (directional signal too sparse) | DROPPED (controlled null) |
| Train-time logit-adjusted loss | YES | DIAL (debias decoupled from reach) | KEPT (C4 train lever) |
| Targeted step-2 "exploration" decoder fix | NO (killed pre-build) | killed by the oracle (it is not a search error) | DROPPED |
| CARE-style reasoning queries (train-time) | NO | progressive-attention mask hard on MARIUS (sum-fusion); cite as related work | DROPPED as method; cite |
| Debias the COSETTE tokenizer (TA framing) | NO | scooped (CRAB, Ghost, Taming-the-Long-Tail); survives only as comparative refutation | DROPPED as novelty |
| Global-norm / listwise MARIUS fine-tune | NO | scooped (Gryphon/SimGR/Latte) | DROPPED |
| PS-COSETTE (predictive-state tokenizer) | partial (the gate IS its diagnostic) | gate failed -> not justified | DROPPED |
| Large Amazon-2023 scale run (Tier E) | NO (next) | tests scale generality + the "scissors" | PENDING (only thing needing GPU) |
| Oracle on the logit-adj checkpoint | NO (next, cheap) | tests objection #1 (checkpoint circularity) | PENDING (cheap Snellius) |

--------------------------------------------------------------------------------
## 6. Hypotheses (confirmed / refuted / refined / open)

CONFIRMED:
- The reachability collapse is real and structural at top-K (Chao1, exact oracle).
- The collapse is MODEL-bound: the beam is near-optimal over the model.
- Popularity correction at both decode and train time is a bounded dial.
- Popularity de-biasing is decoupled from catalog reach (can even shrink it).
- The behavior is density-dependent (worse on sparse Sports than dense Beauty).

REFUTED:
- "It is a beam-search / decoding error" (exact == beam).
- "The sum-fusion is the bottleneck" (FUSE null).
- "An order-aware tokenizer would help" (directional gate SHELVE).
- "It is a codebook bug / code exhaustion" (audit + occupancy).
- "A mechanism-targeted step-2 decoder would fix accuracy" (killed by the oracle).
- "The mechanism-matched prefix prior beats a generic popularity prior" (item ~= cond2).

REFINED (changed during the session):
- "Structural / expressiveness limit" -> "top-K exposure-ranking ceiling of THIS
  trained model": the model can score buried items into a deeper frontier (top-100
  reaches 61%/83%); they lose the top-K competition for their target users. Do not
  over-claim a function-class impossibility (the oracle measures one checkpoint).

OPEN:
- Does the collapse persist / change regime on a larger Amazon-2023 catalog (Tier E)?
- Does a paper-faithful (higher-accuracy) MARIUS collapse less (the reproduction-gap
  confound)? Partial evidence: SASRec++ at comparable accuracy reaches ~74% vs
  MARIUS ~39% on the same data, so accuracy alone does not explain the gap.
- Could a NON-popularity intervention (e.g. CRAB-style codebook rebalancing) move the
  bound? Out of our scope; flagged as the live counter-hypothesis.

--------------------------------------------------------------------------------
## 7. Novelty and positioning (adversarially verified 2026-06-17)

VERDICT: the package -- exact-scoring oracle + structural-ceiling estimator +
two-lever bounded-dial + debias/coverage decoupling + density inversion -- is NOT
scooped. Nearest works each cover at most one facet and most reach the opposite
"it is fixable" conclusion.

Three MANDATORY framing fixes:
1. Cite and contrast GHOST / "Echoes in Filter Bubble" (arXiv:2605.16825): the
   nearest GR-popularity-bias paper; it CURES the bias (asymmetric unlikelihood +
   skeleton tokenization) on different datasets. We are its negative-result
   counterweight.
2. Cite LATTE / "Expressiveness Limits" (arXiv:2605.06331): the real scoop risk. It
   already proves THEORETICALLY that AR semantic-ID generation has a model-level
   expressiveness limit. So do NOT headline "the model, not the beam" as our
   discovery; cite Latte as concurrent theoretical support and lead with the
   empirical oracle, the coverage numbers, the density inversion, and the decoupling.
3. The debias/coverage DECOUPLING is only partially novel: the general Gini-vs-coverage
   divergence is known (Abdollahpouri et al. arXiv:2103.06364, 1901.07555 -- a
   budget-saturation re-ranking effect). Our novel slice: decoupling under a
   top-K/structural ceiling (not a budget), coverage SHRINKING (Beauty -1.7pp), and
   the same knob bounded at BOTH stages. Cite Abdollahpouri; claim the narrow version.

Other must-cite / differentiate: SimGR (2602.07847; CORRECTED 2026-06-21 after web
verification: SimGR's actual claim is a token-level vs item-level modeling mismatch and it
ranks items directly, NOT "beam pruning"; frame our oracle as independently localizing the
cause in the model ranking, consistent/complementary, not as a refutation),
APAO (2603.02730, fixes prefix-pruning in training),
V-STAR / Spend-Search (2602.10699, value-guided search), CRAB (2604.05113, codebook
rebalancing -- the lever we audited and exonerated), D3 / Decoding-Matters
(2406.14900, decode-time, supports "decoding is a dial"), ActionPiece (2502.13581,
context-aware tokenization -- our directional-gate comparator). The phenomenon
(catalog collapse) itself is NOT novel (multiple 2026 papers); our contribution is
the instrument + the systematic exoneration + the negative results.

--------------------------------------------------------------------------------
## 8. Reviewer objections and our defenses (reviewer-2 pass)

Ranked most-damaging first; "cheap" = artifacts we already have.

1. (HIGH) Oracle/checkpoint circularity: the oracle measures ONE trained model, not
   the function class. DEFENSE: down-scope the claim (done, 2.6); re-run the oracle on
   the logit-adj checkpoint and across seeds (cheap Snellius, PENDING); the SASRec
   74% vs MARIUS 39% same-data same-accuracy contrast.
2. (HIGH) Reproduction-gap confound: we diagnose an underperforming model. DEFENSE:
   accuracy-vs-coverage correlation across our seeds (cheap, PENDING); the SASRec
   contrast; full closure = Tier E / paper-faithful MARIUS.
3. (MED-HIGH) tau single-point for "bounded dial": DEFENSE: the decode-time PMI
   alpha-sweep already covers the decode stage across 7 strengths; add per-strength
   tail-hit-count and a 2nd train-time tau if needed.
4. (MED) Chao1 IID assumptions under a deterministic beam: DEFENSE: the exact-frontier
   coverage curve (2.6) shows the ceiling is K-dependent, addressed.
5. (MED) 2-seed (arm) vs 3-seed (baseline) asymmetry: DEFENSE: report per-seed std and
   n per claim (cheap, PENDING); the big effects (39% vs 74%, ARP 101->55, median
   exact rank ~1300) dwarf seed noise.
6. (MED) Small-catalog external validity (12-18k items): the ONLY objection that
   genuinely needs GPU -> Tier E (Office Products ~77k). Cheap partial: occupancy
   shows it is not code exhaustion.
7. (LOW-MED) Tokenizer exonerated from a single alternative: DEFENSE: missed-prefix
   spread (2.6) shows misses are diffuse across codes; soften the prose.

--------------------------------------------------------------------------------
## 9. Operational notes / gotchas discovered this session

- Quant ids are ACCOUNT-SPECIFIC. This account's baseline used COSETTE_128d_256x4_24cd
  (Sports) / a434 (Beauty), NOT the repo-default 8ed1/f958 (the original author's).
  Training jobs that load the -col parquet by id MUST use the right id; read-only eval
  jobs (exact_catalog, dump_topk --no-support) do not, because they read the checkpoint
  config / committed tables. The committed per-token prior was computed from the
  committed Top-K tables and therefore already matches 24cd/a434 (verified byte-identical).
- Fresh OUTPUT_ROOT trap (training arms): export OUTPUT_ROOT BEFORE sourcing
  jobs/env_*.sh, so env_common builds PATHS_OVERRIDES (paths.model_folder_tplt) against
  the fresh dir. Otherwise checkpoints write into the baseline models dir and
  find_run_for_seed / dump_topk miss them. Fixed in jobs/26_*.
- jobs/25 launcher: GROUPS is a bash readonly builtin; use N_GROUPS (fixed on Snellius).
- The eval steps in training arms must write beyond-accuracy/reach to ISOLATED paths
  (not the default) or they overwrite the committed baseline CSVs (fixed in jobs/26_*).

--------------------------------------------------------------------------------
## 10. Code and artifacts inventory (added this session)

Scripts (scripts/extensions/): reach_estimator.py, exact_catalog.py, pmi_rerank.py,
direction_gate.py, compute_token_prior.py, logitadj_selftest.py. Each has --selftest.
Model: src/models/marius_logitadj.py (subclass). Jobs: jobs/23_exact_catalog_*,
jobs/24_pmi_*, jobs/25_direction_gate, jobs/26_logitadj_*. Results:
reports/extensions/{exact_catalog,pmi,reach_estimator,direction_gate,logitadj,
topk_logitadj_*}/. Pushed to origin/stanislaw (latest de02733 + the Snellius result
commits be417d8 etc.). The exact-scoring + PMI + gate + logit-adj results are committed.

--------------------------------------------------------------------------------
## 11. Open next steps and the pending decision

The diagnostic is complete and reviewer-hardened on most axes. Remaining, in order:

1. Free consolidation (no GPU): accuracy-vs-coverage correlation across seeds (#2),
   per-seed std reporting (#5), formalize the 2.6 robustness checks into a committed
   script. (Recommended to do now.)
2. One cheap Snellius re-run: the exact oracle on the logit-adj checkpoint (#1).
3. Tier E: large Amazon-2023 (Office Products ~77k) -- the course's required
   large-dataset prong and the only objection (#6) that needs GPU. Build pipeline
   (jobs 01-06) + 2-seed train of SASRec + MARIUS + the full instrument.
4. Writeup, with the corrected framing (sections 7-8).

PENDING DECISION (Stanislaw): whether to (a) do the free consolidation now then Tier E,
(b) go straight to Tier E, or (c) freeze and start the writeup with Tier E as future
work. The diagnostic paper is publishable as-is given the framing fixes; Tier E
strengthens generality and satisfies the course rubric's large-dataset requirement.

Course rubric (Ext1) status: paradigm comparison (SASRec++ vs COSETTE/MARIUS) DONE;
popularity-bias mitigation DONE (the PMI/MBR/logit-adj dials, reported honestly as
bounded); large-scale dataset = Tier E (pending).
