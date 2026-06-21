# Extension narrative and methods (TA-facing guide)

Purpose: one place that explains EVERYTHING important we tried, in plain terms AND with the
technical implementation, plus the recommended story to build the report around. Written so you
can study from it and explain the implementation to the TA. The dense chronological log is
EXTENSION_SUMMARY.md; this doc is the readable narrative + the "how it actually works" cheat-sheet.

Plain-ASCII only (repo rule). Numbers below are from our committed runs unless marked PENDING.

--------------------------------------------------------------------------------
## 0. TL;DR (the 60-second version)

- We reproduced the paper, then asked WHY the generative model (MARIUS) recommends only a small
  slice of the catalog ("catalog collapse").
- We built a novel measurement tool (an "exact-scoring oracle") that PROVES the collapse is the
  MODEL's fault, not the search/decoding. This is our strongest, most novel asset.
- We then tried hard to turn this into a POSITIVE result (something that improves a metric). We
  ran three systematic novelty scans and built two concrete things. Honest outcome so far:
  no "fix" for the collapse is structurally possible (we proved it), and the easy positives are
  either scooped by 2026 papers or were falsified by our own data.
- What stands: a strong, honest NEGATIVE-RESULT paper (a diagnosis), with the oracle as the
  headline instrument and a direct refutation of a 2026 paper (SimGR). One real positive bet
  (cross-paradigm distillation) is still running on the cluster; a second (selective
  recommendation) is real but turned out NOT to beat the baseline, so it is a characterization.

The narrative to build around: "It is the model, not the decoder: a measurement study of why
generative recommenders under-serve the catalog, and what does and does not fix it."

--------------------------------------------------------------------------------
## 1. What the project is (plain terms)

We reproduce and extend the paper "Closing the Performance Gap in Generative Recommenders with
Collaborative Tokenization and Efficient Modeling" (Lepage, Mary, Picard, 2025). It has two parts:

- COSETTE = a "tokenizer" for items. Every item (a product) gets a short code, like a 4-letter
  word drawn from an alphabet of 256 letters per position (so 4 levels x 256 codes). These codes
  are "semantic IDs": similar items get similar codes. COSETTE's trick is to bake COLLABORATIVE
  signal (who-bought-what-together) into the codes, not just text similarity.
- MARIUS = a generative recommender. To recommend the next item for a user, it GENERATES that
  item's 4-code word one code at a time (like a tiny language model that writes a 4-letter word),
  using the user's history. It then keeps the highest-probability words via beam search.

  Plain analogy: SASRec (the baseline) is like a search engine that scores every item directly.
  MARIUS is like autocomplete that "writes" the next item's code. Both end up with a ranked list.

Task we were given:
- MANDATORY: reproduce on Amazon-2014 Beauty + Sports, baseline SASRec++, metrics Recall + NDCG,
  5 seeds. (Done, merged to main.)
- EXTENSION (we chose "item-side distribution"): compare the two paradigms (generative vs
  discriminative), add beyond-accuracy metrics (diversity, popularity bias), a large dataset, and
  try to mitigate popularity bias. We went well beyond this with a root-cause diagnosis.

--------------------------------------------------------------------------------
## 2. The reproduction (done, for context)

- Our first reproduction landed ~15-28% below the paper (e.g. Beauty SASRec++ Recall@10 8.24 vs
  paper 9.73; MARIUS 8.17 vs 10.02).
- Root cause (a real finding): we had run the repo's DEFAULT config, not the paper's per-dataset
  Beauty config. The paper's Figure 11 is a trap: subplot (a) for Beauty stars d_model=32 with no
  loss normalization; the bigger d in {128,256} search is for the large datasets only.
- Paper-faithful re-run: Beauty SASRec++ Recall@10 8.24 -> 9.06 (closes ~55% of the gap), and
  d=32 beats d=64 (smaller-is-better on Beauty, confirming the paper). Residual gap (~7% Beauty,
  ~20% Sports) is most consistent with a validation-to-test generalization gap (a hypothesis).
- Honest negative we report: the paper's claim "COSETTE/MARIUS is competitive with or beats
  SASRec++ on small data" did NOT reproduce for us; our MARIUS sits below SASRec++ on Beauty.

Why this matters for the extension: because MARIUS underperforms here, any "MARIUS beats SASRec on
raw accuracy" claim is unlikely, so our positive results have to live elsewhere.

--------------------------------------------------------------------------------
## 3. The core finding (the spine of the paper): catalog collapse is the MODEL's fault

### 3.1 What "catalog collapse" is (plain terms)
Across all users, MARIUS only ever recommends a SMALL fraction of the catalog. On Sports it
reaches ~39% of items in its top-10 lists, while SASRec reaches ~74%. About 22% of Sports demand
(items people actually want) is NEVER recommended by MARIUS. So the generative model is
"collapsed" onto a popular core. (Beauty is milder: 71% vs 81%.)

### 3.2 The exact-scoring oracle (OUR KEY INSTRUMENT) -- plain + technical
The big question: is the collapse because (a) the beam SEARCH is too greedy and prunes good items,
or (b) the MODEL itself thinks those items are unlikely? These have opposite fixes (better search
vs a better model), so distinguishing them matters.

- In plain terms: instead of trusting the beam search, we force the model to score EVERY item in
  the catalog directly, for ~1000 test users, and read off where it ranks the true answer. This
  is the "ground truth of what the model believes," independent of the search shortcut.
- Technically: for each user and each catalog item, we teacher-force that item's 4-code tuple
  through MARIUS and sum the 4 per-code log-probabilities to get the item's exact joint
  log-probability. We then rank all items by that exact score and compare the exact Top-K to the
  beam's Top-K. (Code: scripts/extensions/exact_catalog.py + scripts/extensions/scoring.py.)
- What it found (the result that anchors the paper): the collapse is MODEL-bound. The beam is a
  near-optimal search over the model (exact Top-K == beam Top-K at matched K). The items MARIUS
  misses sit at MEDIAN EXACT RANK ~1328 (Sports) / ~1393 (Beauty) out of ~12-18k; 0% are in the
  exact top-20; only 12-14% are even in the exact top-100. So the model genuinely scores those
  items as very unlikely. Better search cannot help; the model's learned ranking is the problem.
- One important nuance (keeps us honest): the model's DEEPER frontier is not empty. Its exact
  top-100 spans 61% (Sports) / 83% (Beauty) of the catalog. So the buried items are not
  un-representable; they lose the top-K competition for their target users. We call this a "top-K
  exposure-ranking ceiling," not an absolute impossibility.

### 3.3 The sharp contribution: we refute a 2026 paper (SimGR)
SimGR (arXiv:2602.07847, 2026) studied the same phenomenon, but used a beam-of-100 as a PROXY for
"what the model would do" and concluded the problem is SEARCH (premature pruning), then built a
fix on that premise. Our oracle does the EXACT computation and reaches the OPPOSITE conclusion
(model-bound). That is a concrete, citable disagreement with a published paper, settled with a
better instrument. This is the most "publishable" single point we have.

### 3.4 Why every fix is a "bounded dial" (a near-theorem; the reason there is no breakthrough)
Because the locus is the model's learned scores, any intervention that only RE-SHAPES the output
of the same trained model on the same data (re-ranking by popularity, re-weighting the loss, etc.)
cannot drag a rank-1330 item into the top-10. It can shift exposure around among already-reachable
items (cut popularity bias), but it cannot lift the structural ceiling. We confirmed this from two
independent angles (decode-time and train-time, below), and it is consistent with theory (Latte,
arXiv:2605.06331, proves an expressiveness limit for this model family).

### 3.5 What we systematically ruled OUT (the "exoneration", strengthens the diagnosis)
| Suspected cause of collapse | How we tested it | Verdict |
|---|---|---|
| Beam search / pruning | exact-scoring oracle (3.2) | NOT IT (exact == beam) |
| The code-fusion architecture | "un-sum" the 4 code embeddings (FUSE ablation) | NOT IT (3-seed null) |
| A codebook bug / collisions | audit (256/256 codes used, near-uniform) | NOT IT (refuted) |
| Tokenizer order-symmetry | order-aware tokenizer probe (directional gate) | NOT IT (too data-starved) |
| Code exhaustion | code-tuple occupancy ~4e-6 | NOT IT |
| Decode-time popularity | PMI re-rank (3.6) | bounded DIAL |
| Train-time popularity | logit-adjusted loss (3.6) | bounded DIAL |
| The model's learned ranking | by elimination + the oracle | THE CAUSE |

### 3.6 The two popularity "dials" (our mitigation results; honest)
- Decode-time PMI re-rank: re-rank the candidates by joint_logprob - alpha * log p_hat(prior).
  Result: a tunable accuracy-vs-coverage dial; cutting alpha trades recall for coverage. KEY
  honesty point: a GENERIC global-popularity prior works as well as a clever within-prefix prior,
  so the bias is just global-popularity-shaped. (Code: scripts/extensions/pmi_rerank.py.)
- Train-time logit-adjusted loss (MARIUSLogitAdj): train with CE(logits + tau*log pi) where pi is
  the per-code popularity prior (Menon et al. 2021). Result: crushes popularity bias (ARP Sports
  101->55, Beauty 56->33) but accuracy drops and catalog reach barely moves or even SHRINKS.
  KEY FINDING: de-biasing popularity is DECOUPLED from catalog reach. (Code:
  src/models/marius_logitadj.py, a small subclass; inference unchanged.)

--------------------------------------------------------------------------------
## 4. Everything we tried to turn this into a POSITIVE result

After the diagnosis, we wanted a result that IMPROVES something. We ran three adversarial novelty
scans (each idea is checked against 2024-2026 literature so we do not re-invent a published method)
and built two concrete things.

### 4.1 The three novelty scans (meta-conclusion)
1. Can we FIX catalog collapse (incl. cross-domain ideas)? No. Every candidate is scooped or a
   bounded dial. It is structural.
2. Cheap 2026 LLM-lab tricks (Muon optimizer, DeepSeek balancing, Kimi QK-clip, weight-averaging,
   SAM)? None touch coverage; at best modest accuracy gains, and mostly scooped (Muon is already
   "MuonRec"). Their only value: a stronger MARIUS that STILL collapses would defend the reviewer
   objection "you diagnosed an underperforming model."
3. Any OTHER positive behavior (not collapse)? One genuine candidate (selective recommendation);
   most others (diversity, cold-start, cold-user) were FALSIFIED against our own data;
   complementarity is real but scooped.

### 4.2 Master table of directions tried
| Direction | In plain terms | Result | Status |
|---|---|---|---|
| Exact-scoring oracle | force the model to score every item to find what it really believes | proves model-bound | KEEP (headline instrument) |
| Reachability census (Chao1) | statistically estimate the true coverage ceiling | ceiling ~46% Sports (structural) | KEEP (instrument) |
| PMI decode re-rank | nudge popular items down at recommend time | bounded dial | KEEP (mitigation, honest) |
| Logit-adjusted loss | nudge popular items down during training | bounded dial; debias != reach | KEEP (mitigation, honest) |
| FUSE ablation | stop summing the 4 codes into one token | no effect | KEEP (exoneration) |
| Order-aware tokenizer | make the codes care about item ORDER | too data-starved to help | DROPPED (clean null) |
| Latte latent-token fix | architectural escape from the limit | it IS Latte (2605.06331) | CITE only (scooped) |
| Muon/DeepSeek/Kimi tricks | modern LLM training tricks | scooped / accuracy-only | CITE only |
| Cross-paradigm distillation | teach MARIUS using SASRec's rankings | PENDING on GPU | BET (win-either-way) |
| Selective recommendation | let MARIUS abstain when unsure | real lift, but loses to SASRec | COMPANION (characterization) |
| Diversity (ILD) win | MARIUS gives more varied lists | FALSE (SASRec dominates) | DEAD |
| Cold-item / cold-user win | MARIUS better on rare items/users | FALSE (MARIUS worse) | DEAD |
| Paradigm complementarity | combine both models' hits | real but scooped (2603.19809) | CITE / companion |

### 4.3 The two things we BUILT (deep dive)

#### (A) Cross-paradigm distillation: MARIUS learns from SASRec (the main positive bet, PENDING)
- Plain terms: SASRec reaches 74% of the catalog and MARIUS only 39% on the SAME data. So SASRec
  clearly "knows" a per-user ranking signal MARIUS does not. We try to TEACH MARIUS using SASRec
  as a teacher: when MARIUS trains, we also push it to agree with SASRec's ranking of items.
- Why it is "win-either-way": if MARIUS's reach lifts toward SASRec's, the ceiling was a missing
  training signal ("supervision starvation, not the model") -> a positive headline. If reach stays
  flat, it is a THIRD independent proof that the ceiling is structural -> strengthens the negative
  paper. We pre-registered the success criteria before running.
- How it works technically (this is the implementation to explain to the TA):
  - We freeze a trained SASRec++ as the teacher.
  - At each training step, for the item MARIUS is trying to predict, we build a CANDIDATE SET =
    {the true item} + {SASRec's top-M items} + popularity negatives.
  - The teacher gives each candidate a score (its dot-product); the student (MARIUS) gives each
    candidate its joint code-tuple log-probability.
  - We add a distillation loss = KL(teacher distribution || student distribution) over the
    candidate set, on top of the usual next-item cross-entropy: loss = CE + alpha * KL.
  - Key engineering points: the teacher is stored UNREGISTERED (so it is not optimized or saved in
    the checkpoint); the student's per-item log-prob reuses the same scorer as the oracle; the
    item indexes of SASRec, the catalog, and the codes all line up (SASRec index = catalog index
    + 2 special tokens); inference is completely unchanged (it is a training-only add-on).
  - Files: src/models/marius_distill.py (the model subclass), src/data/marius.py
    (MARIUSDistillPrePro, which also emits the teacher's input + the true item's catalog index),
    scripts/extensions/distill_utils.py (the candidate builder + an offline item->code table),
    scripts/extensions/scoring.py (a grad-enabled version of the per-item scorer),
    jobs/27_distill_sports.sbatch (the training launcher).
  - Status: the code is fully validated (the torch numeric self-test passes 6/6 on the cluster:
    gradients flow, alpha=0 reproduces the base loss, the KL is computed in the right direction).
    The training run itself is QUEUED behind a cluster-wide GPU maintenance drain. Our honest
    prior: the bounded-dial outcome is more likely than the lift.

#### (B) Selective recommendation via the model's own confidence (real, but a companion not a headline)
- Plain terms: a generative model produces a real PROBABILITY for its top pick. We rank users by
  that confidence and let MARIUS ABSTAIN on the users it is unsure about. On the users it keeps,
  accuracy jumps. It is like a student answering only the questions they are sure about.
- The MARIUS-only result (real and reproduced): serving only the most-confident 5% of users lifts
  Hit@10 from 0.085 to 0.28 on Beauty (3.29x) and 0.046 to 0.124 on Sports (2.67x); both beat a
  random abstention baseline. We ruled out the obvious confounds: confidence is NOT correlated
  with history length, and it helps tail (rare) items MORE than head items, so it is a genuine
  reliability signal, not a disguised "this is an easy/popular case."
- The honest negative (why it is a companion, not the headline): we then compared MARIUS's
  confidence against SASRec's. We HOPED MARIUS's normalized probability would be a better
  reliability signal than SASRec's raw score "by construction." It is not: SASRec's selective
  curve is actually ABOVE MARIUS's on both datasets (lower error-vs-coverage area is better:
  Beauty MARIUS 0.860 vs SASRec 0.816; Sports 0.927 vs 0.915). So both paradigms support selective
  serving and the discriminative baseline does it at least as well. We report this straight.
- How it works technically: rank users by the top-1 joint log-probability; for each "coverage"
  fraction, compute Hit@10 / NDCG@10 over the most-confident subset; summarize with AURC (area
  under the error-vs-coverage curve) vs a random baseline; run the confound battery (history
  length, within-bucket, popularity tail-vs-head, top1-vs-margin). Pure numpy on the committed
  scored Top-K dumps. File: scripts/extensions/selective_prediction.py (with a --cross-paradigm
  mode for the MARIUS-vs-SASRec comparison + multi-seed error bars). Result CSV/JSON +
  FINDINGS.md under reports/extensions/selective_prediction/.

--------------------------------------------------------------------------------
## 5. What to build the narrative around (recommendation)

Build the report as a MEASUREMENT / DIAGNOSIS paper, not a "we beat SOTA" paper. Title idea:
"It is the model, not the decoder: diagnosing why a generative recommender under-serves the
catalog." Spine:

1. Reproduction + the honest paradigm comparison (MARIUS underperforms SASRec here).
2. The phenomenon: catalog collapse, quantified (coverage, invisible demand, Chao1 ceiling).
3. THE CONTRIBUTION: the exact-scoring oracle proves it is model-bound, and REFUTES SimGR's
   search-bound conclusion. This is the part that is genuinely novel and defensible.
4. Systematic exoneration: it is not the beam, the fusion, the codebook, or tokenizer symmetry.
5. The two popularity dials, reported honestly as bounded (de-bias is decoupled from reach).
6. The honest catalogue of attempted positives (selective serving works but does not beat SASRec;
   diversity/cold-start falsified; complementarity scooped) -> this is good-faith science and
   shows breadth.
7. The distillation result (whichever way it forks) as either the one positive twist OR a third
   confirmation that the ceiling is structural.

What NOT to over-claim: do not say we "fixed" collapse; do not say MARIUS beats SASRec on accuracy
or on selective serving; pitch the oracle as the contribution. Must-cite-and-differentiate: Latte
(2605.06331, the expressiveness-limit theory), SimGR (2602.07847, the one we refute), Ghost
(2605.16825, the opposite "fixable" conclusion), Abdollahpouri (the general bias-vs-coverage
tradeoff), UGR (2602.11719, did selective serving with a TRAINED confidence token, so we
differentiate on using the FROZEN likelihood).

--------------------------------------------------------------------------------
## 6. Technical implementation cheat-sheet (for the TA conversation)

### 6.1 Code provenance (state this clearly; the TA will ask)
- The model, loss, and evaluation code (src/models/marius.py, sasrec.py, src/lightning_module.py,
  metrics) is the ORIGINAL authors', byte-identical. We did NOT change the method.
- main is based on the master TA's fork, which added the Amazon-2014 data pipeline and configs.
- OURS: everything under scripts/extensions/*, the small model SUBCLASSES (marius_logitadj.py,
  marius_distill.py) that only add a loss term and leave inference untouched, the new data prepro
  subclass, the jobs/2x sbatch scripts, and all reports/extensions/* + the markdown reports.
  Design principle: every addition is gated behind a config flag or a subclass, so the
  reproduction stays byte-identical when our features are off.

### 6.2 The pieces, each as (what it is / how it works / file)
- MARIUS forward: temporal transformer over the user's items -> a "mid" vector -> depth
  transformer decodes the 4 codes; the 4 code embeddings are SUM-fused into one item token.
  (src/models/marius.py; authors' code.)
- Exact-scoring oracle: teacher-force every catalog item's code tuple, sum the 4 log-softmaxes =
  joint log-prob, rank, compare to beam. (scripts/extensions/exact_catalog.py + scoring.py;
  score_marius_tuples is the per-item scorer.)
- Reachability ceiling: Good-Turing / Chao1 estimator on the observed coverage.
  (scripts/extensions/reach_estimator.py.)
- PMI decode re-rank: adjusted = joint_logprob - alpha * log prior; sweep alpha.
  (scripts/extensions/pmi_rerank.py.)
- Logit-adjusted loss: CE(logits + tau*log pi, target); predict with raw logits. Subclass that
  overrides only __init__ + get_loss. (src/models/marius_logitadj.py.)
- Distillation: see 4.3(A). KL(teacher||student) over a candidate set + CE; teacher frozen and
  unregistered; inference unchanged. (src/models/marius_distill.py, distill_utils.py,
  the grad-enabled scorer in scoring.py, MARIUSDistillPrePro in src/data/marius.py,
  jobs/27_distill_sports.sbatch.)
- Selective recommendation: rank users by top-1 joint log-prob, abstain on the low-confidence
  tail, risk-coverage + AURC + confound battery. (scripts/extensions/selective_prediction.py.)
- Self-tests: because the dev laptop has no GPU/torch, we verify code two ways: torch-FREE static
  + numpy-mirror checks that run anywhere (scripts/extensions/*_selftest.py,
  distill_selftest.py), and a torch numeric check that runs on the cluster
  (distill_torch_check.py). This is why the implementation could be reviewed before any GPU run.

### 6.3 Key facts a TA might probe
- "Is the oracle just the beam?" No: the oracle scores ALL items (O(catalog)); the beam explores
  ~20 paths. They agree at matched K, which is the POINT (the beam is near-optimal over the model).
- "Did you change the model to get your results?" No. Diagnosis uses read-only scoring; the two
  dials and distillation are additive subclasses; inference paths are byte-identical to the authors'.
- "Why is MARIUS below SASRec here?" Honest: the paper's small-data parity claim did not reproduce
  for us; we report it as a negative and it is WHY our positives are not raw-accuracy wins.

--------------------------------------------------------------------------------
## 7. Honest limitations and current status

- Single-seed for the selective-prediction depth-100 dumps originally; the cluster run added
  seeds 42-44 for error bars (committed).
- The distillation training run is PENDING on a cluster GPU maintenance drain; the code is
  validated and queued, with a pre-registered readout (does median exact rank ~1328 drop? does
  top-100 recovery exceed 12-14%? does reach@10 lift above ~0.39?).
- The selective head-to-head is negative (SASRec selects better); reported straight.
- Everything zero-GPU is committed and pushed to branch stanislaw. The large-dataset prong
  (Amazon-2023 Office Products, "Tier E") is designed but not run.

--------------------------------------------------------------------------------
## 8. Glossary (plain terms)

- Semantic ID / code tuple: the 4-number "word" that names an item (4 levels, 256 options each).
- Catalog coverage / reach: how many distinct items appear across all users' top-K lists.
- Catalog collapse: the model only ever recommends a small popular slice of the catalog.
- Teacher-forcing: feeding the known correct codes in to read off the model's probability for an
  item, instead of letting it generate freely.
- Joint log-prob: the model's total log-probability for an item = sum of its 4 per-code log-probs.
- Model-bound vs search-bound: is the failure the model's scores, or the beam search shortcut?
- Bounded dial: a knob that trades one metric for another within limits but cannot break a ceiling.
- ARP / Gini: popularity-bias metrics (Average Recommendation Popularity; inequality of exposure).
- ILD: intra-list diversity (how different the items within one user's list are from each other).
- Selective prediction / abstention: letting the model decline to answer on low-confidence cases.
- Risk-coverage / AURC: accuracy as a function of how many cases you choose to answer; AURC
  summarizes it (lower error-area is better).
- Distillation: training a student model to imitate a teacher model's outputs.
- Fork A / Fork B/C: our pre-registered outcomes for distillation (lift = positive headline;
  flat = bounded-dial control).
