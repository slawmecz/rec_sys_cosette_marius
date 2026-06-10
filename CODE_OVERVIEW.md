# Code overview: provenance, metrics, and extension pilots

A concrete map of every piece of code in this project: what it does, where it
lives, who wrote it (original authors vs master TA vs our team), how we
modified it, and, for every metric we report, the exact place and snippet that
computes it. Companion documents: REPLICATION_REPORT.md (results and
diagnosis), reports/extensions/PILOTS.md (pilot status and numbers).

Branch: `stanislaw`. Nothing in this document changes the authors' model,
loss, or evaluation code; that guarantee is load-bearing for the reproduction
and is re-stated per file below.

---

## 1. The four provenance layers

| Layer | Who | What | May we edit it? |
|---|---|---|---|
| 1. Original authors | Lepage/Mary/Picard (github.com/Simon-Lepage/cosette_and_marius) | model, loss, eval: src/models/{marius,sasrec,cosette}.py, src/lightning_module.py, src/test.py, src/train.py, src/data/* (core), schedulers | NEVER (byte-identical, verified by diff) |
| 2. Master TA fork | Udit Thakur (github.com/2t2c/cosette_and_marius) | Amazon-2014 ingestion (data_scripts/0_raw_to_parquet.py), 2014 configs, configs/experiment/marius_small.yaml, enabling fixes | no (treated as upstream) |
| 3. Our reproduction harness | team (slawek + Stanislaw) | scripts/{sasrec,marius}_5seed.py, jobs/*, scripts/reporting/*, faithful re-run scaffolding (jobs/20-21, EXTRA_TRAIN_OVERRIDES), reporting-bug fix | yes (ours) |
| 4. Extension layer (this branch) | team (Stanislaw + AI-assisted, reviewed) | scripts/extensions/*, src/models/marius_fuse.py (a NEW file), jobs/22, notebooks/{beyond_accuracy,extension_pilots}.ipynb, reports/extensions/* | yes (ours) |

The one repository-level change worth flagging: `.gitignore` ignores
`models/` (training-output dirs), which also silently matched the source
package `src/models/`. We added a `!src/models/` re-include so the new source
file `marius_fuse.py` is tracked; previously-tracked author files were never
affected.

---

## 2. The authors' code we build on (unmodified)

Everything we measure flows through two author-owned interfaces. We quote
them because every extension hooks into them read-only.

### 2.1 Ranked Top-K generation: `search()`

SASRec++ scores the full vocabulary and returns the arg-top-K
(src/models/sasrec.py:207):

```python
def search(self, batch, n_results):
    query = batch["query"]
    embs = self(query)[:, -1, :]          # last-position user embedding
    voc = self.get_embs()                  # all item embeddings
    logits = embs @ voc.T
    if self.filter_preds:                  # mask items already in the history
        logits[torch.arange(B)[:, None], query] = float("-inf")
    _, top_idx = logits.topk(n_results, dim=-1)
    return top_idx                         # B x K raw vocab ids
```

MARIUS runs a beam search over the L=4 semantic-ID codes
(src/models/marius.py:193-290): the Temporal Transformer encodes the history,
`mid_proj` takes the last position, and the Depth Transformer autoregressively
expands code tuples, keeping the top `n_results` beams by accumulated
log-probability. It returns `B x K x L` TOKEN-space code tuples. Two facts
matter downstream:

- token space: `token[l] = raw_code[l] + l*256 + n_special` (the flattened
  1026-entry vocabulary: 2 specials + 4 levels x 256 codes);
- `filter_preds` over-generates by `seq_len` and -inf-masks tuples already in
  the history before the final top-K cut.

### 2.2 Official accuracy metrics: HR@K and NDCG@K

The numbers in REPLICATION_REPORT.md come from the authors'
src/lightning_module.py:59-95 (unmodified), for K in {1, 5, 10, 20}:

```python
gen = self.net.search(batch, n_results=max(self.Ks))
all_hits = (gen == target).all(dim=-1)          # generative: tuple match
RatK = (all_hits.cumsum(dim=1) > 0).float().mean(dim=0)        # HR@K
DCG  = ((2 ** all_hits.float() - 1) / self.dcg_denom).cumsum(dim=1).mean(dim=0)
# single positive => IDCG = 1, so DCG is already NDCG
```

Under leave-one-out with a single held-out positive, HR@K equals Recall@K;
this is why the repo logs `HR@10` and the report calls it R@10.

### 2.3 The fusion line FUSE targets

src/models/marius.py:135, inside `temporal_forward`:

```python
input_embs = self.temp_emb(input).sum(dim=-2)   # B x L x K x D -> B x L x D
```

This single line is the architectural hypothesis of the FUSE pilot
(section 6.5). We do not edit it; we subclass around it.

### 2.4 COSETTE artifacts we consume

- data_scripts/2_train_cosette.py:382 (TA-modified upstream) names each
  tokenizer run `COSETTE_{d}d_{K}x{L}_{uuid4().hex[:4]}`: the suffix is
  RANDOM per run. This is why our job scripts now parameterize the id
  (section 4.1).
- data_scripts/3_remove_colisions.py reallocates colliding items to nearby
  unused tuples and writes the `-col` parquet (`L0..L3` raw codes +
  `product_id`) that both MARIUS training and our lookup tables use.

---

## 3. Reproduction-era harness code (layer 3, ours)

- scripts/sasrec_5seed.py and scripts/marius_5seed.py: per-seed
  train+test+record loops. Key behaviors a reader must know:
  - results are appended per seed to `*_5seed_full_scores.jsonl`; a re-run
    SKIPS seeds already recorded, and `find_run_for_seed` discovers runs by
    (category, seed) under `OUTPUT_ROOT/models` and "records existing metrics
    instead of training". Consequence: any variant run (faithful config,
    FUSE arm) MUST use a fresh OUTPUT_ROOT or it silently records the wrong
    run. This trap is documented in marius_fuse.py and bit us once via a
    leftover smoke checkpoint.
  - `paper_train_args()` appends `EXTRA_TRAIN_OVERRIDES` (env, space-separated
    hydra overrides), empty by default. This hook, added for the faithful
    re-run, is how FUSE swaps the model class with zero harness edits.
- jobs/01-08: sbatch pipeline (download, parquet, sasrec, embeddings,
  cosette, collision removal, marius, extension benchmarks). jobs/20-21:
  paper-faithful re-runs. scripts/reporting/*: tables for the report;
  scripts/benchmark_extensions.py: inference time/memory benchmarks whose
  `get_best_checkpoint` / `run_directory_for_seed` helpers the dump script
  reuses.

---

## 4. Extension layer part 1: the measurement pipeline

Three files form a chain: dump on GPU once, convert + aggregate offline.

### 4.1 jobs/06 and 07: dynamic COSETTE id (small edit to layer-3 files)

Because the COSETTE id suffix is a random uuid (section 2.4), fresh pipelines
produce NEW ids (this account: Beauty `a434`, Sports `24cd`; slawek's originals
`f958`/`8ed1`). Jobs 06/07 now read the id from the environment with the old
ids as defaults, so existing reproductions are unaffected:

```bash
QUANT_METHOD="${QUANT_METHOD:-COSETTE_128d_256x4_f958}"     # job 06
--quant-id "${QUANT_ID:-COSETTE_128d_256x4_f958-col}"        # job 07
```

### 4.2 scripts/extensions/dump_topk.py: the read-only Top-K dumper

Concept: re-run the authors' evaluation on the test split and RECORD the
ranked lists `search()` produces, without touching any model/eval code. A
Lightning `Callback` rides along `trainer.test`:

```python
class TopKCollector(L.Callback):
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, ...):
        # Callback hooks fire OUTSIDE the bf16 autocast that test_step uses;
        # running the generative beam in fp32 collapses the depth decoder to
        # PAD. Re-enter autocast so dumps match the evaluator's arithmetic.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen = pl_module.net.search(batch, n_results=K)
        self.topk.append(gen.detach().cpu().numpy())
        ...
```

The autocast comment records bug #2 of the validation campaign (section 7).
The script also writes the support tables every metric needs, all derived
from layer-1/2 artifacts:

- `popularity.npy`: per-item TRAIN interaction counts, counted from the train
  timelines parquet;
- `embeddings.npy`: Sentence-T5-XL content embeddings reordered to catalog
  index (for ILD; gitignored, regenerated on demand);
- `tuple_to_item.json`: raw-space code tuple -> catalog index, read from the
  `-col` parquet;
- `meta.json`: `{category, n_special=2, n_catalog, L=4, K}`.

Flags added for the scored pilots: `--method {sasrec,marius,both}` and
`--no-support` (dump models living under different OUTPUT_ROOTs),
`--n-results N` (deeper lists, written as `*_topk{N}.npz` so validated 20-deep
files are never clobbered) and `--with-scores` (attach per-candidate model
scores via scoring.py, section 6.4).

Provenance: written by us; autocast fix contributed during Snellius
validation; --n-results/--with-scores added for the MBR pilot.

### 4.3 scripts/extensions/compute_beyond_accuracy.py: dumps -> tidy CSV

The single source of truth for converting raw dumps into catalog-index lists.
SASRec ids just subtract the special-token offset. MARIUS tokens need the
per-level de-offset (bug #1 of the validation campaign; getting this wrong
silently mismatches items):

```python
# MARIUS emits semantic-ID *token* ids: token[l] = code[l] + l*K + n_special.
# tuple_to_item is keyed by raw per-level codes in [0, K).
K = max(int(x) for key in t2i for x in key.split(",")) + 1     # = 256
level_off = [l * K + ns for l in range(L)]

def to_item(code):
    raw = [int(c) - level_off[l] for l, c in enumerate(code)]
    if any(r < 0 or r >= K for r in raw):
        return ba.HALLUCINATION          # emitted special/garbage token
    return t2i.get(",".join(str(r) for r in raw), ba.HALLUCINATION)
```

`load_support` / `load_model_recs` are imported by the notebook, reach,
conformal, and MBR scripts so the conversion can never drift between
analyses. `compute_table` aggregates mean and std over seeds per (category,
model, k) and writes reports/extensions/beyond_accuracy_<category>.csv.

### 4.4 scripts/extensions/beyond_accuracy.py: EVERY metric, with snippets

Pure numpy; operates on `recs` (per-user ranked catalog-index lists, sentinel
HALLUCINATION = -1 for generated tuples mapping to no item), `item_pop`
(train counts), optional `item_emb`, optional `targets`. All "@K" metrics
truncate to the first k entries. The shared primitive:

```python
def exposure_vector(recs, n_catalog, k=None):
    """How often each catalog item appears across all users' top-k lists."""
    counts = np.zeros(int(n_catalog), dtype=np.int64)
    for row in recs:
        for it in (row if k is None else row[:k]):
            if it is not None and it >= 0:
                counts[it] += 1
    return counts
```

1. Catalog coverage (aggregate diversity): fraction of the catalog
   recommended to at least one user.

```python
def catalog_coverage(recs, n_catalog, k=None):
    counts = exposure_vector(recs, n_catalog, k)
    return float((counts > 0).sum() / float(n_catalog))
```

2. Gini index of exposure: 0 = perfectly equal, -> 1 = concentrated.
   Computed over the FULL catalog (zero-exposure items included), so it
   captures both inequality and narrowness.

```python
def gini(counts):
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, s = x.size, x.sum()
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * s) - (n + 1.0) / n)
```

3. Normalized Shannon entropy of exposure (bits / log2(n_catalog), in
   [0, 1]; higher = exposure spread more evenly):

```python
p = c[c > 0] / total
h = float(-(p * np.log2(p)).sum())
return h / float(np.log2(c.size))
```

4. ARP, average recommendation popularity (mean train count of recommended
   items; lower = leans less on blockbusters):

```python
vals = [pop[it] for row in recs for it in row[:k] if it >= 0]
return float(np.mean(vals))
```

5. Head/tail split helpers. Default head = most popular 20% of items
   (`head_mask_by_item_fraction(item_pop, head_frac=0.2)`); a Pareto
   alternative (`head_mask_by_interaction_share`, fewest items covering 80%
   of interactions) is provided but the reported numbers use the 20% split.

6. APLT, average percentage of long-tail items (per user, the share of the
   top-k that is NOT head; then averaged over users):

```python
per_user.append(float(np.mean([not head[it] for it in items])))
return float(np.mean(per_user))
```

7. Novelty, mean self-information of recommended items (higher = more
   novel/surprising):

```python
-np.log2(pop[it] / total)        # averaged over all recommended real items
```

8. Popularity-decile exposure profile (the tabulated version of the paper's
   informal Figure 4): items are stably argsorted by popularity into 10
   equal-size deciles (D1 = rarest); the function returns the share of all
   top-k slots landing in each decile (sums to 1):

```python
order = np.argsort(pop, kind="stable")
edges = np.linspace(0, n, n_deciles + 1).astype(int)   # decile membership
share[d] = counts[decile_of == d].sum();  share /= share.sum()
```

9. ILD, intra-list diversity (mean over users of mean pairwise cosine
   DISTANCE within the top-k, using the T5 content embeddings):

```python
v = embn[items]                      # L2-normalized item embeddings
sims = v @ v.T
per_user.append(float(np.mean(1.0 - sims[np.triu_indices(len(items), k=1)])))
```

10. Hallucination rate (generative-only): fraction of top-k slots whose
    generated tuple maps to no real item (the -1 sentinels):

```python
hall += (it is None or it < 0);  return hall / total_slots
```

11. Recall@k (leave-one-out; equals HR@k with one positive):

```python
hits += (tgt in row[:k])         # over users with a known target
```

12. Tail-recall@k: Recall@k restricted to users whose held-out target is a
    long-tail (non-head) item. This is the "accuracy-meets-beyond-accuracy"
    quantity: does the model actually get tail items RIGHT, not just expose
    them?

```python
sub = [(row, tgt) for row, tgt in zip(recs, targets) if tgt >= 0 and not head[tgt]]
return recall_at_k(sub_rows, sub_targets, k)
```

13. NDCG@k lives in two places. The OFFICIAL numbers come from the authors'
    lightning_module (section 2.2). For re-ranked lists (MBR), the offline
    equivalent is scripts/extensions/mbr.py:

```python
def ndcg_at_k(recs, targets, k):
    # single positive: NDCG = 1/log2(rank + 2) if hit at 0-based rank, else 0
    vals.append(1.0 / np.log2(top.index(tgt) + 2) if tgt in top else 0.0)
```

`compute_all(...)` bundles 1-12 into one dict; `_demo()` (run the file
directly) sanity-checks the orderings on synthetic recommenders.

Cross-validation of this whole chain: the recall recomputed from the dumps
matches the authors' evaluator's HR@10 TO THE DECIMAL (Beauty MARIUS 8.26 =
8.26; sasrec 9.04; Sports 4.71), which is only possible if the de-offset,
the dump, and the metrics are all correct.

### 4.5 Notebooks and jobs

- notebooks/beyond_accuracy.ipynb (built by
  scripts/extensions/build_beyond_accuracy_notebook.py): part A synthetic
  demo, part B the real tables; imports the loaders from
  compute_beyond_accuracy.py (single source of truth).
- notebooks/extension_pilots.ipynb (built by
  scripts/extensions/build_pilots_notebook.py): the plot-and-explain
  walkthrough of all pilots, executed with outputs baked in.
- jobs/22_dump_topk_{beauty,sports}.sbatch: one GPU eval pass per
  (model, seed); SMOKE=1 dumps two batches to validate shapes cheaply.

---

## 5. What we measured with it (pointers, not numbers)

Full tables: reports/extensions/beyond_accuracy_*.csv (mean_/std_ columns
per metric), decile profiles inside both notebooks, headline numbers in
reports/extensions/PILOTS.md and REPLICATION_REPORT.md. The audit feeds the
REACH/conformal/MBR pilots below.

---

## 6. Extension layer part 2: the four pilot codebases

All four follow the same rules: numpy-only if they run locally (torch is
broken on the laptop), import the validated loaders instead of re-deriving
conversions, ship a selftest, never touch layer-1/2 files.

### 6.1 scripts/extensions/reach_analysis.py (REACH: collapse mechanism)

Concept: treat "can the beam ever write this item's code?" as a measurable
property. Key constructs:

- Reachability: an item is REACHED if it appears in any user's top-k;
  consistently-unreached = unreached in every seed.

```python
def reached_mask(recs_arr, k, n_catalog):
    flat = recs_arr[:, :k].ravel(); flat = flat[flat >= 0]
    mask = np.zeros(n_catalog, dtype=bool); mask[np.unique(flat)] = True
    return mask
```

- Prefix mass: per item, the total train popularity of all items sharing its
  L1 code (and its L1:L2 pair), via `invert_tuple_to_item` + bincount:

```python
c0 = codes[:, 0]; pair = c0 * k_cb + codes[:, 1]
mass1_by_code = np.bincount(c0, weights=pop, minlength=k_cb)
mass2_by_pair = np.bincount(pair, weights=pop, minlength=k_cb * k_cb)
```

- The mechanism table (the 5x2 test): bin items into popularity quintiles
  (`equal_freq_bins`, same stable-argsort binning as the decile profile);
  within each quintile, median-split by prefix mass (`rank_half_split`) and
  compare unreached% in the low vs high cell (`mechanism_table`). Run for L1
  mass and L1:L2 mass separately; the L1 table came out INVERTED (sibling
  competition), the L1:L2 table as hypothesized.
- Predictor strength: `point_biserial` (scipy pearsonr of the binary
  unreached flag vs log-mass / log-popularity / log own-share-within-prefix).
- Beam concentration: distinct L1 / L1:L2 prefixes ever emitted
  (`marius_prefix_counts` de-offsets `topk_codes[:, :, 0]` token-space ids)
  vs the catalog's own prefix distribution, plus top-5 slot shares.
- Decomposition: items whose L1:L2 pair is NEVER emitted in any seed are
  unreachable by construction; their count over the unreached set is the
  "step-2 pruning" share.

Outputs: reach_{category}.md, reach_summary.md, reach_results_*.json (all
numbers, machine-readable), one decile PNG.

### 6.2 scripts/extensions/conformal.py (calibrated recommendation sets)

Concept: split conformal prediction on RANK nonconformity. Per (model, seed):

```python
# nonconformity = 1-based rank of the held-out target in the ranked list;
# ABSENT (= depth + 1) if missing; -1 targets excluded.
def conformal_k_star(cal_ranks, coverage, depth=DEPTH):
    m = ceil((n_cal + 1) * coverage)        # standard split-conformal quantile
    if m > n_cal: return ABSENT, m          # unachievable at this depth
    return int(np.sort(cal)[m - 1]), m      # set = top-k*
```

Users are split 50/50 calibration/evaluation (`split_indices`, fixed rng
seed 0); empirical coverage is reported on the eval half. The guaranteeable
ceiling is Recall@depth. The Mondrian variant recalibrates k* inside
history-length buckets (hist <= 5 / 6-15 / >= 16; MARIUS hist_len is
SASRec hist_len + 1 and is offset-corrected in `canonical_hist`), making set
size a per-user uncertainty signal. `mean_effective_set_size` excludes
hallucinated slots from MARIUS's sets. Outputs: split_conformal_*.csv,
mondrian_*.csv, per-category md + summary md.

### 6.3 scripts/extensions/mbr.py (Minimum Bayes Risk re-ranking)

Concept: re-rank the beam by consensus instead of raw likelihood. The utility
is RQ-tree prefix overlap, computed directly on the dumped tuples (valid in
token space because the per-level offset is a bijection):

```python
eq = codes[:, :, None, :] == codes[:, None, :, :]   # U x C x C x L
prefix = np.cumprod(eq, axis=-1)                    # level counts only if all
return prefix.sum(-1) / L                           # shallower levels match

# MBR score per candidate i: sum_j w_j * U(i, j)
score = util.mean(axis=2)                 # uniform weights (tonight's variant)
score = np.einsum("uij,uj->ui", util, w)  # softmax(log-prob / tau) weights
```

`mbr_order` returns a stable permutation (ties keep beam order);
`mbr_topm` reranks only the first m slots. Evaluation reuses
`ba.compute_all` plus the local `ndcg_at_k`, comparing the ORIGINAL beam
order vs the MBR order on identical candidates: any change is attributable
to ranking alone. The uniform variant is the documented lower bound; the
score-weighted variant activates automatically when the npz has a "scores"
key (produced on Snellius). Outputs: mbr_{category}.{md,csv}.

### 6.4 scripts/extensions/scoring.py (torch; GPU-side only)

The bridge that makes scored MBR possible without touching author code: both
functions call only PUBLIC author modules, mirroring what search() and
train_forward() already do.

```python
# log p(candidate tuple | history): teacher-force the depth decoder
mid = net.mid_proj(net.temporal_forward(batch_input))[:, -1, :]
dec_embs = torch.cat([mid_rep, net.depth_emb(flat[:, :-1])], dim=1)
logits = net.depth_forward(dec_embs)
tok_lp = F.log_softmax(logits.float(), -1).gather(2, flat[..., None]).squeeze(-1)
return tok_lp.sum(-1).view(B, C)          # float32[B, C]
```

`score_sasrec_topk` recomputes `embs @ voc.T` and gathers at the dumped item
ids (it does NOT re-apply filter_preds; the dumped items were already
filtered, and re-masking would corrupt their logits). The caller (dump_topk)
wraps both in the same bf16 autocast as search(), so scores match the
arithmetic that produced the rankings. Never imported by local CLIs.

### 6.5 src/models/marius_fuse.py (FUSE: the architectural probe)

A NEW file; the authors' MARIUS is subclassed, never edited. Overrides ONLY
`__init__` (adds fusion parameters) and `temporal_forward` (replaces the one
fusion line of section 2.3). Four arms, all EXACTLY equal to the baseline at
initialization so training alone decides:

```python
# "sum": delegates to super() -- the ablation/identity arm.

# "level_bias": fused = sum_l (e_l + b_l), b zero-init.  PROVEN DEGENERATE:
# the unweighted sum makes the biases collapse to one learned constant
# (sum_l b_l), so the function class equals the baseline. Kept as an
# optimization-null control arm.

# "level_gain": fused = sum_l (g_l * e_l), g ones-init.  The cheapest
# NON-degenerate arm: per-level multiplicative gains do not sum out and
# break level-permutation invariance.  (+1024 params)
fused = (code_embs * self.fuse_level_gain[None, None, :K, :]).sum(dim=-2)

# "attn": learned-query attention pool, q zero-init, scale init = 4.0
# (uniform weights * 4 = the sum at init).  Content-adaptive level
# weighting.  (+257 params)
scores = torch.einsum("blkd,d->blk", code_embs, self.fuse_query) / sqrt(D)
fused = self.fuse_scale * torch.einsum("blk,blkd->bld", softmax(scores), code_embs)
```

Contracts preserved (each checked by the selftest): padded history positions
stay EXACTLY zero before the positional embedding (`* pad_mask`), positional
embedding / dropout / causal mask / key-padding mask are verbatim from the
base, and the new parameter names avoid the "temp_emb"/"depth_emb"
substrings so the authors' `get_param_groups` keeps them in the weight-decay
group.

How it runs with zero harness edits: hydra instantiates the model from
`model.net._target_`, and the layer-3 hook passes overrides through:

```bash
EXTRA_TRAIN_OVERRIDES="model.net._target_=src.models.marius_fuse.MARIUSFuse +model.net.fusion=attn"
# AND a fresh OUTPUT_ROOT per arm (see the section-3 trap), e.g.
OUTPUT_ROOT=/scratch-shared/$USER/cosette_marius/outputs_fuse_attn
```

### 6.6 scripts/extensions/fuse_selftest.py

Torch-free verification of marius_fuse.py (33 checks): AST/static checks
(subclasses MARIUS; overrides only the two methods; ASCII; decay-group-safe
parameter names; pad mask present in all three new arms) plus a numpy mirror
of the fusion math (init-equality of every arm with the baseline, exact-zero
pads, the level_bias degeneracy proof, level_gain permutation-SENSITIVITY,
attn weight adaptivity, parameter counts). This selftest is what caught the
level_bias degeneracy before any GPU time was spent on it.

---

## 7. Validation methodology (how we know the numbers are right)

1. Every locally-run script has a selftest with fabricated data and known
   answers: beyond_accuracy `_demo` (metric orderings on synthetic
   recommenders), compute_beyond_accuracy `--selftest` (round-trips
   token-space codes through the de-offset; asserts the injected 10%
   hallucination rate is recovered), reach `--selftest` (a toy catalog where
   exactly the blocked-prefix items must be unreachable), conformal
   `--selftest` (brute-force cross-check of k*; identical k* from item-space
   and token-space inputs; coverage and Mondrian assertions), mbr
   `--selftest` (hand-computed utility matrix; outlier demotion; tie
   stability), fuse_selftest (33 checks).
2. End-to-end cross-checks against independent ground truth: dumped recall
   == the authors' evaluator's HR to the decimal (the decisive check);
   reach's per-seed reached fractions reproduce the audit CSV coverage to 3
   decimals; MBR's "original" row reproduces the validated audit numbers
   exactly.
3. Three real bugs were caught and fixed by this discipline, all in OUR
   code, none in the authors': (i) the MARIUS token-space de-offset missing
   from the first lookup; (ii) search() running outside bf16 autocast in the
   dump callback (fp32 collapses the depth decoder to PAD); (iii) the
   level_bias degeneracy (a flaw in the experiment design itself).

---

## 8. Complete file-by-file provenance

| File | Layer | Status |
|---|---|---|
| src/models/marius.py, sasrec.py, cosette.py | 1 | byte-identical, never edited |
| src/lightning_module.py, src/test.py, src/train.py | 1 | byte-identical, never edited |
| src/data/* | 1+2 | authors' core + TA enabling fixes; untouched by us |
| data_scripts/* | 2 | TA's Amazon-2014 pipeline; untouched by us |
| configs/* (existing) | 1+2 | untouched |
| scripts/{sasrec,marius}_5seed.py, scripts/reporting/*, jobs/01-21 | 3 | ours; jobs 06/07 gained QUANT_METHOD/QUANT_ID env parameterization this week |
| scripts/benchmark_extensions.py | 3 | ours (inference benchmarks); helpers reused by dump_topk |
| scripts/extensions/beyond_accuracy.py | 4 | ours, new |
| scripts/extensions/dump_topk.py | 4 | ours, new (+ autocast fix, + --n-results/--with-scores) |
| scripts/extensions/compute_beyond_accuracy.py | 4 | ours, new (+ token de-offset fix) |
| scripts/extensions/reach_analysis.py | 4 | ours, new (REACH pilot) |
| scripts/extensions/conformal.py | 4 | ours, new (conformal pilot) |
| scripts/extensions/mbr.py | 4 | ours, new (MBR pilot) |
| scripts/extensions/scoring.py | 4 | ours, new (GPU scoring for scored MBR) |
| scripts/extensions/fuse_selftest.py | 4 | ours, new |
| src/models/marius_fuse.py | 4 | ours, NEW FILE subclassing the authors' MARIUS |
| scripts/extensions/build_*_notebook.py, notebooks/{beyond_accuracy,extension_pilots}.ipynb | 4 | ours, new |
| jobs/22_dump_topk_*.sbatch | 4 | ours, new |
| .gitignore | 3 | ours; + !src/models/ re-include this week |
| reports/extensions/* | 4 | generated results + PILOTS.md |

---

## 9. Re-running everything (quick reference)

Local (no GPU, no torch; runs on the committed dumps):

```bash
python3 scripts/extensions/beyond_accuracy.py                  # metric demo
python3 scripts/extensions/compute_beyond_accuracy.py --category Beauty --seeds "42 43 44"
python3 scripts/extensions/reach_analysis.py --category "Beauty Sports_and_Outdoors" --seeds "42 43 44"
python3 scripts/extensions/conformal.py --category Beauty
python3 scripts/extensions/mbr.py --category Sports_and_Outdoors --seeds "42 43 44" --k 10
python3 scripts/extensions/fuse_selftest.py
jupyter nbconvert --to notebook --execute notebooks/extension_pilots.ipynb
```

Snellius (GPU): jobs/22 for 20-deep dumps; dump_topk.py --n-results 100
--with-scores for scored dumps; marius_5seed.py with EXTRA_TRAIN_OVERRIDES +
fresh OUTPUT_ROOT for FUSE arms (full command blocks: src/models/
marius_fuse.py docstring and reports/extensions/PILOTS.md).
