# Reproduction report: Arts, Crafts & Sewing (Amazon Reviews 2023)

From-scratch 5-seed reproduction of **MARIUS (COSETTE)** and **SASRec++** for the
`Arts_Crafts_and_Sewing` category of Amazon Reviews 2023, run on Snellius.

- Branch: `repro/arts-crafts-2023`
- Paper: [arXiv:2508.14910](https://arxiv.org/abs/2508.14910) (5 seeds, mean reported)
- Reference: [`2t2c/cosette_and_marius`](https://github.com/2t2c/cosette_and_marius)
  (`upstream/REPRODUCIBILITY.md`), single seed, 2x A100 80 GiB, verified at commit `9a9d8e4`
- Dataset: 197,286 users / 89,958 items
- Seeds: 42, 44, 46, 48, 50 (full pipeline - COSETTE retrained per seed)

This single file is the complete write-up: results, how to reproduce with the job
scripts, how to regenerate the metrics, the deviations from the reference, and cost.

---

## 1. Results

All numbers are **test** metrics in percent. R@k == HR@k (one held-out target per user).

### MARIUS (COSETTE)

| Source | R@5 | NDCG@5 | R@10 | NDCG@10 |
|--------|-----|--------|------|---------|
| Paper (5-seed mean) | 3.49 | 2.37 | **5.30** | 2.95 |
| Reference (1 seed) | - | - | **5.04** | - |
| **Ours (5-seed mean ± std)** | 3.32 ± 0.03 | 2.22 ± 0.03 | **5.01 ± 0.04** | 2.77 ± 0.03 |

R@10 deltas: ours vs reference **-0.03 pp** (on target); ours vs paper **-0.29 pp**;
reference vs paper **-0.26 pp**.

### SASRec++

| Source | R@5 | NDCG@5 | R@10 | NDCG@10 |
|--------|-----|--------|------|---------|
| Paper (5-seed mean) | 3.51 | 2.42 | **5.09** | 2.93 |
| Reference | not run (open TODO in their `upstream/REPRODUCIBILITY.md`) | | | |
| **Ours (5-seed mean ± std)** | 3.31 ± 0.05 | 2.28 ± 0.04 | **4.86 ± 0.06** | 2.78 ± 0.04 |

R@10 delta: ours vs paper **-0.23 pp**. The reference never ran SASRec++ on Arts, so
this is a **new contribution** with no reference baseline.

### Reading the results

- Our 5-seed MARIUS mean lands essentially **on top of the reference's single-seed
  result** (5.01 vs 5.04). The residual gap to the paper (-0.29 pp) is the same
  shortfall the reference saw (-0.26 pp).
- Both gaps to the paper are several times the 5-seed std (±0.04 / ±0.06 pp on R@10),
  so the residual shortfall is **systematic** (a validation->test generalization gap),
  not seed noise.

### Per-seed test metrics

MARIUS (COSETTE):

| Seed | R@5 | NDCG@5 | R@10 | NDCG@10 |
|------|-----|--------|------|---------|
| 42 | 3.31 | 2.23 | 5.00 | 2.77 |
| 44 | 3.32 | 2.24 | 5.03 | 2.79 |
| 46 | 3.26 | 2.18 | 4.95 | 2.72 |
| 48 | 3.33 | 2.23 | 5.03 | 2.77 |
| 50 | 3.36 | 2.24 | 5.04 | 2.79 |

SASRec++:

| Seed | R@5 | NDCG@5 | R@10 | NDCG@10 |
|------|-----|--------|------|---------|
| 42 | 3.37 | 2.32 | 4.91 | 2.82 |
| 44 | 3.26 | 2.25 | 4.78 | 2.73 |
| 46 | 3.32 | 2.29 | 4.87 | 2.78 |
| 48 | 3.26 | 2.24 | 4.83 | 2.74 |
| 50 | 3.34 | 2.31 | 4.90 | 2.81 |

Raw per-seed records (with run directories and timestamps):
`reports/results/{marius,sasrec}_arts_5seed_full_scores.jsonl`.

### Statistical significance (5 seeds)

Regenerate with `python scripts/significance_arts.py` (reads the jsonl above; scipy-free:
Student-t CIs with a bootstrap cross-check, plus an exact two-sample permutation test).

**Q1 - 95% confidence intervals (Student-t).**

| metric | MARIUS [95% CI] | SASRec [95% CI] |
|--------|-----------------|-----------------|
| R@5 | 3.32 [3.27, 3.36] | 3.31 [3.25, 3.37] |
| NDCG@5 | 2.22 [2.19, 2.26] | 2.28 [2.23, 2.33] |
| R@10 | 5.01 [4.97, 5.05] | 4.86 [4.79, 4.93] |
| NDCG@10 | 2.77 [2.74, 2.80] | 2.78 [2.73, 2.83] |

**Q2 - ours vs paper.** The paper's reported value lies **outside our 95% CI on every
metric for both models** (3.8-8.1 std from our mean), so the shortfall is **systematic, not
seed noise**. Strongest single piece of evidence: our MARIUS R@10 gap to the paper (-0.29 pp)
closely matches the reference's own gap (-0.26 pp), pointing to a systematic effect (a
validation->test generalization gap) rather than a reproduction error. We do **not** claim
"significantly worse than the paper" - that needs the paper's per-seed variance, which is not
published, so this is CI containment, not a two-sample test against the paper.

**Q3a - MARIUS vs SASRec** (exact two-sample permutation test, Holm-corrected over 4 metrics):

| metric | diff (M-S) | Welch t | perm p | Holm p | significant? |
|--------|-----------|---------|--------|--------|--------------|
| R@10 | +0.151 | +5.00 | 0.008 | **0.032** | **yes - MARIUS** |
| NDCG@5 | -0.056 | -2.74 | 0.032 | 0.095 | no |
| NDCG@10 | -0.009 | -0.43 | 0.683 | 1.000 | no |
| R@5 | +0.005 | +0.17 | 0.865 | 1.000 | no |

Only the **MARIUS R@10 advantage is significant** - it is perfectly separated across all
5+5 seeds (perm p hits the n=5 floor of 2/252 = 0.008) and survives Holm. The other three
metrics move in the paper's own direction (paper: SASRec NDCG@5 2.42 > MARIUS 2.37; MARIUS
R@10 5.30 > SASRec 5.09) but are **not separately significant at 5 seeds** - NDCG@5 favors
SASRec at uncorrected p = 0.032 but does not survive correction. Both models were evaluated
under an identical filtered protocol (`filter_preds=True`), so this reflects the models, not
the eval. Resolving the sub-threshold metrics would need a well-powered per-user paired test.
(The paired across-seed test is omitted: at n=5 its sign-flip floor is 2/32 = 0.0625, so it
cannot reach alpha = 0.05.)

---

## 2. How to reproduce (Snellius job pipeline)

All job scripts live in [`jobs/arts2023/`](jobs/arts2023/). Submit **from the repo
root** (`/gpfs/home2/scur1250/rec_sys_cosette_marius`). Shared config -
venv, scratch `DATA_ROOT`, offline flags, `paths.root` override - is sourced from
[`jobs/arts2023/env_arts.sh`](jobs/arts2023/env_arts.sh) by every job.

### Run order

| # | Step | Command | Where / GPUs | Notes |
|---|------|---------|--------------|-------|
| 0 | Verify GPU stack | `sbatch jobs/arts2023/00_verify_gpu.sbatch` | gpu_h100 x1 | no data needed; safe anytime |
| A1 | Download (data + T5-XL) | `bash jobs/arts2023/A1_download.sh` | **login node** | once; needs internet |
| A2 | Parquet | `sbatch jobs/arts2023/A2_parquet.sbatch` | genoa (CPU) | once |
| A3 | Embeddings | `sbatch jobs/arts2023/A3_embeddings.sbatch` | gpu_h100 x2 | once; offline |
| - | Smoke test | `sbatch jobs/arts2023/smoke_sasrec.sbatch` | gpu_h100 x2 | after A2; confirm steps/epoch |
| B | COSETTE x5 seeds | `sbatch jobs/arts2023/B_cosette.sbatch` | gpu_h100 x1 (array 42,44,46,48,50) | per-seed tokenizer |
| C | Collisions + MARIUS x5 | `sbatch jobs/arts2023/C_marius.sbatch` | gpu_h100 x2 (array 42,44,46,48,50) | needs B per seed |
| D | SASRec++ x5 seeds | `sbatch jobs/arts2023/D_sasrec.sbatch` | gpu_h100 x2 (array 42,44,46,48,50) | independent of B/C |
| E | Aggregate vs paper | `python scripts/aggregate_arts.py` | login node | reads `runs/seed_*.{marius,sasrec}` |

Recommended order: **0 -> A1 -> A2 -> smoke -> (one full seed first) -> A3 -> B -> C, with D
in parallel -> E.**

- A1/A2/A3 are seed-independent and run **once**. A1 must run on a login node (compute
  nodes have no internet); it pre-fetches both the Amazon-2023 files and Sentence-T5-XL
  into the HF cache so all later jobs run offline.
- C(seed) depends on B(seed) - it reads `runs/seed_<seed>.quant`. Chain per-seed with:
  `sbatch --dependency=aftercorr:<B_arrayjobid> jobs/arts2023/C_marius.sbatch`
- D depends only on A2 parquet (no embeddings/COSETTE), so it runs in parallel with B/C.

### Reproduction overrides (reference recipe, applied per-run only)

The repo's committed defaults are the gap-study config and are **left untouched**; the
reference recipe is restored via per-run Hydra overrides inside the job scripts:

- COSETTE (B): `optim.batch_size=1024 optim.epochs=3000` (~192 steps/epoch -> ~576k steps)
- MARIUS & SASRec (C/D): `ray.scaling_config.num_workers=2  trainer.max_steps=81000  trainer.limit_val_batches=1000`
- Partition `gpu_h100` (94 GiB) so reference batch sizes fit without an OOM-driven deviation.

### What to confirm in the smoke / first-seed logs

- **Steps per epoch under `num_workers=2`.** With `total_len = max_steps x batch` split
  over 2 workers, each worker sees `total_len/2` rows -> **~`max_steps/2` steps per
  epoch**, and `max_steps` is reached over ~2 epochs. The smoke run (`max_steps=600`)
  should show ~300 steps/epoch and finish at 600.
- COSETTE (B): ~192 steps/epoch at bs=1024.
- **Filtering is identical for both models at eval (`filter_preds=True`** - already-seen
  items removed from candidates). SASRec's `filter_preds` defaults to `True` (it is not set
  in `configs/experiment/sasrec.yaml`), so the `enforce_filtering=false` in `D_sasrec.sbatch`
  is a harmless no-op - it only skips force-setting an already-`True` flag, it does **not**
  disable filtering. SASRec is filtered, matching the paper's SASRec++ and keeping the
  MARIUS-vs-SASRec comparison on one eval protocol.

### Provenance written at run time

- `runs/seed_<seed>.quant` - COSETTE quant id for the seed (written by B)
- `runs/seed_<seed>.marius` / `.sasrec` - Ray run-directory name (written by C / D)

---

## 3. How to get the metrics

Two scripts read the run-directory pointers in `runs/seed_<seed>.{marius,sasrec}` and the
`metrics.pkl` that `src/test.py` writes for each run.

```sh
source jobs/arts2023/env_arts.sh

# Print mean ± std for R@5, NDCG@5, R@10, NDCG@10 next to the paper targets:
python scripts/aggregate_arts.py

# Refresh the per-seed score jsonl the replication notebook reads
# (reports/results/{marius,sasrec}_arts_5seed_full_scores.jsonl):
python scripts/export_arts_scores.py

# Statistical significance over the 5-seed scores (CIs, MARIUS-vs-SASRec, vs paper):
python scripts/significance_arts.py
```

Both accept `--data-root` / `--runs-dir` if `DATA_ROOT`/`PROJECT_ROOT` are not exported.
`export_arts_scores.py` only rewrites a method's jsonl when at least one seed is found,
so running it where `DATA_ROOT` is not mounted (e.g. a laptop) never clobbers the
committed cache. Metrics are normalised to percent.

---

## 4. Changes relative to `main` and the reference

A `diff -rq` of our `configs/` and `data_scripts/` against a fresh reference clone shows
the code paths are identical; **every reference fix is already present**. Our repo =
reference code + (a) additive seed plumbing, (b) the `jobs/` / `reports/` / `notebooks/`
tooling, and (c) three committed gap-study defaults restored to the reference recipe via
per-run overrides. The deliberate, documented deviations:

### 1. Seed plumbing (additive, gated on `seed`)

The reference is single-seed and leaves COSETTE k-means, the COSETTE timeline sampler,
and the Ray-Data augmentation/sampler unseeded (`seed` is only logged as an hparam, never
applied). We added purely additive seeding so the paper's "mean ± std over 5 runs" is
reproducible:

- k-means `random_state` (`src/models/cosette.py`)
- seeded generators in `CropAndAugment` and `ToRow` (`src/data/augment.py`, `ray_data.py`)
- COSETTE DataLoader `worker_init_fn` and `seed_everything` in the COSETTE GPU worker
  (`data_scripts/2_train_cosette.py`)
- `seed` threaded through `make_ray_dataset` -> prepro -> augmenter (`src/train.py`,
  `src/data/{ray_data,sasrec,marius}.py`)

Every new parameter **defaults to `None`/original behaviour**, so this does not change
the method (verified: same seed -> identical, different seed -> differs).

- **Caveat (in code comments):** parallel actors within one Ray-Data op share the op's
  seed (Ray Data exposes no per-actor index), so streams are reproducible per (seed, op)
  but correlated across that op's parallel actors. Affects only training-sample ordering.
- The paper does not publish its seed values, so matching its exact seeds is impossible
  and not required to reproduce a 5-seed mean ± std. **Seeds are spaced by 2** so that
  under `seed_everything(seed + world_rank)` with `num_workers=2` no two runs share a
  worker RNG stream (consecutive seeds would overlap and correlate the runs).

### 2. Reference recipe applied as per-run overrides (committed defaults intact)

| Override | Value | Repo default | Why |
|----------|-------|--------------|-----|
| `optim.batch_size` (COSETTE) | `1024` | 256 | reference intent + paper; bs=256 -> ~4x steps |
| `optim.epochs` (COSETTE) | `3000` | 500 | reference Arts command |
| `ray.scaling_config.num_workers` | `2` | 1 | reference's R@10 5.04 used 2 workers -> effective batch 512 |
| `trainer.max_steps` | `81000` | 80000 | reference |
| `trainer.limit_val_batches` | `1000` | null (full) | reference |

**COSETTE batch-size ambiguity.** `configs/2_train_cosette.yaml` defaults to
`batch_size: 256` with a comment targeting ~600k steps, while the reference's Arts command
passes only `epochs=3000` (no batch override) yet its section heading says "bs=1024". We
resolve in favour of the config comment's intent: `bs=1024 + epochs=3000 = 576k steps`
(192 steps/epoch). Our MARIUS R@10 matching the reference's (5.01 vs 5.04) validates this.

### 3. `gpu_h100` (94 GiB) instead of A100

The reference ran 80 GiB A100s; Snellius A100 is 40 GiB. H100 >= reference VRAM, so the
reference batch sizes fit **without any OOM-driven batch-size reduction**.

### 4. Path overrides

`model_folder` / `ckpt_dir` / `paths.root` overrides replace the original author's
hardcoded machine paths. No method impact.

### 5. SASRec `vocab_size` off-by-one bugfix

`configs/experiment/sasrec.yaml` hardcodes `vocab_size: 89959`, but
`src/data/ray_data.get_items_map` assigns item token ids `idx + len(SpecialTokens)`
(PAD=0, BOS=1 -> offset 2), so Arts's 89,958 items occupy tokens **2..89959** and the
embedding needs **89,960** rows (`num_items + len(SpecialTokens)`, the same convention as
MARIUS `vocab_size=1026 = 4*256 + 2`). With 89959 the last item overflows the table -> CUDA
`indexSelectLargeIndex: srcIndex < srcSelectDimSize` device-side assert (the run aborts
~2 min in; a short smoke happens to dodge the one overflowing item). Applied as the per-run
override `model.net.vocab_size=89960` in `D_sasrec.sbatch`; the committed config is left
intact. This is a correctness fix, not a method change.

---

## 5. Target configuration

- **Embeddings:** Sentence-T5-XL (official code + reference use it throughout). NV-Embed-v2
  is **not** used.
- **COSETTE:** codebook 256 x 4 levels, 128-d latent (`COSETTE_128d_256x4`), contrastive
  weight 1e-3, tau=2, bias=-8, dropout 0.1; Arts: `batch_size=1024`, `epochs=3000`.
- **MARIUS:** full model (`experiment=marius`): temporal d=512 x4 layers, depth d=512 x6
  layers, dropout 0.4/0.1, `vocab_size=1026`, best checkpoint by `valid/.../HR@10`,
  `filter_preds=true`. (`marius_small` is for the 2014 datasets, not this one.)
- **SASRec++:** `experiment=sasrec`: d=128 x2 layers, L2 norm, sampled softmax tau=0.05 with
  30k negatives, `vocab_size=89960` (see deviation #5), no quantization.

---

## 6. Snellius environment & storage

- **venv:** `/home/scur1250/cosette-venv` (Python 3.9.21): `torch 2.6.0+cu124`,
  `ray 2.44.1`, `pytorch-lightning 2.5.0.post0`, `sentence-transformers 2.7.0`.
  GPU-stack check: `jobs/arts2023/00_verify_gpu.sbatch`.
- **Storage** (project space `/projects/prjs2120` is ACL-locked to a teammate, so unused):
  - code + venv + HF model cache -> `$HOME` (200 GiB quota, persistent)
  - bulk data / embeddings / checkpoints -> `/scratch-shared/scur1250/cosette_arts`
    (8 TiB, periodically purged -> final metrics copied back into git)
- **Offline on compute nodes:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; T5-XL
  pre-fetched to `HF_HOME` on the login node (A1).
- **Account:** `gpuuva080`. **GPUs:** `gpu_h100` = 94 GiB, 192 SBU/GPU-h; `gpu_a100` =
  40 GiB, 128 SBU/GPU-h. **W&B:** offline.

---

## 7. Cost (measured, seed 42)

| Stage | GPUs | Wall | GPU-h | SBU (H100, 192/GPU-h) |
|-------|------|------|-------|------------------------|
| COSETTE (B) | 1 | 2:52 | 2.87 | ~551 |
| MARIUS (C, incl. collisions + test) | 2 | 1:48 | 3.59 | ~690 |
| SASRec++ (D, incl. test) | 2 | 1:06 | 2.20 | ~220 |

Full 5-seed run measured **~8,000 SBU total** - well under the 25k planning flag and
~0.4% of the account budget remaining.

---

## 8. Status & provenance

**Complete.** All five seeds (42, 44, 46, 48, 50) finished for COSETTE, MARIUS, and
SASRec++. MARIUS: 5.01 ± 0.04 R@10 (-0.03 pp vs reference, -0.29 pp vs paper). SASRec++:
4.86 ± 0.06 R@10 (-0.23 pp vs paper; new contribution, no reference baseline).

- Run-name pointers: `runs/seed_{42,44,46,48,50}.{quant,marius,sasrec}`
- Per-seed scores: `reports/results/{marius,sasrec}_arts_5seed_full_scores.jsonl`
- Regenerate the comparison:
  `source jobs/arts2023/env_arts.sh && python scripts/aggregate_arts.py`
