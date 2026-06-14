# Reproducing Arts, Crafts & Sewing (Amazon Reviews 2023) on Snellius

5-seed reproduction of **MARIUS (COSETTE)** and **SASRec++** for the
`Arts_Crafts_and_Sewing` category, from scratch, following the reference recipe in
[`2t2c/cosette_and_marius`](https://github.com/2t2c/cosette_and_marius) (`REPRODUCIBILITY.md`)
and the paper [arXiv:2508.14910](https://arxiv.org/abs/2508.14910).

Branch: `repro/arts-crafts-2023`. Job scripts: [`jobs/arts2023/`](jobs/arts2023/README.md).

---

## 1. Reconciliation: our repo vs the reference

A full `diff -rq` of our repo against a fresh clone of `2t2c/cosette_and_marius`:
**our repo contains every file the reference has** (nothing is "only in reference"),
and the two differ in only four files:

| File | Difference | Verdict |
|------|-----------|---------|
| `configs/paths.yaml` | one comment character | functionally identical |
| `configs/train.yaml` | we **added** `seed: 42`; `limit_val_batches: null` (ref `1000`); `max_steps: 80_000` (ref `81_000`); `num_workers: 1` (ref `2`) | our gap-study defaults |
| `src/train.py` | we **added** `seed_everything(seed+world_rank)` + seed in run name | our addition |
| `REPRODUCIBILITY.md` | different (superseded) document | not code |

**Every reference fix is already present and byte-identical** — there is nothing to
port. Specifically, all of these are already in our `data_scripts/`, `scripts/`,
`configs/`:

- download via `hf_hub_download` per file (`scripts/download_data.py`)
- `makedirs` before writing parquet/embeddings (`0_raw_to_parquet.py`, `1_make_embeddings.py`)
- `ray.init(num_gpus=torch.cuda.device_count())` (`2_train_cosette.py`)
- COSETTE DataLoader `num_workers=4` (`2_train_cosette.py`)
- `tqdm`-wrapped embeddings (`1_make_embeddings.py`)
- `paths.yaml` `protocol: file`, `root: ${oc.env:PWD}/datasets`

So our repo = reference code + (a) the seed plumbing, (b) `jobs/`, `reports/`,
`notebooks/`, and (c) the three committed defaults above (kept as our gap-study config;
the reference recipe is restored via per-run overrides — see §4).

### Job → reference-stage mapping

| Our job | Reference stage / command |
|---------|---------------------------|
| `A1_download.sh` | `scripts/download_data.py --categories Arts_Crafts_and_Sewing` |
| `A2_parquet.sbatch` | `data_scripts/0_raw_to_parquet.py paths.skip_download=true` |
| `A3_embeddings.sbatch` | `data_scripts/1_make_embeddings.py category=…` (+ `model_folder` override) |
| `B_cosette.sbatch` | `data_scripts/2_train_cosette.py data.category=… optim.epochs=3000` (+ `batch_size=1024`) |
| `C_marius.sbatch` | `data_scripts/3_remove_colisions.py …` → `src/train.py experiment=marius …` |
| `D_sasrec.sbatch` | `src/train.py experiment=sasrec` |
| `scripts/aggregate_arts.py` | new (collects `src/test.py` outputs vs paper) |

---

## 2. Target configuration (confirmed against paper + reference)

- **Embeddings:** Sentence-T5-XL (the official code + reference use it throughout;
  reproduced Arts to within 0.26 pp). NV-Embed-v2 is **not** used.
- **COSETTE:** codebook 256 × 4 levels, 128-d latent (`COSETTE_128d_256x4`), contrastive
  weight 1e-3, τ=2, bias=−8, dropout 0.1; Arts: `batch_size=1024`, `epochs=3000`
  (~192 steps/epoch → ~576k steps).
- **MARIUS:** full model (`experiment=marius`): temporal d=512 ×4 layers, depth d=512 ×6
  layers, dropout 0.4/0.1, `vocab_size=1026`, best checkpoint by `valid/…/HR@10`,
  `filter_preds=true`. (`marius_small` is for the 2014 datasets, not this one.)
- **SASRec++:** `experiment=sasrec`: d=128 ×2 layers, L2 norm, sampled softmax τ=0.05 with
  30k negatives, `vocab_size=89959`, no quantization.

Dataset: 197,286 users / 89,958 items.

---

## 3. Snellius environment & storage

- **venv:** `/home/scur1250/cosette-venv` (Python 3.9.21). `torch 2.6.0+cu124`,
  `ray 2.44.1`, `pytorch-lightning 2.5.0.post0`, `sentence-transformers 2.7.0` — all
  installed and import-verified. (GPU-level check: `00_verify_gpu.sbatch`.)
- **Storage** (project space `/projects/prjs2120` is **ACL-locked to a teammate**, so not used):
  - code + venv + HF model cache → `$HOME` (200 GiB quota, persistent)
  - bulk data / embeddings / checkpoints → `/scratch-shared/scur1250/cosette_arts`
    (8 TiB; periodically purged → copy final metrics back into git when done)
- **Offline on compute nodes:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; T5-XL
  pre-fetched to `HF_HOME` on the login node (A1).
- **Account:** `gpuuva080` (~2,136,636 SBU remaining as of 2026-06-14).
- **GPUs:** `gpu_h100` = 94 GiB, 192 SBU/GPU·h; `gpu_a100` = 40 GiB, 128 SBU/GPU·h; both
  120 h max walltime; 18 cores/A100, 16 cores/H100.
- **W&B:** offline (no account needed).

---

## 4. Documented deviations (deliberate, faithful)

1. **Seed plumbing.** The reference is single-seed and leaves COSETTE k-means, the
   COSETTE timeline sampler, and the Ray-Data augmentation/sampler unseeded. We added
   purely additive seeding (verified: same seed → identical, different seed → differs):
   k-means `random_state`, seeded generators in `CropAndAugment`/`ToRow`, COSETTE
   DataLoader `worker_init_fn`, and `seed_everything` in the COSETTE GPU worker. Gated on
   `seed` (default `None` = original behaviour). **Does not change the method** — it makes
   the paper's "mean ± std over 5 runs" reproducible. (See `git diff` of the 8 files.)
   - Caveat (in code comments): parallel actors within one Ray-Data op share the op's
     seed (Ray Data exposes no per-actor index), so streams are reproducible per
     (seed, op) but correlated across the op's parallel actors. Affects only
     training-sample ordering, not the method.

2. **Reference recipe applied as per-run overrides** (committed defaults left intact):

   | Override | Value | Repo default | Why |
   |----------|-------|--------------|-----|
   | `optim.batch_size` (COSETTE) | `1024` | 256 | reference intent + paper; literal bs=256 → ~4× steps |
   | `optim.epochs` (COSETTE) | `3000` | 500 | reference Arts command |
   | `ray.scaling_config.num_workers` | `2` | 1 | reference's R@10 5.04 came from 2 workers → **effective batch 512** |
   | `trainer.max_steps` | `81000` | 80000 | reference |
   | `trainer.limit_val_batches` | `1000` | null (full) | reference |

3. **`gpu_h100` (94 GiB) instead of A100.** The reference ran 80 GiB A100s; Snellius
   A100 is 40 GiB. H100 ≥ reference VRAM, so the reference batch sizes fit **without any
   OOM-driven batch-size reduction**. We do **not** shrink batch size to fit 40 GiB.

4. **`model_folder` / `ckpt_dir` / `paths.root` overrides** replace the original author's
   hardcoded machine paths. No method impact.

---

## 5. Run plan

Phases (see [`jobs/arts2023/README.md`](jobs/arts2023/README.md) for commands):

- **0** `00_verify_gpu` — GPU stack check (no data). *Submitted (job 23823033).*
- **A1** download (login) → **A2** parquet (CPU) → **A3** embeddings (GPU×2). Run **once**.
- **Smoke** `smoke_sasrec` (GPU×2, 600 steps) — confirm env/Ray/data/DDP + steps-per-epoch.
- **First full seed** — run B(42) → C(42) and D(42), inspect logs, *then* fan out.
- **B** COSETTE array 42–46 (GPU×1 each) → **C** collisions+MARIUS array 42–46 (GPU×2 each).
- **D** SASRec++ array 42–46 (GPU×2 each) — independent of B/C, run in parallel.
- **E** `aggregate_arts.py` — mean ± std vs paper.

Dependencies: C(seed) needs B(seed) (`runs/seed_<seed>.quant`). Chain with
`--dependency=aftercorr:<B_arrayjobid>`. D needs only A2.

### What to confirm in the smoke / first-seed logs

- **Steps per epoch under `num_workers=2`.** With `total_len = max_steps × batch` and 2
  workers, each worker sees `total_len/2` rows → **~`max_steps/2` steps per epoch**, and
  `max_steps` is reached over ~2 epochs. Smoke (`max_steps=600`) should show ~300
  steps/epoch and finish at 600. *This determines whether C/D do 1× or ~2× the
  step-compute — and therefore the SBU at the high vs low end of the estimate below.*
- COSETTE (B): `~192 steps/epoch` at bs=1024.
- SASRec test runs with `enforce_filtering=false`; MARIUS test with filtering (default).

---

## 6. Runtime & SBU estimates

Estimates are rough (esp. COSETTE/MARIUS wall-time and the `num_workers=2` step-count);
**calibrate from the first full seed.** H100 = 192 SBU/GPU·h.

| Phase | Part. ×GPU | wall (est) | seeds | GPU·h | SBU (H100) |
|-------|-----------|-----------|-------|-------|-----------|
| A2 parquet | genoa CPU | ~0.5–1 h | 1 | — | ~15 (CPU) |
| A3 embeddings | h100 ×2 | ~0.5 h | 1 | ~1 | ~190 |
| B COSETTE | h100 ×1 | ~3–6 h | 5 | 15–30 | ~2,900–5,800 |
| C collisions+MARIUS+test | h100 ×2 | ~3–6 h | 5 | 30–60 | ~5,800–11,500 |
| D SASRec+test | h100 ×2 | ~2–4 h | 5 | 20–40 | ~3,800–7,700 |
| smoke + verify | h100 | — | — | ~0.7 | ~130 |
| **Total (full 5-seed)** | | | | **~66–131** | **≈ 13k–25k SBU** |

- **Budget for ~20k SBU** for the full pipeline (≈1% of the ~2.14M remaining).
- **Cheaper fallback** (one COSETTE shared by all seeds — `B` with `--array=42`, then `C`
  over 5 seeds on the fixed IDs): B drops to ~600–1,150 SBU, total **≈ 10.5k–20.5k SBU**.
  This captures only downstream-training variance, **not** the COSETTE-draw variance, so
  its std understates the true spread. We use the **full** 5-seed pipeline by default.
- **A100 alternative:** 0.67× the rate but ~1.3–1.5× slower → net SBU ≈ similar, **but
  40 GiB risks OOM** on MARIUS/COSETTE → not recommended (would force a batch deviation).

**Wall-clock (full):** critical path A2→A3→B→C ≈ 11–13 h compute; D (~3 h) overlaps B/C;
plus scheduling. Realistically **~1 day** (gpu_h100 had 32 idle nodes at planning time).

---

## 7. Aggregation & paper targets

`python scripts/aggregate_arts.py` prints mean ± std for R@5, NDCG@5, R@10, NDCG@10 vs:

| Method | R@5 | NDCG@5 | R@10 | NDCG@10 |
|--------|-----|--------|------|---------|
| MARIUS (COSETTE) — paper | 3.49 | 2.37 | 5.30 | 2.95 |
| SASRec++ — paper | 3.51 | 2.42 | 5.09 | 2.93 |

Reference single-seed got MARIUS R@10 **5.04** (−0.26 pp); 5-seed averaging is expected
to close the gap toward the paper's reported spread.
