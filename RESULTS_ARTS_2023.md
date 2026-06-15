# Results overview: Arts, Crafts & Sewing (Amazon Reviews 2023)

Single-seed (seed 42) calibration results for our from-scratch reproduction,
compared against the master TA's single-seed reproduction and the paper. This is
the calibration checkpoint before the 5-seed fan-out (seeds 44, 46, 48, 50); it is
updated to mean +/- std once those land. Seeds are spaced by 2 so that, under
`seed_everything(seed + world_rank)` with `num_workers=2`, no two runs share a worker
RNG stream (consecutive seeds would overlap, correlating runs and understating std).

- Our runs: Snellius `gpu_h100` (94 GiB), venv `cosette-venv` (torch 2.6.0+cu124, ray 2.44.1).
- TA reference: [`2t2c/cosette_and_marius`](https://github.com/2t2c/cosette_and_marius)
  (`REPRODUCIBILITY.md`), single seed, 2x A100 80 GiB. Verified current at commit `9a9d8e4`.
- Paper: [arXiv:2508.14910](https://arxiv.org/abs/2508.14910), 5 seeds, mean reported.
- Dataset: 197,286 users / 89,958 items.

All numbers are **test** metrics in percent.

---

## 1. MARIUS (COSETTE) - test

| Source | R@5 | NDCG@5 | R@10 | NDCG@10 |
|--------|-----|--------|------|---------|
| Paper (5-seed mean) | 3.49 | 2.37 | **5.30** | 2.95 |
| TA reproduction (1 seed) | - | - | **5.04** | - |
| **Ours (seed 42)** | 3.31 | 2.23 | **5.00** | 2.77 |

R@10 percentage-point differences:

| Comparison | R@10 delta (pp) |
|------------|-----------------|
| Ours vs TA reproduction | **-0.04** (on target) |
| Ours vs paper | **-0.30** |
| TA reproduction vs paper | **-0.26** |

- Our seed-42 MARIUS lands essentially **on top of the TA's single-seed result**
  (5.00 vs 5.04, -0.04 pp). Our gap to the paper (-0.30 pp) is the same single-seed
  shortfall the TA observed (-0.26 pp); the paper's number is a 5-seed mean.
- The TA reported only HR@10 for Arts (no R@5 / NDCG breakdown), so those cells are
  blank. Our full row is provided for completeness.

---

## 2. SASRec++ - test

| Source | R@5 | NDCG@5 | R@10 | NDCG@10 |
|--------|-----|--------|------|---------|
| Paper (5-seed mean) | 3.51 | 2.42 | **5.09** | 2.93 |
| TA reproduction | not run (open TODO) | | | |
| **Ours (seed 42)** | 3.37 | 2.32 | **4.91** | 2.82 |

R@10 percentage-point differences:

| Comparison | R@10 delta (pp) |
|------------|-----------------|
| Ours vs paper | **-0.18** |
| Ours vs TA reproduction | n/a (TA did not run SASRec on Arts) |

- **The TA never ran SASRec++ on Arts** - it is listed as an open TODO in their
  `REPRODUCIBILITY.md`. Our SASRec Arts result is therefore a **new contribution**
  with no TA baseline, compared only against the paper.
- Our R@10 sits -0.18 pp under the paper's 5-seed mean - a normal single-seed
  shortfall, slightly tighter than our MARIUS gap. It required the `vocab_size`
  off-by-one bugfix (see deviation #5 in `REPRO_ARTS_2023.md`); without it the run
  aborts with a CUDA device-side assert.

---

## 3. Procedure conformance vs the TA repository

A `diff -rq` of our `configs/` and `data_scripts/` against the reference clone
(current at `origin/main` `9a9d8e4`) shows the code paths are identical; the only
differences are our deliberate, documented deviations:

| Stage | TA recipe (Arts) | Ours | Note |
|-------|------------------|------|------|
| download | `download_data.py --categories Arts_Crafts_and_Sewing` | same | A1 |
| parquet | `0_raw_to_parquet.py paths.skip_download=true` | same | A2 |
| embeddings | `1_make_embeddings.py category=...` (Sentence-T5-XL) | same | A3 |
| COSETTE | `2_train_cosette.py ... optim.epochs=3000` | `+ optim.batch_size=1024` | see below |
| collisions | `3_remove_colisions.py ...` | same | C |
| MARIUS | `experiment=marius` (full d=512) | same | C |
| SASRec | (not run) | `experiment=sasrec` `+ model.net.vocab_size=89960` | D, new |

**COSETTE batch-size ambiguity in the TA recipe.** `configs/2_train_cosette.yaml`
defaults to `batch_size: 256`, with a comment targeting ~600k steps
("192 steps/epoch -> 3125 epochs = 600k steps, adapt to category"). The TA's Arts
command passes only `epochs=3000` (no batch override), which at the default bs=256
does not land on that step budget, while their section heading says "bs=1024". We
resolve the ambiguity in favour of the config comment's intent: `bs=1024 +
epochs=3000 = 576k steps` (empirically confirmed at 192 steps/epoch). Our MARIUS
R@10 matching the TA's (5.00 vs 5.04) validates this reading.

**Seeding.** The released reference code does not seed training - `seed` is only
logged as an hparam, never applied (no `seed_everything`). The paper states "5 seeds"
but does not publish the seed values. Our additive seed plumbing (deviation #1) is
therefore our own; matching the paper's exact seeds is not possible and not required
to reproduce a 5-seed mean +/- std. See `REPRO_ARTS_2023.md` section 4 for the seed
caveat.

Other documented deviations (full detail in `REPRO_ARTS_2023.md` section 4): reference
recipe applied as per-run overrides with committed defaults intact (#2), `gpu_h100`
instead of 40 GiB A100 so reference batch sizes fit without OOM-driven reduction (#3),
path overrides replacing hardcoded machine paths (#4), and the SASRec `vocab_size`
fix (#5).

---

## 4. Cost (seed 42, measured)

| Stage | GPUs | Wall | GPU.h | SBU (H100, 192/GPU.h) |
|-------|------|------|-------|------------------------|
| COSETTE (B) | 1 | 2:52 | 2.87 | ~551 |
| MARIUS (C, incl. collisions + test) | 2 | 1:48 | 3.59 | ~690 |
| SASRec++ (D, incl. test) | 2 | 1:06 | 2.20 | ~220 |

Projected full 5-seed total **~8,000 SBU**, well under the 25k flag and ~1% of the
~2.13M account budget remaining.

---

## 5. Status

- Seed 42 (calibration) complete for both MARIUS and SASRec++; both land within the
  expected single-seed shortfall of the paper, and MARIUS matches the TA's
  single-seed Arts result to within 0.04 pp.
- **Next:** fan out seeds 44, 46, 48, 50 (full B+C+D pipeline) and re-run `scripts/aggregate_arts.py`
  to produce the mean +/- std this table is designed to hold.

Provenance: `runs/seed_42.{quant,marius,sasrec}`. Regenerate this comparison with
`source jobs/arts2023/env_arts.sh && python scripts/aggregate_arts.py`.
