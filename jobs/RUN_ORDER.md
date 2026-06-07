# SASRec++ job order (Amazon 2014)

Each dataset: **3 jobs** — download → parquet → 5-seed train+test+tables.

Data: `${DATA_ROOT}` — Outputs: `${OUTPUT_ROOT}`

---

## Beauty

```bash
cd /home/scur1266/rec_sys_cosette_marius
# optional: source jobs/env_beauty.sh
```

| Step | Command |
|------|---------|
| 1 | `sbatch jobs/01_download_beauty.sbatch` |
| 2 | `sbatch jobs/02_parquet_beauty.sbatch` |
| 3 | `sbatch jobs/03_sasrec_beauty_5seed_full.sbatch` |

Paper test SASRec++: R@5 6.66±0.08, NDCG@5 4.58±0.08, R@10 9.73±0.10, NDCG@10 5.57±0.04

- Vocab **12103**, seeds **42–46**, ~6–8 h, Slurm **12 h**
- Results: `results/sasrec_beauty_5seed_full_scores.jsonl`, `table5_beauty_5seed_*_full_latest.txt`
- Re-submit job 03 skips seeds already `"status": "ok"` in the scores file

---

## Sports_and_Outdoors

```bash
cd /home/scur1266/rec_sys_cosette_marius
# optional: source jobs/env_sports.sh
```

| Step | Command |
|------|---------|
| 1 | `sbatch jobs/01_download_sports.sbatch` |
| 2 | `sbatch jobs/02_parquet_sports.sbatch` |
| 3 | `sbatch jobs/03_sasrec_sports_5seed_full.sbatch` |

Paper test SASRec++: R@5 4.37±0.09, NDCG@5 2.96±0.05, R@10 6.44±0.10, NDCG@10 3.62±0.04

- Vocab **18359**, seeds **42–46**, ~7–10 h, Slurm **14 h**
- Results: `results/sasrec_sports_5seed_full_scores.jsonl`, `table5_sports_5seed_*_full_latest.txt`

---

## Shared config

- `jobs/env_common.sh` — shared paths, conda, WandB, training defaults
- `jobs/env_beauty.sh` / `jobs/env_sports.sh` — dataset category + vocab (each sbatch job sources the matching one)
- `jobs/wandb.env.example` → copy to `jobs/wandb.env` (gitignored)

Training: 80k steps, batch 256, 1 GPU, full validation, WandB names `SASRec_{Category}_seed{N}_...`

---

## COSETTE / MARIUS pipeline (MARIUS row in Table 5)

After **01 download** + **02 parquet** (same as SASRec). Embeddings are **one-time per dataset** and reused for COSETTE → collision removal → MARIUS.

### Beauty

| Step | Command | Notes |
|------|---------|--------|
| 4 | `sbatch jobs/04_embeddings_beauty.sbatch` | ~6 min; skips if already built |
| 5 | `sbatch jobs/05_cosette_beauty.sbatch` | ~30 min; 1000 epochs, 1 GPU |
| 6 | `sbatch jobs/06_remove_collisions_beauty.sbatch` | ~5–15 min; quant `COSETTE_128d_256x4_f958` → `-col` |
| 7 | `sbatch jobs/07_marius_beauty_5seed_full.sbatch` | **5 seeds**, ~6–10 h total |

Output: `${DATA_ROOT}/data/embeddings/sentence-t5-xl/Beauty/embeddings.parquet`

Beauty COSETTE run: **`COSETTE_128d_256x4_f958`** → after job 06: **`COSETTE_128d_256x4_f958-col`**

Results: `marius_beauty_5seed_full_scores.jsonl`, `table5_beauty_marius_5seed_*_full_latest.txt`

### Sports_and_Outdoors

| Step | Command | Notes |
|------|---------|--------|
| 4 | `sbatch jobs/04_embeddings_sports.sbatch` | ~9 min; skips if already built |
| 5 | `sbatch jobs/05_cosette_sports.sbatch` | ~50 min; 1000 epochs |
| 6 | `sbatch jobs/06_remove_collisions_sports.sbatch` | ~10–20 min; quant `COSETTE_128d_256x4_8ed1` → `-col` |
| 7 | `sbatch jobs/07_marius_sports_5seed_full.sbatch` | **5 seeds**, ~5–9 h total |

Sports COSETTE run: **`COSETTE_128d_256x4_8ed1`** → after job 06: **`COSETTE_128d_256x4_8ed1-col`**

Results: `marius_sports_5seed_full_scores.jsonl`, `table5_sports_marius_5seed_*_full_latest.txt`

Output: `${DATA_ROOT}/data/embeddings/sentence-t5-xl/Sports_and_Outdoors/embeddings.parquet`

First run downloads `sentence-transformers/sentence-t5-xl` from HuggingFace (~3 GB).

---

## Extensions (optional, after replication)

Inference time + GPU memory on **existing checkpoints** (seed 42, ~30 min each):

| Dataset | Command |
|---------|---------|
| Beauty | `sbatch jobs/08_extensions_beauty.sbatch` |
| Sports | `sbatch jobs/08_extensions_sports.sbatch` |

Outputs: `reports/extensions/inference_benchmark.csv`, `wall_clock.json`, `model_vocab.json`  
Notebook: `notebooks/extended_report.ipynb`
