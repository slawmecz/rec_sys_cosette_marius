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
