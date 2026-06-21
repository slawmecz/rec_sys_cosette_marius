# Notebooks

The canonical notebooks for the project. Extension results are consolidated in
[`EXTENSION_RESULTS.md`](../EXTENSION_RESULTS.md); the reproduction in
[`REPLICATION_REPORT.md`](../REPLICATION_REPORT.md).

## `extension_story.ipynb`

The complete extension story (RQ1-4): reproducibility at scale, the
beyond-accuracy / popularity-bias characterization, the model-bound diagnosis
(exact full-catalog oracle), and mitigation across pipeline stages with the
accuracy-vs-bias trade-off. Built by
`scripts/extensions/build_extension_story_notebook.py`; figures come from
`scripts/extensions/extension_figures.py` (also a CLI that writes
`reports/figures/fig_rq*.{png,svg}`).

## `replication_report.ipynb`

Table 5 metrics, pipeline status, and learning curves for Beauty and Sports
(the mandatory reproduction).

## `codebook_entropy.ipynb`

The codebook-health audit (256/256 codes used, near-uniform usage entropy) that
exonerates the tokenizer as a cause of the collapse (supports RQ3).

## `extended_report.ipynb`

Extra tables/plots: inference time & GPU memory, training wall-clock, val-test gap, vocab sizes, COSETTE collisions, per-seed spread. GPU benchmarks skipped by default - run `sbatch jobs/08_extensions_beauty.sbatch` and `08_extensions_sports.sbatch` first.

---

## Replicating experiments (full pipeline)

See also `jobs/RUN_ORDER.md` and `REPRODUCIBILITY.md`.

### Snellius (recommended)

Replace `/home/scur1266` with your username everywhere (especially `#SBATCH --output` / `--error` in `jobs/*.sbatch`).

```bash
cd /home/<USER>/rec_sys_cosette_marius
conda activate recsys
export SCRATCH=/home/<USER>/scratch
cp jobs/wandb.env.example jobs/wandb.env   # optional
```

**Per dataset (Beauty or Sports), submit in order:**

| Step | Beauty | Sports |
|------|--------|--------|
| 1 Download | `sbatch jobs/01_download_beauty.sbatch` | `sbatch jobs/01_download_sports.sbatch` |
| 2 Parquet | `sbatch jobs/02_parquet_beauty.sbatch` | `sbatch jobs/02_parquet_sports.sbatch` |
| 3 SASRec++ 5-seed | `sbatch jobs/03_sasrec_beauty_5seed_full.sbatch` | `sbatch jobs/03_sasrec_sports_5seed_full.sbatch` |
| 4 Embeddings | `sbatch jobs/04_embeddings_beauty.sbatch` | `sbatch jobs/04_embeddings_sports.sbatch` |
| 5 COSETTE | `sbatch jobs/05_cosette_beauty.sbatch` | `sbatch jobs/05_cosette_sports.sbatch` |
| 6 Collision removal | `sbatch jobs/06_remove_collisions_beauty.sbatch` | `sbatch jobs/06_remove_collisions_sports.sbatch` |
| 7 MARIUS 5-seed | `sbatch jobs/07_marius_beauty_5seed_full.sbatch` | `sbatch jobs/07_marius_sports_5seed_full.sbatch` |

Steps 1-3: SASRec++ baseline. Steps 4-7: COSETTE -> MARIUS. Jobs 03 and 07 skip seeds already recorded as `"status": "ok"`.

After job 05, note the `COSETTE_128d_256x4_*` run id for job 06 and the `-col` id for job 07.

### Export results for the notebook

```bash
source jobs/wandb.env          # only for --source wandb
conda activate recsys
export OUTPUT_ROOT=$SCRATCH/cosette_marius/outputs
python scripts/export_metrics_for_report.py --copy-results --source wandb
jupyter notebook notebooks/replication_report.ipynb
```

### Manual / interactive runs

Section 0 of the notebook lists the same shell commands as the job files (markdown). Copy into a code cell and run one step at a time on a GPU node if not using Slurm.

---

## WandB

| Task | Need `jobs/wandb.env`? |
|------|-------------------------|
| Open notebook (`reports/` in repo) | No |
| Training (sbatch jobs) | Optional (logging) |
| `export_metrics_for_report.py --source wandb` | Yes |

Copy `jobs/wandb.env.example` -> `jobs/wandb.env`; set `WANDB_API_KEY` (https://wandb.ai/authorize) and `WANDB_ENTITY`.
