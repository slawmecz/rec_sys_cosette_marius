#!/usr/bin/env python3
"""Generate notebooks/extension_story.ipynb: the complete RQ1-4 extension story.

The narrative (markdown) plus every figure, where the figures come from the shared
library scripts/extensions/extension_figures.py (single source of truth, also a CLI
that exports the PNGs/SVGs to reports/figures/). The notebook loads only committed
artifacts under reports/, so it re-executes anywhere the repo is checked out; no GPU.

This is the consolidated story for the paper: reproduction at scale (RQ1), the
beyond-accuracy / popularity-bias characterization (RQ2), the model-bound diagnosis
via the exact oracle (RQ3), and mitigation across pipeline stages with the
accuracy-vs-bias trade-off (RQ4).

Run: python3 scripts/extensions/build_extension_story_notebook.py
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

# =========================================================================== #
md(
    "# COSETTE & MARIUS: a beyond-accuracy study of generative recommendation\n"
    "\n"
    "This notebook is the consolidated extension story for our reproduction of "
    "*Closing the Performance Gap in Generative Recommenders with Collaborative "
    "Tokenization and Efficient Modeling* (Lepage, Mary, Picard; arXiv:2508.14910). "
    "It is organized around four research questions, following the FairDiverse "
    "(SIGIR'25) pre/in/post-processing taxonomy for the mitigation half.\n"
    "\n"
    "**RQ1.** Is COSETTE/MARIUS reproducible against SASRec++, and how does the "
    "verdict change with catalog scale?\n\n"
    "**RQ2.** How do the discriminative (SASRec++) and generative (MARIUS) paradigms "
    "compare on diversity and popularity bias (coverage, Gini, entropy, ILD)?\n\n"
    "**RQ3.** What in the generative pipeline causes the popularity bias - the "
    "semantic-ID distribution, the training data, the model, or the beam search?\n\n"
    "**RQ4.** How well can the bias be mitigated at each pipeline stage (training loss, "
    "beam, re-ranking, cross-paradigm distillation), and at what accuracy cost?\n"
    "\n"
    "Every figure is computed from committed artifacts in `reports/`. Model accuracy "
    "comes from the authors' unmodified evaluation code; beyond-accuracy metrics are "
    "recomputed from the models' own ranked outputs and match the official Recall@10 "
    "to the decimal. Datasets: Amazon-2014 Beauty (12k items) and Sports (18k), and "
    "Amazon-2023 Arts_Crafts_and_Sewing (90k) for the large-scale prong.\n"
    "\n"
    "All numbers here are cross-checked in `EXTENSION_RESULTS.md`."
)

code(
    "import sys\n"
    "from pathlib import Path\n"
    "import numpy as np, pandas as pd\n"
    "\n"
    "REPO = Path.cwd()\n"
    "while not (REPO / 'scripts' / 'extensions' / 'extension_figures.py').exists() and REPO != REPO.parent:\n"
    "    REPO = REPO.parent\n"
    "sys.path.insert(0, str(REPO))\n"
    "from scripts.extensions import extension_figures as ef\n"
    "import matplotlib.pyplot as plt\n"
    "from IPython.display import display\n"
    "\n"
    "# Figures are defined once in extension_figures.py and reused here. The same\n"
    "# module, run as a script, writes every figure to reports/figures/ as PNG+SVG\n"
    "# for direct inclusion in the paper:  python scripts/extensions/extension_figures.py\n"
    "def show(fig):\n"
    "    \"\"\"Render a figure exactly once (display, then close so the inline backend\n"
    "    does not also auto-flush it).\"\"\"\n"
    "    display(fig); plt.close(fig)\n"
    "\n"
    "pd.set_option('display.float_format', lambda v: f'{v:.4f}')"
)

# =========================================================================== #
md(
    "## RQ1. Reproducibility, and the small-to-large flip\n"
    "\n"
    "The paper claims MARIUS+COSETTE is competitive with the tuned SASRec++ baseline "
    "at small scale and beats it at large scale. We reproduce both ends.\n"
    "\n"
    "On the small 2014 datasets the claim does **not** reproduce: a paper-faithful "
    "SASRec++ (per-dataset sizing, no L2 normalization) beats MARIUS on Beauty "
    "(9.06 vs 8.17 Recall@10) and Sports (5.15 vs 4.87). But on the large 2023 Arts "
    "dataset (90k items) the claim **does** reproduce: MARIUS 5.01 vs SASRec++ 4.86, "
    "a separation that holds across all five seeds (exact permutation p=0.008). The "
    "generative paradigm's advantage is scale-dependent."
)
code("show(ef.fig_rq1_reproduction())")

# =========================================================================== #
md(
    "## RQ2. Diversity and popularity bias\n"
    "\n"
    "From each model's ranked top-10 we measure catalog coverage (fraction of items "
    "ever recommended), the Gini coefficient of exposure, intra-list diversity (ILD, "
    "the average pairwise content distance within a list), and APLT (the average share "
    "of long-tail items per list).\n"
    "\n"
    "The result is more specific than 'MARIUS is less diverse'. MARIUS's individual "
    "lists are about as diverse as SASRec's (ILD tied) and lean **more** to the long "
    "tail (higher APLT, lower ARP). Yet in **aggregate** it reaches far fewer distinct "
    "items (Arts coverage 0.40 vs 0.59; Sports 0.39 vs 0.74) at equal-or-higher Gini. "
    "This is the aggregate-vs-individual diversity distinction (Adomavicius & Kwon "
    "2012): MARIUS serves a reasonably varied but largely overlapping popular slice of "
    "the catalog across users. The collapse is worst on the sparsest data and persists "
    "at 90k scale."
)
code("show(ef.fig_rq2_beyond_accuracy())")
code(
    "# The Arts (90k) beyond-accuracy table, 5-seed mean (k=10).\n"
    "ab = ef.arts_beyond(); reach = ef.arts_reach()\n"
    "cols = ['coverage', 'gini', 'entropy_norm', 'ild', 'aplt', 'arp', 'recall', 'tail_recall']\n"
    "tbl = ab[ab.k == 10].set_index('model')[cols].copy()\n"
    "tbl['chao1_ceiling'] = [reach.loc[(m, 10), 'chao1_coverage_mean'] for m in tbl.index]\n"
    "tbl.round(4)"
)

# =========================================================================== #
md(
    "## RQ3. Where does the bias come from? The exact-oracle diagnosis\n"
    "\n"
    "A generative recommender can only surface an item whose semantic-ID code its beam "
    "writes, so the collapse could live in the codebook, the data, the model's learned "
    "ranking, or the beam search. We settle it with an **exact full-catalog oracle**: "
    "for ~1000 test users we teacher-force MARIUS to score *every* catalog item's code "
    "tuple, giving the model's exact ranking independent of the beam.\n"
    "\n"
    "The verdict is **model-bound**. Exact scoring recovers essentially the same "
    "Recall@10 as the beam (so the beam is near-optimal over the model - not the "
    "culprit), while the true held-out items sit at a median exact rank of ~1956 out "
    "of 90k on Arts (and ~1000/18k on Sports). The model's learned ranking buries the "
    "tail. This replicates the small-scale finding at 90k and refutes accounts that "
    "blame premature beam pruning (e.g. SimGR). The codebook audit (256/256 codes used, "
    "near-uniform) independently rules out the tokenizer.\n"
    "\n"
    "The distilled checkpoint (see RQ4) is shown here too: distillation spreads exposure "
    "but pushes the true targets *deeper* (median 1000 to 1405), so the verdict stays "
    "model-bound even after the strongest intervention."
)
code("show(ef.fig_rq3_oracle())")
code(
    "# Oracle verdicts (filtered where available; distilled is unfiltered-only).\n"
    "import json\n"
    "def verdict(p):\n"
    "    d = json.loads(Path(p).read_text()); k10 = {r['k']: r for r in d['per_k']}[10]\n"
    "    return {'verdict': d['verdict'], 'recall_exact@10': k10['recall_exact'],\n"
    "            'recall_beam@10': k10['recall_beam'],\n"
    "            'median_target_rank': d['target_exact_rank_median_all'],\n"
    "            'n_catalog': int(np.load(str(p).replace('.json', '.npz'))['n_catalog'])}\n"
    "rows = {\n"
    "    'Arts MARIUS': ef.ARTS_DIR / 'seed42' / 'exact_catalog' / f'exact_catalog_{ef.ARTS}_seed42_filtered.json',\n"
    "    'Sports baseline': ef.EXT / 'exact_catalog' / 'exact_catalog_Sports_and_Outdoors_seed42_filtered.json',\n"
    "    'Sports distilled': ef.EXT / 'exact_catalog_distill' / 'exact_catalog_Sports_and_Outdoors_seed42.json',\n"
    "}\n"
    "pd.DataFrame({k: verdict(v) for k, v in rows.items()}).T"
)

# =========================================================================== #
md(
    "## RQ4. Mitigation across pipeline stages, and the trade-off\n"
    "\n"
    "We test one mitigation per FairDiverse stage and compare their accuracy-vs-bias "
    "trade-offs:\n"
    "\n"
    "- **in-processing** (training loss): logit-adjusted cross-entropy (MARIUSLogitAdj), "
    "a per-token popularity correction, swept over tau on Beauty and Sports;\n"
    "- **beam stage**: MBR re-ranking of the depth-100 candidates, swept over tau;\n"
    "- **post-processing**: PMI / logit-adjusted decode-time re-rank (priors cond2 / pair "
    "/ item), swept over alpha, on Arts;\n"
    "- **cross-paradigm**: KL distillation of a frozen SASRec++ teacher into MARIUS.\n"
    "\n"
    "The unified result: **every lever is a bounded dial** that buys catalog coverage "
    "with recall, and **none moves the model-bound ceiling**. Panel A shows the clean "
    "Arts Pareto for the re-rankers; panel B normalizes every arm to its own baseline "
    "so all four stages share one axis. Distillation is the extreme - it nearly matches "
    "the teacher's spread (coverage 0.39 to 0.87, +123%) but costs 22% of Recall@10, "
    "and the RQ3 oracle shows it does this by smearing mass, not by ranking the correct "
    "items higher (their exact rank gets *worse*). Aggregate coverage can be inflated "
    "without improving the model's ability to serve the tail; only the oracle "
    "distinguishes genuine reach from cosmetic spreading."
)
code("show(ef.fig_rq4_pareto())")
code(
    "# Representative trade-off points (% change vs each arm's own baseline, k=10).\n"
    "out = []\n"
    "# post: PMI cond2 @ alpha 0.5 (Arts)\n"
    "p = ef.arts_pmi('cond2'); b = p.loc[0.0]\n"
    "out.append(['post: PMI cond2 a=0.5 (Arts)', 100*(p.loc[0.5].recall-b.recall)/b.recall,\n"
    "            100*(p.loc[0.5].coverage-b.coverage)/b.coverage])\n"
    "# beam: MBR tau1.0 (Arts)\n"
    "bm, dm = ef.arts_mbr()\n"
    "out.append(['beam: MBR tau=1.0 (Arts)', 100*(dm.loc[1.0].recall-bm['recall'])/bm['recall'],\n"
    "            100*(dm.loc[1.0].coverage-bm['coverage'])/bm['coverage']])\n"
    "# in: logit-adj tau1.5 (Sports)\n"
    "lb, ld = ef.logitadj('sports', 'Sports_and_Outdoors')\n"
    "out.append(['in: logit-adj tau=1.5 (Sports)', 100*(ld.loc[1.5].recall_mean-lb.recall_mean)/lb.recall_mean,\n"
    "            100*(ld.loc[1.5].coverage_mean-lb.coverage_mean)/lb.coverage_mean])\n"
    "# distillation (Sports)\n"
    "db, dd = ef.distill()\n"
    "out.append(['cross-paradigm: distillation (Sports)', 100*(dd.recall_mean-db.recall_mean)/db.recall_mean,\n"
    "            100*(dd.coverage_mean-db.coverage_mean)/db.coverage_mean])\n"
    "pd.DataFrame(out, columns=['arm (stage)', 'dRecall@10 %', 'dCoverage %']).round(1)"
)

# =========================================================================== #
md(
    "## Synthesis\n"
    "\n"
    "| Question | Finding |\n"
    "|---|---|\n"
    "| RQ1 Reproducible? | Yes, scale-dependent: MARIUS < SASRec++ on 2014 (Beauty 8.17 vs 9.06), MARIUS > SASRec++ on Arts-2023 (5.01 vs 4.86, p=0.008). |\n"
    "| RQ2 Diversity / bias? | Aggregate collapse (Arts coverage 0.40 vs 0.59, Gini ~0.93) despite competitive per-list diversity (ILD tied, APLT higher). Worsens with sparsity; persists at 90k. |\n"
    "| RQ3 Source? | The model's learned ranking. Exact oracle ~= beam (not search); true targets at median exact rank ~1956/90k; codebook audit clears the tokenizer. |\n"
    "| RQ4 Mitigation? | Every stage (loss / beam / re-rank / distillation) is a bounded dial trading recall for coverage; none lifts the ceiling. Distillation spreads most (coverage +123%) but buries the true targets deeper. |\n"
    "\n"
    "The through-line: in semantic-ID generative recommendation the popularity collapse "
    "is a property of the trained model's ranking, not of decoding or tokenization, and "
    "popularity-correction levers redistribute exposure without expanding genuine reach. "
    "Measuring aggregate coverage alone is misleading; the exact oracle is needed to tell "
    "real reach from cosmetic spreading.\n"
    "\n"
    "Verified numbers and provenance: `EXTENSION_RESULTS.md`. Reproduction methodology: "
    "`REPLICATION_REPORT.md`, `FAITHFUL_RERUN.md`. The earlier diagnostic log is "
    "`EXTENSION_SUMMARY.md`. Figures are regenerated by "
    "`python scripts/extensions/extension_figures.py`."
)

nb["cells"] = cells
out = Path(__file__).resolve().parents[2] / "notebooks" / "extension_story.ipynb"
nbf.write(nb, str(out))
print("wrote", out)
