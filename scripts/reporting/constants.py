"""Paper targets and replication layout shared by export script and notebook."""

from __future__ import annotations

EXPECTED_SEEDS = [42, 43, 44, 45, 46]

CATEGORIES = {
    "Beauty": {"slug": "beauty"},
    "Sports_and_Outdoors": {"slug": "sports"},
}

PAPER_SASREC_PP: dict[str, dict[str, tuple[float, float]]] = {
    "Beauty": {
        "R@5": (6.66, 0.08),
        "NDCG@5": (4.58, 0.08),
        "R@10": (9.73, 0.10),
        "NDCG@10": (5.57, 0.04),
    },
    "Sports_and_Outdoors": {
        "R@5": (4.37, 0.09),
        "NDCG@5": (2.96, 0.05),
        "R@10": (6.44, 0.10),
        "NDCG@10": (3.62, 0.04),
    },
}

PAPER_MARIUS_COSETTE: dict[str, dict[str, tuple[float, float]]] = {
    "Beauty": {
        "R@5": (6.58, 0.09),
        "NDCG@5": (4.35, 0.07),
        "R@10": (10.02, 0.08),
        "NDCG@10": (5.46, 0.05),
    },
    "Sports_and_Outdoors": {
        "R@5": (4.31, 0.08),
        "NDCG@5": (2.83, 0.06),
        "R@10": (6.72, 0.08),
        "NDCG@10": (3.62, 0.06),
    },
}

METRIC_COLUMNS = ["R@5", "NDCG@5", "R@10", "NDCG@10"]

COSETTE_RUNS = {
    "Beauty": "COSETTE_128d_256x4_f958-col",
    "Sports_and_Outdoors": "COSETTE_128d_256x4_8ed1-col",
}

# Checklist rows for the notebook (artifact paths are resolved at runtime).
PIPELINE_STEPS = [
    ("download", "Raw Amazon 2014 data"),
    ("parquet", "Timelines parquet"),
    ("embeddings", "Sentence-T5-XL embeddings"),
    ("cosette", "COSETTE tokenizer"),
    ("collisions", "Collision removal (-col)"),
    ("sasrec", "SASRec++ 5-seed (full)"),
    ("marius", "MARIUS (COSETTE) 5-seed (full)"),
]
