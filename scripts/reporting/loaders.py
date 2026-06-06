"""Load replication scores and compute summary statistics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .constants import CATEGORIES, COSETTE_RUNS, EXPECTED_SEEDS, METRIC_COLUMNS


def load_scores_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                records.append(record)
    return records


def summarize_method(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    summary: dict[str, Any] = {"n_seeds": len(records), "seeds": sorted(int(r["seed"]) for r in records)}
    for metric in METRIC_COLUMNS:
        values = [float(r[metric]) for r in records if metric in r]
        if not values:
            continue
        summary[f"{metric}_mean"] = statistics.mean(values)
        summary[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
    return summary


def _exists(path: Path) -> bool:
    return path.exists()


def pipeline_status(
    *,
    data_root: Path,
    output_root: Path | None,
    results_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(step: str, label: str, ok: bool, detail: str) -> None:
        rows.append(
            {
                "step": step,
                "label": label,
                "status": "done" if ok else "pending",
                "detail": detail,
            }
        )

    for category, meta in CATEGORIES.items():
        slug = meta["slug"]

        raw_glob = list((data_root / "amazon-2014").glob(f"*{category}*"))
        if not raw_glob and category == "Sports_and_Outdoors":
            raw_glob = list((data_root / "amazon-2014").glob("*Sports*"))
        add(
            f"download_{slug}",
            f"Download ({category})",
            bool(raw_glob),
            str(raw_glob[0]) if raw_glob else f"missing under {data_root / 'amazon-2014'}",
        )

        timelines = data_root / "data" / "timelines" / f"{category}.train.parquet"
        add(
            f"parquet_{slug}",
            f"Parquet ({category})",
            _exists(timelines),
            str(timelines),
        )

        emb = (
            data_root
            / "data"
            / "embeddings"
            / "sentence-t5-xl"
            / category
            / "embeddings.parquet"
        )
        add(
            f"embeddings_{slug}",
            f"Embeddings ({category})",
            _exists(emb),
            str(emb),
        )

        quant_id = COSETTE_RUNS[category]
        cosette_ckpt = (
            data_root
            / "data"
            / "embeddings"
            / "sentence-t5-xl"
            / category
            / quant_id.replace("-col", "")
        )
        cosette_col = (
            data_root
            / "data"
            / "embeddings"
            / "sentence-t5-xl"
            / category
            / f"{quant_id}.parquet"
        )
        add(
            f"cosette_{slug}",
            f"COSETTE ({category})",
            _exists(cosette_ckpt) or _exists(cosette_col),
            quant_id,
        )
        add(
            f"collisions_{slug}",
            f"Collision removal ({category})",
            _exists(cosette_col),
            str(cosette_col),
        )

        sasrec_scores = results_dir / f"sasrec_{slug}_5seed_full_scores.jsonl"
        sasrec_records = load_scores_jsonl(sasrec_scores)
        sasrec_ok = len(sasrec_records) >= len(EXPECTED_SEEDS)
        add(
            f"sasrec_{slug}",
            f"SASRec++ 5-seed ({category})",
            sasrec_ok,
            f"{len(sasrec_records)}/{len(EXPECTED_SEEDS)} seeds in {sasrec_scores.name}",
        )

        marius_scores = results_dir / f"marius_{slug}_5seed_full_scores.jsonl"
        marius_records = load_scores_jsonl(marius_scores)
        marius_ok = len(marius_records) >= len(EXPECTED_SEEDS)
        add(
            f"marius_{slug}",
            f"MARIUS 5-seed ({category})",
            marius_ok,
            f"{len(marius_records)}/{len(EXPECTED_SEEDS)} seeds in {marius_scores.name}",
        )

    if output_root is not None:
        _ = output_root  # reserved for future model-dir checks

    return rows
