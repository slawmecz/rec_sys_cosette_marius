#!/usr/bin/env python3
"""GPU inference benchmarks + extension artifacts for extended_report.ipynb."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import fsspec
import hydra
import pytorch_lightning as L
import ray
import torch
from omegaconf import OmegaConf

from scripts.reporting.constants import CATEGORIES, COSETTE_RUNS, EXPECTED_SEEDS
from scripts.reporting.loaders import load_scores_jsonl
from src.utils.tools import patch_fsspec

FILENAMES = {
    "config": "config.yaml",
    "checkpoint": "checkpoint.ckpt",
    "progress": "checkpoint_manager_snapshot.json",
}

MODEL_VOCAB = {
    "Beauty": {"sasrec_vocab": 12103, "marius_code_vocab": 1026},
    "Sports_and_Outdoors": {"sasrec_vocab": 18359, "marius_code_vocab": 1026},
}


def get_best_checkpoint(models_root: Path, run_directory: str) -> tuple[OmegaConf, str]:
    root = models_root / run_directory
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")

    ckpt_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")]
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint_* folders under {root}")

    with (root / FILENAMES["progress"]).open() as handle:
        progress = json.load(handle)

    metric_name = None
    with (ckpt_dirs[0] / FILENAMES["config"]).open() as handle:
        probe = OmegaConf.load(handle)
    metric_name = probe.ray.run_config.checkpoint_config.checkpoint_score_attribute

    best_name = None
    best_metric = float("-inf")
    for ckpt in progress["checkpoint_results"]:
        if ckpt["metrics"][metric_name] > best_metric:
            best_metric = ckpt["metrics"][metric_name]
            best_name = ckpt["checkpoint_dir_name"]

    best_dir = root / best_name
    with (best_dir / FILENAMES["config"]).open() as handle:
        cfg = OmegaConf.load(handle)

    ckpt_path = str(best_dir / FILENAMES["checkpoint"])
    return cfg, ckpt_path


def timed_eval(cfg: OmegaConf, ckpt_path: str, *, split: str) -> tuple[float, float, int]:
    patch_fsspec()
    cfg.data.ray_datasets.which = ["valid", "test"]
    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    model.full_hydra_config = cfg
    n_params = sum(p.numel() for p in model.parameters())

    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    protocol = str(cfg.paths.protocol)
    ckpt_uri = ckpt_path if "://" in ckpt_path else f"{protocol}://{ckpt_path}"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    fn = trainer.validate if split == "valid" else trainer.test
    fn(model, datamodule=datamodule, ckpt_path=ckpt_uri)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return elapsed, mem_gb, n_params


def run_directory_for_seed(
    results_dir: Path,
    *,
    method: str,
    slug: str,
    seed: int,
) -> str:
    fname = f"{method}_{slug}_5seed_full_scores.jsonl"
    for record in load_scores_jsonl(results_dir / fname):
        if int(record["seed"]) == int(seed):
            return str(record["run_directory"])
    raise FileNotFoundError(f"No run_directory for {method} seed {seed} in {fname}")


def wall_clock_hours(results_dir: Path, *, method: str, slug: str) -> float | None:
    records = load_scores_jsonl(results_dir / f"{method}_{slug}_5seed_full_scores.jsonl")
    if len(records) < 2:
        return None
    times = sorted(datetime.fromisoformat(r["timestamp"]) for r in records)
    return (times[-1] - times[0]).total_seconds() / 3600.0


def write_csv_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = ("category", "method", "seed", "split")
    merged = {(r["category"], r["method"], str(r["seed"]), r["split"]): r for r in existing}
    for row in rows:
        merged[(row["category"], row["method"], str(row["seed"]), row["split"])] = row
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(merged.values(), key=lambda r: (r["category"], r["method"], int(r["seed"]))):
            writer.writerow(row)


def update_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(payload)
    path.write_text(json.dumps(data, indent=2) + "\n")


def benchmark_method(
    *,
    method: str,
    category: str,
    slug: str,
    seed: int,
    output_root: Path,
    results_dir: Path,
    extensions_dir: Path,
) -> None:
    run_directory = run_directory_for_seed(
        results_dir, method=method, slug=slug, seed=seed
    )
    cfg, ckpt_path = get_best_checkpoint(output_root / "models", run_directory)

    rows = []
    for split in ("valid", "test"):
        seconds, mem_gb, n_params = timed_eval(cfg, ckpt_path, split=split)
        rows.append(
            {
                "category": category,
                "method": method,
                "seed": seed,
                "split": split,
                "seconds": f"{seconds:.3f}",
                "peak_mem_gb": f"{mem_gb:.3f}",
                "n_params": n_params,
                "run_directory": run_directory,
            }
        )
        print(
            f"{category} {method} seed {seed} {split}: "
            f"{seconds:.2f}s, peak GPU mem {mem_gb:.2f} GB, params {n_params:,}",
            flush=True,
        )

    write_csv_rows(
        extensions_dir / "inference_benchmark.csv",
        rows,
        ["category", "method", "seed", "split", "seconds", "peak_mem_gb", "n_params", "run_directory"],
    )


def write_static_artifacts(extensions_dir: Path) -> None:
    update_json(
        extensions_dir / "model_vocab.json",
        {
            cat: {
                **MODEL_VOCAB[cat],
                "cosette_quant_id": COSETTE_RUNS[cat],
            }
            for cat in CATEGORIES
        },
    )


def write_wall_clock(extensions_dir: Path, results_dir: Path, category: str, slug: str) -> None:
    payload = {
        category: {
            "sasrec_5seed_hours": wall_clock_hours(results_dir, method="sasrec", slug=slug),
            "marius_5seed_hours": wall_clock_hours(results_dir, method="marius", slug=slug),
            "note": "Approx. span from first to last seed timestamp in scores jsonl.",
        }
    }
    update_json(extensions_dir / "wall_clock.json", payload)
    print(f"Wall-clock {category}: {payload[category]}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extension benchmarks (inference + metadata)")
    parser.add_argument("--category", required=True, choices=list(CATEGORIES))
    parser.add_argument("--category-slug", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--extensions-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true", help="Only write wall-clock / vocab JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root or Path(
        os.environ.get("OUTPUT_ROOT", repo_root / "outputs")
    )
    results_dir = args.results_dir or (
        output_root / "results"
        if (output_root / "results").exists()
        else repo_root / "reports" / "results"
    )
    extensions_dir = args.extensions_dir or (repo_root / "reports" / "extensions")

    write_static_artifacts(extensions_dir)
    write_wall_clock(extensions_dir, results_dir, args.category, args.category_slug)

    if args.skip_gpu:
        return 0

    ray.init(ignore_reinit_error=True)
    fsspec.filesystem("file")

    for method in ("sasrec", "marius"):
        benchmark_method(
            method=method,
            category=args.category,
            slug=args.category_slug,
            seed=args.seed,
            output_root=output_root,
            results_dir=results_dir,
            extensions_dir=extensions_dir,
        )

    ray.shutdown()
    print(f"Wrote {extensions_dir / 'inference_benchmark.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
