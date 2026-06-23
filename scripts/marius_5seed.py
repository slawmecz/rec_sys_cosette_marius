#!/usr/bin/env python3
"""
Train + evaluate MARIUS (COSETTE) for multiple seeds on an Amazon 2014 category.

Saves one JSON line per completed seed so a crash does not lose prior results.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

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

CATEGORY_DEFAULTS = {
    "Beauty": {
        "slug": "beauty",
        "quant_id": "COSETTE_128d_256x4_f958-col",
    },
    "Sports_and_Outdoors": {
        "slug": "sports",
        "quant_id": "COSETTE_128d_256x4_8ed1-col",
    },
}

RUN_PREFIX = "MARIUS_small"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def category_slug(category: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if category in CATEGORY_DEFAULTS:
        return CATEGORY_DEFAULTS[category]["slug"]
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def metric_keys(category: str) -> list[tuple[str, str]]:
    return [
        ("R@5", f"test/{category}/HR@5"),
        ("NDCG@5", f"test/{category}/NDCG@5"),
        ("R@10", f"test/{category}/HR@10"),
        ("NDCG@10", f"test/{category}/NDCG@10"),
    ]


def _split_overrides(value: str) -> list[str]:
    return [part for part in value.split() if part]


def load_completed_scores(scores_file: Path) -> dict[int, dict]:
    if not scores_file.exists():
        return {}
    completed: dict[int, dict] = {}
    with scores_file.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") == "ok":
                completed[int(record["seed"])] = record
    return completed


def append_score(scores_file: Path, record: dict) -> None:
    scores_file.parent.mkdir(parents=True, exist_ok=True)
    with scores_file.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_category(cfg: OmegaConf) -> str | None:
    try:
        return str(cfg.data.ray_datasets.category)
    except Exception:
        return None


def find_run_for_seed(
    output_root: Path,
    seed: int,
    category: str,
    since_mtime: float | None = None,
) -> str | None:
    models_dir = output_root / "models"
    if not models_dir.exists():
        return None

    best_name: str | None = None
    best_mtime = 0.0
    for run_dir in models_dir.glob(f"{RUN_PREFIX}_*"):
        if not run_dir.is_dir():
            continue
        mtime = run_dir.stat().st_mtime
        if since_mtime is not None and mtime < since_mtime:
            continue
        for cfg_path in run_dir.rglob("config.yaml"):
            try:
                cfg = OmegaConf.load(cfg_path)
            except Exception:
                continue
            if int(cfg.get("seed", -1)) != int(seed):
                continue
            if _run_category(cfg) != category:
                continue
            if mtime > best_mtime:
                best_mtime = mtime
                best_name = run_dir.name
            break
    return best_name


def has_metrics(output_root: Path, run_dir: str) -> bool:
    root = output_root / "models" / run_dir
    return any(root.rglob("metrics.pkl"))


def extract_metrics(
    output_root: Path, run_dir: str, metrics: list[tuple[str, str]]
) -> dict[str, float]:
    root = output_root / "models" / run_dir
    metric_paths = list(root.rglob("metrics.pkl"))
    if not metric_paths:
        raise FileNotFoundError(f"No metrics.pkl under {root}")

    with metric_paths[0].open("rb") as handle:
        blob = pickle.load(handle)

    test = blob["test_metrics"][0]
    return {label: 100.0 * float(test[key]) for label, key in metrics}


def record_seed_result(
    *,
    seed: int,
    run_dir: str,
    output_root: Path,
    scores_file: Path,
    mode: str,
    category: str,
    metrics: list[tuple[str, str]],
) -> dict:
    values = extract_metrics(output_root, run_dir, metrics)
    record = {
        "status": "ok",
        "mode": mode,
        "category": category,
        "seed": seed,
        "run_directory": run_dir,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **values,
    }
    append_score(scores_file, record)
    return record


def run_train(
    *,
    project_root: Path,
    python_bin: str,
    experiment: str = "marius_small",
    paths_overrides: str,
    train_overrides: str,
    seed: int,
    category: str,
    quant_id: str,
    extra_args: list[str],
) -> int:
    cmd = [
        python_bin,
        "src/train.py",
        f"experiment={experiment}",
        *_split_overrides(paths_overrides),
        *_split_overrides(train_overrides),
        f"seed={seed}",
        f"data.ray_datasets.category={category}",
        "data.ray_datasets.emb_id=sentence-t5-xl",
        f"data.ray_datasets.quant_id={quant_id}",
        *extra_args,
    ]
    print(f"[seed {seed}] train:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def run_test(
    *,
    project_root: Path,
    python_bin: str,
    paths_overrides: str,
    run_dir: str,
) -> int:
    cmd = [
        python_bin,
        "src/test.py",
        *_split_overrides(paths_overrides),
        f"run_directory={run_dir}",
    ]
    print(f"[run {run_dir}] test:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=project_root).returncode


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f}% ±{std:.2f}"


def _metric_stats(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _render_table(rows: list[dict], cols: list[str]) -> str:
    def fmt_cell(row: dict, col: str) -> str:
        value = row[col]
        if col in {"method", "seed", "run"}:
            return str(value)
        if isinstance(value, str):
            return value
        return _fmt_pct(float(value))

    widths = {
        col: max(len(col), *(len(fmt_cell(row, col)) for row in rows)) for col in cols
    }

    def render_row(row: dict) -> str:
        return " | ".join(fmt_cell(row, col).ljust(widths[col]) for col in cols)

    sep = "-+-".join("-" * widths[col] for col in cols)
    lines = [render_row({col: col for col in cols}), sep]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def _write_table_files(
    *,
    results_dir: Path,
    mode: str,
    stem: str,
    header: str,
    body: str,
) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text = header + body + "\n"
    stamped = results_dir / f"{stem}_{mode}_{stamp}.txt"
    latest = results_dir / f"{stem}_{mode}_latest.txt"
    stamped.write_text(text)
    latest.write_text(text)
    return stamped, latest


def write_summary_table(
    *,
    completed: dict[int, dict],
    results_dir: Path,
    mode: str,
    expected_seeds: list[int],
    category: str,
    slug: str,
    metrics: list[tuple[str, str]],
    paper_targets: dict[str, tuple[float, float]],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    metric_cols = [label for label, _ in metrics]
    display_name = category.replace("_", " ")
    common_header = (
        f"Amazon 2014 {display_name} - MARIUS (COSETTE) test metrics ({mode})\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Expected seeds: {expected_seeds}\n"
        f"Completed seeds: {sorted(completed)} ({len(completed)}/{len(expected_seeds)})\n\n"
    )

    individual_rows: list[dict] = []
    for seed in expected_seeds:
        if seed not in completed:
            individual_rows.append(
                {
                    "seed": str(seed),
                    "run": "(pending)",
                    **{label: "-" for label in metric_cols},
                }
            )
            continue
        row = completed[seed]
        individual_rows.append(
            {
                "seed": str(seed),
                "run": row["run_directory"],
                **{label: row[label] for label in metric_cols},
            }
        )

    individual_cols = ["seed", "run", *metric_cols]
    individual_body = _render_table(individual_rows, individual_cols)
    ind_stamped, ind_latest = _write_table_files(
        results_dir=results_dir,
        mode=mode,
        stem=f"table5_{slug}_marius_5seed_individual",
        header=common_header + "Per-seed results (R@5, NDCG@5, R@10, NDCG@10 on test)\n\n",
        body=individual_body,
    )

    paper_row = {
        "method": "MARIUS (COSETTE) paper",
        "seed": "mean ± std (5 runs)",
        **{
            label: _fmt_mean_std(mean, std)
            for label, (mean, std) in paper_targets.items()
        },
    }

    summary_rows = [paper_row]
    if completed:
        n = len(completed)
        agg = {
            "method": "MARIUS + COSETTE (replication)",
            "seed": f"mean ± std ({n} runs)",
        }
        for label, _ in metrics:
            values = [completed[s][label] for s in sorted(completed)]
            mean, std = _metric_stats(values)
            agg[label] = _fmt_mean_std(mean, std)
        summary_rows.append(agg)

    summary_cols = ["method", "seed", *metric_cols]
    summary_body = _render_table(summary_rows, summary_cols)
    sum_stamped, sum_latest = _write_table_files(
        results_dir=results_dir,
        mode=mode,
        stem=f"table5_{slug}_marius_5seed_summary",
        header=common_header + "Table 5 - paper vs replication (mean ± std on test)\n\n",
        body=summary_body,
    )

    combined_body = (
        "=== Individual runs ===\n\n"
        + individual_body
        + "\n\n=== Paper comparison (mean ± std) ===\n\n"
        + summary_body
        + "\n"
    )
    comb_stamped, comb_latest = _write_table_files(
        results_dir=results_dir,
        mode=mode,
        stem=f"table5_{slug}_marius_5seed",
        header=common_header,
        body=combined_body,
    )

    print(combined_body, flush=True)
    print(
        "Saved:\n"
        f"  {ind_latest}\n"
        f"  {sum_latest}\n"
        f"  {comb_latest}\n"
        f"  (stamped copies: {ind_stamped.name}, {sum_stamped.name}, {comb_stamped.name})",
        flush=True,
    )


def _extra_overrides() -> list[str]:
    """Optional Hydra overrides for paper-faithful re-runs (e.g. finer checkpointing).

    Set via EXTRA_TRAIN_OVERRIDES (space-separated Hydra overrides).
    Empty by default, so existing jobs are unaffected.
    """
    return os.environ.get("EXTRA_TRAIN_OVERRIDES", "").split()


def paper_train_args() -> list[str]:
    batch = os.environ.get("TRAIN_BATCH_PER_GPU", "256")
    workers = os.environ.get("RAY_NUM_WORKERS", "1")
    max_steps = os.environ.get("TRAIN_MAX_STEPS", "80000")
    val_check = os.environ.get("TRAIN_VAL_CHECK_INTERVAL", "10000")
    return [
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_check}",
        f"data.datamodule.train_batch_size={batch}",
        f"data.datamodule.valid_batch_size={batch}",
        f"ray.scaling_config.num_workers={workers}",
        "ray.scaling_config.resources_per_worker={CPU:4,GPU:1}",
    ] + _extra_overrides()


def smoke_train_args() -> list[str]:
    batch = os.environ.get("TRAIN_BATCH_PER_GPU", "256")
    return [
        "trainer.max_steps=15",
        "trainer.val_check_interval=5",
        "trainer.limit_val_batches=2",
        "model.scheduler.warmup_steps=5",
        "model.scheduler.cosine_steps=10",
        f"data.datamodule.train_batch_size={batch}",
        f"data.datamodule.valid_batch_size={batch}",
        "ray.scaling_config.num_workers=1",
        "ray.scaling_config.resources_per_worker={CPU:4,GPU:1}",
    ] + _extra_overrides()


def process_seed(
    *,
    seed: int,
    mode: str,
    experiment: str = "marius_small",
    category: str,
    metrics: list[tuple[str, str]],
    project_root: Path,
    output_root: Path,
    python_bin: str,
    paths_overrides: str,
    train_overrides: str,
    scores_file: Path,
    train_args: list[str],
    quant_id: str,
) -> bool:
    completed = load_completed_scores(scores_file)
    if seed in completed:
        print(f"[seed {seed}] already recorded in {scores_file}, skipping", flush=True)
        return True

    existing_run = find_run_for_seed(output_root, seed, category)
    if existing_run and has_metrics(output_root, existing_run):
        print(
            f"[seed {seed}] found existing metrics in {existing_run}, recording only",
            flush=True,
        )
        record_seed_result(
            seed=seed,
            run_dir=existing_run,
            output_root=output_root,
            scores_file=scores_file,
            mode=mode,
            category=category,
            metrics=metrics,
        )
        return True

    since_mtime = time.time() - 5
    train_rc = run_train(
        project_root=project_root,
        python_bin=python_bin,
        experiment=experiment,
        paths_overrides=paths_overrides,
        train_overrides=train_overrides,
        seed=seed,
        category=category,
        quant_id=quant_id,
        extra_args=train_args,
    )
    if train_rc != 0:
        append_score(
            scores_file,
            {
                "status": "train_failed",
                "mode": mode,
                "category": category,
                "seed": seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "return_code": train_rc,
            },
        )
        print(f"[seed {seed}] training failed with exit code {train_rc}", flush=True)
        return False

    run_dir = find_run_for_seed(output_root, seed, category, since_mtime=since_mtime)
    if not run_dir:
        run_dir = find_run_for_seed(output_root, seed, category)
    if not run_dir:
        append_score(
            scores_file,
            {
                "status": "run_not_found",
                "mode": mode,
                "category": category,
                "seed": seed,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(f"[seed {seed}] could not locate MARIUS run directory", flush=True)
        return False

    if not has_metrics(output_root, run_dir):
        test_rc = run_test(
            project_root=project_root,
            python_bin=python_bin,
            paths_overrides=paths_overrides,
            run_dir=run_dir,
        )
        if test_rc != 0:
            append_score(
                scores_file,
                {
                    "status": "test_failed",
                    "mode": mode,
                    "category": category,
                    "seed": seed,
                    "run_directory": run_dir,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "return_code": test_rc,
                },
            )
            print(f"[seed {seed}] test failed with exit code {test_rc}", flush=True)
            return False

    record_seed_result(
        seed=seed,
        run_dir=run_dir,
        output_root=output_root,
        scores_file=scores_file,
        mode=mode,
        category=category,
        metrics=metrics,
    )
    print(f"[seed {seed}] saved results for {run_dir}", flush=True)
    return True


def parse_seeds(raw: str) -> list[int]:
    return [int(part) for part in raw.split() if part.strip()]


def resolve_quant_id(category: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    if category in CATEGORY_DEFAULTS:
        return CATEGORY_DEFAULTS[category]["quant_id"]
    env_val = os.environ.get("MARIUS_QUANT_ID")
    if env_val:
        return env_val
    raise ValueError(
        f"Set --quant-id or MARIUS_QUANT_ID for category {category}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MARIUS (COSETTE) multi-seed train+test")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        required=True,
        help="smoke: quick pipeline check; full: paper-aligned 5-seed run",
    )
    parser.add_argument(
        "--category",
        default=os.environ.get("CATEGORY", "Beauty"),
        help="Amazon 2014 category (e.g. Beauty, Sports_and_Outdoors)",
    )
    parser.add_argument(
        "--category-slug",
        default=os.environ.get("CATEGORY_SLUG"),
        help="short name for result files (default: beauty, sports, ...)",
    )
    parser.add_argument(
        "--quant-id",
        default=None,
        help="deduped COSETTE quant id (e.g. COSETTE_128d_256x4_f958-col)",
    )
    parser.add_argument(
        "--experiment",
        default=os.environ.get("MARIUS_EXPERIMENT", "marius_small"),
        help="Hydra experiment config (marius_small for ~10-18k catalogs, marius for large)",
    )
    parser.add_argument(
        "--project-root",
        default=os.environ.get(
            "PROJECT_ROOT", "/home/scur1266/rec_sys_cosette_marius"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "OUTPUT_ROOT", "/home/scur1266/scratch/cosette_marius/outputs"
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
    )
    parser.add_argument(
        "--seeds",
        default=os.environ.get("SASREC_5SEEDS", " ".join(map(str, DEFAULT_SEEDS))),
        help="space-separated seed list",
    )
    parser.add_argument(
        "--paths-overrides",
        default=os.environ.get("PATHS_OVERRIDES", ""),
    )
    parser.add_argument(
        "--train-overrides",
        default=os.environ.get("TRAIN_OVERRIDES", ""),
    )
    args = parser.parse_args(argv)

    category = args.category
    slug = category_slug(category, args.category_slug)
    quant_id = resolve_quant_id(category, args.quant_id)
    metrics = metric_keys(category)
    paper_targets = PAPER_MARIUS_COSETTE.get(category, {})

    project_root = Path(args.project_root)
    output_root = Path(args.output_root)
    results_dir = output_root / "results"
    scores_file = results_dir / f"marius_{slug}_5seed_{args.mode}_scores.jsonl"

    if args.mode == "smoke":
        seeds = [parse_seeds(args.seeds)[0]]
        train_args = smoke_train_args()
    else:
        seeds = parse_seeds(args.seeds)
        train_args = paper_train_args()

    print(f"Mode: {args.mode}", flush=True)
    print(f"Category: {category}", flush=True)
    print(f"Quant id: {quant_id}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Scores file: {scores_file}", flush=True)

    failures = 0
    for seed in seeds:
        ok = process_seed(
            seed=seed,
            mode=args.mode,
            experiment=args.experiment,
            category=category,
            metrics=metrics,
            project_root=project_root,
            output_root=output_root,
            python_bin=args.python_bin,
            paths_overrides=args.paths_overrides,
            train_overrides=args.train_overrides,
            scores_file=scores_file,
            train_args=train_args,
            quant_id=quant_id,
        )
        if not ok:
            failures += 1
        completed = load_completed_scores(scores_file)
        write_summary_table(
            completed=completed,
            results_dir=results_dir,
            mode=args.mode,
            expected_seeds=seeds,
            category=category,
            slug=slug,
            metrics=metrics,
            paper_targets=paper_targets,
        )

    if failures:
        print(f"Finished with {failures} failed seed(s).", flush=True)
        return 1

    print("All requested seeds completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
