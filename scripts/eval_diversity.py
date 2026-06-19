#!/usr/bin/env python3
"""Run the best MARIUS / SASRec checkpoint per seed on the real test set and
record the Gini index and Shannon entropy of their recommendations
(popularity/code-usage skew and diversity).

For each (method, seed) under --models-root, picks the run directory with the
most training progress (guards against partial/restarted runs - e.g. the Arts
SASRec seed42 has several aborted folders alongside the completed one), loads
its best-by-valid-HR@10 checkpoint, runs net.search() over the whole test set,
and writes one JSON summary per run to --output-dir.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import fsspec
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from scripts.benchmark_extensions import FILENAMES, get_best_checkpoint
from src.data.ray_data import get_items_map, get_quantized
from src.models import SpecialTokens
from src.utils.metrics import (
    category_diversity,
    summarize_dense,
    summarize_dense_entropy,
    summarize_dense_ild,
    summarize_generative,
    summarize_generative_entropy,
    summarize_generative_ild,
    summarize_generative_item,
    summarize_generative_item_ild,
    summarize_generative_item_entropy,
    summarize_generative_support,
)
from src.utils.tools import patch_fsspec

RUN_DIR_RE = re.compile(r"^(?P<method>MARIUS|SASRec)_(?P<category>.+)_seed(?P<seed>\d+)_")


def discover_runs(models_root: Path, category: str) -> dict[tuple[str, int], Path]:
    best: dict[tuple[str, int], tuple[Path, int]] = {}

    for run_dir in sorted(models_root.iterdir()):
        match = RUN_DIR_RE.match(run_dir.name)
        if not match or match.group("category") != category:
            continue

        progress_path = run_dir / FILENAMES["progress"]
        if not progress_path.exists():
            continue
        with progress_path.open() as f:
            progress = json.load(f)

        steps = [c["metrics"].get("step", 0) for c in progress["checkpoint_results"]]
        if not steps:
            continue

        key = (match.group("method"), int(match.group("seed")))
        max_step = max(steps)
        if key not in best or max_step > best[key][1]:
            best[key] = (run_dir, max_step)

    return {key: run_dir for key, (run_dir, _) in best.items()}


def best_valid_hr10(run_dir: Path) -> float:
    with (run_dir / FILENAMES["progress"]).open() as f:
        progress = json.load(f)
    scores = [
        v
        for c in progress["checkpoint_results"]
        for k, v in c["metrics"].items()
        if k.startswith("valid/") and k.endswith("/HR@10")
    ]
    return max(scores) if scores else None


@torch.no_grad()
def collect_generations(cfg, ckpt_path: str, n_results: int, device: str, limit_batches: int | None):
    patch_fsspec()
    cfg = cfg.copy()
    cfg.data.ray_datasets.which = ["valid", "test"]
    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    model.full_hydra_config = cfg

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    model.eval().to(device)

    gens, n_examples = [], 0
    for i, batch in enumerate(datamodule.test_dataloader()):
        batch = {k: v.to(device) for k, v in batch.items()}
        gen = model.net.search(batch, n_results=n_results)
        gens.append(gen.cpu())
        n_examples += gen.shape[0]
        if limit_batches and i + 1 >= limit_batches:
            break

    return torch.cat(gens, dim=0).numpy(), n_examples


def catalog_size(cfg) -> int:
    """Number of distinct items in the category catalog.

    Read from the COSETTE semantic-ID table (one row per item, unique after
    collision removal). Matches SASRec's `vocab_size - 2` for the same dataset,
    so the item-level Gini/entropy are comparable across methods.
    """
    rd = cfg.data.ray_datasets
    path = cfg.paths.semantic_ids_tplt.format(
        emb_method=rd.emb_id, category=rd.category, quant_method=rd.quant_id
    )
    fs = fsspec.filesystem(cfg.paths.protocol)
    return len(get_quantized(fs, path))


UNKNOWN_CATEGORY = "__unknown__"


def _cat_label(cats, level):
    """Pick one category label from an item's hierarchy array.

    `level` is either "leaf" (most specific known category) or an int depth.
    Items with no/short hierarchy fall back to a shared unknown sentinel.
    """
    try:
        n = len(cats)
    except TypeError:
        return UNKNOWN_CATEGORY
    if n == 0:
        return UNKNOWN_CATEGORY
    if level == "leaf":
        return str(cats[-1])
    return str(cats[level]) if n > level else UNKNOWN_CATEGORY


def category_map(cfg, level) -> dict[str, str]:
    """product_id -> category label, from the category metadata parquet."""
    fs = fsspec.filesystem(cfg.paths.protocol)
    meta_path = cfg.paths.meta_tplt.format(category=cfg.data.ray_datasets.category)
    meta = pd.read_parquet(meta_path, filesystem=fs, columns=["parent_asin", "categories"])
    return {row.parent_asin: _cat_label(row.categories, level) for row in meta.itertuples()}


def recommended_categories(cfg, gen, mode, level):
    """Map each recommended item to its category label. Returns (B, K) of labels.

    SASRec gen is item ids (offset by len(SpecialTokens)); MARIUS gen is Remapped
    semantic-ID tuples (token = raw_code + level*K + len(SpecialTokens)), which we
    un-remap and look up in the COSETTE code table to recover the product_id.
    """
    pid_to_cat = category_map(cfg, level)
    fs = fsspec.filesystem(cfg.paths.protocol)
    gen = np.asarray(gen)

    if mode == "generative":
        rd = cfg.data.ray_datasets
        quant_path = cfg.paths.semantic_ids_tplt.format(
            emb_method=rd.emb_id, category=rd.category, quant_method=rd.quant_id
        )
        quant_df = get_quantized(fs, quant_path)
        K = int(quant_df.values.max()) + 1
        L = quant_df.shape[1]
        tuple_to_pid = {
            tuple(int(v) for v in row): pid
            for pid, row in zip(quant_df.index, quant_df.values)
        }
        offset = np.arange(L) * K + len(SpecialTokens)
        raw = gen - offset  # un-remap to raw RVQ codes
        labels = [
            [pid_to_cat.get(tuple_to_pid.get(tuple(int(v) for v in rec)), UNKNOWN_CATEGORY) for rec in user]
            for user in raw
        ]
    else:
        items_path = cfg.paths.unique_items_tplt.format(category=cfg.data.ray_datasets.category)
        id_to_item = get_items_map(fs, items_path)["id_to_item"]
        labels = [
            [pid_to_cat.get(id_to_item.get(int(item)), UNKNOWN_CATEGORY) for item in user]
            for user in gen
        ]

    return np.asarray(labels, dtype=object)


def evaluate_run(
    method: str,
    seed: int,
    run_dir: Path,
    models_root: Path,
    *,
    n_results: int,
    device: str,
    enforce_filtering: bool,
    limit_batches: int | None,
    category_level,
) -> dict:
    cfg, ckpt_path = get_best_checkpoint(models_root, run_dir.name)

    if enforce_filtering:
        cfg.model.net.filter_preds = True

    gen, n_examples = collect_generations(cfg, ckpt_path, n_results, device, limit_batches)

    result = {
        "method": method,
        "seed": seed,
        "category": cfg.data.ray_datasets.category,
        "mode": cfg.model.mode,
        "run_directory": run_dir.name,
        "checkpoint": ckpt_path,
        "n_results": n_results,
        "n_test_examples": n_examples,
        "valid_HR10": best_valid_hr10(run_dir),
    }

    # Category diversity (distinct categories / K, per user) - same computation for
    # both methods, so MARIUS and SASRec are directly comparable on diversity.
    rec_cats = recommended_categories(cfg, gen, cfg.model.mode, category_level)
    result["category_level"] = str(category_level)
    result["category_diversity"] = category_diversity(rec_cats)

    if cfg.model.mode == "generative":
        # K = codes per level (256 in the paper's COSETTE_128d_256x4 setup):
        # vocab_size = L * K + len(SpecialTokens).
        k_per_level = (cfg.model.net.depth_cfg.vocab_size - 2) // gen.shape[-1]
        result["k_per_level"] = k_per_level
        result["gini_per_level"] = summarize_generative(gen, n_total_per_level=k_per_level)
        result["entropy_per_level"] = summarize_generative_entropy(gen, n_total_per_level=k_per_level)
        # Support diagnostics so a per-level Gini/entropy change can be told apart
        # from a shrinking-support artifact (fewer recs/codes per group at depth).
        result["support_per_level"] = summarize_generative_support(gen)
        result["ild"] = summarize_generative_ild(gen)

        # Item-level skew (full semantic-ID tuple = one item), so MARIUS is
        # directly comparable to SASRec's item-level gini/entropy/ILD below.
        n_items = catalog_size(cfg)
        result["n_items"] = n_items
        result["gini"] = summarize_generative_item(gen, n_items=n_items)
        result["entropy"] = summarize_generative_item_entropy(gen, n_items=n_items)
        # Binary item-level ILD (item identity, not RVQ codes), comparable to SASRec.
        result["item_ild"] = summarize_generative_item_ild(gen)
    else:
        n_items = cfg.model.net.vocab_size - 2  # minus PAD/BOS
        result["n_items"] = n_items
        result["gini"] = summarize_dense(gen, n_items=n_items)
        result["entropy"] = summarize_dense_entropy(gen, n_items=n_items)
        result["ild"] = summarize_dense_ild(gen)

    return result


def main():
    parser = argparse.ArgumentParser(description="Diversity evaluation (Gini + entropy) per checkpoint")
    parser.add_argument("--category", default="Arts_Crafts_and_Sewing")
    parser.add_argument(
        "--models-root", type=Path, default=Path("/scratch-shared/scur1250/cosette_arts/models")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diversity"))
    parser.add_argument("--n-results", type=int, default=20, help="max(Ks) in LitModule")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-filter-preds", action="store_true")
    parser.add_argument(
        "--limit-batches", type=int, default=None, help="Smoke-test: only eval the first N batches"
    )
    parser.add_argument("--methods", nargs="+", default=["MARIUS", "SASRec"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument(
        "--category-level",
        default="leaf",
        help="Taxonomy depth for category diversity: 'leaf' (most specific) or an int depth (e.g. 1).",
    )
    args = parser.parse_args()

    category_level = args.category_level if args.category_level == "leaf" else int(args.category_level)

    fsspec.filesystem("file")
    runs = discover_runs(args.models_root, args.category)

    out_dir = args.output_dir / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    for (method, seed), run_dir in sorted(runs.items()):
        if method not in args.methods:
            continue
        if args.seeds is not None and seed not in args.seeds:
            continue

        print(f"[{method} seed={seed}] evaluating {run_dir.name} on {args.device}", flush=True)
        result = evaluate_run(
            method,
            seed,
            run_dir,
            args.models_root,
            n_results=args.n_results,
            device=args.device,
            enforce_filtering=not args.no_filter_preds,
            limit_batches=args.limit_batches,
            category_level=category_level,
        )
        print(f"  -> {result}", flush=True)

        out_path = out_dir / f"{method.lower()}_seed{seed}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
