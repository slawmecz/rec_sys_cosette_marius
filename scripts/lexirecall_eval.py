#!/usr/bin/env python3
"""Export test rankings and compute TSE / lexirecall metrics (Diaz et al., TOIS 2026).

Total Search Efficiency (TSE) with reciprocal-rank exposure equals 1/rank of the
last (here: only) relevant item; with one relevant item per query this matches MRR.

Lexicographic recall compares two systems per query (worst-case / recall-oriented).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import pickle
from pathlib import Path

import fsspec
import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as L
import ray
import torch
from omegaconf import OmegaConf

from scripts.benchmark_extensions import get_best_checkpoint, run_directory_for_seed
from scripts.reporting.constants import CATEGORIES, COSETTE_RUNS
from src.models import SpecialTokens
from src.utils.tools import patch_fsspec

SPECIAL_OFFSET = len(SpecialTokens)


def load_items_map(data_root: Path, category: str) -> dict:
    path = data_root / "data" / "timelines" / f"{category}_items.pkl"
    with path.open("rb") as handle:
        items = pickle.load(handle)
    id_to_item = {i + SPECIAL_OFFSET: item for i, item in enumerate(items)}
    return id_to_item


def load_quantizer(data_root: Path, category: str, quant_id: str) -> pd.DataFrame:
    path = (
        data_root
        / "data"
        / "embeddings"
        / "sentence-t5-xl"
        / category
        / f"{quant_id}.parquet"
    )
    df = pd.read_parquet(path).set_index("product_id")
    sorted_cols = sorted(col for col in df.columns if col.startswith("L"))
    return df[sorted_cols]


def build_sid_to_product(quant_df: pd.DataFrame) -> dict[tuple[int, ...], str]:
    mapping: dict[tuple[int, ...], str] = {}
    for product_id, row in quant_df.iterrows():
        mapping[tuple(int(v) for v in row.values)] = str(product_id)
    return mapping


def remapped_to_raw(codes: np.ndarray, *, n_levels: int, codebook_size: int) -> tuple[int, ...]:
    raw = []
    for level, value in enumerate(codes):
        raw.append(int(value) - SPECIAL_OFFSET - level * codebook_size)
    return tuple(raw)


def raw_to_product(
    raw: tuple[int, ...], sid_to_product: dict[tuple[int, ...], str]
) -> str | None:
    return sid_to_product.get(raw)


def product_from_sasrec_target(item_id: int, id_to_item: dict[int, str]) -> str | None:
    if int(item_id) in (SpecialTokens.PAD.value, -100):
        return None
    return str(id_to_item.get(int(item_id)))


def product_from_marius_target(
    codes: np.ndarray,
    *,
    n_levels: int,
    codebook_size: int,
    sid_to_product: dict[tuple[int, ...], str],
) -> str | None:
    if (codes == -100).all():
        return None
    raw = remapped_to_raw(codes, n_levels=n_levels, codebook_size=codebook_size)
    return raw_to_product(raw, sid_to_product)


def preds_to_products_sasrec(
    pred_ids: torch.Tensor, id_to_item: dict[int, str]
) -> list[str]:
    docs: list[str] = []
    for item_id in pred_ids.tolist():
        doc = id_to_item.get(int(item_id))
        if doc is not None:
            docs.append(str(doc))
    return docs


def preds_to_products_marius(
    pred_codes: torch.Tensor,
    *,
    n_levels: int,
    codebook_size: int,
    sid_to_product: dict[tuple[int, ...], str],
) -> list[str]:
    docs: list[str] = []
    for row in pred_codes.cpu().numpy():
        raw = remapped_to_raw(row, n_levels=n_levels, codebook_size=codebook_size)
        doc = sid_to_product.get(raw)
        if doc is not None:
            docs.append(doc)
    return docs


def rank_of_relevant(ranked_docs: list[str], relevant: str) -> int | None:
    for i, doc in enumerate(ranked_docs, start=1):
        if doc == relevant:
            return i
    return None


def tse_rr(rank: int | None) -> float:
    """TSE with reciprocal-rank exposure (Eq. 3, Diaz et al.; one relevant item)."""
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def recall_at_k(rank: int | None, k: int) -> float:
    if rank is None:
        return 0.0
    return 1.0 if rank <= k else 0.0


def lexirecall_single(rank_a: int | None, rank_b: int | None) -> float:
    """Preference for A over B: +1 prefer A, -1 prefer B, 0 tie."""
    found_a = rank_a is not None
    found_b = rank_b is not None
    if found_a and not found_b:
        return 1.0
    if found_b and not found_a:
        return -1.0
    if not found_a and not found_b:
        return 0.0
    assert rank_a is not None and rank_b is not None
    if rank_a < rank_b:
        return 1.0
    if rank_a > rank_b:
        return -1.0
    return 0.0


def write_trec_qrels(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for qid, docid in rows:
            handle.write(f"{qid} 0 {docid} 1\n")


def write_trec_run(path: Path, run_name: str, rankings: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for qid, docs in rankings.items():
            for rank, docid in enumerate(docs, start=1):
                score = 1000 - rank
                handle.write(f"{qid} 0 {docid} {rank} {score} {run_name}\n")


class _RankingCallback(L.Callback):
    """Collect per-query rankings during Lightning test (matches src/test.py inference)."""

    def __init__(
        self,
        *,
        method: str,
        topk: int,
        id_to_item: dict[int, str] | None,
        sid_to_product: dict[tuple[int, ...], str] | None,
        n_levels: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.method = method
        self.topk = topk
        self.id_to_item = id_to_item
        self.sid_to_product = sid_to_product
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.qrels: dict[str, str] = {}
        self.rankings: dict[str, list[str]] = {}
        self.item_hr10: list[float] = []
        self._qid = 0

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        batch = trainer.strategy.batch_to_device(batch, dataloader_idx=dataloader_idx)
        with trainer.precision_plugin.forward_context():
            preds = pl_module.net.search(batch, n_results=self.topk)

        if self.method == "sasrec":
            assert self.id_to_item is not None
            targets = batch["target"][:, -1]
            target_ids = batch["target"][:, -1:]
            self.item_hr10.extend(
                (preds[:, :10] == target_ids).any(dim=1).float().tolist()
            )
            for i in range(targets.shape[0]):
                qid_str = str(self._qid)
                relevant = product_from_sasrec_target(
                    targets[i].item(), self.id_to_item
                )
                if relevant is None:
                    self._qid += 1
                    continue
                ranked = preds_to_products_sasrec(preds[i], self.id_to_item)
                self.qrels[qid_str] = relevant
                self.rankings[qid_str] = ranked
                self._qid += 1
        else:
            assert self.sid_to_product is not None
            targets = batch["target"][:, -1, :]
            target_codes = batch["target"][:, -1:, :]
            code_hits = (preds == target_codes).all(dim=-1)
            self.item_hr10.extend(code_hits[:, :10].any(dim=1).float().tolist())
            for i in range(targets.shape[0]):
                qid_str = str(self._qid)
                relevant = product_from_marius_target(
                    targets[i].cpu().numpy(),
                    n_levels=self.n_levels,
                    codebook_size=self.codebook_size,
                    sid_to_product=self.sid_to_product,
                )
                if relevant is None:
                    self._qid += 1
                    continue
                ranked = preds_to_products_marius(
                    preds[i],
                    n_levels=self.n_levels,
                    codebook_size=self.codebook_size,
                    sid_to_product=self.sid_to_product,
                )
                self.qrels[qid_str] = relevant
                self.rankings[qid_str] = ranked
                self._qid += 1


def collect_test_rankings(
    cfg: OmegaConf,
    ckpt_path: str,
    *,
    method: str,
    topk: int,
    id_to_item: dict[int, str] | None,
    sid_to_product: dict[tuple[int, ...], str] | None,
    n_levels: int,
    codebook_size: int,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    patch_fsspec()
    # RayDataModule.setup(stage="test") still loads valid; keep both splits.
    cfg.data.ray_datasets.which = ["valid", "test"]
    ray_datasets = hydra.utils.instantiate(cfg.data.ray_datasets, paths=cfg.paths)
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, ray_datasets=ray_datasets)
    datamodule.setup(stage="test")

    model = hydra.utils.instantiate(cfg["model"])
    model.full_hydra_config = cfg

    callback = _RankingCallback(
        method=method,
        topk=topk,
        id_to_item=id_to_item,
        sid_to_product=sid_to_product,
        n_levels=n_levels,
        codebook_size=codebook_size,
    )

    use_gpu = torch.cuda.is_available()
    trainer = L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        precision="bf16-mixed" if use_gpu else "32-true",
        logger=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        callbacks=[callback],
    )

    protocol = str(cfg.paths.protocol)
    ckpt_uri = ckpt_path if "://" in ckpt_path else f"{protocol}://{ckpt_path}"
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_uri)

    if callback.item_hr10:
        mean_item_hr10 = float(np.mean(callback.item_hr10))
        print(
            f"{method} item-id HR@10 sanity: {mean_item_hr10 * 100:.4f}%",
            flush=True,
        )

    return callback.qrels, callback.rankings


def summarize_method(
    qrels: dict[str, str], rankings: dict[str, list[str]], *, topk: int
) -> dict[str, float]:
    tse_vals: list[float] = []
    rr_vals: list[float] = []
    r10_vals: list[float] = []
    r20_vals: list[float] = []
    rk_vals: list[float] = []

    for qid, relevant in qrels.items():
        ranked = rankings.get(qid, [])
        rank = rank_of_relevant(ranked, relevant)
        tse_vals.append(tse_rr(rank))
        rr_vals.append(tse_rr(rank))
        r10_vals.append(recall_at_k(rank, 10))
        r20_vals.append(recall_at_k(rank, 20))
        rk_vals.append(recall_at_k(rank, topk))

    n = len(tse_vals)
    return {
        "n_queries": n,
        "mean_tse": float(np.mean(tse_vals)) if n else 0.0,
        "mean_rr": float(np.mean(rr_vals)) if n else 0.0,
        "mean_r_at_10": float(np.mean(r10_vals)) if n else 0.0,
        "mean_r_at_20": float(np.mean(r20_vals)) if n else 0.0,
        f"mean_r_at_{topk}": float(np.mean(rk_vals)) if n else 0.0,
    }


def pairwise_lexirecall(
    qrels: dict[str, str],
    rankings_a: dict[str, list[str]],
    rankings_b: dict[str, list[str]],
) -> dict[str, float]:
    prefs: list[float] = []
    wins_a = wins_b = ties = 0
    for qid, relevant in qrels.items():
        rank_a = rank_of_relevant(rankings_a.get(qid, []), relevant)
        rank_b = rank_of_relevant(rankings_b.get(qid, []), relevant)
        pref = lexirecall_single(rank_a, rank_b)
        prefs.append(pref)
        if pref > 0:
            wins_a += 1
        elif pref < 0:
            wins_b += 1
        else:
            ties += 1
    n = len(prefs)
    return {
        "n_queries": n,
        "mean_lexirecall_pref": float(np.mean(prefs)) if n else 0.0,
        "fraction_sasrec_preferred": wins_a / n if n else 0.0,
        "fraction_marius_preferred": wins_b / n if n else 0.0,
        "fraction_ties": ties / n if n else 0.0,
    }


def update_csv(path: Path, row: dict, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = (row["category"], row["method"], str(row["seed"]))
    merged = {(r["category"], r["method"], str(r["seed"])): r for r in existing}
    merged[key] = row
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(merged.values(), key=lambda x: (x["category"], x["method"], int(x["seed"]))):
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def update_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if path.exists():
        data = json.loads(path.read_text())
    data.update(payload)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="TSE / lexirecall evaluation on test split")
    parser.add_argument("--category", required=True, choices=list(CATEGORIES))
    parser.add_argument("--category-slug", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Export depth for rankings (20 matches LitModule Ks; MARIUS beam OOMs above ~20).",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--extensions-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root or Path(
        os.environ.get("OUTPUT_ROOT", repo_root / "outputs")
    )
    data_root = args.data_root or Path(
        os.environ.get("DATA_ROOT", output_root.parent / "data")
    )
    results_dir = args.results_dir or (
        output_root / "results"
        if (output_root / "results").exists()
        else repo_root / "reports" / "results"
    )
    extensions_dir = args.extensions_dir or (repo_root / "reports" / "extensions")
    trec_dir = extensions_dir / "trec" / args.category_slug

    id_to_item = load_items_map(data_root, args.category)
    quant_df = load_quantizer(data_root, args.category, COSETTE_RUNS[args.category])
    n_levels = len(quant_df.columns)
    codebook_size = int(quant_df.values.max()) + 1
    sid_to_product = build_sid_to_product(quant_df)

    ray.init(ignore_reinit_error=True)
    fsspec.filesystem("file")

    method_rankings: dict[str, dict[str, list[str]]] = {}
    method_summaries: dict[str, dict] = {}
    sasrec_qrels: dict[str, str] = {}

    for method in ("sasrec", "marius"):
        method_topk = args.topk
        run_directory = run_directory_for_seed(
            results_dir, method=method, slug=args.category_slug, seed=args.seed
        )
        cfg, ckpt_path = get_best_checkpoint(output_root / "models", run_directory)
        qrels, rankings = collect_test_rankings(
            cfg,
            ckpt_path,
            method=method,
            topk=method_topk,
            id_to_item=id_to_item if method == "sasrec" else None,
            sid_to_product=sid_to_product if method == "marius" else None,
            n_levels=n_levels,
            codebook_size=codebook_size,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if method == "sasrec":
            sasrec_qrels = qrels
        method_rankings[method] = rankings
        method_summaries[method] = summarize_method(qrels, rankings, topk=method_topk)

        write_trec_qrels(trec_dir / f"test_{args.category_slug}.qrels", list(qrels.items()))
        write_trec_run(
            trec_dir / f"test_{method}_seed{args.seed}.run",
            f"{method}_seed{args.seed}",
            rankings,
        )

        row = {
            "category": args.category,
            "method": method,
            "seed": args.seed,
            "topk": method_topk,
            **{
                k: f"{v:.6f}" if isinstance(v, float) else v
                for k, v in method_summaries[method].items()
            },
        }
        fieldnames = [
            "category",
            "method",
            "seed",
            "topk",
            "n_queries",
            "mean_tse",
            "mean_rr",
            "mean_r_at_10",
            "mean_r_at_20",
            f"mean_r_at_{method_topk}",
        ]
        update_csv(extensions_dir / "lexirecall_metrics.csv", row, fieldnames)
        print(f"{args.category} {method} seed {args.seed}: {method_summaries[method]}", flush=True)

    pairwise = pairwise_lexirecall(
        sasrec_qrels,
        method_rankings["sasrec"],
        method_rankings["marius"],
    )
    pairwise_row = {
        "category": args.category,
        "seed": args.seed,
        "topk": args.topk,
        **{k: f"{v:.6f}" if isinstance(v, float) else v for k, v in pairwise.items()},
    }
    update_json(
        extensions_dir / "lexirecall.json",
        {
            args.category: {
                "seed": args.seed,
                "topk": args.topk,
                "note": (
                    "mean_tse is Total Search Efficiency with reciprocal-rank exposure "
                    "(1/rank of relevant item; Diaz et al. TOIS 2026). "
                    "With one relevant item per query, mean_tse equals MRR."
                ),
                "sasrec": method_summaries["sasrec"],
                "marius": method_summaries["marius"],
                "pairwise_lexirecall_sasrec_vs_marius": pairwise,
            }
        },
    )

    pairwise_path = extensions_dir / "lexirecall_pairwise.csv"
    pairwise_path.parent.mkdir(parents=True, exist_ok=True)
    pairwise_fields = [
        "category",
        "seed",
        "topk",
        "n_queries",
        "mean_lexirecall_pref",
        "fraction_sasrec_preferred",
        "fraction_marius_preferred",
        "fraction_ties",
    ]
    existing_pairwise: list[dict] = []
    if pairwise_path.exists():
        with pairwise_path.open(newline="") as handle:
            existing_pairwise = list(csv.DictReader(handle))
    merged_pairwise = {
        (r["category"], str(r["seed"])): r for r in existing_pairwise
    }
    merged_pairwise[(args.category, str(args.seed))] = pairwise_row
    with pairwise_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pairwise_fields)
        writer.writeheader()
        for r in sorted(merged_pairwise.values(), key=lambda x: (x["category"], int(x["seed"]))):
            writer.writerow(r)

    print(f"Pairwise lexirecall ({args.category}): {pairwise}", flush=True)
    print(f"Wrote {extensions_dir / 'lexirecall_metrics.csv'}", flush=True)
    print(f"Wrote {extensions_dir / 'lexirecall.json'}", flush=True)
    print(f"TREC files under {trec_dir}", flush=True)

    ray.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())