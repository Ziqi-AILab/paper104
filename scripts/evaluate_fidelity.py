#!/usr/bin/env python
"""Evaluate fixed-decoy retrieval and topology separation for a checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.encoder import extract_embeddings, load_encoder_checkpoint
from polyfid.evaluation.fidelity import (
    evaluate_relation_embeddings,
    evaluate_topology_controls,
)
from polyfid.io import atomic_write_parquet, atomic_write_text
from polyfid.paths import load_paths
from polyfid.records import RunRecord, measured_resources, write_run_record


def _embedding_blocks(
    model: object,
    tokenizer: object,
    frame: pd.DataFrame,
    *,
    row_id_column: str,
    method: str,
    seed: int,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[object, object, object]:
    blocks = []
    for role in ("anchor", "positive", "negative"):
        blocks.append(
            extract_embeddings(
                model,
                tokenizer,
                [f"{row_id}:{role}" for row_id in frame[row_id_column].astype(str)],
                frame[f"{role}_psmiles"].astype(str).tolist(),
                method=method,
                representation_seed=seed,
                input_view=role,
                max_length=max_length,
                batch_size=batch_size,
                device=device,
            ).embeddings
        )
    return tuple(blocks)


def _supplemental_frame(relations: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    controls = pd.read_parquet(path).copy()
    controls["control_row_id"] = [f"control:{index}" for index in range(len(controls))]
    positives = relations[
        ["anchor_sample_id", "positive_psmiles", "positive_sample_id", "positive_relation"]
    ].drop_duplicates("anchor_sample_id")
    controls = controls.merge(positives, on="anchor_sample_id", how="inner")
    if controls.empty:
        return controls
    controls["anchor_sample_id"] = controls["anchor_sample_id"].astype(str)
    return controls


def _decoy_bank_embeddings(
    model: object,
    tokenizer: object,
    bank: pd.DataFrame,
    *,
    method: str,
    seed: int,
    batch_size: int,
    max_length: int,
    device: str,
) -> object:
    ordered = bank.sort_values("decoy_bank_index", kind="stable").reset_index(drop=True)
    if ordered["decoy_bank_index"].astype(int).tolist() != list(range(len(ordered))):
        raise ValueError("decoy_bank_index must be contiguous from zero")
    return extract_embeddings(
        model,
        tokenizer,
        [f"decoy_bank:{value}" for value in ordered["decoy_bank_index"]],
        ordered["decoy_psmiles"].astype(str).tolist(),
        method=method,
        representation_seed=seed,
        input_view="retrieval_decoy",
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    ).embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Score a verified relation set using its stored matched retrieval decoys; "
            "write one output row per triplet."
        )
    )
    parser.add_argument("--relations", required=True, help="PI1M triplet parquet")
    parser.add_argument(
        "--decoy-bank",
        help=(
            "Fixed decoy-bank parquet from relation construction; defaults to "
            "retrieval_decoy_bank.parquet beside --relations"
        ),
    )
    parser.add_argument(
        "--supplemental-controls",
        help="Optional endpoint-relocation/closure-rewiring control parquet",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()

    paths = load_paths()
    relation_path = Path(args.relations)
    relations = pd.read_parquet(relation_path).reset_index(drop=True)
    if "retrieval_decoy_bank_indices" not in relations.columns:
        raise ValueError(
            "relation table must contain retrieval_decoy_bank_indices; rebuild it "
            "with the partition-level fixed decoy bank"
        )
    decoy_bank_path = (
        Path(args.decoy_bank)
        if args.decoy_bank
        else relation_path.with_name("retrieval_decoy_bank.parquet")
    )
    if not decoy_bank_path.is_file():
        raise FileNotFoundError(
            f"fixed retrieval decoy bank is missing: {decoy_bank_path}"
        )
    decoy_bank = pd.read_parquet(decoy_bank_path)
    base_checkpoint = args.base_checkpoint or str(
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint,
        base_checkpoint=base_checkpoint,
        device=args.device,
    )
    anchors, positives, negatives = _embedding_blocks(
        model,
        tokenizer,
        relations,
        row_id_column="triplet_id",
        method=args.method,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    decoy_embeddings = _decoy_bank_embeddings(
        model,
        tokenizer,
        decoy_bank,
        method=args.method,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )
    details, metrics = evaluate_relation_embeddings(
        relations, anchors, positives, negatives, decoy_embeddings
    )
    details["method"] = args.method
    details["representation_seed"] = args.seed

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(details, destination / "triplet_metrics.parquet")

    control_summaries: dict[str, dict[str, float | int]] = {}
    if args.supplemental_controls:
        controls = _supplemental_frame(relations, args.supplemental_controls)
        if not controls.empty:
            control_embeddings = _embedding_blocks(
                model,
                tokenizer,
                controls,
                row_id_column="control_row_id",
                method=args.method,
                seed=args.seed,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=args.device,
            )
            control_details, control_summaries = evaluate_topology_controls(
                controls, *control_embeddings
            )
            control_details["method"] = args.method
            control_details["representation_seed"] = args.seed
            atomic_write_parquet(
                control_details,
                destination / "topology_control_metrics.parquet",
            )

    report = {
        **metrics,
        "method": args.method,
        "representation_seed": args.seed,
        "decoy_bank_path": str(decoy_bank_path),
        "topology_controls_by_family": control_summaries,
    }
    atomic_write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        destination / "metrics.json",
    )
    scalar_metrics = {
        str(key): float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    write_run_record(
        RunRecord(
            run_id=f"fidelity_{args.method}_seed-{args.seed}",
            stage="fidelity_evaluation",
            method=args.method,
            seed=args.seed,
            config_path="configs/analysis/fidelity.yaml",
            command=tuple(sys.argv),
            input_paths=tuple(
                str(value)
                for value in (
                    relation_path,
                    decoy_bank_path,
                    args.supplemental_controls,
                    args.checkpoint,
                )
                if value is not None
            ),
            output_path=str(destination),
            status="complete",
            metrics=scalar_metrics,
            resources=measured_resources(
                started,
                device=args.device,
                batch_size=args.batch_size,
            ),
        ),
        destination / "run_record.json",
    )


if __name__ == "__main__":
    main()
