#!/usr/bin/env python
"""Evaluate atom-group MLM by attachment distance and cache hidden layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.boundary_mlm import evaluate_atom_group_mlm
from polyfid.evaluation.embeddings import save_embedding_table
from polyfid.evaluation.encoder import (
    extract_hidden_layer_embeddings,
    load_encoder_checkpoint,
)
from polyfid.io import atomic_write_csv, atomic_write_parquet, atomic_write_text
from polyfid.paths import load_paths
from polyfid.records import RunRecord, measured_resources, write_run_record


def _read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_parquet(source)


def _layers(value: str) -> tuple[int, ...]:
    layers = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not layers:
        raise argparse.ArgumentTypeError("hidden layer list cannot be empty")
    return layers


def _layer_name(layer: int) -> str:
    return f"m{abs(layer)}" if layer < 0 else str(layer)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mask every eligible atom group once; report token NLL and accuracy "
            "at attachment distances 0, 1, 2, 3+ and the explicit interior region."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scope", choices=("repeat", "molecule"), default="repeat")
    parser.add_argument("--text-column", default="canonical_psmiles")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--split")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--hidden-layers", type=_layers, default=_layers("0,3,6,9,12"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()

    frame = _read_frame(args.dataset)
    if args.split is not None:
        frame = frame.loc[frame[args.split_column].astype(str).eq(args.split)]
    if args.max_rows is not None:
        frame = frame.iloc[: args.max_rows]
    frame = frame.reset_index(drop=True)
    sample_ids = frame[args.sample_id_column].astype(str).tolist()
    texts = frame[args.text_column].astype(str).tolist()

    paths = load_paths()
    base_checkpoint = args.base_checkpoint or str(
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint, base_checkpoint=base_checkpoint, device=args.device
    )
    details, summary, metadata = evaluate_atom_group_mlm(
        model,
        tokenizer,
        sample_ids,
        texts,
        scope=args.scope,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    details["method"] = args.method
    details["representation_seed"] = args.seed
    summary["method"] = args.method
    summary["representation_seed"] = args.seed

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(details, destination / "token_metrics.parquet")
    atomic_write_csv(summary, destination / "local_mlm_summary.csv")

    layer_tables = extract_hidden_layer_embeddings(
        model,
        tokenizer,
        sample_ids,
        texts,
        layers=args.hidden_layers,
        method=args.method,
        representation_seed=args.seed,
        input_view=f"{args.scope}_reference",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    for layer, table in layer_tables.items():
        save_embedding_table(
            table, destination / f"reference_layer_{_layer_name(layer)}.npz"
        )

    report = {
        **metadata,
        "method": args.method,
        "representation_seed": args.seed,
        "selected_hidden_layers": list(args.hidden_layers),
        "local_mlm_summary": json.loads(summary.to_json(orient="records")),
    }
    atomic_write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        destination / "metrics.json",
    )
    write_run_record(
        RunRecord(
            run_id=f"boundary_mlm_{args.method}_seed-{args.seed}_{args.scope}",
            stage="boundary_mlm_evaluation",
            method=args.method,
            seed=args.seed,
            config_path="configs/analysis/fidelity.yaml",
            command=tuple(sys.argv),
            input_paths=(str(args.dataset), str(args.checkpoint)),
            output_path=str(destination),
            status="complete",
            metrics={
                "records": float(len(frame)),
                "token_rows": float(len(details)),
                "summary_rows": float(len(summary)),
            },
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
