#!/usr/bin/env python3
"""Cache all formal downstream equivalent-view partitions in one A800 run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from polyfid.evaluation.embeddings import save_embedding_table
from polyfid.evaluation.encoder import extract_embeddings, load_encoder_checkpoint
from polyfid.io import atomic_write_csv, atomic_write_text
from polyfid.paths import load_paths
from polyfid.records import RunRecord, measured_resources, write_run_record


def main() -> None:
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=paths.project_root / "data/derived/property_views/registry.tsv",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-record", type=Path)
    args = parser.parse_args()

    started = time.time()
    if args.device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    registry = pd.read_csv(args.registry, sep="\t")
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint,
        base_checkpoint=(
            args.base_checkpoint
            or str(paths.shared_model_root / "chemberta_100m_mlm")
        ),
        device=args.device,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    for row in registry.itertuples(index=False):
        frame = pd.read_parquet(row.view_rows)
        output_path = args.output / f"{row.partition_id}.npz"
        if frame.empty:
            output_rows.append(
                {
                    "partition_id": row.partition_id,
                    "view_rows": row.view_rows,
                    "view_embeddings": "",
                    "views": 0,
                    "status": "no_eligible_views",
                }
            )
            continue
        table = extract_embeddings(
            model,
            tokenizer,
            frame["view_id"].astype(str).tolist(),
            frame["view_text"].astype(str).tolist(),
            method=args.method,
            representation_seed=args.seed,
            input_view="verified_equivalent_property_view",
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        save_embedding_table(table, output_path)
        output_rows.append(
            {
                "partition_id": row.partition_id,
                "view_rows": row.view_rows,
                "view_embeddings": str(output_path),
                "views": int(len(frame)),
                "status": "complete",
            }
        )

    output_registry = pd.DataFrame(output_rows)
    atomic_write_csv(
        output_registry,
        args.output / "registry.tsv",
        index=False,
        sep="\t",
    )
    report = {
        "method": args.method,
        "representation_seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": str(
            args.base_checkpoint
            or paths.shared_model_root / "chemberta_100m_mlm"
        ),
        "partitions": int(len(output_registry)),
        "encoded_partitions": int(output_registry["status"].eq("complete").sum()),
        "encoded_views": int(output_registry["views"].sum()),
        "output_registry": str(args.output / "registry.tsv"),
    }
    atomic_write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        args.output / "metrics.json",
    )
    run_record_path = args.run_record or args.output / "run_record.json"
    write_run_record(
        RunRecord(
            run_id=f"property_views_{args.method}_seed-{args.seed}",
            stage="property_view_embedding_cache",
            method=args.method,
            seed=int(args.seed),
            config_path=str(
                paths.project_root / "configs/analysis/property_views.yaml"
            ),
            command=tuple([sys.executable, *sys.argv]),
            input_paths=tuple(
                [
                    str(args.registry),
                    str(args.checkpoint),
                    str(
                        args.base_checkpoint
                        or paths.shared_model_root / "chemberta_100m_mlm"
                    ),
                ]
            ),
            output_path=str(args.output),
            status="complete",
            metrics={
                "partitions": float(report["partitions"]),
                "encoded_partitions": float(report["encoded_partitions"]),
                "encoded_views": float(report["encoded_views"]),
            },
            resources=measured_resources(
                started,
                device=args.device,
                batch_size=int(args.batch_size),
                max_length=int(args.max_length),
            ),
        ),
        run_record_path,
    )


if __name__ == "__main__":
    main()
