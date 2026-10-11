#!/usr/bin/env python
"""Evaluate fixed-mask ZINC50k MLM retention for one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.encoder import load_encoder_checkpoint
from polyfid.evaluation.retention import evaluate_molecular_mlm_retention
from polyfid.io import atomic_write_parquet, atomic_write_text
from polyfid.paths import load_paths
from polyfid.records import RunRecord, measured_resources, write_run_record


def main() -> None:
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(
            paths.shared_data_root
            / "standardized/v1/zinc20/zinc20_1m.parquet"
        ),
    )
    parser.add_argument("--split", default="retention")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--representation-seed", type=int, required=True)
    parser.add_argument("--mask-seed", type=int, default=10_407)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()

    frame = pd.read_parquet(
        args.dataset, columns=["sample_id", "canonical_text", "split"]
    )
    frame = frame.loc[frame["split"].astype(str).eq(args.split)].reset_index(drop=True)
    if args.max_rows is not None:
        frame = frame.iloc[: args.max_rows].copy()
    base_checkpoint = args.base_checkpoint or str(
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint, base_checkpoint=base_checkpoint, device=args.device
    )
    details, summary = evaluate_molecular_mlm_retention(
        model,
        tokenizer,
        frame["sample_id"].astype(str).tolist(),
        frame["canonical_text"].astype(str).tolist(),
        seed=args.mask_seed,
        mask_probability=args.mask_probability,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    details["method"] = args.method
    details["representation_seed"] = args.representation_seed
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(details, output / "token_metrics.parquet")
    report = {
        **summary,
        "method": args.method,
        "representation_seed": args.representation_seed,
        "dataset": str(args.dataset),
        "split": args.split,
    }
    atomic_write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        output / "metrics.json",
    )
    write_run_record(
        RunRecord(
            run_id=(
                f"zinc_retention_{args.method}_seed-{args.representation_seed}"
            ),
            stage="molecular_mlm_retention",
            method=args.method,
            seed=args.representation_seed,
            config_path="configs/analysis/fidelity.yaml",
            command=tuple(sys.argv),
            input_paths=(str(args.dataset), str(args.checkpoint)),
            output_path=str(output),
            status="complete",
            metrics={
                str(key): float(value)
                for key, value in summary.items()
                if isinstance(value, (int, float))
            },
            resources=measured_resources(
                started,
                device=args.device,
                batch_size=args.batch_size,
            ),
        ),
        output / "run_record.json",
    )


if __name__ == "__main__":
    main()
