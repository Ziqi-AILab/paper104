#!/usr/bin/env python
"""Evaluate a fixed uniform whole-atom MLM mask on any declared split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.encoder import load_encoder_checkpoint
from polyfid.evaluation.retention import evaluate_uniform_atom_group_mlm
from polyfid.paths import load_paths
from polyfid.records import RunRecord, measured_resources, write_run_record


def _checkpoint_token_progress(path: str | Path) -> int | None:
    """Read the actual non-padding token counter from a training checkpoint."""

    source = Path(path)
    if not source.is_file():
        return None
    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("training_state", {})
    value = state.get("non_padding_tokens")
    return None if value is None else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--representation-seed", type=int, required=True)
    parser.add_argument("--checkpoint-fraction", type=float)
    parser.add_argument("--checkpoint-tokens", type=int)
    parser.add_argument("--text-column", default="canonical_psmiles")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--split")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--mask-seed", type=int, default=10_407)
    parser.add_argument("--mask-probability", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()

    columns = [args.sample_id_column, args.text_column]
    if args.split is not None:
        columns.append(args.split_column)
    frame = pd.read_parquet(args.dataset, columns=list(dict.fromkeys(columns)))
    if args.split is not None:
        frame = frame.loc[
            frame[args.split_column].astype(str).eq(args.split)
        ].copy()
    if args.max_rows is not None:
        frame = frame.iloc[: args.max_rows].copy()
    frame = frame.reset_index(drop=True)

    paths = load_paths()
    base_checkpoint = args.base_checkpoint or str(
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint, base_checkpoint=base_checkpoint, device=args.device
    )
    details, summary = evaluate_uniform_atom_group_mlm(
        model,
        tokenizer,
        frame[args.sample_id_column].astype(str).tolist(),
        frame[args.text_column].astype(str).tolist(),
        seed=args.mask_seed,
        mask_probability=args.mask_probability,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    details["method"] = args.method
    details["representation_seed"] = args.representation_seed
    details["checkpoint_fraction"] = args.checkpoint_fraction
    checkpoint_tokens = (
        args.checkpoint_tokens
        if args.checkpoint_tokens is not None
        else _checkpoint_token_progress(args.checkpoint)
    )
    details["checkpoint_tokens"] = checkpoint_tokens
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    details.to_parquet(output / "token_metrics.parquet", index=False)
    report = {
        **summary,
        "method": args.method,
        "representation_seed": args.representation_seed,
        "checkpoint_fraction": args.checkpoint_fraction,
        "checkpoint_tokens": checkpoint_tokens,
        "dataset": str(args.dataset),
        "split": args.split,
        "text_column": args.text_column,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scalar_metrics = {
        key: float(report[key])
        for key in (
            "records_scored",
            "masked_atom_groups",
            "supervised_tokens",
            "token_mean_nll",
            "token_accuracy",
            "atom_group_mean_nll",
            "atom_group_accuracy",
        )
    }
    write_run_record(
        RunRecord(
            run_id=(
                f"uniform_mlm_{args.method}_seed-{args.representation_seed}_"
                f"fraction-{args.checkpoint_fraction}"
            ),
            stage="fixed_mask_mlm_evaluation",
            method=args.method,
            seed=args.representation_seed,
            config_path="configs/analysis/fidelity.yaml",
            command=tuple(sys.argv),
            input_paths=(str(args.dataset), str(args.checkpoint)),
            output_path=str(output),
            status="complete",
            metrics=scalar_metrics,
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
