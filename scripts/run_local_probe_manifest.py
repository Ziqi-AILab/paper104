#!/usr/bin/env python3
"""Run selected rows from the Paper104 frozen-probe manifest on local CPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from polyfid.evaluation.probe_selection import PROPERTY_HEAD_LEARNING_RATES


HEAD_SEEDS = (11, 22, 33)


def selected_learning_rate(path: str | Path) -> float:
    source = Path(path)
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    recipe = values.get("selected_recipe", values)
    learning_rate = float(recipe["property_head_learning_rate"])
    if learning_rate not in PROPERTY_HEAD_LEARNING_RATES:
        raise ValueError(
            "selected property-head learning rate is outside the declared candidates"
        )
    return learning_rate


def command_for_row(row: pd.Series, *, python: str | Path) -> list[str]:
    declared_learning_rate = row.get("learning_rate")
    if declared_learning_rate is not None and not pd.isna(declared_learning_rate) and str(declared_learning_rate).strip():
        learning_rate = float(declared_learning_rate)
    else:
        learning_rate = selected_learning_rate(str(row["learning_rate_recipe"]))
    command = [
        str(python),
        "scripts/evaluate_downstream.py",
        "--dataset",
        str(row["dataset"]),
        "--split-manifest",
        str(row["split_manifest"]),
        "--embeddings",
        str(row["embeddings"]),
        "--task",
        str(row["task"]),
        "--protocol",
        str(row["protocol"]),
        "--label-column",
        str(row["label_column"]),
        "--fold",
        str(int(row["fold"])),
        "--label-fraction",
        f"{float(row['label_fraction']):g}",
        "--subsample-id",
        str(int(row["subsample_id"])),
        "--learning-rate",
        f"{learning_rate:g}",
        "--task-type",
        str(row["task_type"]),
        "--device",
        "cpu",
        "--output",
        str(row["output"]),
    ]
    if _present(row.get("probe_state_dir")):
        command.extend(("--probe-state-dir", str(row["probe_state_dir"])))
    metrics_output = row.get("metrics_output")
    if metrics_output is not None and not pd.isna(metrics_output) and str(metrics_output).strip():
        command.extend(("--metrics-output", str(metrics_output)))
    return command


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value) and bool(str(value).strip())


def row_is_complete(row: pd.Series) -> bool:
    output = Path(str(row["output"]))
    metrics = (
        Path(str(row["metrics_output"]))
        if _present(row.get("metrics_output"))
        else output.with_name(f"{output.stem}.validation.parquet")
    )
    run_record = output.with_suffix(".run_record.json")
    required = [output, metrics, run_record]
    if _present(row.get("probe_state_dir")):
        state_dir = Path(str(row["probe_state_dir"]))
        required.extend(state_dir / f"head_seed-{seed}.npz" for seed in HEAD_SEEDS)
    if not all(path.is_file() for path in required):
        return False
    try:
        status = json.loads(run_record.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return status == "complete"


def _row_indices(args: argparse.Namespace, row_count: int) -> range:
    if args.row_index is not None:
        if args.row_index < 0 or args.row_index >= row_count:
            raise ValueError(f"row index {args.row_index} is outside 0..{row_count - 1}")
        return range(args.row_index, args.row_index + 1)
    if not args.all:
        raise ValueError("choose --row-index for one probe or --all for a row range")
    start = args.start_index
    stop = row_count if args.stop_index is None else args.stop_index
    if start < 0 or stop < start or stop > row_count:
        raise ValueError(f"requested row range {start}:{stop} is outside 0:{row_count}")
    return range(start, stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row-index", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip rows whose predictions, validation metrics, head states and final run record exist.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Reuse loaded datasets and embeddings across a local row range.",
    )
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.manifest, sep="\t")
    indices = _row_indices(args, len(frame))
    evaluator = None
    input_cache = None
    parse_evaluation_args = None
    if args.in_process and not args.dry_run:
        import torch

        from evaluate_downstream import (
            DownstreamInputCache,
            evaluate,
            parse_args as parse_downstream_args,
        )

        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        torch.set_num_interop_threads(1)
        evaluator = evaluate
        input_cache = DownstreamInputCache()
        parse_evaluation_args = parse_downstream_args
    processed = 0
    for index in indices:
        if args.skip_complete and row_is_complete(frame.iloc[index]):
            print(f"row_index={index} already complete; skipping", flush=True)
        else:
            command = command_for_row(frame.iloc[index], python=args.python)
            if args.dry_run:
                print(subprocess.list2cmdline(command))
            elif args.in_process:
                assert evaluator is not None
                assert input_cache is not None
                assert parse_evaluation_args is not None
                evaluator(
                    parse_evaluation_args(command[2:]),
                    input_cache=input_cache,
                    command=tuple(command[1:]),
                )
            else:
                subprocess.run(command, check=True)
        processed += 1
        if args.progress_every > 0 and processed % args.progress_every == 0:
            print(
                f"local probe rows processed in this shard: {processed}",
                flush=True,
            )


if __name__ == "__main__":
    main()
