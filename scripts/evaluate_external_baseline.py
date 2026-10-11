#!/usr/bin/env python3
"""Prepare, run, or convert same-split external polymer baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from polyfid.baselines.external import (
    external_model_text,
    materialize_external_fold,
    read_external_predictions,
)
from polyfid.baselines.grin import GRINRunConfig, probe_grin_load, run_grin_fold
from polyfid.baselines.periogt import (
    PerioGTRunConfig,
    prepare_periogt_cache,
    probe_periogt_load,
    run_periogt_fold,
)
from polyfid.baselines.transpolymer import (
    TransPolymerRunConfig,
    probe_transpolymer_load,
    run_transpolymer_fold,
)
from polyfid.evaluation.adaptation import prepare_downstream_frame
from polyfid.evaluation.predictions import write_predictions


def _label_column(task: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return {"EA": "ea_ev", "IP": "ip_ev"}.get(task.upper(), "label")


def _read_columns(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)
    return set(pd.read_csv(path, nrows=0).columns)


def _text_column(dataset: Path, baseline: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    columns = _read_columns(dataset)
    if baseline.casefold() == "transpolymer":
        candidates = ("structure_only_text", "canonical_text", "canonical_psmiles")
    else:
        candidates = ("canonical_text", "structure_only_text", "canonical_psmiles", "raw_text")
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError("no compatible structure text column was found")


def _frame(args: argparse.Namespace) -> pd.DataFrame:
    return prepare_downstream_frame(
        args.dataset,
        args.split_manifest,
        task=args.task,
        protocol=args.protocol,
        fold=args.fold,
        label_column=_label_column(args.task, args.label_column),
        text_column=_text_column(args.dataset, args.baseline, args.text_column),
    )


def _write_prepare_summary(
    frame: pd.DataFrame, files: tuple[Path, Path, Path], output_dir: Path, baseline: str
) -> None:
    payload = {
        "baseline": baseline,
        "rows": {
            split: int(frame["split"].eq(split).sum())
            for split in ("train", "validation", "test")
        },
        "files": [str(path) for path in files],
        "split_source": "Paper104 project manifest",
    }
    (output_dir / "prepared_fold.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _add_fold_arguments(
    parser: argparse.ArgumentParser, *, include_baseline: bool = True
) -> None:
    if include_baseline:
        parser.add_argument("--baseline", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label-column")
    parser.add_argument("--text-column")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    _add_fold_arguments(prepare)
    prepare.add_argument("--output-dir", type=Path, required=True)

    convert = commands.add_parser("convert")
    _add_fold_arguments(convert)
    convert.add_argument("--prediction-input", type=Path, required=True)
    convert.add_argument("--representation-path")
    convert.add_argument("--seed", type=int, required=True)
    convert.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run-transpolymer")
    _add_fold_arguments(run, include_baseline=False)
    run.set_defaults(baseline="transpolymer")
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--epochs", type=int, default=20)
    run.add_argument("--patience", type=int, default=5)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-length", type=int, default=512)
    run.add_argument("--num-workers", type=int, default=0)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--checkpoint-dir", type=Path)
    run.add_argument(
        "--checkpoint-retention",
        choices=("trainable", "metadata"),
        default="trainable",
    )
    run.add_argument("--output", type=Path, required=True)

    probe = commands.add_parser("probe-transpolymer")
    probe.add_argument("--model-root", type=Path, required=True)
    probe.add_argument("--output", type=Path)

    for name in ("periogt", "grin"):
        graph_probe = commands.add_parser(f"probe-{name}")
        graph_probe.add_argument("--model-root", type=Path, required=True)
        graph_probe.add_argument("--output", type=Path)

    prepare_periogt = commands.add_parser("prepare-periogt")
    _add_fold_arguments(prepare_periogt, include_baseline=False)
    prepare_periogt.set_defaults(baseline="periogt")
    prepare_periogt.add_argument("--model-root", type=Path, required=True)
    prepare_periogt.add_argument("--preprocess-workers", type=int, default=8)
    prepare_periogt.add_argument("--cache-dir", type=Path, required=True)

    run_periogt = commands.add_parser("run-periogt")
    _add_fold_arguments(run_periogt, include_baseline=False)
    run_periogt.set_defaults(baseline="periogt")
    run_periogt.add_argument("--model-root", type=Path, required=True)
    run_periogt.add_argument("--seed", type=int, required=True)
    run_periogt.add_argument("--epochs", type=int, default=50)
    run_periogt.add_argument("--patience", type=int, default=20)
    run_periogt.add_argument("--batch-size", type=int, default=32)
    run_periogt.add_argument("--learning-rate", type=float, default=1e-4)
    run_periogt.add_argument("--dropout", type=float, default=0.05)
    run_periogt.add_argument("--num-workers", type=int, default=2)
    run_periogt.add_argument("--cache-dir", type=Path, required=True)
    run_periogt.add_argument("--output-dir", type=Path, required=True)
    run_periogt.add_argument("--checkpoint-dir", type=Path)
    run_periogt.add_argument(
        "--checkpoint-retention",
        choices=("trainable", "metadata"),
        default="trainable",
    )
    run_periogt.add_argument("--output", type=Path, required=True)

    run_grin = commands.add_parser("run-grin")
    _add_fold_arguments(run_grin, include_baseline=False)
    run_grin.set_defaults(baseline="grin")
    run_grin.add_argument("--model-root", type=Path, required=True)
    run_grin.add_argument("--seed", type=int, required=True)
    run_grin.add_argument("--epochs", type=int, default=500)
    run_grin.add_argument("--patience", type=int, default=50)
    run_grin.add_argument("--batch-size", type=int, default=128)
    run_grin.add_argument("--learning-rate", type=float, default=1e-3)
    run_grin.add_argument("--output-dir", type=Path, required=True)
    run_grin.add_argument("--checkpoint-dir", type=Path)
    run_grin.add_argument(
        "--checkpoint-retention",
        choices=("trainable", "metadata"),
        default="trainable",
    )
    run_grin.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command.startswith("probe-"):
        probe_functions = {
            "probe-transpolymer": probe_transpolymer_load,
            "probe-periogt": probe_periogt_load,
            "probe-grin": probe_grin_load,
        }
        payload = probe_functions[args.command](args.model_root)
        rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return

    frame = _frame(args)
    if args.command == "prepare-periogt":
        inputs = pd.DataFrame(
            {
                "sample_id": frame["sample_id"].astype(str),
                "smiles": external_model_text(frame, "periogt", args.task),
                "value": frame["label"].astype(float),
            }
        )
        prepare_periogt_cache(
            inputs,
            args.cache_dir,
            model_root=args.model_root,
            task=args.task,
            preprocess_workers=args.preprocess_workers,
        )
        return
    if args.command == "prepare":
        files = materialize_external_fold(
            frame, args.output_dir, baseline=args.baseline, task=args.task
        )
        _write_prepare_summary(frame, files, args.output_dir, args.baseline)
        return
    if args.command == "convert":
        records = read_external_predictions(
            frame,
            args.prediction_input,
            baseline=args.baseline,
            task=args.task,
            protocol=args.protocol,
            fold=args.fold,
            seed=args.seed,
            representation_path=args.representation_path,
        )
        write_predictions(records, args.output)
        return

    baseline = str(args.baseline)
    files = materialize_external_fold(
        frame, args.output_dir / "data", baseline=baseline, task=args.task
    )
    _write_prepare_summary(frame, files, args.output_dir / "data", baseline)
    checkpoint_dir = args.checkpoint_dir or args.output_dir / "model"
    if args.command == "run-transpolymer":
        predictions = run_transpolymer_fold(
            args.output_dir / "data",
            args.output_dir / "run",
            model_root=args.model_root,
            seed=args.seed,
            config=TransPolymerRunConfig(
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_workers=args.num_workers,
            ),
            checkpoint_dir=checkpoint_dir,
            checkpoint_retention=args.checkpoint_retention,
        )
        representation = checkpoint_dir / (
            "transpolymer_best.pt"
            if args.checkpoint_retention == "trainable"
            else "model_metadata.json"
        )
    elif args.command == "run-periogt":
        predictions = run_periogt_fold(
            args.output_dir / "data",
            args.cache_dir,
            args.output_dir / "run",
            model_root=args.model_root,
            seed=args.seed,
            config=PerioGTRunConfig(
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dropout=args.dropout,
                num_workers=args.num_workers,
            ),
            checkpoint_dir=checkpoint_dir,
            checkpoint_retention=args.checkpoint_retention,
        )
        representation = checkpoint_dir / (
            "periogt_best.pth"
            if args.checkpoint_retention == "trainable"
            else "model_metadata.json"
        )
    elif args.command == "run-grin":
        predictions = run_grin_fold(
            args.output_dir / "data",
            args.output_dir / "run",
            model_root=args.model_root,
            seed=args.seed,
            config=GRINRunConfig(
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            ),
            checkpoint_dir=checkpoint_dir,
            checkpoint_retention=args.checkpoint_retention,
        )
        representation = checkpoint_dir / (
            "grin.pt"
            if args.checkpoint_retention == "trainable"
            else "model_metadata.json"
        )
    else:
        raise ValueError(f"unsupported execution command {args.command!r}")
    records = read_external_predictions(
        frame,
        predictions,
        baseline=baseline,
        task=args.task,
        protocol=args.protocol,
        fold=args.fold,
        seed=args.seed,
        representation_path=str(representation),
    )
    write_predictions(records, args.output)


if __name__ == "__main__":
    main()
