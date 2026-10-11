#!/usr/bin/env python3
"""Run one predeclared ChemBERTa LoRA or full fine-tuning adaptation job."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from polyfid.evaluation.adaptation import (
    AdaptationConfig,
    build_adaptation_model,
    prediction_records,
    prepare_downstream_frame,
    train_adaptation,
    trainable_state,
    validate_adaptation_scope,
)
from polyfid.evaluation.predictions import write_predictions
from polyfid.paths import load_paths
from polyfid.records import RunRecord


def _label_column(task: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    return {"EA": "ea_ev", "IP": "ip_ev"}.get(task.upper(), "label")


def _text_column(dataset: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    import pyarrow.parquet as pq

    columns = set(pq.read_schema(dataset).names)
    for candidate in ("structure_only_text", "canonical_text", "canonical_psmiles"):
        if candidate in columns:
            return candidate
    raise ValueError("no supported structure text column was found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lora", "full"), required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--representation-seed", type=int, required=True)
    parser.add_argument("--checkpoint-fraction", type=float)
    parser.add_argument("--checkpoint-tokens", type=int)
    parser.add_argument("--adaptation-seed", type=int, choices=(11, 22, 33), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label-column")
    parser.add_argument("--text-column")
    parser.add_argument("--task-type", choices=("regression", "binary"), default="regression")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-retention",
        choices=("trainable", "metadata"),
        default="trainable",
        help=(
            "Store the learned trainable tensors, or only compact metadata after "
            "held-out predictions have been written."
        ),
    )
    parser.add_argument("--run-record", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    validate_adaptation_scope(
        arm=args.arm,
        task=args.task,
        protocol=args.protocol,
        method=args.method,
    )
    frame = prepare_downstream_frame(
        args.dataset,
        args.split_manifest,
        task=args.task,
        protocol=args.protocol,
        fold=args.fold,
        label_column=_label_column(args.task, args.label_column),
        text_column=_text_column(args.dataset, args.text_column),
    )
    paths = load_paths()
    base_checkpoint = args.base_checkpoint or (
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    model, tokenizer, lora_layers = build_adaptation_model(
        args.checkpoint,
        base_checkpoint=base_checkpoint,
        method=args.method,
        task_type=args.task_type,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    config = AdaptationConfig(
        method=args.method,
        task_type=args.task_type,
        seed=args.adaptation_seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
    )
    started = time.time()
    result = train_adaptation(frame, model, tokenizer, config)
    test = frame.loc[frame["split"].eq("test")]
    records = prediction_records(
        test,
        result.predictions,
        task=args.task,
        method=args.method,
        arm=args.arm,
        protocol=args.protocol,
        fold=args.fold,
        representation_seed=args.representation_seed,
        adaptation_seed=args.adaptation_seed,
        representation_path=str(args.checkpoint),
        checkpoint_fraction=args.checkpoint_fraction,
        checkpoint_tokens=args.checkpoint_tokens,
    )
    write_predictions(records, args.output)
    model_payload = {
        "artifact_type": "adaptation_trainable_state",
        "checkpoint_retention": args.checkpoint_retention,
        "arm": args.arm,
        "method": args.method,
        "task": args.task,
        "protocol": args.protocol,
        "fold": args.fold,
        "representation_seed": args.representation_seed,
        "adaptation_seed": args.adaptation_seed,
        "lora_layers": lora_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "source_checkpoint": str(args.checkpoint),
        "base_checkpoint": str(base_checkpoint),
        "best_validation_loss": result.best_validation_loss,
        "completed_epochs": result.completed_epochs,
        "trainable_parameters": result.trainable_parameters,
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_retention == "trainable":
        torch.save({**model_payload, "model": trainable_state(model)}, args.model_output)
    else:
        model_payload["artifact_type"] = "adaptation_metadata"
        args.model_output.write_text(
            json.dumps(model_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    model_output_bytes = args.model_output.stat().st_size
    metrics = {
        "best_validation_loss": result.best_validation_loss,
        "completed_epochs": float(result.completed_epochs),
        "test_rows": float(len(test)),
    }
    record = RunRecord(
        run_id=(
            f"adapt_{args.method}_{args.arm}_{args.task}_{args.protocol}"
            f"_fold-{args.fold}_rep-{args.representation_seed}_seed-{args.adaptation_seed}"
        ),
        stage="downstream_adaptation",
        method=f"{args.arm}_{args.method}",
        seed=args.adaptation_seed,
        config_path="configs/downstream/tasks.yaml",
        command=tuple(__import__("sys").argv),
        input_paths=(
            str(args.dataset),
            str(args.split_manifest),
            str(args.checkpoint),
            str(base_checkpoint),
        ),
        output_path=str(args.output),
        status="complete",
        metrics=metrics,
        resources={
            "device": args.device,
            "wall_seconds": time.time() - started,
            "trainable_parameters": result.trainable_parameters,
            "peak_cuda_memory_bytes": result.peak_cuda_memory_bytes,
            "checkpoint_retention": args.checkpoint_retention,
            "model_output_bytes": model_output_bytes,
        },
    )
    args.run_record.parent.mkdir(parents=True, exist_ok=True)
    args.run_record.write_text(
        json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
