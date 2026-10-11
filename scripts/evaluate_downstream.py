#!/usr/bin/env python
"""Train the common frozen probe and emit one row per held-out sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from polyfid.evaluation.embeddings import (
    EmbeddingTable,
    align_embeddings,
    load_embedding_table,
)
from polyfid.evaluation.predictions import write_predictions
from polyfid.evaluation.probes import save_probe_state, train_frozen_probe
from polyfid.io import atomic_write_parquet
from polyfid.records import (
    PredictionRecord,
    RunRecord,
    measured_resources,
    write_run_record,
)


HEAD_SEEDS = (11, 22, 33)


class DownstreamInputCache:
    """Reuse immutable datasets, split tables and embeddings within one CPU worker."""

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._embeddings: dict[str, EmbeddingTable] = {}

    def read_parquet(self, path: str | Path) -> pd.DataFrame:
        key = str(path)
        if key not in self._frames:
            self._frames[key] = pd.read_parquet(key)
        return self._frames[key]

    def load_embeddings(self, path: str | Path) -> EmbeddingTable:
        key = str(path)
        if key not in self._embeddings:
            self._embeddings[key] = load_embedding_table(key)
        return self._embeddings[key]


def _select_training_fraction(
    frame: pd.DataFrame,
    fraction: float,
    *,
    repeat: int,
) -> pd.DataFrame:
    if fraction >= 1.0:
        return frame
    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    rng = np.random.default_rng(3107 + repeat)
    ordered = groups[rng.permutation(len(groups))]
    count = max(1, int(np.ceil(fraction * len(groups))))
    return frame.loc[frame["group_id"].astype(str).isin(set(ordered[:count]))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--subsample-id", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--task-type", choices=("regression", "binary"), default="regression")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metrics-output",
        help="Validation-only fit summary; defaults beside the prediction parquet.",
    )
    parser.add_argument(
        "--probe-state-dir",
        help="Optional directory for one reusable .npz state per head seed.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Frozen heads train locally on cached embeddings; CPU is the default.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def evaluate(
    args: argparse.Namespace,
    *,
    input_cache: DownstreamInputCache | None = None,
    command: tuple[str, ...] | None = None,
) -> None:
    started = time.time()
    inputs = input_cache or DownstreamInputCache()

    records = inputs.read_parquet(args.dataset)
    if "task" in records:
        records = records.loc[records["task"].astype(str).str.casefold().eq(args.task.casefold())]
    roles = inputs.read_parquet(args.split_manifest)
    if "task" in roles:
        roles = roles.loc[roles["task"].astype(str).str.casefold().eq(args.task.casefold())]
    if "evaluation_fold" in roles:
        roles = roles.loc[roles["evaluation_fold"].eq(args.fold)]
    elif "fold" in roles and roles["fold"].nunique() > 1:
        roles = roles.loc[roles["fold"].eq(args.fold)]
    role_columns = [column for column in roles.columns if column != "group_id"]
    merge_keys = ["sample_id"]
    if "task" in records and "task" in roles:
        merge_keys.append("task")
    frame = records.merge(
        roles.loc[:, role_columns],
        on=merge_keys,
        how="inner",
        suffixes=("", "_role"),
    )
    if "split_role" in frame:
        frame["split"] = frame["split_role"]
    if "monomer_pair_group_id" in frame and args.protocol in {
        "pair_id", "family_ood", "butina_cluster", "random"
    }:
        frame["group_id"] = frame["monomer_pair_group_id"]
    elif "group_id" not in frame:
        frame["group_id"] = frame["sample_id"]
    embedding_table = inputs.load_embeddings(args.embeddings)
    frame_embeddings = align_embeddings(
        embedding_table, frame["sample_id"].astype(str).to_numpy()
    )
    frame = frame.reset_index(drop=True)
    frame["embedding_row"] = np.arange(len(frame))
    train = _select_training_fraction(
        frame.loc[frame["split"].eq("train")],
        args.label_fraction,
        repeat=args.subsample_id,
    )
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            f"split {args.protocol}/{args.fold} has train={len(train)}, "
            f"validation={len(validation)}, test={len(test)} rows"
        )
    outputs: list[PredictionRecord] = []
    validation_rows: list[dict[str, object]] = []
    for head_seed in HEAD_SEEDS:
        result = train_frozen_probe(
            frame_embeddings[train["embedding_row"]],
            train[args.label_column].to_numpy(dtype=float),
            frame_embeddings[validation["embedding_row"]],
            validation[args.label_column].to_numpy(dtype=float),
            frame_embeddings[test["embedding_row"]],
            seed=head_seed,
            learning_rate=args.learning_rate,
            task_type=args.task_type,
            device=args.device,
        )
        if args.probe_state_dir:
            save_probe_state(
                result.state,
                Path(args.probe_state_dir) / f"head_seed-{head_seed}.npz",
            )
        validation_rows.append(
            {
                "split": "validation",
                "task": args.task,
                "protocol": args.protocol,
                "fold": str(args.fold),
                "label_fraction": float(args.label_fraction),
                "subsample_id": int(args.subsample_id),
                "method": embedding_table.method,
                "representation_seed": int(embedding_table.representation_seed),
                "head_seed": int(head_seed),
                "learning_rate": float(args.learning_rate),
                "validation_loss": float(result.validation_loss),
                "completed_epochs": int(result.epochs),
            }
        )
        for row, prediction in zip(test.to_dict(orient="records"), result.predictions, strict=True):
            eligibility = next(
                (
                    row.get(column)
                    for column in (
                        "structural_fidelity_eligible",
                        "repeat_eligible",
                        "structural_fidelity_eligible_role",
                        "repeat_eligible_role",
                    )
                    if row.get(column) is not None and not pd.isna(row.get(column))
                ),
                None,
            )
            if eligibility is not None and not pd.isna(eligibility):
                eligibility = bool(eligibility)
            else:
                eligibility = None
            scope_class = row.get("scope_class", row.get("scope_class_role"))
            if scope_class is not None and pd.isna(scope_class):
                scope_class = None
            checkpoint_fraction = row.get(
                "checkpoint_fraction", embedding_table.checkpoint_fraction
            )
            if checkpoint_fraction is not None and pd.isna(checkpoint_fraction):
                checkpoint_fraction = embedding_table.checkpoint_fraction
            checkpoint_tokens = row.get(
                "checkpoint_tokens", embedding_table.checkpoint_tokens
            )
            if checkpoint_tokens is not None and pd.isna(checkpoint_tokens):
                checkpoint_tokens = embedding_table.checkpoint_tokens
            outputs.append(
                PredictionRecord(
                    sample_id=str(row["sample_id"]),
                    task=args.task,
                    split="test",
                    group_id=str(row["group_id"]),
                    y_true=float(row[args.label_column]),
                    y_pred=float(prediction),
                    method=embedding_table.method,
                    representation_seed=embedding_table.representation_seed,
                    head_seed=head_seed,
                    protocol=args.protocol,
                    family_id=None if row.get("family_id") is None else str(row["family_id"]),
                    cluster_id=None if row.get("cluster_id") is None else str(row["cluster_id"]),
                    fold=str(args.fold),
                    label_fraction=args.label_fraction,
                    subsample_id=args.subsample_id,
                    input_view=embedding_table.input_view,
                    representation_path=args.embeddings,
                    structural_fidelity_eligible=eligibility,
                    scope_class=None if scope_class is None else str(scope_class),
                    checkpoint_fraction=(
                        None
                        if checkpoint_fraction is None
                        else float(checkpoint_fraction)
                    ),
                    checkpoint_tokens=(
                        None if checkpoint_tokens is None else int(checkpoint_tokens)
                    ),
                )
            )
    write_predictions(outputs, args.output)
    metrics_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else Path(args.output).with_name(
            f"{Path(args.output).stem}.validation.parquet"
        )
    )
    atomic_write_parquet(pd.DataFrame(validation_rows), metrics_path, index=False)
    write_run_record(
        RunRecord(
            run_id=(
                f"frozen_probe_{embedding_table.method}_{args.task}_{args.protocol}_"
                f"fold-{args.fold}_fraction-{args.label_fraction}_"
                f"subsample-{args.subsample_id}"
            ),
            stage="frozen_probe_evaluation",
            method=embedding_table.method,
            seed=embedding_table.representation_seed,
            config_path="configs/downstream/tasks.yaml",
            command=command or tuple(sys.argv),
            input_paths=(
                str(args.dataset),
                str(args.split_manifest),
                str(args.embeddings),
            ),
            output_path=str(args.output),
            status="complete",
            metrics={
                "prediction_rows": float(len(outputs)),
                "test_records": float(len(test)),
                "head_seeds": float(len(HEAD_SEEDS)),
            },
            resources=measured_resources(
                started,
                device=args.device,
            ),
        ),
        Path(args.output).with_suffix(".run_record.json"),
    )


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
