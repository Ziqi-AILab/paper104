#!/usr/bin/env python3
"""Run same-split classical or Chemprop baselines and emit held-out rows."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from polyfid.baselines.chemprop import (
    check_retained_chemprop,
    chemprop_train_arguments,
    locate_test_predictions,
    prepare_chemprop_files,
    probe_chemprop,
    read_chemprop_predictions,
    run_official_chemprop_train,
    write_chemprop_run_description,
)
from polyfid.baselines.classical import (
    ecfp_features,
    metadata_features,
    nearest_neighbor_predictions,
    rdkit_2d_features,
    structure_smiles,
    train_classical,
)
from polyfid.baselines.external import detect_external_assets
from polyfid.evaluation.adaptation import prediction_records, prepare_downstream_frame
from polyfid.evaluation.predictions import write_predictions
from polyfid.io import atomic_save_npz, atomic_write_text
from polyfid.records import RunRecord


METHODS = (
    "ecfp-ridge",
    "ecfp-xgboost",
    "rdkit-ridge",
    "rdkit-xgboost",
    "tanimoto-nn",
    "nearest-neighbor",
    "chemprop",
)


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


def _seed_values(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("at least one seed is required")
    return values


def _records(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    *,
    task: str,
    method: str,
    protocol: str,
    fold: int,
    seed: int,
    label_fraction: float,
    subsample_id: int,
    input_view: str,
    representation_path: str,
) -> list:
    return prediction_records(
        frame.loc[frame["split"].eq("test")],
        predictions,
        task=task,
        method=method,
        arm="baseline",
        protocol=protocol,
        fold=fold,
        representation_seed=0,
        adaptation_seed=seed,
        representation_path=representation_path,
        label_fraction=label_fraction,
        subsample_id=subsample_id,
        input_view=input_view,
    )


def _select_training_fraction(
    frame: pd.DataFrame,
    *,
    fraction: float,
    subsample_id: int,
) -> pd.DataFrame:
    """Use the same fixed nested group order as the frozen-probe curves."""

    if fraction >= 1.0:
        return frame
    groups = np.asarray(sorted(frame["group_id"].astype(str).unique()))
    rng = np.random.default_rng(3107 + int(subsample_id))
    ordered = groups[rng.permutation(len(groups))]
    count = max(1, int(np.ceil(float(fraction) * len(groups))))
    return frame.loc[frame["group_id"].astype(str).isin(set(ordered[:count]))].copy()


def _metadata_matrix(
    train: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[np.ndarray, ...]:
    if "metadata_sections" not in train:
        raise ValueError("structure_plus_metadata needs the standardized metadata_sections column")
    _, arrays = metadata_features(
        train["metadata_sections"].tolist(),
        *(frame["metadata_sections"].tolist() for frame in others),
    )
    return arrays


def _metric(task: str, task_type: str, truth: np.ndarray, prediction: np.ndarray) -> tuple[str, float]:
    if task_type == "binary":
        from sklearn.metrics import roc_auc_score

        return "roc_auc", float(roc_auc_score(truth, prediction))
    if str(task).upper() in {"EA", "IP"}:
        return "mae", float(np.mean(np.abs(truth - prediction)))
    return "rmse", float(np.sqrt(np.mean(np.square(truth - prediction))))


def _write_run_records(records: list[RunRecord], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    if path.suffix == ".json" and len(payload) == 1:
        atomic_write_text(json.dumps(payload[0], indent=2) + "\n", path)
        return
    atomic_write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload),
        path,
    )


def _classical_features(
    frame: pd.DataFrame,
    *,
    method: str,
    cache_path: Path | None,
) -> np.ndarray:
    feature_function = rdkit_2d_features if method.startswith("rdkit") else ecfp_features
    # Store text as a NumPy Unicode array so the shared cache remains readable
    # with ``allow_pickle=False``.
    sample_ids = np.asarray(frame["sample_id"].astype(str).tolist(), dtype=np.str_)
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_ids = cached["sample_ids"].astype(str)
            cached_features = cached["features"].astype(np.float32, copy=False)
        if set(cached_ids) == set(sample_ids):
            locations = {sample_id: index for index, sample_id in enumerate(cached_ids)}
            return cached_features[[locations[sample_id] for sample_id in sample_ids]]
    features = feature_function(structure_smiles(frame))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_save_npz(
            cache_path,
            {"sample_ids": sample_ids, "features": features},
        )
    return features


def run_baseline(args: argparse.Namespace) -> None:
    started = time.time()
    text_column = _text_column(args.dataset, args.text_column)
    frame = prepare_downstream_frame(
        args.dataset,
        args.split_manifest,
        task=args.task,
        protocol=args.protocol,
        fold=args.fold,
        label_column=_label_column(args.task, args.label_column),
        text_column=text_column,
    )
    frame["_baseline_feature_row"] = np.arange(len(frame), dtype=np.int64)
    full_frame = frame
    train = _select_training_fraction(
        frame.loc[frame["split"].eq("train")],
        fraction=args.label_fraction,
        subsample_id=args.subsample_id,
    )
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    frame = pd.concat([train, validation, test], ignore_index=True)
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    seeds = _seed_values(args.seeds)
    output_records = []
    completed_runs: list[RunRecord] = []
    metadata = args.input_view == "structure_plus_metadata"
    run_cell = (
        f"{args.task}:{args.protocol}:{args.input_view}:fold-{args.fold}:"
        f"fraction-{args.label_fraction:g}:subsample-{args.subsample_id}"
    )

    if args.method == "chemprop":
        import torch

        work_root = args.work_dir
        all_metadata = None
        if metadata:
            encoder, _ = metadata_features(train["metadata_sections"].tolist())
            all_metadata = encoder.transform(frame["metadata_sections"].tolist())
        for seed in seeds:
            seed_started = time.time()
            torch.cuda.reset_peak_memory_stats()
            seed_root = work_root / f"seed-{seed}"
            persistent_model_dir = (
                args.checkpoint_dir / f"seed-{seed}"
                if args.checkpoint_dir is not None
                else seed_root / "model"
            )
            temporary_runtime = (
                tempfile.TemporaryDirectory(
                    prefix="paper104_chemprop_",
                    dir=os.environ.get("SLURM_TMPDIR"),
                )
                if args.checkpoint_retention == "metadata"
                else nullcontext(None)
            )
            with temporary_runtime as temporary_root:
                runtime_root = (
                    Path(temporary_root) if temporary_root is not None else seed_root
                )
                files = prepare_chemprop_files(
                    frame,
                    runtime_root / "data",
                    additional_features=all_metadata,
                )
                model_dir = (
                    runtime_root / "model"
                    if temporary_root is not None
                    else persistent_model_dir
                )
                arguments = chemprop_train_arguments(
                    files,
                    save_dir=model_dir,
                    seed=seed,
                    task_type=args.task_type,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    cuda=True,
                )
                probe = probe_chemprop()
                write_chemprop_run_description(
                    runtime_root / "run.json",
                    probe=probe,
                    arguments=arguments,
                    files=files,
                )
                run_official_chemprop_train(arguments)
                peak_device_memory_bytes = int(torch.cuda.max_memory_allocated())
                predictions = read_chemprop_predictions(
                    locate_test_predictions(model_dir), expected_rows=len(test)
                )
                if args.checkpoint_retention == "trainable":
                    check_retained_chemprop(
                        model_dir, files, predictions,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                    )
                checkpoint_bytes = int(
                    sum(
                        path.stat().st_size
                        for path in model_dir.rglob("*")
                        if path.is_file()
                    )
                )
            if args.checkpoint_retention == "metadata":
                persistent_model_dir.mkdir(parents=True, exist_ok=True)
                model_artifact = persistent_model_dir / "model_metadata.json"
                model_artifact.write_text(
                    json.dumps(
                        {
                            "artifact_type": "chemprop_model_metadata",
                            "checkpoint_retention": "metadata",
                            "method": "chemprop_dpnn",
                            "task": args.task,
                            "protocol": args.protocol,
                            "fold": args.fold,
                            "seed": seed,
                            "label_fraction": args.label_fraction,
                            "subsample_id": args.subsample_id,
                            "epochs": args.epochs,
                            "validation_selected_model_bytes": checkpoint_bytes,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                model_artifact = persistent_model_dir
            model_output_bytes = int(
                model_artifact.stat().st_size
                if model_artifact.is_file()
                else sum(
                    path.stat().st_size
                    for path in model_artifact.rglob("*")
                    if path.is_file()
                )
            )
            output_records.extend(
                _records(
                    frame,
                    predictions,
                    task=args.task,
                    method="chemprop_dpnn",
                    protocol=args.protocol,
                    fold=args.fold,
                    seed=seed,
                    label_fraction=args.label_fraction,
                    subsample_id=args.subsample_id,
                    input_view=args.input_view,
                    representation_path=str(model_artifact),
                )
            )
            metric_name, metric_value = _metric(
                args.task,
                args.task_type,
                test["label"].to_numpy(dtype=float),
                predictions,
            )
            completed_runs.append(
                RunRecord(
                    run_id=f"{args.method}:{run_cell}:seed-{seed}",
                    stage="downstream_controlled_baseline",
                    method="chemprop_dpnn",
                    seed=seed,
                    config_path=str(args.config_path),
                    command=tuple(sys.argv),
                    input_paths=(str(args.dataset), str(args.split_manifest)),
                    output_path=str(args.output),
                    status="complete",
                    metrics={metric_name: metric_value},
                    resources={
                        "execution_path": "[GPU-A800 → sbatch]",
                        "gpu_count": 1,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "peak_device_memory_bytes": peak_device_memory_bytes,
                        "checkpoint_retention": args.checkpoint_retention,
                        "validation_selected_model_bytes": checkpoint_bytes,
                        "model_output_bytes": model_output_bytes,
                        "wall_seconds": time.time() - seed_started,
                    },
                )
            )
    else:
        full_features = _classical_features(
            full_frame,
            method=args.method,
            cache_path=args.feature_cache,
        )
        train_x = full_features[train["_baseline_feature_row"].to_numpy(dtype=int)]
        test_x = full_features[test["_baseline_feature_row"].to_numpy(dtype=int)]
        if metadata:
            train_metadata, test_metadata = _metadata_matrix(train, test)
            train_x = np.concatenate((train_x, train_metadata), axis=1)
            test_x = np.concatenate((test_x, test_metadata), axis=1)
        train_y = train["label"].to_numpy(dtype=float)
        if args.method in {"tanimoto-nn", "nearest-neighbor"}:
            predictions, _ = nearest_neighbor_predictions(
                train_x,
                train_y,
                test_x,
                metric="euclidean" if metadata else "tanimoto",
            )
            seed = seeds[0]
            output_records.extend(
                _records(
                    frame,
                    predictions,
                    task=args.task,
                    method=args.method.replace("-", "_"),
                    protocol=args.protocol,
                    fold=args.fold,
                    seed=seed,
                    label_fraction=args.label_fraction,
                    subsample_id=args.subsample_id,
                    input_view=args.input_view,
                    representation_path="",
                )
            )
            metric_name, metric_value = _metric(
                args.task,
                args.task_type,
                test["label"].to_numpy(dtype=float),
                predictions,
            )
            completed_runs.append(
                RunRecord(
                    run_id=f"nearest_neighbor:{run_cell}:seed-{seed}",
                    stage="downstream_controlled_baseline",
                    method="nearest_neighbor",
                    seed=seed,
                    config_path=str(args.config_path),
                    command=tuple(sys.argv),
                    input_paths=tuple(
                        path
                        for path in (
                            str(args.dataset),
                            str(args.split_manifest),
                            None if args.feature_cache is None else str(args.feature_cache),
                        )
                        if path is not None
                    ),
                    output_path=str(args.output),
                    status="complete",
                    metrics={metric_name: metric_value},
                    resources={
                        "execution_path": "[CPU → local, no sbatch]",
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "wall_seconds": time.time() - started,
                    },
                )
            )
        else:
            estimator = args.method.rsplit("-", 1)[-1]
            for seed in seeds:
                seed_started = time.time()
                predictions = train_classical(
                    train_x,
                    train_y,
                    test_x,
                    estimator=estimator,
                    seed=seed,
                    task_type=args.task_type,
                )
                output_records.extend(
                    _records(
                        frame,
                        predictions,
                        task=args.task,
                        method=args.method.replace("-", "_"),
                        protocol=args.protocol,
                        fold=args.fold,
                        seed=seed,
                        label_fraction=args.label_fraction,
                        subsample_id=args.subsample_id,
                        input_view=args.input_view,
                        representation_path="",
                    )
                )
                metric_name, metric_value = _metric(
                    args.task,
                    args.task_type,
                    test["label"].to_numpy(dtype=float),
                    predictions,
                )
                completed_runs.append(
                    RunRecord(
                        run_id=f"{args.method}:{run_cell}:seed-{seed}",
                        stage="downstream_controlled_baseline",
                        method=args.method.replace("-", "_"),
                        seed=seed,
                        config_path=str(args.config_path),
                        command=tuple(sys.argv),
                        input_paths=tuple(
                            path
                            for path in (
                                str(args.dataset),
                                str(args.split_manifest),
                                None if args.feature_cache is None else str(args.feature_cache),
                            )
                            if path is not None
                        ),
                        output_path=str(args.output),
                        status="complete",
                        metrics={metric_name: metric_value},
                        resources={
                            "execution_path": "[CPU → local, no sbatch]",
                            "train_rows": len(train),
                            "validation_rows": len(validation),
                            "test_rows": len(test),
                            "wall_seconds": time.time() - seed_started,
                        },
                    )
                )
    write_predictions(output_records, args.output)
    _write_run_records(completed_runs, args.run_record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--split-manifest", type=Path, required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--protocol", required=True)
    run.add_argument("--fold", type=int, default=0)
    run.add_argument("--label-column")
    run.add_argument("--text-column")
    run.add_argument("--task-type", choices=("regression", "binary"), default="regression")
    run.add_argument("--seeds", default="11,22,33")
    run.add_argument("--label-fraction", type=float, default=1.0)
    run.add_argument("--subsample-id", type=int, default=0)
    run.add_argument("--input-view", default="structure_only")
    run.add_argument("--feature-cache", type=Path)
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--batch-size", type=int, default=50)
    run.add_argument("--num-workers", type=int, default=8)
    run.add_argument("--work-dir", type=Path, default=Path("runs/baselines/chemprop"))
    run.add_argument("--checkpoint-dir", type=Path)
    run.add_argument(
        "--checkpoint-retention",
        choices=("trainable", "metadata"),
        default="trainable",
    )
    run.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/downstream/controlled_baselines.yaml"),
    )
    run.add_argument("--run-record", type=Path)
    run.add_argument("--output", type=Path, required=True)

    probe = subparsers.add_parser("probe-external")
    probe.add_argument("--model-root", type=Path, required=True)
    probe.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run_baseline(args)
        return
    payload = [status.as_dict() for status in detect_external_assets(args.model_root)]
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
