#!/usr/bin/env python
"""Apply one saved frozen probe to legal equivalent and semantic-control views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.adaptation import prepare_downstream_frame
from polyfid.evaluation.embeddings import load_embedding_table
from polyfid.evaluation.probes import load_probe_state
from polyfid.evaluation.property_views import (
    predict_equivalent_property_views,
    predict_property_applicability_controls,
    write_applicability_control_predictions,
    write_equivalent_view_predictions,
)
from polyfid.records import RunRecord, measured_resources, write_run_record


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    return pd.read_parquet(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--probe-state", required=True)
    parser.add_argument("--reference-embeddings", required=True)
    parser.add_argument("--view-rows", required=True)
    parser.add_argument("--view-embeddings", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--head-seed", type=int, required=True)
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--text-column", default="canonical_text")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--group-id-column", default="group_id")
    parser.add_argument("--semantic-control-rows")
    parser.add_argument("--semantic-control-embeddings")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.time()

    semantic_arguments = (
        args.semantic_control_rows,
        args.semantic_control_embeddings,
    )
    if any(semantic_arguments) and not all(semantic_arguments):
        raise ValueError(
            "semantic-control rows and embeddings must be supplied together"
        )

    frame = prepare_downstream_frame(
        args.dataset,
        args.split_manifest,
        task=args.task,
        protocol=args.protocol,
        fold=args.fold,
        label_column=args.label_column,
        text_column=args.text_column,
    )
    test = frame.loc[frame["split"].eq("test")].copy()
    state = load_probe_state(args.probe_state)
    reference_embeddings = load_embedding_table(args.reference_embeddings)
    view_rows = _read_table(args.view_rows)
    view_embeddings = load_embedding_table(args.view_embeddings)
    result = predict_equivalent_property_views(
        state,
        test,
        reference_embeddings,
        view_rows,
        view_embeddings,
        task=args.task,
        head_seed=args.head_seed,
        protocol=args.protocol,
        sample_id_column=args.sample_id_column,
        group_id_column=args.group_id_column,
        label_column="label",
        reference_representation_path=args.reference_embeddings,
        view_representation_path=args.view_embeddings,
        batch_size=args.batch_size,
        device=args.device,
    )
    destination = Path(args.output)
    written = write_equivalent_view_predictions(result, destination)

    semantic_rows = 0
    semantic_output: Path | None = None
    if all(semantic_arguments):
        controls = predict_property_applicability_controls(
            state,
            test,
            reference_embeddings,
            _read_table(args.semantic_control_rows),
            load_embedding_table(args.semantic_control_embeddings),
            task=args.task,
            head_seed=args.head_seed,
            protocol=args.protocol,
            sample_id_column=args.sample_id_column,
            group_id_column=args.group_id_column,
            label_column="label",
            reference_representation_path=args.reference_embeddings,
            control_representation_path=args.semantic_control_embeddings,
            batch_size=args.batch_size,
            device=args.device,
        )
        semantic_rows = int(len(controls))
        semantic_output = write_applicability_control_predictions(
            controls,
            destination / "semantic_control_property_predictions.parquet",
        )

    report = {
        "task": args.task,
        "protocol": args.protocol,
        "fold": args.fold,
        "head_seed": args.head_seed,
        "probe_state": str(args.probe_state),
        "equivalent_view_rows": int(len(result.per_view)),
        "equivalent_record_family_rows": int(len(result.per_record)),
        "families": int(result.family_summary["view_family"].nunique())
        if not result.family_summary.empty
        else 0,
        "semantic_control_rows": semantic_rows,
        "outputs": {name: str(path) for name, path in written.items()},
        "semantic_control_output": None
        if semantic_output is None
        else str(semantic_output),
    }
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_run_record(
        RunRecord(
            run_id=(
                f"property_views_{reference_embeddings.method}_{args.task}_"
                f"{args.protocol}_fold-{args.fold}_head-{args.head_seed}"
            ),
            stage="property_view_evaluation",
            method=reference_embeddings.method,
            seed=reference_embeddings.representation_seed,
            config_path="configs/analysis/fidelity.yaml",
            command=tuple(sys.argv),
            input_paths=tuple(
                str(value)
                for value in (
                    args.dataset,
                    args.split_manifest,
                    args.probe_state,
                    args.reference_embeddings,
                    args.view_rows,
                    args.view_embeddings,
                    args.semantic_control_rows,
                    args.semantic_control_embeddings,
                )
                if value is not None
            ),
            output_path=str(destination),
            status="complete",
            metrics={
                "equivalent_view_rows": float(len(result.per_view)),
                "equivalent_record_family_rows": float(len(result.per_record)),
                "semantic_control_rows": float(semantic_rows),
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
