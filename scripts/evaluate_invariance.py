#!/usr/bin/env python
"""Evaluate representation consistency and cache selected hidden layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

from polyfid.evaluation.embeddings import align_embeddings, save_embedding_table
from polyfid.evaluation.encoder import (
    extract_hidden_layer_embeddings,
    load_encoder_checkpoint,
)
from polyfid.evaluation.invariance import (
    build_invariance_views,
    score_applicability_controls,
    score_layer_drift,
    score_view_consistency,
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
        raise argparse.ArgumentTypeError("--layers must contain at least one integer")
    return layers


def _layer_name(layer: int) -> str:
    return f"m{abs(layer)}" if layer < 0 else str(layer)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure randomized-SMILES, periodic-cut, and atom-order consistency. "
            "H-capped and marker-substituted inputs are written separately as "
            "applicability controls."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--text-column", default="canonical_psmiles")
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--layers", type=_layers, default=_layers("0,3,6,9,12"))
    parser.add_argument("--randomized-views", type=int, default=10)
    parser.add_argument("--periodic-views", type=int, default=10)
    parser.add_argument("--atom-permutations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.time()

    frame = _read_frame(args.dataset)
    if args.max_rows is not None:
        frame = frame.iloc[: args.max_rows].copy()
    frame = frame.reset_index(drop=True)
    sample_ids = frame[args.sample_id_column].astype(str).tolist()
    texts = frame[args.text_column].astype(str).tolist()
    view_rows, semantic_rows, scope = build_invariance_views(
        frame,
        sample_id_column=args.sample_id_column,
        text_column=args.text_column,
        randomized_count=args.randomized_views,
        periodic_count=args.periodic_views,
        atom_permutation_count=args.atom_permutations,
        seed=args.seed,
    )

    paths = load_paths()
    base_checkpoint = args.base_checkpoint or str(
        paths.shared_model_root / "chemberta_100m_mlm"
    )
    model, tokenizer = load_encoder_checkpoint(
        args.checkpoint, base_checkpoint=base_checkpoint, device=args.device
    )
    original_tables = extract_hidden_layer_embeddings(
        model,
        tokenizer,
        sample_ids,
        texts,
        layers=args.layers,
        method=args.method,
        representation_seed=args.seed,
        input_view="reference",
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(view_rows, destination / "equivalent_views.parquet")
    atomic_write_parquet(
        semantic_rows,
        destination / "applicability_control_views.parquet",
    )
    for layer, table in original_tables.items():
        save_embedding_table(
            table, destination / f"reference_layer_{_layer_name(layer)}.npz"
        )

    if not view_rows.empty:
        view_tables = extract_hidden_layer_embeddings(
            model,
            tokenizer,
            view_rows["view_id"].astype(str).tolist(),
            view_rows["view_text"].astype(str).tolist(),
            layers=args.layers,
            method=args.method,
            representation_seed=args.seed,
            input_view="verified_equivalent",
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        reference_arrays = {
            layer: align_embeddings(
                original_tables[layer], view_rows["sample_id"].astype(str).to_numpy()
            )
            for layer in args.layers
        }
        view_arrays = {layer: view_tables[layer].embeddings for layer in args.layers}
        for layer, table in view_tables.items():
            save_embedding_table(
                table, destination / f"equivalent_view_layer_{_layer_name(layer)}.npz"
            )
        view_details, view_summary = score_view_consistency(
            view_rows, reference_arrays, view_arrays
        )
    else:
        view_details = view_rows.assign(
            layer=pd.Series(dtype=int), reference_view_cosine=pd.Series(dtype=float)
        )
        view_summary = pd.DataFrame(
            columns=["view_family", "layer", "n_views", "n_samples", "mean_cosine"]
        )
    view_details["method"] = args.method
    view_details["representation_seed"] = args.seed
    atomic_write_parquet(view_details, destination / "view_metrics.parquet")
    atomic_write_csv(view_summary, destination / "view_summary.csv")

    if not semantic_rows.empty:
        control_tables = extract_hidden_layer_embeddings(
            model,
            tokenizer,
            semantic_rows["control_id"].astype(str).tolist(),
            semantic_rows["control_text"].astype(str).tolist(),
            layers=args.layers,
            method=args.method,
            representation_seed=args.seed,
            input_view="applicability_control",
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        control_references = {
            layer: align_embeddings(
                original_tables[layer], semantic_rows["sample_id"].astype(str).to_numpy()
            )
            for layer in args.layers
        }
        control_arrays = {
            layer: control_tables[layer].embeddings for layer in args.layers
        }
        for layer, table in control_tables.items():
            save_embedding_table(
                table,
                destination / f"applicability_control_layer_{_layer_name(layer)}.npz",
            )
        control_details, control_summary = score_applicability_controls(
            semantic_rows, control_references, control_arrays
        )
    else:
        control_details = semantic_rows.assign(
            layer=pd.Series(dtype=int), reference_control_cosine=pd.Series(dtype=float)
        )
        control_summary = pd.DataFrame(
            columns=["control_family", "layer", "n_controls", "n_samples", "mean_cosine"]
        )
    control_details["method"] = args.method
    control_details["representation_seed"] = args.seed
    atomic_write_parquet(control_details, destination / "semantic_controls.parquet")
    atomic_write_csv(control_summary, destination / "semantic_control_summary.csv")

    drift_summary = pd.DataFrame()
    if args.reference_checkpoint:
        reference_model, reference_tokenizer = load_encoder_checkpoint(
            args.reference_checkpoint,
            base_checkpoint=base_checkpoint,
            device=args.device,
        )
        reference_tables = extract_hidden_layer_embeddings(
            reference_model,
            reference_tokenizer,
            sample_ids,
            texts,
            layers=args.layers,
            method="reference_checkpoint",
            representation_seed=args.seed,
            input_view="reference",
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
        drift_details, drift_summary = score_layer_drift(
            sample_ids,
            {layer: original_tables[layer].embeddings for layer in args.layers},
            {layer: reference_tables[layer].embeddings for layer in args.layers},
        )
        drift_details["method"] = args.method
        drift_details["representation_seed"] = args.seed
        atomic_write_parquet(drift_details, destination / "layer_drift.parquet")
        atomic_write_csv(drift_summary, destination / "layer_drift_summary.csv")

    scope["method"] = args.method
    scope["representation_seed"] = args.seed
    atomic_write_parquet(scope, destination / "scope.parquet")
    report = {
        "method": args.method,
        "representation_seed": args.seed,
        "records": int(len(frame)),
        "repeat_eligible_records": int(scope["repeat_eligible"].sum()),
        "equivalent_views": int(len(view_rows)),
        "applicability_controls": int(len(semantic_rows)),
        "selected_hidden_layers": list(args.layers),
        "equivalence_summary": _json_records(view_summary),
        "applicability_summary": _json_records(control_summary),
        "layer_drift_summary": _json_records(drift_summary),
    }
    atomic_write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        destination / "metrics.json",
    )
    write_run_record(
        RunRecord(
            run_id=f"invariance_{args.method}_seed-{args.seed}",
            stage="invariance_evaluation",
            method=args.method,
            seed=args.seed,
            config_path="configs/analysis/fidelity.yaml",
            command=tuple(sys.argv),
            input_paths=tuple(
                str(value)
                for value in (
                    args.dataset,
                    args.checkpoint,
                    args.reference_checkpoint,
                )
                if value is not None
            ),
            output_path=str(destination),
            status="complete",
            metrics={
                "records": float(len(frame)),
                "repeat_eligible_records": float(scope["repeat_eligible"].sum()),
                "equivalent_views": float(len(view_rows)),
                "applicability_controls": float(len(semantic_rows)),
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
