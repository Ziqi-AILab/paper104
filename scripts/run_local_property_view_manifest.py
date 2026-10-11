#!/usr/bin/env python3
"""Apply all 100%-data frozen heads for one representation to cached views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from polyfid.evaluation.adaptation import prepare_downstream_frame
from polyfid.evaluation.embeddings import EmbeddingTable, load_embedding_table
from polyfid.io import atomic_write_csv, atomic_write_text
from polyfid.evaluation.probes import load_probe_state
from polyfid.evaluation.property_views import (
    predict_equivalent_property_views,
    write_equivalent_view_predictions,
)


def _text_column(frame: pd.DataFrame, input_view: str) -> str:
    if input_view == "structure_plus_metadata":
        return "structure_with_metadata_text"
    for candidate in ("structure_only_text", "canonical_text", "canonical_psmiles"):
        if candidate in frame:
            return candidate
    raise ValueError("downstream dataset lacks a structure text column")


def _empty_view_table(reference: EmbeddingTable) -> EmbeddingTable:
    return EmbeddingTable(
        sample_ids=np.asarray([], dtype=str),
        embeddings=np.zeros((0, reference.embeddings.shape[1]), dtype=np.float32),
        method=reference.method,
        representation_seed=reference.representation_seed,
        input_view="verified_equivalent_property_view",
        checkpoint_fraction=reference.checkpoint_fraction,
        checkpoint_tokens=reference.checkpoint_tokens,
    )


def run_representation(frame: pd.DataFrame, *, representation_label: str) -> dict[str, object]:
    selected = frame.loc[
        frame["representation_label"].astype(str).eq(representation_label)
    ].copy()
    if selected.empty:
        raise ValueError(f"manifest has no rows for {representation_label}")

    datasets: dict[str, pd.DataFrame] = {}
    manifests: dict[str, pd.DataFrame] = {}
    references: dict[str, EmbeddingTable] = {}
    views: dict[str, pd.DataFrame] = {}
    view_embeddings: dict[str, EmbeddingTable] = {}
    summary_blocks: list[pd.DataFrame] = []
    output_count = 0

    for row in selected.itertuples(index=False):
        if row.dataset not in datasets:
            datasets[row.dataset] = pd.read_parquet(row.dataset)
        dataset = datasets[row.dataset]
        if row.split_manifest not in manifests:
            manifests[row.split_manifest] = pd.read_parquet(row.split_manifest)
        split_manifest = manifests[row.split_manifest]
        if row.reference_embeddings not in references:
            references[row.reference_embeddings] = load_embedding_table(
                row.reference_embeddings
            )
        reference = references[row.reference_embeddings]
        if row.view_rows not in views:
            views[row.view_rows] = pd.read_parquet(row.view_rows)
        view_rows = views[row.view_rows]
        if view_rows.empty:
            transformed = _empty_view_table(reference)
        else:
            if row.view_embeddings not in view_embeddings:
                view_embeddings[row.view_embeddings] = load_embedding_table(
                    row.view_embeddings
                )
            transformed = view_embeddings[row.view_embeddings]
        task_frame = prepare_downstream_frame(
            dataset,
            split_manifest,
            task=row.task,
            protocol=row.protocol,
            fold=int(row.fold),
            label_column=row.label_column,
            text_column=_text_column(dataset, row.input_view),
        )
        test = task_frame.loc[task_frame["split"].eq("test")].copy()
        for head_seed in (11, 22, 33):
            state_path = Path(row.probe_state_dir) / f"head_seed-{head_seed}.npz"
            result = predict_equivalent_property_views(
                load_probe_state(state_path),
                test,
                reference,
                view_rows,
                transformed,
                task=row.task,
                head_seed=head_seed,
                protocol=row.protocol,
                label_column="label",
                reference_representation_path=row.reference_embeddings,
                view_representation_path=(
                    None if view_rows.empty else row.view_embeddings
                ),
                device="cpu",
            )
            destination = Path(row.output_root) / f"head-seed-{head_seed}"
            write_equivalent_view_predictions(result, destination)
            scope = pd.read_parquet(row.scope_rows)
            eligible_test = int(
                (
                    scope["sample_id"]
                    .astype(str)
                    .isin(set(test["sample_id"].astype(str)))
                    & scope["structural_fidelity_eligible"].astype(bool)
                ).sum()
            )
            report = {
                "representation_label": representation_label,
                "task": row.task,
                "protocol": row.protocol,
                "fold": int(row.fold),
                "input_view": row.input_view,
                "head_seed": head_seed,
                "test_records": int(len(test)),
                "eligible_test_records": eligible_test,
                "equivalent_view_predictions": int(len(result.per_view)),
                "record_axis_rows": int(len(result.per_record)),
                "view_families": sorted(
                    result.family_summary["view_family"].astype(str).unique().tolist()
                )
                if not result.family_summary.empty
                else [],
            }
            atomic_write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                destination / "metrics.json",
            )
            if not result.family_summary.empty:
                block = result.family_summary.copy()
                block["input_view"] = row.input_view
                block["fold"] = int(row.fold)
                block["representation_label"] = representation_label
                summary_blocks.append(block)
            output_count += 1

    summary_path = Path(selected.iloc[0]["batch_summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(summary_blocks, ignore_index=True) if summary_blocks else pd.DataFrame()
    atomic_write_csv(summary, summary_path)
    return {
        "representation_label": representation_label,
        "analysis_units": int(len(selected)),
        "head_outputs": output_count,
        "family_summary_rows": int(len(summary)),
        "family_summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--representation-label", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest, sep="\t")
    print(
        json.dumps(
            run_representation(
                frame, representation_label=args.representation_label
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
