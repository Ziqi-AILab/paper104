"""Apply one fitted property head to verified structural view caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import EmbeddingTable, align_embeddings
from .probes import ProbeState, predict_with_probe
from polyfid.io import atomic_write_csv, atomic_write_parquet


@dataclass(frozen=True)
class EquivalentViewPredictionResult:
    """Detailed predictions and record-equal summaries for legal equivalents."""

    per_view: pd.DataFrame
    per_record: pd.DataFrame
    family_summary: pd.DataFrame


_RECORD_METADATA = (
    "split",
    "family_id",
    "cluster_id",
    "fold",
    "label_fraction",
    "subsample_id",
    "structural_fidelity_eligible",
    "scope_class",
    "checkpoint_fraction",
    "checkpoint_tokens",
)


def _task_records(
    records: pd.DataFrame,
    *,
    sample_id_column: str,
    group_id_column: str,
    label_column: str,
    default_split: str,
) -> pd.DataFrame:
    columns = [sample_id_column, label_column]
    columns.extend(
        column
        for column in (group_id_column, *_RECORD_METADATA, "repeat_eligible")
        if column in records
    )
    frame = records.loc[:, list(dict.fromkeys(columns))].copy()
    frame = frame.rename(
        columns={
            sample_id_column: "sample_id",
            group_id_column: "group_id",
            label_column: "y_true",
        }
    )
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError("property records must contain one row per sample for one task")
    if "group_id" not in frame:
        frame["group_id"] = frame["sample_id"]
    else:
        frame["group_id"] = frame["group_id"].astype(str)
    if "split" not in frame:
        frame["split"] = default_split
    if "repeat_eligible" in frame:
        if "structural_fidelity_eligible" not in frame:
            frame["structural_fidelity_eligible"] = frame["repeat_eligible"]
        else:
            frame["structural_fidelity_eligible"] = frame[
                "structural_fidelity_eligible"
            ].fillna(frame["repeat_eligible"])
    return frame


def _confirm_same_representation(
    reference: EmbeddingTable,
    transformed: EmbeddingTable,
) -> None:
    if (
        reference.method != transformed.method
        or reference.representation_seed != transformed.representation_seed
    ):
        raise ValueError("reference and transformed caches must use the same representation")
    for field in ("checkpoint_fraction", "checkpoint_tokens"):
        reference_value = getattr(reference, field)
        transformed_value = getattr(transformed, field)
        if (
            reference_value is not None
            and transformed_value is not None
            and reference_value != transformed_value
        ):
            raise ValueError(f"reference and transformed caches differ in {field}")


def _carry_cache_progress(frame: pd.DataFrame, table: EmbeddingTable) -> pd.DataFrame:
    output = frame.copy()
    for column in ("checkpoint_fraction", "checkpoint_tokens"):
        value = getattr(table, column)
        if column not in output:
            output[column] = value
        elif value is not None:
            output[column] = output[column].fillna(value)
    return output


def _merge_transformed_rows(
    transformed: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Join task metadata while retaining view-specific scope annotations."""

    metadata_values = metadata.copy()
    for column in ("structural_fidelity_eligible", "scope_class"):
        if column in transformed and column in metadata_values:
            metadata_values = metadata_values.drop(columns=column)
    return transformed.merge(
        metadata_values, on="sample_id", how="inner", validate="many_to_one"
    )


def _empty_equivalent_result() -> EquivalentViewPredictionResult:
    per_view = pd.DataFrame(
        columns=[
            "sample_id",
            "task",
            "split",
            "group_id",
            "y_true",
            "method",
            "representation_seed",
            "head_seed",
            "protocol",
            "task_type",
            "view_id",
            "view_family",
            "view_index",
            "input_view",
            "reference_y_pred",
            "y_pred",
            "prediction_delta",
            "analysis_role",
            "structural_fidelity_eligible",
            "scope_class",
            "checkpoint_fraction",
            "checkpoint_tokens",
        ]
    )
    per_record = pd.DataFrame(
        columns=[
            "sample_id",
            "task",
            "split",
            "group_id",
            "method",
            "representation_seed",
            "head_seed",
            "protocol",
            "task_type",
            "view_family",
            "n_equivalent_views",
            "n_predictions",
            "reference_y_pred",
            "mean_y_pred",
            "property_prediction_std",
            "mean_absolute_view_delta",
            "max_absolute_view_delta",
            "structural_fidelity_eligible",
            "scope_class",
            "checkpoint_fraction",
            "checkpoint_tokens",
        ]
    )
    family_summary = pd.DataFrame(
        columns=[
            "task",
            "method",
            "representation_seed",
            "head_seed",
            "protocol",
            "task_type",
            "view_family",
            "n_records",
            "n_equivalent_views",
            "mean_property_prediction_std",
            "median_property_prediction_std",
            "mean_absolute_view_delta",
            "structural_fidelity_eligible",
            "scope_class",
            "checkpoint_fraction",
            "checkpoint_tokens",
        ]
    )
    return EquivalentViewPredictionResult(per_view, per_record, family_summary)


def predict_equivalent_property_views(
    state: ProbeState,
    records: pd.DataFrame,
    reference_embeddings: EmbeddingTable,
    view_rows: pd.DataFrame,
    view_embeddings: EmbeddingTable,
    *,
    task: str,
    head_seed: int,
    protocol: str = "default",
    sample_id_column: str = "sample_id",
    group_id_column: str = "group_id",
    label_column: str = "label",
    default_split: str = "test",
    reference_representation_path: str | None = None,
    view_representation_path: str | None = None,
    batch_size: int = 4096,
    device: str = "cpu",
) -> EquivalentViewPredictionResult:
    """Predict legal equivalents with exactly the same fitted property head.

    The within-record standard deviation contains the valid reference once and
    every available transformed view in that family.  Regression values remain
    in the task's original units; binary values are predicted probabilities.
    """

    _confirm_same_representation(reference_embeddings, view_embeddings)
    if "analysis_role" in view_rows and not view_rows["analysis_role"].eq(
        "equivalence_consistency"
    ).all():
        raise ValueError("applicability controls must be predicted separately")
    metadata = _task_records(
        records,
        sample_id_column=sample_id_column,
        group_id_column=group_id_column,
        label_column=label_column,
        default_split=default_split,
    )
    selected = view_rows.loc[
        view_rows["sample_id"].astype(str).isin(set(metadata["sample_id"]))
    ].copy()
    if selected.empty:
        return _empty_equivalent_result()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected = _merge_transformed_rows(selected, metadata)
    selected = _carry_cache_progress(selected, reference_embeddings)

    reference_x = align_embeddings(
        reference_embeddings, metadata["sample_id"].to_numpy(dtype=str)
    )
    reference_prediction = predict_with_probe(
        state, reference_x, batch_size=batch_size, device=device
    )
    reference_by_id = dict(zip(metadata["sample_id"], reference_prediction, strict=True))
    view_x = align_embeddings(
        view_embeddings, selected["view_id"].astype(str).to_numpy()
    )
    selected["reference_y_pred"] = selected["sample_id"].map(reference_by_id).astype(float)
    selected["y_pred"] = predict_with_probe(
        state, view_x, batch_size=batch_size, device=device
    ).astype(float)
    selected["prediction_delta"] = selected["y_pred"] - selected["reference_y_pred"]
    selected["task"] = str(task)
    selected["method"] = reference_embeddings.method
    selected["representation_seed"] = reference_embeddings.representation_seed
    selected["head_seed"] = int(head_seed)
    selected["protocol"] = str(protocol)
    selected["task_type"] = state.task_type
    selected["input_view"] = selected["view_family"].map(
        lambda family: f"verified_equivalent:{family}"
    )
    selected["reference_representation_path"] = reference_representation_path
    selected["representation_path"] = view_representation_path

    leading = [
        "sample_id",
        "task",
        "split",
        "group_id",
        "y_true",
        "method",
        "representation_seed",
        "head_seed",
        "protocol",
        "task_type",
        "view_id",
        "view_family",
        "view_index",
        "input_view",
        "reference_y_pred",
        "y_pred",
        "prediction_delta",
        "analysis_role",
        "reference_representation_path",
        "representation_path",
    ]
    optional = [column for column in _RECORD_METADATA if column in selected and column not in leading]
    remainder = [column for column in selected if column not in leading and column not in optional]
    per_view = selected.loc[:, leading + optional + remainder].reset_index(drop=True)

    record_rows: list[dict[str, object]] = []
    group_columns = [
        "sample_id",
        "task",
        "split",
        "group_id",
        "y_true",
        "method",
        "representation_seed",
        "head_seed",
        "protocol",
        "task_type",
        "view_family",
    ]
    group_columns.extend(
        column for column in _RECORD_METADATA if column in per_view and column not in group_columns
    )
    for keys, cell in per_view.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys, strict=True))
        reference_value = float(cell["reference_y_pred"].iloc[0])
        view_values = cell["y_pred"].to_numpy(dtype=float)
        all_predictions = np.concatenate(([reference_value], view_values))
        deltas = np.abs(view_values - reference_value)
        row.update(
            {
                "n_equivalent_views": int(len(view_values)),
                "n_predictions": int(len(all_predictions)),
                "reference_y_pred": reference_value,
                "mean_y_pred": float(np.mean(all_predictions)),
                "property_prediction_std": float(np.std(all_predictions, ddof=0)),
                "mean_absolute_view_delta": float(np.mean(deltas)),
                "max_absolute_view_delta": float(np.max(deltas)),
            }
        )
        record_rows.append(row)
    per_record = pd.DataFrame(record_rows)

    summary_groups = [
        "task",
        "method",
        "representation_seed",
        "head_seed",
        "protocol",
        "task_type",
        "view_family",
    ]
    summary_groups.extend(
        column
        for column in (
            "structural_fidelity_eligible",
            "scope_class",
            "checkpoint_fraction",
            "checkpoint_tokens",
        )
        if column in per_record
    )
    family_summary = (
        per_record.groupby(summary_groups, as_index=False, dropna=False)
        .agg(
            n_records=("sample_id", "nunique"),
            n_equivalent_views=("n_equivalent_views", "sum"),
            mean_property_prediction_std=("property_prediction_std", "mean"),
            median_property_prediction_std=("property_prediction_std", "median"),
            mean_absolute_view_delta=("mean_absolute_view_delta", "mean"),
        )
        .sort_values(summary_groups)
        .reset_index(drop=True)
    )
    return EquivalentViewPredictionResult(per_view, per_record, family_summary)


def predict_property_applicability_controls(
    state: ProbeState,
    records: pd.DataFrame,
    reference_embeddings: EmbeddingTable,
    control_rows: pd.DataFrame,
    control_embeddings: EmbeddingTable,
    *,
    task: str,
    head_seed: int,
    protocol: str = "default",
    sample_id_column: str = "sample_id",
    group_id_column: str = "group_id",
    label_column: str = "label",
    default_split: str = "test",
    reference_representation_path: str | None = None,
    control_representation_path: str | None = None,
    batch_size: int = 4096,
    device: str = "cpu",
) -> pd.DataFrame:
    """Predict semantic controls without treating them as equivalents."""

    _confirm_same_representation(reference_embeddings, control_embeddings)
    if "analysis_role" in control_rows and not control_rows["analysis_role"].eq(
        "applicability_only"
    ).all():
        raise ValueError("equivalent views belong in predict_equivalent_property_views")
    metadata = _task_records(
        records,
        sample_id_column=sample_id_column,
        group_id_column=group_id_column,
        label_column=label_column,
        default_split=default_split,
    )
    selected = control_rows.loc[
        control_rows["sample_id"].astype(str).isin(set(metadata["sample_id"]))
    ].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "task",
                "split",
                "group_id",
                "y_true",
                "method",
                "representation_seed",
                "head_seed",
                "protocol",
                "task_type",
                "control_id",
                "control_family",
                "reference_y_pred",
                "y_pred",
                "prediction_delta",
                "analysis_role",
                "structural_fidelity_eligible",
                "scope_class",
                "checkpoint_fraction",
                "checkpoint_tokens",
            ]
        )
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected = _merge_transformed_rows(selected, metadata)
    selected = _carry_cache_progress(selected, reference_embeddings)
    reference_x = align_embeddings(
        reference_embeddings, metadata["sample_id"].to_numpy(dtype=str)
    )
    reference_prediction = predict_with_probe(
        state, reference_x, batch_size=batch_size, device=device
    )
    reference_by_id = dict(zip(metadata["sample_id"], reference_prediction, strict=True))
    control_x = align_embeddings(
        control_embeddings, selected["control_id"].astype(str).to_numpy()
    )
    selected["reference_y_pred"] = selected["sample_id"].map(reference_by_id).astype(float)
    selected["y_pred"] = predict_with_probe(
        state, control_x, batch_size=batch_size, device=device
    ).astype(float)
    selected["prediction_delta"] = selected["y_pred"] - selected["reference_y_pred"]
    selected["task"] = str(task)
    selected["method"] = reference_embeddings.method
    selected["representation_seed"] = reference_embeddings.representation_seed
    selected["head_seed"] = int(head_seed)
    selected["protocol"] = str(protocol)
    selected["task_type"] = state.task_type
    selected["input_view"] = selected["control_family"].map(
        lambda family: f"applicability_control:{family}"
    )
    selected["reference_representation_path"] = reference_representation_path
    selected["representation_path"] = control_representation_path
    leading = [
        "sample_id",
        "task",
        "split",
        "group_id",
        "y_true",
        "method",
        "representation_seed",
        "head_seed",
        "protocol",
        "task_type",
        "control_id",
        "control_family",
        "input_view",
        "reference_y_pred",
        "y_pred",
        "prediction_delta",
        "analysis_role",
        "reference_representation_path",
        "representation_path",
    ]
    remainder = [column for column in selected if column not in leading]
    return selected.loc[:, leading + remainder].reset_index(drop=True)


def write_equivalent_view_predictions(
    result: EquivalentViewPredictionResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write detailed view predictions and both summary levels."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_view": destination / "equivalent_view_predictions.parquet",
        "per_record": destination / "equivalent_view_within_record.parquet",
        "family_summary": destination / "equivalent_view_family_summary.csv",
    }
    atomic_write_parquet(result.per_view, paths["per_view"])
    atomic_write_parquet(result.per_record, paths["per_record"])
    atomic_write_csv(result.family_summary, paths["family_summary"])
    return paths


def write_applicability_control_predictions(
    predictions: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write semantic-control predictions to a separate table."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(predictions, destination)
    return destination


__all__ = [
    "EquivalentViewPredictionResult",
    "predict_equivalent_property_views",
    "predict_property_applicability_controls",
    "write_applicability_control_predictions",
    "write_equivalent_view_predictions",
]
