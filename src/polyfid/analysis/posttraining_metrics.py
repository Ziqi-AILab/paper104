"""Stream completed adaptation and baseline predictions into raw-unit tables.

Only one prediction file is loaded at a time. Representation and downstream
seeds keep their recorded meaning; a baseline's representation seed of zero
does not become a set of independent pretrained models.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

from polyfid.analysis.statistics import infer_task_metric
from polyfid.evaluation.metrics import compute_metric
from polyfid.evaluation.predictions import load_predictions
from polyfid.io import atomic_write_csv, atomic_write_text


METRIC_CELL_COLUMNS = (
    "method",
    "representation_seed",
    "head_seed",
    "task",
    "split",
    "label_fraction",
    "subsample_id",
    "input_view",
    "protocol",
    "fold",
)


def discover_manifests(project: Path) -> dict[str, Path]:
    """Return the six formal manifests, excluding overlapping combined lists."""

    project = Path(project)
    return {
        "adaptation": project / "slurm/generated/postpretraining/adaptation/adaptation_manifest.tsv",
        "chemprop": project / "slurm/generated/controlled_baselines/chemprop/chemprop_dpnn_manifest.tsv",
        "classical": project / "runs/manifests/controlled_baselines/classical_baseline_manifest.tsv",
        "transpolymer": project / "slurm/generated/external_reference/transpolymer/transpolymer_external_reference_manifest.tsv",
        "periogt": project / "slurm/generated/external_reference/graph_models/periogt_external_reference_manifest.tsv",
        "grin": project / "slurm/generated/external_reference/graph_models/grin_external_reference_manifest.tsv",
    }


def normalize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Align task/view spelling without changing observations or seed labels."""

    normalized = frame.copy()
    normalized["task"] = normalized["task"].astype(str).str.strip().str.upper()
    if "input_view" not in normalized:
        normalized["input_view"] = "valid"
    normalized["input_view"] = normalized["input_view"].fillna("valid").replace(
        {
            "structure_only": "valid",
            "metadata": "structure_plus_metadata_secondary",
            "structure_plus_metadata": "structure_plus_metadata_secondary",
        }
    )
    defaults = {
        "protocol": None,
        "fold": None,
        "label_fraction": 1.0,
        "subsample_id": 0,
    }
    for column, default in defaults.items():
        if column not in normalized:
            normalized[column] = default
        elif default is not None:
            normalized[column] = normalized[column].fillna(default)
    if normalized["fold"].notna().any():
        normalized["fold"] = normalized["fold"].map(
            lambda value: None if pd.isna(value) else str(value)
        )
    return normalized


def _reporting_table(row: dict[str, Any], source_group: str) -> str:
    supplied = row.get("reporting_table")
    if supplied is not None and not pd.isna(supplied):
        return str(supplied)
    if source_group == "adaptation":
        return "controlled_adaptation"
    if source_group in {"transpolymer", "periogt", "grin"}:
        return "same_split_external_reference"
    return "controlled_common_table"


def stream_metrics(
    manifest_path: str | Path,
    source_group: str,
    output_dir: str | Path,
    progress_every: int = 250,
) -> dict[str, Any]:
    """Process every manifest output into complete held-out fold metrics.

    Family, cluster, eligibility, and scope labels remain observation metadata;
    none splits a fold's error metric. A completed source can be reused without
    loading its prediction files again. Partial runs restart this source only.
    """

    started = time.monotonic()
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    manifest = pd.read_csv(
        manifest_path,
        sep="\t" if manifest_path.suffix.lower() in {".tsv", ".txt"} else ",",
    )
    if "output" not in manifest:
        raise ValueError("prediction manifest needs an output column")
    if manifest["output"].duplicated().any():
        raise ValueError("prediction manifest contains duplicate output paths")
    paths = {
        "task_metrics": output_dir / "task_metrics.csv",
        "prediction_inventory": output_dir / "prediction_inventory.csv",
        "summary": output_dir / "summary.json",
    }
    if paths["summary"].is_file():
        previous = json.loads(paths["summary"].read_text())
        if (
            previous.get("status") == "complete"
            and previous.get("manifest_path") == str(manifest_path)
            and previous.get("source_group") == source_group
            and previous.get("manifest_rows") == len(manifest)
            and paths["task_metrics"].is_file()
            and paths["prediction_inventory"].is_file()
        ):
            return previous

    metric_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    total_input_rows = 0
    for row_index, row in enumerate(manifest.to_dict(orient="records")):
        prediction_path = Path(str(row["output"]))
        predictions = normalize_predictions(load_predictions(prediction_path))
        total_input_rows += len(predictions)
        test = predictions.loc[predictions["split"].eq("test")]
        if test.empty:
            raise ValueError(f"prediction file has no held-out test rows: {prediction_path}")
        if test.duplicated([*METRIC_CELL_COLUMNS, "sample_id"]).any():
            raise ValueError(f"duplicate held-out sample in one prediction cell: {prediction_path}")
        context = {
            "source_group": source_group,
            "source_path": str(prediction_path),
            "reporting_table": _reporting_table(row, source_group),
        }
        before = len(metric_rows)
        for keys, cell in test.groupby(list(METRIC_CELL_COLUMNS), dropna=False, sort=True):
            values = dict(zip(METRIC_CELL_COLUMNS, keys, strict=True))
            metric = infer_task_metric(str(values["task"]))
            metric_rows.append(
                {
                    **context,
                    **values,
                    "metric": metric,
                    "value": compute_metric(
                        cell["y_true"].to_numpy(dtype=float),
                        cell["y_pred"].to_numpy(dtype=float),
                        metric,
                    ),
                    "n_rows": len(cell),
                    "n_groups": cell["group_id"].astype(str).nunique(),
                }
            )
        inventory_rows.append(
            {
                **context,
                "manifest_row": row_index,
                "job_id": row.get("job_id", ""),
                "n_input_rows": len(predictions),
                "n_rows": len(test),
                "n_groups": test["group_id"].astype(str).nunique(),
                "metric_rows": len(metric_rows) - before,
                "methods": ",".join(sorted(test["method"].astype(str).unique())),
                "tasks": ",".join(sorted(test["task"].astype(str).unique())),
                "representation_seeds": ",".join(str(value) for value in sorted(test["representation_seed"].unique())),
                "head_seeds": ",".join(str(value) for value in sorted(test["head_seed"].unique())),
            }
        )
        if progress_every > 0 and (row_index + 1) % progress_every == 0:
            print(
                f"{source_group}: {row_index + 1}/{len(manifest)} prediction files processed",
                flush=True,
            )

    metrics = pd.DataFrame(metric_rows)
    inventory = pd.DataFrame(inventory_rows)
    atomic_write_csv(metrics, paths["task_metrics"])
    atomic_write_csv(inventory, paths["prediction_inventory"])
    summary: dict[str, Any] = {
        "status": "complete",
        "source_group": source_group,
        "manifest_path": str(manifest_path),
        "manifest_rows": len(manifest),
        "processed_files": len(inventory_rows),
        "input_rows": total_input_rows,
        "prediction_rows": sum(row["n_rows"] for row in inventory_rows),
        "metric_rows": len(metric_rows),
        "methods": sorted(metrics["method"].unique().tolist()) if len(metrics) else [],
        "tasks": sorted(metrics["task"].unique().tolist()) if len(metrics) else [],
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {name: str(path) for name, path in paths.items()},
    }
    atomic_write_text(json.dumps(summary, indent=2) + "\n", paths["summary"])
    return summary


__all__ = ["discover_manifests", "normalize_predictions", "stream_metrics"]
