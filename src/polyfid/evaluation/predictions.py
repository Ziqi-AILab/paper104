"""Lossless per-sample prediction I/O."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polyfid.io import atomic_write_parquet, atomic_write_text
from polyfid.records import PredictionRecord


PREDICTION_COLUMNS = tuple(field.name for field in fields(PredictionRecord))
REQUIRED_COLUMNS = {
    "sample_id",
    "task",
    "split",
    "group_id",
    "y_true",
    "y_pred",
    "method",
    "representation_seed",
    "head_seed",
}


def prediction_frame(
    records: pd.DataFrame | Iterable[PredictionRecord | Mapping[str, Any]],
) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame(
            record.as_dict() if isinstance(record, PredictionRecord) else dict(record)
            for record in records
        )
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction rows are missing columns: {missing}")
    for column in PREDICTION_COLUMNS:
        if column not in frame:
            frame[column] = None
    numeric = frame[["y_true", "y_pred"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("prediction values must be finite")
    extras = [column for column in frame if column not in PREDICTION_COLUMNS]
    return frame.loc[:, list(PREDICTION_COLUMNS) + extras]


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def write_predictions(
    records: pd.DataFrame | Iterable[PredictionRecord | Mapping[str, Any]],
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = prediction_frame(records)
    if destination.suffix == ".parquet":
        atomic_write_parquet(frame, destination, index=False)
    elif destination.suffix == ".jsonl":
        text = "".join(
            json.dumps(
                {key: _native(value) for key, value in row.items()},
                ensure_ascii=False,
            )
            + "\n"
            for row in frame.to_dict(orient="records")
        )
        atomic_write_text(text, destination)
    else:
        raise ValueError("prediction path must end in .parquet or .jsonl")
    return destination


def load_predictions(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return prediction_frame(pd.read_parquet(source))
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            return prediction_frame(pd.DataFrame(json.loads(line) for line in handle if line.strip()))
    raise ValueError("prediction path must end in .parquet or .jsonl")
