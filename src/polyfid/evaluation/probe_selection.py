"""Validation-only learning-rate selection for the common property head."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from polyfid.io import atomic_write_text


PROPERTY_HEAD_LEARNING_RATES = (1e-4, 3e-4, 1e-3)


def select_property_head_learning_rate(
    results: pd.DataFrame,
    *,
    candidates: Sequence[float] = PROPERTY_HEAD_LEARNING_RATES,
) -> dict[str, Any]:
    """Select the lowest mean validation loss among the declared candidates.

    If a ``split`` column is present, only rows explicitly labelled
    ``validation`` are considered. Test rows can therefore coexist in a run
    table without entering model selection.
    """

    required = {"learning_rate", "validation_loss"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"property-head selection input lacks columns: {missing}")
    validation = results.copy()
    if "split" in validation:
        validation = validation.loc[
            validation["split"].astype(str).str.casefold().eq("validation")
        ].copy()
    validation["learning_rate"] = pd.to_numeric(
        validation["learning_rate"], errors="raise"
    )
    validation["validation_loss"] = pd.to_numeric(
        validation["validation_loss"], errors="raise"
    )
    validation = validation.loc[np.isfinite(validation["validation_loss"])].copy()
    if validation.empty:
        raise ValueError("property-head selection has no finite validation losses")

    candidate_values = tuple(float(value) for value in candidates)
    summaries: list[dict[str, float | int]] = []
    for learning_rate in candidate_values:
        cell = validation.loc[
            np.isclose(
                validation["learning_rate"].to_numpy(dtype=float),
                learning_rate,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if cell.empty:
            raise ValueError(
                f"no validation loss was supplied for candidate {learning_rate:g}"
            )
        summaries.append(
            {
                "learning_rate": learning_rate,
                "mean_validation_loss": float(cell["validation_loss"].mean()),
                "observations": int(len(cell)),
            }
        )
    selected = min(
        summaries,
        key=lambda row: (
            float(row["mean_validation_loss"]),
            candidate_values.index(float(row["learning_rate"])),
        ),
    )
    return {
        "selection_split": "validation",
        "candidate_learning_rates": list(candidate_values),
        "validation_summary": summaries,
        "selected_recipe": {
            "property_head_learning_rate": float(selected["learning_rate"])
        },
    }


def write_property_head_selection(
    results: pd.DataFrame, output: str | Path
) -> dict[str, Any]:
    payload = select_property_head_learning_rate(results)
    destination = Path(output)
    atomic_write_text(
        yaml.safe_dump(payload, sort_keys=False),
        destination,
    )
    return payload


__all__ = [
    "PROPERTY_HEAD_LEARNING_RATES",
    "select_property_head_learning_rate",
    "write_property_head_selection",
]
