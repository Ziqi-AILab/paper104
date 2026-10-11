"""Small retained-model metadata and a prediction round-trip check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from polyfid.io import atomic_write_text


def write_retained_description(
    directory: str | Path,
    *,
    config: Mapping[str, Any],
    target_scaler: Mapping[str, Any],
) -> None:
    """Keep the training recipe and target transform beside selected weights."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for filename, values in (("config.json", config), ("target_scaler.json", target_scaler)):
        atomic_write_text(json.dumps(dict(values), indent=2, default=str) + "\n", root / filename)


def verify_prediction_roundtrip(
    reference: Any, reloaded: Any, output_path: str | Path,
) -> dict[str, Any]:
    """Check that the saved selected model reproduces held-out predictions.

    This compares a model with its own reloaded state, not with an earlier
    stochastic training run. It is not a performance-selection condition.
    """

    expected = np.asarray(reference, dtype=float).reshape(-1)
    observed = np.asarray(reloaded, dtype=float).reshape(-1)
    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-6)
    report = {
        "comparison": "selected_in_memory_model_vs_reloaded_saved_model",
        "test_rows": int(expected.size),
        "maximum_absolute_difference": float(np.max(np.abs(observed - expected))),
        "rtol": 1e-5,
        "atol": 1e-6,
    }
    atomic_write_text(json.dumps(report, indent=2) + "\n", output_path)
    return report
