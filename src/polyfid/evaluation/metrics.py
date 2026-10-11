"""Property metrics and task-equal paired effects."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


LOWER_IS_BETTER = {"mae", "rmse"}


def compute_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    truth = np.asarray(y_true)
    prediction = np.asarray(y_pred)
    name = metric.casefold()
    if name == "mae":
        return float(mean_absolute_error(truth, prediction))
    if name == "rmse":
        return float(np.sqrt(mean_squared_error(truth, prediction)))
    if name == "r2":
        return float(r2_score(truth, prediction))
    if name in {"auroc", "roc_auc"}:
        return float(roc_auc_score(truth, prediction))
    if name in {"auprc", "average_precision"}:
        return float(average_precision_score(truth, prediction))
    if name == "accuracy":
        return float(accuracy_score(truth, prediction >= 0.5))
    raise ValueError(f"unsupported metric: {metric}")


def log_error_ratio(candidate: float, baseline: float, metric: str) -> float:
    name = metric.casefold()
    if name in LOWER_IS_BETTER:
        candidate_error, baseline_error = candidate, baseline
    else:
        candidate_error, baseline_error = 1.0 - candidate, 1.0 - baseline
    return float(np.log(max(candidate_error, 1e-12) / max(baseline_error, 1e-12)))


def task_equal_effect(
    task_values: Mapping[str, tuple[float, float, str]],
) -> float:
    """Mean paired log error ratio, giving every task one vote."""

    return float(
        np.mean(
            [log_error_ratio(candidate, baseline, metric) for candidate, baseline, metric in task_values.values()]
        )
    )

