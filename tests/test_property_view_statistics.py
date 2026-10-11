from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyfid.analysis.property_view_statistics import (
    _multi_structure_task_scope,
    compute_property_view_task_effects,
    load_primary_paired_property_views,
    paired_property_view_hierarchical_bootstrap,
    stream_property_view_task_metrics,
)
from polyfid.analysis.figures import figure4d_property_view_stability


def _record_frame(
    method: str,
    *,
    task: str = "EGC",
    seed: int = 42,
    head: int = 11,
    scale: float = 1.0,
) -> pd.DataFrame:
    rows = []
    for family in ("traversal", "periodic_cut"):
        for index in range(4):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "task": task,
                    "split": "test",
                    "group_id": f"group-{index // 2}",
                    "y_true": float(index),
                    "method": method,
                    "representation_seed": seed,
                    "head_seed": head,
                    "protocol": "group_ood",
                    "task_type": "regression",
                    "view_family": family,
                    "n_equivalent_views": 3,
                    "n_predictions": 4,
                    "reference_y_pred": float(index) + 0.2,
                    "mean_y_pred": float(index) + 0.1,
                    "property_prediction_std": scale,
                    "mean_absolute_view_delta": scale * 1.5,
                    "max_absolute_view_delta": scale * 2.0,
                    "structural_fidelity_eligible": True,
                    "scope_class": "linear_dual_attachment_repeat",
                    "checkpoint_fraction": np.nan,
                    "checkpoint_tokens": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _manifest_row(root: Path, method: str) -> dict[str, object]:
    return {
        "arm": method,
        "representation_label": f"{method}_seed-42",
        "representation_seed": 42,
        "task": "EGC",
        "protocol": "group_ood",
        "fold": 0,
        "input_view": "structure_only",
        "partition_id": "transpolymer__structure_only",
        "head_seeds": "11",
        "label_fraction": 1.0,
        "subsample_id": 0,
        "output_root": str(root),
    }


def _write_record_output(root: Path, frame: pd.DataFrame) -> None:
    destination = root / "head-seed-11"
    destination.mkdir(parents=True)
    frame.to_parquet(destination / "equivalent_view_within_record.parquet", index=False)


def test_streaming_keeps_view_axes_separate_and_records_empty_scope(
    tmp_path: Path,
) -> None:
    full_root = tmp_path / "full"
    empty_root = tmp_path / "empty"
    _write_record_output(full_root, _record_frame("A0"))
    _write_record_output(empty_root, _record_frame("A0").iloc[0:0])
    manifest = pd.DataFrame(
        [
            _manifest_row(full_root, "A0"),
            {**_manifest_row(empty_root, "A0"), "task": "OPV"},
        ]
    )
    metrics, inventory = stream_property_view_task_metrics(
        manifest, progress_every=0
    )
    assert set(metrics["view_family"]) == {"traversal", "periodic_cut"}
    assert len(metrics) == 2
    assert inventory["status"].value_counts().to_dict() == {
        "nonempty": 1,
        "no_eligible_views": 1,
    }


def test_task_effects_use_exact_log_ratio_for_each_axis() -> None:
    rows = []
    for method, value in (("B0", 2.0), ("A0", 1.0)):
        for family in ("traversal", "periodic_cut"):
            rows.append(
                {
                    "method": method,
                    "representation_seed": 42,
                    "head_seed": 11,
                    "task": "EGC",
                    "protocol": "group_ood",
                    "fold": 0,
                    "input_view": "structure_only",
                    "label_fraction": 1.0,
                    "subsample_id": 0,
                    "view_family": family,
                    "n_records": 4,
                    "n_groups": 2,
                    "n_equivalent_views": 12,
                    "mean_property_prediction_std": value,
                    "mean_absolute_view_delta": value * 1.5,
                    "reference_rmse": 0.5,
                    "reference_prediction_std": 1.0,
                }
            )
    effects = compute_property_view_task_effects(pd.DataFrame(rows))
    molecular = effects.loc[
        effects["comparison"].eq("molecular_initialization")
    ]
    assert len(molecular) == 2
    assert np.allclose(
        molecular["within_record_prediction_std_paired_log_ratio"],
        np.log(0.5),
    )


def test_property_view_bootstrap_is_task_equal() -> None:
    rows = []
    for task, count, candidate, baseline in (
        ("large", 100, 0.5, 1.0),
        ("small", 2, 2.0, 1.0),
    ):
        for head in (11, 22):
            for index in range(count):
                rows.append(
                    {
                        "sample_id": f"{task}-{index}",
                        "group_id": f"{task}-group-{index // 2}",
                        "view_family": "traversal",
                        "n_equivalent_views": 3,
                        "property_prediction_std_candidate": candidate,
                        "property_prediction_std_baseline": baseline,
                        "mean_absolute_view_delta_candidate": candidate,
                        "mean_absolute_view_delta_baseline": baseline,
                        "representation_seed": 42,
                        "head_seed": head,
                        "task": task,
                        "protocol": "group_ood",
                        "fold": 0,
                        "input_view": "structure_only",
                    }
                )
    result = paired_property_view_hierarchical_bootstrap(
        pd.DataFrame(rows), n_resamples=30, seed=7
    )
    assert abs(result.estimate) < 1e-12


def test_single_structure_task_stays_out_of_cross_task_inference() -> None:
    frame = pd.DataFrame(
        {
            "task": ["PE-I", "PE-I", "EGC", "EGC"],
            "fold": [0, 0, 0, 1],
            "group_id": ["same", "same", "g1", "g2"],
        }
    )
    selected = _multi_structure_task_scope(frame)
    assert set(selected["task"]) == {"EGC"}


def test_record_pairing_rejects_changed_view_count(tmp_path: Path) -> None:
    candidate_root = tmp_path / "candidate"
    baseline_root = tmp_path / "baseline"
    candidate = _record_frame("A0")
    baseline = _record_frame("B0", scale=2.0)
    baseline.loc[0, "n_equivalent_views"] = 2
    _write_record_output(candidate_root, candidate)
    _write_record_output(baseline_root, baseline)
    manifest = pd.DataFrame(
        [
            _manifest_row(candidate_root, "A0"),
            _manifest_row(baseline_root, "B0"),
        ]
    )
    with pytest.raises(ValueError, match="different view counts"):
        load_primary_paired_property_views(manifest, "A0", "B0")


def test_property_view_companion_figure_uses_both_axes(tmp_path: Path) -> None:
    statistics = tmp_path / "statistics"
    statistics.mkdir()
    comparisons = [
        "molecular_initialization",
        "boundary_supervision",
        "verified_relations",
    ]
    families = ["periodic_cut", "traversal"]
    pd.DataFrame(
        [
            {
                "comparison": comparison,
                "view_family": family,
                "task": task,
                "paired_log_ratio": -0.1,
            }
            for comparison in comparisons
            for family in families
            for task in ("EGC", "XC")
        ]
    ).to_csv(statistics / "figure4d_task_data.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison": comparison,
                "view_family": family,
                "estimate": -0.1,
                "ci_lower": -0.2,
                "ci_upper": 0.02,
            }
            for comparison in comparisons
            for family in families
        ]
    ).to_csv(statistics / "comparison_aggregate.csv", index=False)
    outcome = figure4d_property_view_stability(
        statistics, tmp_path / "figures"
    )
    assert outcome.status == "generated"
    assert (tmp_path / "figures/figure4d_property_view_stability.pdf").is_file()
