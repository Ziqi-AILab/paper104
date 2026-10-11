from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyfid.analysis.multiple_testing import holm_adjust
from polyfid.analysis.statistics import (
    _prepare_bootstrap,
    compute_task_effects,
    compute_task_metrics,
    pair_method_predictions,
    paired_task_equal_hierarchical_bootstrap,
    run_primary_analysis,
    select_primary_scope,
    stream_formal_summaries_from_manifest,
    stream_task_metrics_from_manifest,
    write_analysis_outputs,
    write_manifest_analysis_outputs,
)


def _formal_predictions() -> pd.DataFrame:
    errors = {
        "B0": 1.00,
        "A0": 0.70,
        "A_att": 0.50,
        "A_gcl": 0.60,
        "A_eqs": 0.35,
    }
    rows = []
    for method, error in errors.items():
        for representation_seed in (42, 123):
            for head_seed in (11, 22):
                for task, protocol in (("EA", "family_ood"), ("Egc", "group_ood")):
                    for index in range(6):
                        truth = float(index) / 3
                        rows.append(
                            {
                                "sample_id": f"{task}:{index}",
                                "task": task,
                                "split": "test",
                                "group_id": f"group:{index // 2}",
                                "y_true": truth,
                                "y_pred": truth + error + 0.01 * (head_seed == 22),
                                "method": method,
                                "representation_seed": representation_seed,
                                "head_seed": head_seed,
                                "protocol": protocol,
                                "family_id": None,
                                "cluster_id": None,
                                "fold": str(index // 3) if task == "EA" else "0",
                                "label_fraction": 1.0,
                                "subsample_id": 0,
                                "input_view": "valid",
                            }
                        )
                # A deliberately poor random-split EA result must stay separate
                # from the declared family-OOD primary scope.
                if method in {"A0", "B0"}:
                    for index in range(6):
                        rows.append(
                            {
                                "sample_id": f"EA-random:{index}",
                                "task": "EA",
                                "split": "test",
                                "group_id": f"random-group:{index // 2}",
                                "y_true": float(index),
                                "y_pred": float(index) + (4.0 if method == "A0" else 1.0),
                                "method": method,
                                "representation_seed": representation_seed,
                                "head_seed": head_seed,
                                "protocol": "random",
                                "family_id": None,
                                "cluster_id": None,
                                "fold": "0",
                                "label_fraction": 1.0,
                                "subsample_id": 0,
                                "input_view": "valid",
                            }
                        )
    return pd.DataFrame(rows)


def _prediction_manifest(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = _formal_predictions().loc[
        lambda frame: ~frame["protocol"].eq("random")
    ].copy()
    predictions["scope_class"] = "eligible"
    rows = []
    group_columns = [
        "method",
        "representation_seed",
        "task",
        "protocol",
        "fold",
        "label_fraction",
        "subsample_id",
    ]
    for index, (keys, cell) in enumerate(
        predictions.groupby(group_columns, sort=True)
    ):
        values = dict(zip(group_columns, keys, strict=True))
        output = tmp_path / "predictions" / f"{index}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        cell.to_parquet(output, index=False)
        rows.append(
            {
                "arm": values["method"],
                "representation_seed": values["representation_seed"],
                "task": values["task"],
                "protocol": values["protocol"],
                "input_view": "structure_only",
                "fold": values["fold"],
                "label_fraction": values["label_fraction"],
                "subsample_id": values["subsample_id"],
                "output": str(output),
                # This adjacent table is deliberately not a prediction input.
                "metrics_output": str(tmp_path / "validation" / f"{index}.parquet"),
            }
        )
    return pd.DataFrame(rows), predictions


def test_task_metrics_keep_protocols_separate() -> None:
    metrics = compute_task_metrics(_formal_predictions())
    ea = metrics.loc[(metrics["method"].eq("A0")) & metrics["task"].eq("EA")]
    assert set(ea["protocol"]) == {"family_ood", "random"}
    assert set(ea["metric"]) == {"mae"}
    assert set(ea.loc[ea["protocol"].eq("family_ood"), "fold"]) == {"0", "1"}
    effects = compute_task_effects(_formal_predictions())
    ea_effects = effects.loc[
        effects["comparison"].eq("molecular_initialization")
        & effects["task"].eq("EA")
        & effects["protocol"].eq("family_ood")
    ]
    assert set(ea_effects["fold"]) == {"0", "1"}


def test_family_bootstrap_is_enabled_only_for_family_ood_protocol() -> None:
    rows = _formal_predictions()
    group_rows = rows.loc[
        rows["task"].eq("Egc") & rows["method"].isin(["A0", "B0"])
    ].copy()
    group_rows["family_id"] = np.where(
        group_rows["sample_id"].str.endswith(("0", "1", "2")), "f1", "f2"
    )
    paired = pair_method_predictions(group_rows, "A0", "B0")
    prepared = _prepare_bootstrap(paired, None)
    assert all(task.family_groups is None for task in prepared[0].tasks)

    family_rows = group_rows.copy()
    family_rows["protocol"] = "family_ood"
    paired_family = pair_method_predictions(family_rows, "A0", "B0")
    prepared_family = _prepare_bootstrap(paired_family, None)
    assert all(task.family_groups is not None for task in prepared_family[0].tasks)


def test_prediction_pairing_rejects_equal_size_different_sample_sets() -> None:
    predictions = _formal_predictions().loc[
        lambda frame: frame["method"].isin(["A0", "B0"])
        & frame["task"].eq("Egc")
    ].copy()
    baseline_index = predictions.index[predictions["method"].eq("B0")][0]
    predictions.loc[baseline_index, "sample_id"] = "different-sample"
    with pytest.raises(ValueError, match="same held-out rows"):
        pair_method_predictions(
            predictions, "A0", "B0", require_full_coverage=True
        )


def test_prediction_pairing_rejects_changed_hierarchy_metadata() -> None:
    predictions = _formal_predictions().loc[
        lambda frame: frame["method"].isin(["A0", "B0"])
        & frame["task"].eq("Egc")
    ].copy()
    baseline_index = predictions.index[predictions["method"].eq("B0")][0]
    predictions.loc[baseline_index, "group_id"] = "different-group"
    with pytest.raises(ValueError, match="group_id"):
        pair_method_predictions(predictions, "A0", "B0")


def test_primary_analysis_uses_one_protocol_per_task() -> None:
    aggregates, unavailable = run_primary_analysis(
        _formal_predictions(), n_resamples=40, bootstrap_seed=7
    )
    assert not unavailable
    assert set(aggregates["comparison"]) == {
        "molecular_initialization",
        "boundary_supervision",
        "verified_relations",
    }
    assert (aggregates["estimate"] < 0).all()
    molecular = aggregates.loc[
        aggregates["comparison"].eq("molecular_initialization")
    ].iloc[0]
    assert "EA=family_ood" in molecular["task_protocols"]
    assert "EA=random" not in molecular["task_protocols"]
    assert aggregates["holm_adjusted_p"].notna().all()


def test_task_equal_effect_does_not_weight_large_task_by_rows() -> None:
    rows = []
    for task, count, candidate_error in (("large", 100, 0.5), ("small", 2, 2.0)):
        for index in range(count):
            for method, error in (("candidate", candidate_error), ("baseline", 1.0)):
                rows.append(
                    {
                        "sample_id": f"{task}:{index}",
                        "task": task,
                        "split": "test",
                        "group_id": f"{task}:{index // 2}",
                        "y_true": 0.0,
                        "y_pred": error,
                        "method": method,
                        "representation_seed": 42,
                        "head_seed": 11,
                        "protocol": "group_ood",
                        "family_id": None,
                        "cluster_id": None,
                        "fold": "0",
                        "label_fraction": 1.0,
                        "subsample_id": 0,
                        "input_view": "valid",
                    }
                )
    paired = pair_method_predictions(pd.DataFrame(rows), "candidate", "baseline")
    result = paired_task_equal_hierarchical_bootstrap(paired, n_resamples=20, seed=3)
    assert abs(result.estimate) < 1e-12
    assert result.valid_resamples == 20


def test_external_applicability_and_retention_stay_out_of_primary_aggregate() -> None:
    rows = []
    for task in ("OpenPoly:Tg_K", "MoleculeNet:ESOL", "Egc"):
        for method, error in (("candidate", 0.5), ("baseline", 1.0)):
            rows.append(
                {
                    "sample_id": f"{task}:0",
                    "task": task,
                    "split": "test",
                    "group_id": f"{task}:group",
                    "y_true": 0.0,
                    "y_pred": error,
                    "method": method,
                    "representation_seed": 42,
                    "head_seed": 11,
                    "protocol": "group_ood",
                    "family_id": None,
                    "cluster_id": None,
                    "fold": "0",
                    "label_fraction": 1.0,
                    "subsample_id": 0,
                    "input_view": "valid",
                }
            )
    paired = pair_method_predictions(pd.DataFrame(rows), "candidate", "baseline")
    selected = select_primary_scope(paired)
    assert set(selected["task"]) == {"Egc"}


def test_statistics_outputs_are_derived_from_prediction_rows(tmp_path: Path) -> None:
    output = tmp_path / "statistics"
    summary = write_analysis_outputs(
        _formal_predictions(),
        [tmp_path / "formal_predictions.parquet"],
        output,
        n_resamples=20,
        bootstrap_seed=5,
    )
    assert summary["source"] == "per-sample prediction records"
    assert len(summary["completed_primary_comparisons"]) == 3
    assert (output / "task_metrics.csv").is_file()
    assert (output / "task_effects.csv").is_file()
    assert len(pd.read_csv(output / "primary_aggregate.csv")) == 3
    loaded = json.loads((output / "summary.json").read_text())
    assert loaded["bootstrap_resamples"] == 20


def test_manifest_streaming_matches_in_memory_metrics_and_effects(tmp_path: Path) -> None:
    manifest, predictions = _prediction_manifest(tmp_path)
    metrics, effects, inventory = stream_formal_summaries_from_manifest(
        manifest, progress_every=0
    )
    expected_metrics = compute_task_metrics(predictions)
    sort_keys = [
        "method",
        "representation_seed",
        "head_seed",
        "task",
        "protocol",
        "fold",
    ]
    pd.testing.assert_frame_equal(
        metrics.sort_values(sort_keys).reset_index(drop=True),
        expected_metrics.sort_values(sort_keys).reset_index(drop=True),
        check_like=True,
    )
    expected_effects = compute_task_effects(predictions)
    effect_keys = ["comparison", "representation_seed", "head_seed", "task", "protocol", "fold"]
    columns = sorted(set(effects.columns) & set(expected_effects.columns))
    pd.testing.assert_frame_equal(
        effects.sort_values(effect_keys).reset_index(drop=True)[columns],
        expected_effects.sort_values(effect_keys).reset_index(drop=True)[columns],
        check_like=True,
    )
    assert int(inventory["rows"].sum()) == len(predictions)


def test_manifest_metric_is_one_fold_cell_not_one_score_per_cluster(
    tmp_path: Path,
) -> None:
    predictions = _formal_predictions().loc[
        lambda frame: frame["method"].eq("A0")
        & frame["task"].eq("Egc")
        & frame["representation_seed"].eq(42)
        & frame["fold"].eq("0")
    ].copy()
    predictions["protocol"] = "butina_cluster"
    predictions["cluster_id"] = np.where(
        predictions["sample_id"].str.endswith(("0", "1", "2")), "c1", "c2"
    )
    output = tmp_path / "predictions.parquet"
    predictions.to_parquet(output, index=False)
    manifest = pd.DataFrame(
        [
            {
                "arm": "A0",
                "representation_seed": 42,
                "task": "Egc",
                "protocol": "butina_cluster",
                "input_view": "structure_only",
                "fold": "0",
                "label_fraction": 1.0,
                "subsample_id": 0,
                "output": str(output),
            }
        ]
    )
    metrics, _ = stream_task_metrics_from_manifest(manifest, progress_every=0)
    assert len(metrics) == predictions[["representation_seed", "head_seed"]].drop_duplicates().shape[0]
    assert "cluster_id" not in metrics.columns


def test_manifest_statistics_reads_only_output_column(tmp_path: Path) -> None:
    manifest, _ = _prediction_manifest(tmp_path)
    manifest_path = tmp_path / "frozen_probe_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    output = tmp_path / "statistics" / "formal_frozen_probes"
    summary = write_manifest_analysis_outputs(
        manifest_path,
        output,
        n_resamples=20,
        bootstrap_seed=5,
        primary_filters={"scope_class": "eligible"},
        progress_every=0,
    )
    assert summary["prediction_files"] == len(manifest)
    assert len(summary["completed_primary_comparisons"]) == 3
    assert len(pd.read_csv(output / "task_metrics.csv")) > 0
    assert len(pd.read_csv(output / "task_effects.csv")) > 0
    assert len(pd.read_csv(output / "primary_aggregate.csv")) == 3


def test_manifest_rejects_duplicate_formal_cell(tmp_path: Path) -> None:
    manifest, _ = _prediction_manifest(tmp_path)
    duplicate = manifest.iloc[[0]].copy()
    duplicate["output"] = str(tmp_path / "another_predictions.parquet")
    manifest = pd.concat([manifest, duplicate], ignore_index=True)
    manifest_path = tmp_path / "duplicate_manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    with pytest.raises(ValueError, match="duplicate formal run cell"):
        write_manifest_analysis_outputs(
            manifest_path,
            tmp_path / "statistics",
            n_resamples=2,
            progress_every=0,
        )


def test_holm_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"one": 0.01, "two": 0.03, "three": 0.04})
    assert adjusted == {"one": 0.03, "two": 0.06, "three": 0.06}
