from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import polyfid.analysis.draft_closure_statistics as closure_module

from polyfid.analysis.draft_closure_statistics import (
    BRIDGE_METRICS, BRIDGE_TASKS, EDIT_FAMILIES, GEOMETRY_METRICS,
    METHODS, POSITIVE_FAMILIES, SEEDS, bridge_bootstrap, geometry_bootstrap,
    prepare_bridge, sample_shared_group_weights, write_geometry_analysis,
)
from polyfid.analysis.figures import (
    _fidelity_utility_cells, _learning_curve_effects, _method_ranks, _retention_matrix,
)


def _bridge_rows(tasks: tuple[str, ...] = ("EEA", "EGC")) -> pd.DataFrame:
    return pd.DataFrame([
        {"task": task, "protocol": "group_ood", "fold": index, "sample_id": f"{task}-{index}",
         "group_id": f"global-{index}", "representation_seed": rep, "head_seed": head,
         "periodic_cut_sd_candidate": 0.1, "periodic_cut_sd_baseline": 0.2,
         "traversal_sd_candidate": 0.3, "traversal_sd_baseline": 0.2,
         "squared_error_candidate": (1.0, 9.0)[index], "squared_error_baseline": 4.0}
        for rep in (42, 123) for head in (11, 22) for task in tasks for index in (0, 1)
    ])


def test_bridge_uses_shared_structure_draws_across_tasks_and_folds() -> None:
    data = prepare_bridge(_bridge_rows())
    rng = np.random.default_rng(9)
    for _ in range(30):
        weights = sample_shared_group_weights(data, rng)
        for _, indices in data.records.groupby("group_id").indices.items():
            assert len(set(weights[indices])) == 1
        assert weights.sum() == len(data.records)
    assert len(set(data.group_codes)) == 2  # not four task/structure clusters


def test_bridge_complete_seed_head_common_cohort() -> None:
    rows = _bridge_rows()
    rows = rows.drop(rows[(rows.task == "EEA") & (rows.sample_id == "EEA-0")
                          & (rows.representation_seed == 42) & (rows.head_seed == 11)].index)
    data = prepare_bridge(rows)
    assert len(data.records) == 3
    assert "EEA-0" not in set(data.records.sample_id)


def test_bridge_has_24_joint_intervals_and_pools_squared_errors(tmp_path: Path) -> None:
    data = prepare_bridge(_bridge_rows(BRIDGE_TASKS))
    intervals, points = bridge_bootstrap(data, n_resamples=12, chunk_size=5, seed=7,
                                         output_root=tmp_path)
    assert len(intervals) == 24
    assert set(intervals.metric) == set(BRIDGE_METRICS)
    reference = intervals[intervals.metric == "reference_rmse"]
    assert reference.estimate.to_numpy() == pytest.approx(np.log(np.sqrt(5.0) / 2.0))
    cuts = intervals[intervals.metric == "periodic_cut_prediction_sd"]
    assert cuts.estimate.to_numpy() == pytest.approx(np.log(0.5))
    assert cuts.ci_lower.to_numpy() == pytest.approx(np.log(0.5))
    assert set(points.head_seed) == {11, 22}
    resumed, _ = bridge_bootstrap(data, n_resamples=12, chunk_size=5, seed=7,
                                  output_root=tmp_path)
    pd.testing.assert_frame_equal(intervals, resumed)


def test_bridge_writer_records_scope_and_serializable_seeds(tmp_path: Path, monkeypatch) -> None:
    data = prepare_bridge(_bridge_rows(BRIDGE_TASKS))
    monkeypatch.setattr(closure_module, "load_property_bridge", lambda project: data)
    summary = closure_module.write_property_bridge(tmp_path, tmp_path / "output",
        n_resamples=6, chunk_size=3, seed=8)
    assert summary["n_records"] == 14
    assert summary["n_global_structure_groups"] == 2
    assert summary["intervals"] == 24
    assert (tmp_path / "output/summary.json").is_file()


def test_geometry_seed_anchor_cluster_retains_nested_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        {"representation_seed": rep, "anchor_sample_id": anchor,
         "topology_margin_candidate": baseline + 0.25, "topology_margin_baseline": baseline}
        for rep in (42, 123) for anchor, baseline in (("a", 0.0), ("a", 0.1), ("b", 0.2))
    ])
    intervals, points = geometry_bootstrap(frame, {"topology_margin": "topology_margin"},
        n_resamples=11, chunk_size=5, seed=3, output_root=tmp_path)
    assert intervals.iloc[0].estimate == pytest.approx(0.25)
    assert intervals.iloc[0].ci_lower == pytest.approx(0.25)
    assert len(points) == 2


def test_geometry_24_intervals_separate_five_seed_scopes(tmp_path: Path) -> None:
    root = tmp_path / "results/metrics/fidelity/relations"
    for method in METHODS:
        for seed in SEEDS:
            target = root / f"{method}_seed-{seed}"
            target.mkdir(parents=True)
            records = [
                {"representation_seed": seed, "anchor_sample_id": f"a-{i}",
                 "positive_sample_id": f"p-{i}", "negative_sample_id": f"n-{i}",
                 "positive_relation": family, "triplet_id": i,
                 **{column: 0.2 + (0.1 if method == "A_eqs" else 0.0)
                    for column in GEOMETRY_METRICS.values()}}
                for i, family in enumerate(POSITIVE_FAMILIES)
            ]
            pd.DataFrame(records).to_parquet(target / "triplet_metrics.parquet")
            controls = [
                {"representation_seed": seed, "anchor_sample_id": "a-0",
                 "positive_sample_id": "p-0", "negative_sample_id": f"edit-{i}",
                 "negative_relation": family, "control_row_id": i,
                 "topology_margin": 0.2, "triplet_correct": True}
                for i, family in enumerate(EDIT_FAMILIES)
            ]
            pd.DataFrame(controls).to_parquet(target / "topology_control_metrics.parquet")
    summary = write_geometry_analysis(tmp_path, tmp_path / "output", n_resamples=6,
                                       chunk_size=3, seed=4)
    assert summary["intervals"] == 24
    assert summary["representation_seeds"] == list(SEEDS)
    scope = pd.read_csv(tmp_path / "output/scope.csv")
    assert set(scope.n_representation_seeds) == {5}
    assert set(scope.scope) == {"overall", *POSITIVE_FAMILIES, *EDIT_FAMILIES}


def _fold_metrics() -> pd.DataFrame:
    rows = []
    # A0 has an excellent small fold and a poor large fold. Averaging fold
    # RMSEs would incorrectly rank A0 above B0, whose error is always 3.
    for method, values in (("C", (3.0, 3.0)), ("A0", (1.0, 4.0)), ("B0", (3.0, 3.0)),
                           ("A_att", (2.0, 3.0)), ("A_gcl", (2.0, 3.0)), ("A_eqs", (1.0, 3.0))):
        for task in ("EGC", "ESOL"):
            for fold, (value, n_rows) in enumerate(zip(values, (1, 9), strict=True)):
                rows.append({"method": method, "representation_seed": 0 if method == "C" else 42,
                             "head_seed": 11, "task": task, "protocol": "group_ood",
                             "split": "test", "input_view": "valid", "label_fraction": 1.0,
                             "subsample_id": 0, "metric": "rmse", "fold": fold,
                             "n_rows": n_rows, "value": value})
    return pd.DataFrame(rows)


def test_figure5_ranks_retention_and_curve_pool_folds() -> None:
    metrics = _fold_metrics()
    ranks = _method_ranks(metrics).set_index("method")
    assert ranks.loc["A0", "mean_rank"] > ranks.loc["B0", "mean_rank"]
    expected = np.log(np.sqrt(14.5) / 3.0)
    retention = _retention_matrix(metrics)
    assert retention.loc["A0", "esol"] == pytest.approx(expected)
    curve = _learning_curve_effects(metrics).set_index("comparison")
    assert curve.loc["molecular_initialization", "paired_log_error_ratio"] == pytest.approx(expected)


def test_figure5_association_pairs_heads_after_fold_pooling() -> None:
    metrics = _fold_metrics()
    fidelity = pd.DataFrame([{"method": "A0", "representation_seed": 42,
                              "anchor_positive_cosine": 0.9, "topology_margin": 0.2}])
    cells = _fidelity_utility_cells(metrics, fidelity)
    assert cells.iloc[0].task_equal_log_error_ratio_vs_C == pytest.approx(np.log(np.sqrt(14.5) / 3.0))
    assert cells.iloc[0].downstream_tasks == 1


def test_closure_plots_consume_only_completed_csvs(tmp_path: Path, monkeypatch) -> None:
    def bootstrap_not_allowed(*args, **kwargs):
        raise AssertionError("plotting must not run bootstrap")

    monkeypatch.setattr(closure_module, "run_draft_closure", bootstrap_not_allowed)
    geometry_root = tmp_path / "geometry"
    bridge_root = tmp_path / "property_bridge"
    geometry_root.mkdir()
    bridge_root.mkdir()
    geometry = [
        {"scope": scope, "metric": metric, "estimate": 0.1, "ci_lower": 0.05, "ci_upper": 0.15}
        for scope in ("overall", *POSITIVE_FAMILIES) for metric in GEOMETRY_METRICS
    ] + [
        {"scope": scope, "metric": metric, "estimate": 0.1, "ci_lower": 0.05, "ci_upper": 0.15}
        for scope in EDIT_FAMILIES for metric in ("topology_margin", "triplet_accuracy")
    ]
    bridge = [{"task": task, "metric": metric, "estimate": -0.1, "ci_lower": -0.2, "ci_upper": 0.05}
              for task in (*BRIDGE_TASKS, "aggregate") for metric in BRIDGE_METRICS]
    pd.DataFrame(geometry).to_csv(geometry_root / "intervals.csv", index=False)
    pd.DataFrame(bridge).to_csv(bridge_root / "intervals.csv", index=False)
    pd.DataFrame([{"task": "aggregate", "n_records": 4765, "n_global_structure_groups": 2801}]).to_csv(
        bridge_root / "scope.csv", index=False)
    outputs = closure_module.render_draft_closure_figures(tmp_path, tmp_path / "figures")
    assert len(outputs) == 6
    assert all(path.is_file() for path in outputs)
    plot_data = pd.read_csv(bridge_root / "plot_data.csv")
    assert plot_data.plot_estimate.to_numpy() == pytest.approx(100 * np.expm1(-0.1))
