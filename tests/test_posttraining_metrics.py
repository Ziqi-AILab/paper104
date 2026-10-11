from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polyfid.analysis import posttraining_metrics
from polyfid.analysis.posttraining_metrics import (
    discover_manifests,
    normalize_predictions,
    stream_metrics,
)


def _predictions(task: str = "Egc", *, head_seed: int = 11) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "task": task,
            "split": "test",
            "group_id": ["g1", "g2"],
            "y_true": [0.0, 0.0],
            "y_pred": [1.0, 3.0],
            "method": "baseline_chemprop_dpnn",
            "representation_seed": 0,
            "head_seed": head_seed,
            "protocol": "group_ood",
            "fold": "0",
            "input_view": "structure_only",
            "label_fraction": 1.0,
            "subsample_id": 0,
            "family_id": ["f1", "f2"],
            "cluster_id": ["c1", "c2"],
            "structural_fidelity_eligible": [True, False],
            "scope_class": ["eligible", "ineligible"],
        }
    )


def _manifest(tmp_path: Path, frames: list[pd.DataFrame]) -> Path:
    rows = []
    for index, frame in enumerate(frames):
        path = tmp_path / f"predictions_{index}.parquet"
        frame.to_parquet(path, index=False)
        rows.append({"job_id": f"job-{index}", "output": str(path)})
    path = tmp_path / "manifest.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def test_discover_manifests_does_not_include_overlapping_graph_lists() -> None:
    paths = discover_manifests(Path("/project"))
    assert set(paths) == {"adaptation", "chemprop", "classical", "transpolymer", "periogt", "grin"}
    assert len(set(paths.values())) == 6
    assert paths["classical"] == Path("/project/runs/manifests/controlled_baselines/classical_baseline_manifest.tsv")
    assert paths["periogt"].name == "periogt_external_reference_manifest.tsv"


def test_normalize_spelling_keeps_metadata_and_real_seeds_separate() -> None:
    original = _predictions()
    original.loc[1, "input_view"] = "structure_plus_metadata"
    normalized = normalize_predictions(original)
    assert normalized["task"].tolist() == ["EGC", "EGC"]
    assert normalized["input_view"].tolist() == ["valid", "structure_plus_metadata_secondary"]
    assert normalized["representation_seed"].tolist() == [0, 0]
    assert normalized["head_seed"].tolist() == [11, 11]
    assert original["task"].tolist() == ["Egc", "Egc"]
    assert original["input_view"].iloc[0] == "structure_only"


def test_stream_metrics_uses_full_fold_and_each_seed_without_scope_splitting(tmp_path: Path) -> None:
    egc = pd.concat([_predictions(), _predictions(head_seed=22)], ignore_index=True)
    ea = _predictions("ea")
    validation = ea.iloc[:1].assign(split="validation", y_pred=100.0)
    manifest = _manifest(tmp_path, [egc, pd.concat([ea, validation], ignore_index=True)])
    summary = stream_metrics(manifest, "chemprop", tmp_path / "summary", progress_every=0)
    metrics = pd.read_csv(summary["outputs"]["task_metrics"])
    assert summary["processed_files"] == 2
    assert summary["prediction_rows"] == 6
    assert summary["input_rows"] == 7
    assert summary["metric_rows"] == 3
    assert set(metrics["representation_seed"]) == {0}
    assert set(metrics["head_seed"]) == {11, 22}
    assert set(metrics["n_rows"]) == {2}
    assert set(metrics["n_groups"]) == {2}
    assert "family_id" not in metrics and "scope_class" not in metrics
    assert metrics.loc[metrics.task.eq("EA"), "value"].item() == 2.0
    assert np.allclose(metrics.loc[metrics.task.eq("EGC"), "value"], np.sqrt(5.0))
    assert set(metrics["reporting_table"]) == {"controlled_common_table"}
    inventory = pd.read_csv(summary["outputs"]["prediction_inventory"])
    assert inventory["metric_rows"].tolist() == [2, 1]


def test_external_metadata_is_not_combined_with_structure_only(tmp_path: Path) -> None:
    frame = pd.concat(
        [_predictions("PE-I"), _predictions("PE-I").assign(input_view="structure_plus_metadata")],
        ignore_index=True,
    )
    manifest = _manifest(tmp_path, [frame])
    summary = stream_metrics(manifest, "periogt", tmp_path / "summary")
    metrics = pd.read_csv(summary["outputs"]["task_metrics"])
    assert len(metrics) == 2
    assert set(metrics.input_view) == {"valid", "structure_plus_metadata_secondary"}
    assert set(metrics.reporting_table) == {"same_split_external_reference"}


def test_completed_source_resume_does_not_reread_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path, [_predictions()])
    destination = tmp_path / "summary"
    first = stream_metrics(manifest, "chemprop", destination)

    def unexpected_load(path: Path) -> pd.DataFrame:
        raise AssertionError("completed source must not reload predictions")

    monkeypatch.setattr(posttraining_metrics, "load_predictions", unexpected_load)
    assert stream_metrics(manifest, "chemprop", destination) == first
    assert json.loads((destination / "summary.json").read_text())["status"] == "complete"


def test_missing_output_table_rebuilds_source(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [_predictions()])
    destination = tmp_path / "summary"
    stream_metrics(manifest, "chemprop", destination)
    (destination / "task_metrics.csv").unlink()
    summary = stream_metrics(manifest, "chemprop", destination)
    assert Path(summary["outputs"]["task_metrics"]).is_file()


def test_duplicate_rows_do_not_inflate_fold_metric(tmp_path: Path) -> None:
    frame = _predictions()
    manifest = _manifest(tmp_path, [pd.concat([frame, frame.iloc[:1]], ignore_index=True)])
    with pytest.raises(ValueError, match="duplicate held-out sample"):
        stream_metrics(manifest, "chemprop", tmp_path / "summary")


def test_nonfinite_prediction_is_reported(tmp_path: Path) -> None:
    frame = _predictions()
    frame.loc[0, "y_pred"] = float("nan")
    manifest = _manifest(tmp_path, [frame])
    with pytest.raises(ValueError, match="finite"):
        stream_metrics(manifest, "chemprop", tmp_path / "summary")
