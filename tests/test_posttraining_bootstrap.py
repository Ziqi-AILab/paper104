from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from polyfid.analysis.posttraining_bootstrap import (
    collect_bootstrap_outputs,
    declared_comparisons,
    normalized_manifest,
    run_bootstrap_item,
)


def test_manifest_keeps_three_adaptation_seeds_in_one_representation(tmp_path: Path) -> None:
    source = tmp_path / "adaptation.tsv"
    pd.DataFrame([
        {"arm": "A0", "method": "lora", "representation_seed": 42, "adaptation_seed": seed, "task": "Egc"}
        for seed in (11, 22, 33)
    ]).to_csv(source, sep="\t", index=False)
    manifest = normalized_manifest(source, adaptation=True)
    assert manifest["arm"].tolist() == ["A0_lora"] * 3
    assert manifest["task"].tolist() == ["EGC"] * 3
    assert manifest["representation_seed"].nunique() == 1
    assert len(manifest) == 3


def test_only_controlled_arms_enter_bootstrap_comparisons() -> None:
    comparisons = declared_comparisons()
    assert len(comparisons) == 11
    assert sum(item.analysis_family == "frozen_task_intervals" for item in comparisons) == 3
    assert all("baseline_" not in item.candidate for item in comparisons)
    assert all("baseline_" not in item.baseline for item in comparisons)


def test_prediction_level_nested_interval_and_collection(tmp_path: Path) -> None:
    paths = []
    for method, error in (("A0_lora", 0.5), ("B0_lora", 1.0)):
        rows = []
        for representation_seed in (42, 123, 777, 2025, 2026):
            for head_seed in (11, 22, 33):
                for index in range(6):
                    rows.append({
                        "sample_id": str(index), "task": "EA", "split": "test", "group_id": f"g{index // 2}",
                        "y_true": float(index), "y_pred": index + error, "method": method,
                        "representation_seed": representation_seed, "head_seed": head_seed,
                        "protocol": "family_ood", "family_id": f"family_{index // 2}", "cluster_id": None,
                        "fold": str(index // 2), "label_fraction": 1.0, "subsample_id": 0, "input_view": "valid",
                    })
        path = tmp_path / f"{method}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        paths.append(str(path))
    item = {
        "work_id": "lora__molecular_initialization__EA", "name": "molecular_initialization",
        "candidate": "A0_lora", "baseline": "B0_lora", "analysis_family": "lora", "scope": "EA",
        "prediction_files": paths, "n_resamples": 25, "bootstrap_seed": 7,
        "output": str(tmp_path / "result.json"), "source_manifest": "synthetic_fixture.tsv",
    }
    result = run_bootstrap_item(item)
    assert result["n_representation_seeds"] == 5
    assert result["n_head_seeds"] == 3
    assert result["n_paired_rows"] == 90
    assert result["estimate"] == np.log(0.5)
    assert np.isclose(result["ci_lower"], result["ci_upper"])
    assert "holm_adjusted_p" not in result
    assert run_bootstrap_item(item) == result
    frame = collect_bootstrap_outputs([item], tmp_path)
    assert len(frame) == 1
    assert json.loads((tmp_path / "bootstrap_progress.json").read_text())["status"] == "complete"
