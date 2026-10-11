#!/usr/bin/env python3
"""Select the Paper104 formal recipe from the two development seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_SEEDS = frozenset((13, 37))
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "slurm" / "generated" / "development" / "development_manifest.tsv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "pretraining" / "selected_recipe.yaml"


def _resolve_from_project(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def final_validation_loss(metrics_path: Path) -> float:
    validation_values: list[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "validation_mlm_loss" in row:
            validation_values.append(float(row["validation_mlm_loss"]))
    if not validation_values:
        raise ValueError(f"no validation_mlm_loss in {metrics_path}")
    return validation_values[-1]


def _read_development_results(manifest_path: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        for row in rows:
            if row["matrix"] != "development":
                continue
            seed = int(row["seed"])
            if seed not in DEVELOPMENT_SEEDS:
                continue
            run_result_path = _resolve_from_project(row["output_dir"]) / "run_result.json"
            if not run_result_path.is_file():
                raise FileNotFoundError(f"development result is incomplete: {run_result_path}")
            run_result = json.loads(run_result_path.read_text(encoding="utf-8"))
            metrics_path = _resolve_from_project(str(run_result["metrics_path"]))
            observations.append(
                {
                    "arm": row["arm"],
                    "seed": seed,
                    "learning_rate": float(row["learning_rate"]),
                    "boundary_fraction": float(row["boundary_fraction"]),
                    "interior_fraction": float(row["interior_fraction"]),
                    "validation_mlm_loss": final_validation_loss(metrics_path),
                }
            )
    return observations


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    arm: str,
    fixed_learning_rate: float | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm:
            continue
        if fixed_learning_rate is not None and row["learning_rate"] != fixed_learning_rate:
            continue
        key = (
            row["learning_rate"],
            row["boundary_fraction"],
            row["interior_fraction"],
        )
        grouped[key].append(row)

    candidates: list[dict[str, Any]] = []
    for key, values in grouped.items():
        seeds = {row["seed"] for row in values}
        if seeds != DEVELOPMENT_SEEDS:
            continue
        candidates.append(
            {
                "arm": arm,
                "learning_rate": key[0],
                "boundary_fraction": key[1],
                "interior_fraction": key[2],
                "mean_validation_mlm_loss": mean(
                    row["validation_mlm_loss"] for row in values
                ),
                "seed_losses": {
                    int(row["seed"]): row["validation_mlm_loss"]
                    for row in sorted(values, key=lambda item: item["seed"])
                },
            }
        )
    if not candidates:
        detail = "" if fixed_learning_rate is None else f" at lr={fixed_learning_rate:g}"
        raise ValueError(f"no complete two-seed {arm} candidate{detail}")
    return sorted(
        candidates,
        key=lambda row: (
            row["mean_validation_mlm_loss"],
            row["learning_rate"],
            row["boundary_fraction"],
            row["interior_fraction"],
        ),
    )


def select_recipe(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    observations = _read_development_results(manifest_path)
    selected_warm = _aggregate(observations, arm="A0")[0]
    selected_scratch = _aggregate(observations, arm="B0")[0]
    selected_boundary = _aggregate(
        observations,
        arm="A_att",
        fixed_learning_rate=selected_warm["learning_rate"],
    )[0]
    result = {
        "version": 1,
        "selection": {
            "source": str(manifest_path),
            "development_seeds": sorted(DEVELOPMENT_SEEDS),
            "metric": "final_validation_mlm_loss",
            "aggregation": "mean_across_development_seeds",
            "A0_warm_learning_rate": selected_warm,
            "B0_scratch_learning_rate": selected_scratch,
            "A_att_boundary_ratio_at_selected_warm_lr": selected_boundary,
        },
        "selected_recipe": {
            "warm_learning_rate": selected_warm["learning_rate"],
            "scratch_learning_rate": selected_scratch["learning_rate"],
            "boundary_fraction": selected_boundary["boundary_fraction"],
            "interior_fraction": selected_boundary["interior_fraction"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(result, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = select_recipe(args.manifest.resolve(), args.output.resolve())
    print(yaml.safe_dump(result["selected_recipe"], sort_keys=False).strip())


if __name__ == "__main__":
    main()
