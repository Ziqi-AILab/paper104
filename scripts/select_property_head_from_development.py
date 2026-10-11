#!/usr/bin/env python3
"""Select the common frozen-head LR from completed development-manifest rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from polyfid.evaluation.probe_selection import write_property_head_selection
from polyfid.io import atomic_write_parquet, atomic_write_text


def collect_validation_metrics(manifest: str | Path) -> pd.DataFrame:
    plan = pd.read_csv(manifest, sep="\t")
    frames: list[pd.DataFrame] = []
    for row in plan.to_dict(orient="records"):
        metrics = pd.read_parquet(row["metrics_output"]).copy()
        metrics["input_view"] = row["input_view"]
        metrics["protocol_role"] = row["protocol_role"]
        frames.append(metrics)
    if not frames:
        raise ValueError("property-head development manifest is empty")
    return pd.concat(frames, ignore_index=True)


def selection_units(metrics: pd.DataFrame) -> pd.DataFrame:
    """Give every task/protocol/input-view unit equal weight in LR selection."""

    keys = ["learning_rate", "task", "protocol", "input_view"]
    return (
        metrics.groupby(keys, as_index=False, dropna=False)["validation_loss"]
        .mean()
        .assign(split="validation")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/downstream/selected_property_head.yaml"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(
            "runs/manifests/property_head_development/validation_metrics.parquet"
        ),
    )
    parser.add_argument(
        "--selection-units-output",
        type=Path,
        default=Path(
            "runs/manifests/property_head_development/validation_selection_units.parquet"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = collect_validation_metrics(args.manifest)
    development_seeds = sorted(
        metrics["representation_seed"].astype(int).unique().tolist()
    )
    if development_seeds != [13, 37]:
        raise ValueError(
            "property-head selection must use development representation seeds 13 and 37"
        )
    atomic_write_parquet(metrics, args.metrics_output, index=False)
    units = selection_units(metrics)
    atomic_write_parquet(units, args.selection_units_output, index=False)
    payload = write_property_head_selection(units, args.output)
    payload["development_representation_seeds"] = development_seeds
    atomic_write_text(
        yaml.safe_dump(payload, sort_keys=False),
        args.output,
    )
    print(payload["selected_recipe"]["property_head_learning_rate"])


if __name__ == "__main__":
    main()
