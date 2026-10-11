#!/usr/bin/env python3
"""Select the common property-head learning rate from validation losses."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from polyfid.evaluation.probe_selection import write_property_head_selection


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-losses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_property_head_selection(
        _read_table(args.validation_losses), args.output
    )
    print(payload["selected_recipe"]["property_head_learning_rate"])


if __name__ == "__main__":
    main()
