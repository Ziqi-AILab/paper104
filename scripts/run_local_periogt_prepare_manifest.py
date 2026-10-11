#!/usr/bin/env python3
"""Run or report the seven local-CPU PerioGT cache preparations."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT
    / "slurm/generated/external_reference/graph_models/periogt_prepare_manifest.tsv"
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def complete(row: dict[str, str]) -> bool:
    record = Path(row["output"]) / "run.json"
    if not record.is_file():
        return False
    return json.loads(record.read_text(encoding="utf-8")).get("status") == "complete"


def report(rows: list[dict[str, str]]) -> None:
    finished = sum(complete(row) for row in rows)
    print(f"periogt_prepare complete={finished} pending={len(rows) - finished} total={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    rows = _rows(args.manifest.resolve())
    if args.status or (not args.all and args.row_index is None):
        report(rows)
        return
    selected = rows if args.all else [rows[args.row_index]]
    for row in selected:
        if complete(row):
            print(f"complete; skipping {row['job_id']}")
            continue
        print(f"running {row['job_id']}", flush=True)
        subprocess.run(shlex.split(row["command"]), cwd=PROJECT, check=True)
    report(rows)


if __name__ == "__main__":
    main()
