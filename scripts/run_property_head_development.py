#!/usr/bin/env python3
"""Wait for development embeddings, run all local probes, then select one LR."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def embedding_is_complete(output: str | Path) -> bool:
    output_path = Path(output)
    run_record = output_path.with_suffix(".run_record.json")
    if not output_path.is_file() or not run_record.is_file():
        return False
    try:
        status = json.loads(run_record.read_text(encoding="utf-8")).get("status")
    except (OSError, json.JSONDecodeError):
        return False
    return status == "complete"


def wait_for_embeddings(manifest: Path, poll_seconds: int) -> None:
    rows = pd.read_csv(manifest, sep="\t")
    outputs = rows["output_path"].astype(str).tolist()
    while True:
        ready = sum(embedding_is_complete(path) for path in outputs)
        print(f"development embeddings complete: {ready}/{len(outputs)}", flush=True)
        if ready == len(outputs):
            return
        time.sleep(poll_seconds)


def _probe_command(
    index: int,
    *,
    manifest: Path,
    python: Path,
) -> list[str]:
    return [
        str(python),
        "scripts/run_local_probe_manifest.py",
        "--manifest",
        str(manifest),
        "--row-index",
        str(index),
        "--skip-complete",
    ]


def run_probes(manifest: Path, *, python: Path, workers: int) -> None:
    rows = pd.read_csv(manifest, sep="\t")
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "2",
            "PYTHONUNBUFFERED": "1",
        }
    )

    def run(index: int) -> int:
        subprocess.run(
            _probe_command(index, manifest=manifest, python=python),
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )
        return index

    failures: list[tuple[int, str]] = []
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run, index): index for index in range(len(rows))}
        for future in as_completed(futures):
            index = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append((index, str(error)))
                print(f"probe row {index} failed: {error}", flush=True)
            finished += 1
            if finished % 25 == 0 or finished == len(rows):
                print(
                    f"probe rows processed: {finished}/{len(rows)}; failures={len(failures)}",
                    flush=True,
                )
    if failures:
        failed_rows = ",".join(str(index) for index, _ in failures)
        raise RuntimeError(f"property-head development failed rows: {failed_rows}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "slurm/generated/postpretraining/property_head_development"
        / "property_head_development_embedding_manifest.tsv",
    )
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "runs/manifests/property_head_development"
        / "property_head_development_manifest.tsv",
    )
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".conda/env/bin/python")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    wait_for_embeddings(args.embedding_manifest, args.poll_seconds)
    run_probes(args.probe_manifest, python=args.python, workers=args.workers)
    subprocess.run(
        [
            str(args.python),
            "scripts/select_property_head_from_development.py",
            "--manifest",
            str(args.probe_manifest),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
