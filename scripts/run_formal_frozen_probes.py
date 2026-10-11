#!/usr/bin/env python3
"""Run the complete formal frozen-probe manifest with persistent CPU workers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def balanced_ranges(row_count: int, workers: int) -> list[tuple[int, int]]:
    base, remainder = divmod(row_count, workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    for worker in range(workers):
        stop = start + base + (1 if worker < remainder else 0)
        if start < stop:
            ranges.append((start, stop))
        start = stop
    return ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "runs/manifests/frozen_probes/frozen_probe_manifest.tsv",
    )
    parser.add_argument(
        "--python", type=Path, default=PROJECT_ROOT / ".conda/env/bin/python"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("workers and threads-per-worker must be positive")
    row_count = len(pd.read_csv(args.manifest, sep="\t", usecols=["row_index"]))
    ranges = balanced_ranges(row_count, args.workers)
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.threads_per_worker),
            "MKL_NUM_THREADS": str(args.threads_per_worker),
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    processes: list[tuple[int, int, subprocess.Popen[bytes]]] = []
    for start, stop in ranges:
        command = [
            str(args.python),
            "scripts/run_local_probe_manifest.py",
            "--manifest",
            str(args.manifest),
            "--all",
            "--start-index",
            str(start),
            "--stop-index",
            str(stop),
            "--skip-complete",
            "--in-process",
            "--progress-every",
            str(args.progress_every),
        ]
        print(f"starting local probe shard {start}:{stop}", flush=True)
        processes.append(
            (
                start,
                stop,
                subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment),
            )
        )
    failures: list[tuple[int, int, int]] = []
    for start, stop, process in processes:
        return_code = process.wait()
        print(
            f"local probe shard {start}:{stop} exited with {return_code}",
            flush=True,
        )
        if return_code != 0:
            failures.append((start, stop, return_code))
    if failures:
        raise RuntimeError(f"formal frozen-probe shard failures: {failures}")
    print(f"formal frozen probes complete: {row_count}/{row_count}", flush=True)


if __name__ == "__main__":
    main()
