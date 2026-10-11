#!/usr/bin/env python3
"""Execute controlled classical baselines locally with resumable concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import shlex
import subprocess
import time
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RowResult:
    index: int
    job_id: str
    returncode: int
    wall_seconds: float


def _is_complete(row: pd.Series) -> bool:
    return Path(str(row["output"])).is_file() and Path(str(row["run_record"])).is_file()


def _run_row(index: int, row: pd.Series, environment: dict[str, str]) -> RowResult:
    command = shlex.split(str(row["command"]))
    started = time.monotonic()
    result = subprocess.run(command, check=False, env=environment)
    return RowResult(
        index=index,
        job_id=str(row["job_id"]),
        returncode=int(result.returncode),
        wall_seconds=time.monotonic() - started,
    )


def _write_status(
    path: Path | None,
    *,
    selected: int,
    skipped: int,
    finished: int,
    succeeded: int,
    failed: int,
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                f"selected={selected}",
                f"skipped_complete={skipped}",
                f"finished_this_run={finished}",
                f"succeeded_this_run={succeeded}",
                f"failed_this_run={failed}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def _indices(args: argparse.Namespace, row_count: int) -> range:
    if args.row_index is not None:
        if args.row_index < 0 or args.row_index >= row_count:
            raise ValueError(f"row index {args.row_index} is outside 0..{row_count - 1}")
        return range(args.row_index, args.row_index + 1)
    if not args.all:
        raise ValueError("choose --row-index or --all")
    stop = row_count if args.stop_index is None else args.stop_index
    if args.start_index < 0 or stop < args.start_index or stop > row_count:
        raise ValueError(f"requested range {args.start_index}:{stop} is outside 0:{row_count}")
    return range(args.start_index, stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--row-index", type=int)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--failure-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    if args.progress_every < 1:
        raise ValueError("progress-every must be at least one")
    frame = pd.read_csv(args.manifest, sep="\t")
    selected = list(_indices(args, len(frame)))
    if args.dry_run:
        for index in selected:
            print(shlex.join(shlex.split(str(frame.iloc[index]["command"]))))
        return

    skipped = 0
    pending: list[tuple[int, pd.Series]] = []
    for index in selected:
        row = frame.iloc[index]
        if args.skip_complete and _is_complete(row):
            skipped += 1
        else:
            pending.append((index, row))

    environment = os.environ.copy()
    # Four workers with these limits use at most 32 compute threads on the
    # 48-core local host, including XGBoost's declared eight-thread fits.
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment.setdefault(name, "8")

    if args.failure_log is not None:
        args.failure_log.parent.mkdir(parents=True, exist_ok=True)
        args.failure_log.write_text(
            "row_index\tjob_id\treturncode\n", encoding="utf-8"
        )

    total = len(selected)
    print(
        f"selected={total} skipped_complete={skipped} "
        f"pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    _write_status(
        args.status_file,
        selected=total,
        skipped=skipped,
        finished=0,
        succeeded=0,
        failed=0,
    )

    succeeded = 0
    failed = 0
    finished = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures: dict[Future[RowResult], int] = {
            executor.submit(_run_row, index, row, environment): index
            for index, row in pending
        }
        for future in as_completed(futures):
            result = future.result()
            finished += 1
            if result.returncode == 0:
                succeeded += 1
            else:
                failed += 1
                if args.failure_log is not None:
                    with args.failure_log.open("a", encoding="utf-8") as handle:
                        handle.write(
                            f"{result.index}\t{result.job_id}\t{result.returncode}\n"
                        )
                print(
                    f"FAILED row={result.index} job={result.job_id} "
                    f"rc={result.returncode}",
                    flush=True,
                )
            if finished % args.progress_every == 0 or finished == len(pending):
                print(
                    f"progress={finished}/{len(pending)} succeeded={succeeded} "
                    f"failed={failed} last_row={result.index} "
                    f"last_seconds={result.wall_seconds:.1f}",
                    flush=True,
                )
                _write_status(
                    args.status_file,
                    selected=total,
                    skipped=skipped,
                    finished=finished,
                    succeeded=succeeded,
                    failed=failed,
                )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
