#!/usr/bin/env python3
"""Local CPU integration of completed Paper104 predictions and intervals."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from polyfid.analysis.manuscript import render_observed_tables
from polyfid.analysis.posttraining_bootstrap import build_work_items, run_bootstrap_plan
from polyfid.analysis.posttraining_figures import render_posttraining_outputs
from polyfid.analysis.posttraining_metrics import discover_manifests, stream_metrics
from polyfid.io import atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--stage", choices=("all", "metrics", "bootstrap", "render"), default="all")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = (args.output or project / "results/statistics/completed_experiments").resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {
        "status": "running",
        "stage": args.stage,
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "execution_path": "[CPU → local, no sbatch]",
        "workers": args.workers,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "output": str(output),
        "frozen_primary_reused": str(project / "results/statistics/formal_frozen_probes/primary_aggregate.csv"),
    }
    if args.plan_only:
        items = build_work_items(project, output, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
        atomic_write_text(json.dumps(items, indent=2) + "\n", output / "bootstrap_plan.json")
        print(json.dumps({"work_items": len(items), "sources": {key: str(value) for key, value in discover_manifests(project).items()}}, indent=2))
        return
    record_path = output / f"run_{args.stage}.json"
    atomic_write_text(json.dumps(record, indent=2) + "\n", record_path)
    try:
        if args.stage in {"all", "metrics"}:
            summaries = {}
            for source, manifest in discover_manifests(project).items():
                print(f"metrics_source={source} manifest={manifest}", flush=True)
                summaries[source] = stream_metrics(manifest, source, output / "metrics" / source)
            record["metric_summaries"] = summaries
            # Descriptive tables are available before the longer interval jobs.
            render_posttraining_outputs(project, output)
            render_observed_tables(project)
        if args.stage in {"all", "bootstrap"}:
            items = build_work_items(project, output, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
            intervals = run_bootstrap_plan(items, output, workers=args.workers)
            record["completed_bootstrap_items"] = len(intervals)
        if args.stage in {"all", "bootstrap", "render"}:
            record["rendered"] = [str(path) for path in render_posttraining_outputs(project, output)]
            record["manuscript_tables"] = [str(path) for path in render_observed_tables(project)]
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["updated_at"] = datetime.now().astimezone().isoformat()
        record["wall_seconds"] = time.monotonic() - started
        atomic_write_text(json.dumps(record, indent=2) + "\n", record_path)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
