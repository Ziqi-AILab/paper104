#!/usr/bin/env python3
"""Profile real restored models; cached-head mode is a separate local CPU run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfid.evaluation.cost_profiling import TASKS, generate_timing_jobs, run_task_profile


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--mode", choices=("gpu", "cached-head"), default="gpu")
    parser.add_argument("--recovery-manifest", type=Path, default=project / "runs/manifests/cost_recovery/baseline_recovery_manifest.tsv")
    parser.add_argument("--output-root", type=Path, default=project / "results/metrics/workflow_cost")
    parser.add_argument("--generate-slurm", action="store_true")
    args = parser.parse_args()
    if args.generate_slurm:
        print(json.dumps(generate_timing_jobs(project, args.recovery_manifest.resolve(), args.output_root.resolve()), indent=2))
        return
    if args.mode == "gpu" and args.task is None:
        parser.error("GPU mode needs one --task per single-A800 job")
    tasks = (args.task,) if args.task else TASKS
    for task in tasks:
        result = run_task_profile(project, task=task, recovery_manifest=args.recovery_manifest, output_root=args.output_root, mode=args.mode)
        print(json.dumps({"task": result["task"], "status": result["status"], "methods": result["method_order"]}))


if __name__ == "__main__":
    main()
