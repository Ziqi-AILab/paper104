#!/usr/bin/env python3
"""Collect existing Paper104 cost records locally; never submit GPU work."""
import argparse
import json
from pathlib import Path

from polyfid.analysis.cost_accounting import build_cost_accounting


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reuse-scheduler-snapshot", action="store_true")
    parser.add_argument("--refresh-cpu-timers", action="store_true")
    parser.add_argument("--cpu-workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_cost_accounting(args.project_root, refresh_scheduler=not args.reuse_scheduler_snapshot,
                                           refresh_cpu_timers=args.refresh_cpu_timers,
                                           cpu_workers=args.cpu_workers), indent=2), flush=True)
