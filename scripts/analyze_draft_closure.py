#!/usr/bin/env python
"""Run the finite Paper104 draft-closure intervals on local CPU (resumable)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfid.analysis.draft_closure_statistics import render_draft_closure_figures, run_draft_closure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=("all", "geometry", "property_bridge"), default="all")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=100)
    parser.add_argument("--figures-only", action="store_true", help="Read completed interval CSVs; do not run bootstrap")
    parser.add_argument("--figure-output", type=Path, help="Defaults to project_root/figures")
    args = parser.parse_args()
    if args.figures_only:
        analysis_root = args.output or args.project_root / "results/statistics/draft_closure"
        outputs = render_draft_closure_figures(analysis_root, args.figure_output or args.project_root / "figures")
        print(json.dumps({"figures": [str(path) for path in outputs]}, indent=2), flush=True)
        return
    result = run_draft_closure(args.project_root, args.output, stage=args.stage,
                               n_resamples=args.bootstrap_resamples,
                               chunk_size=args.bootstrap_chunk_size, seed=args.bootstrap_seed)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
