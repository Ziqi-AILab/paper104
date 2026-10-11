#!/usr/bin/env python
"""Build Paper104 mechanism-control supplementary tables on local CPU."""

from __future__ import annotations

import argparse
import json

from polyfid.analysis.mechanism_control_statistics import (
    write_mechanism_control_analysis_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-root", required=True)
    parser.add_argument("--relation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=64)
    args = parser.parse_args()
    summary = write_mechanism_control_analysis_outputs(
        args.boundary_root,
        args.relation_root,
        args.output,
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
