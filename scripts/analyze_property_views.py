#!/usr/bin/env python
"""Summarise formal property-view predictions on the local CPU."""

from __future__ import annotations

import argparse
import json

from polyfid.analysis.property_view_statistics import (
    write_property_view_analysis_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()
    summary = write_property_view_analysis_outputs(
        args.manifest,
        args.output,
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
