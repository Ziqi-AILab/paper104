#!/usr/bin/env python
"""Analyse the formal A0/B0 fixed-mask token-efficiency curve on local CPU."""

from __future__ import annotations

import argparse
import json

from polyfid.analysis.token_efficiency_statistics import (
    write_token_efficiency_analysis_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=16)
    args = parser.parse_args()
    summary = write_token_efficiency_analysis_outputs(
        args.manifest,
        args.output,
        n_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_chunk_size=args.bootstrap_chunk_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
