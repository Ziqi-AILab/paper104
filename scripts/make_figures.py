#!/usr/bin/env python
"""Generate only those Paper104 figures supported by current outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfid.analysis.figures import make_all_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-root", default="figures")
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Create explicitly named engineering previews from smoke relation data.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    outcomes = make_all_figures(
        project_root,
        args.results_root,
        args.output_root,
        include_smoke=args.include_smoke,
    )
    print(json.dumps([outcome.as_dict() for outcome in outcomes], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
