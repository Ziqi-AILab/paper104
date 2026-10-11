#!/usr/bin/env python
"""Render the Figure 4 property-view companion panel."""

from __future__ import annotations

import argparse
import json

from polyfid.analysis.figures import figure4d_property_view_stability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    result = figure4d_property_view_stability(args.statistics, args.output)
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
