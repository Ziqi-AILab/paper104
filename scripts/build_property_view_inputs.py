#!/usr/bin/env python3
"""Build fixed, split-aware downstream equivalent-view inputs on local CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyfid.paths import load_paths
from polyfid.workflows.property_view_inputs import build_property_view_inputs


def main() -> None:
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=paths.project_root / "runs/manifests/frozen_probes/frozen_probe_manifest.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.project_root / "data/derived/property_views",
    )
    parser.add_argument("--randomized-views", type=int, default=10)
    parser.add_argument("--atom-permutations", type=int, default=10)
    parser.add_argument("--periodic-views", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10407)
    args = parser.parse_args()
    report = build_property_view_inputs(
        args.probe_manifest,
        args.output,
        include_openpoly=paths.formal_openpoly,
        randomized_count=args.randomized_views,
        atom_permutation_count=args.atom_permutations,
        periodic_count=args.periodic_views,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
