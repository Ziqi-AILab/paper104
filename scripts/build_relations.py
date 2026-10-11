#!/usr/bin/env python
"""Build PI1M relation indices and verified triplets on the local CPU."""

from __future__ import annotations

import argparse
import json

from polyfid.paths import load_paths
from polyfid.workflows.relations import build_relation_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-triplets", type=int, default=10_000)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--max-index-rows", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Refresh the derived partition index before building relations.",
    )
    parser.add_argument(
        "--allow-near-formula-fallback",
        action="store_true",
        help=(
            "Development diagnostic only: after exact-formula matching, allow "
            "same-atom-count near-formula negatives."
        ),
    )
    args = parser.parse_args()
    result = build_relation_set(
        load_paths(),
        partition=args.partition,
        target_triplets=args.target_triplets,
        max_index_rows=args.max_index_rows,
        workers=args.workers,
        seed=args.seed,
        rebuild_index=args.rebuild_index,
        allow_near_formula_fallback=args.allow_near_formula_fallback,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
