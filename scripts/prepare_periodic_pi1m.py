#!/usr/bin/env python
"""Build Paper104's two-cell PI1M identities and corrected split table."""

from __future__ import annotations

import argparse
import json

from polyfid.paths import load_paths
from polyfid.workflows.periodic_data import prepare_periodic_pi1m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_periodic_pi1m(
                load_paths(), workers=args.workers, force=args.force
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
