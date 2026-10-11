#!/usr/bin/env python
"""Create Paper104 split manifests from the shared standardized datasets."""

from __future__ import annotations

import json

from polyfid.paths import load_paths
from polyfid.workflows.data import prepare_project_data


def main() -> None:
    print(json.dumps(prepare_project_data(load_paths()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

