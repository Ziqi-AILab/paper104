#!/usr/bin/env python
"""Build downstream and fidelity encoder input tables on local CPU."""

from __future__ import annotations

import json

from polyfid.paths import load_paths
from polyfid.workflows.evaluation_inputs import build_evaluation_inputs


if __name__ == "__main__":
    print(
        json.dumps(
            build_evaluation_inputs(load_paths()), indent=2, ensure_ascii=False
        )
    )
