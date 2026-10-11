"""Run PerioGT's released preparation script with CPU-portable weight loading."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Sequence

from polyfid.baselines.periogt import _portable_torch_load


def run_upstream_script(script: str | Path, arguments: Sequence[str]) -> None:
    upstream = Path(script).resolve()
    previous_argv = sys.argv
    sys.argv = [str(upstream), *arguments]
    try:
        with _portable_torch_load(force_cpu=True):
            runpy.run_path(str(upstream), run_name="__main__")
    finally:
        sys.argv = previous_argv


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m polyfid.baselines.periogt_prepare_bridge SCRIPT [ARGS]")
    run_upstream_script(sys.argv[1], sys.argv[2:])


if __name__ == "__main__":
    main()
