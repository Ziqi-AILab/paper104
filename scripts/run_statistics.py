#!/usr/bin/env python
"""Compute Paper104 statistics from held-out, per-sample predictions."""

from __future__ import annotations

import argparse
import json

from polyfid.analysis.statistics import (
    load_prediction_inputs,
    parse_metric_overrides,
    write_analysis_outputs,
    write_manifest_analysis_outputs,
)


def _key_values(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"filter must have COLUMN=VALUE form: {value}")
        key, item = value.split("=", maxsplit=1)
        output[key.strip()] = item.strip()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--predictions",
        action="append",
        help="Per-sample parquet/jsonl file or directory; repeat as needed.",
    )
    inputs.add_argument(
        "--prediction-manifest",
        help="Frozen-probe TSV/CSV whose output column lists prediction files.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument(
        "--metric",
        action="append",
        default=[],
        metavar="TASK=METRIC",
        help="Override the declared default metric for an exact task name.",
    )
    parser.add_argument(
        "--primary-filter",
        action="append",
        default=[],
        metavar="COLUMN=VALUE",
        help="Optional filter defining the single primary aggregate scope.",
    )
    parser.add_argument(
        "--primary-scope",
        action="append",
        default=[],
        metavar="TASK=PROTOCOL",
        help=(
            "Select one protocol per task for the primary aggregate. By default EA/IP use "
            "family_ood when available, other tasks prefer group_ood, and an unambiguous "
            "single protocol is retained."
        ),
    )
    args = parser.parse_args()
    common = {
        "metric_overrides": parse_metric_overrides(args.metric),
        "n_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "primary_filters": _key_values(args.primary_filter),
        "primary_task_protocols": _key_values(args.primary_scope),
    }
    if args.prediction_manifest:
        summary = write_manifest_analysis_outputs(
            args.prediction_manifest,
            args.output,
            progress_every=args.progress_every,
            **common,
        )
    else:
        predictions, files = load_prediction_inputs(args.predictions)
        summary = write_analysis_outputs(
            predictions,
            files,
            args.output,
            **common,
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
