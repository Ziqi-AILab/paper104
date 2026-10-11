#!/usr/bin/env python3
"""Recompute released numerical summaries on local CPU using the standard library.

Point estimates are recalculated from fit-level metrics. Hierarchical confidence
intervals are retained outputs, not recomputed from these aggregate inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("task", "protocol", "split", "label_fraction", "subsample_id", "input_view", "metric")
FIT_KEYS = ("source_group", "method", "representation_seed", "head_seed", *CONDITIONS)
SUMMARY_KEYS = ("source_group", "method", *CONDITIONS, "folds")
DETERMINISTIC = {"baseline_ecfp_ridge", "baseline_rdkit_ridge", "baseline_tanimoto_nn", "baseline_nearest_neighbor"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def groups(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(str(row[key]) for key in keys)].append(row)
    return result


def check_close(actual, expected, description):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{description}: recomputed {actual}, saved {expected}")


def pool_folds(rows):
    """Recover pooled MAE/RMSE from counts, never average fold RMSEs."""
    output = []
    for key, cell in sorted(groups(rows, FIT_KEYS).items()):
        record = dict(zip(FIT_KEYS, key))
        metric = record["metric"]
        if metric not in {"mae", "rmse"}:
            raise ValueError(f"Unsupported pooling metric: {metric}")
        n = sum(int(row["n_rows"]) for row in cell)
        power = 2 if metric == "rmse" else 1
        error = math.fsum(int(row["n_rows"]) * float(row["value"]) ** power for row in cell) / n
        folds = sorted({str(row["fold"]) for row in cell})
        output.append({**record, "value": math.sqrt(error) if power == 2 else error,
                       "n_rows": n, "n_folds": len(folds), "folds": ";".join(folds)})
    return output


def summarize_replicates(pooled):
    """Average heads within representation, then representations equally."""
    output = []
    for key, cell in sorted(groups(pooled, SUMMARY_KEYS).items()):
        record = dict(zip(SUMMARY_KEYS, key))
        reps = groups(cell, ("representation_seed",))
        rep_means = [mean(float(row["value"]) for row in rep) for rep in reps.values()]
        head_counts = [len({row["head_seed"] for row in rep}) for rep in reps.values()]
        deterministic = record["method"] in DETERMINISTIC
        if deterministic:
            repeat_values, basis = [mean(rep_means)], "deterministic fit; no seed SD"
        elif len(reps) > 1:
            repeat_values, basis = rep_means, "representation-seed means; heads nested"
        else:
            repeat_values = [float(row["value"]) for row in cell]
            basis = "downstream training/head seeds; fixed representation"
        output.append({**record, "mean": mean(rep_means),
                       "sd": stdev(repeat_values) if len(repeat_values) > 1 else "",
                       "sd_basis": basis, "n_outer_repeats": len(repeat_values),
                       "n_representation_seeds": sum(int(float(key[0])) != 0 for key in reps),
                       "n_recorded_head_seeds_min": min(head_counts),
                       "n_recorded_head_seeds_max": max(head_counts),
                       "n_seed_fit_cells": 1 if deterministic else len(cell),
                       "n_folds": max(int(row["n_folds"]) for row in cell),
                       "n_prediction_instances_min": min(int(row["n_rows"]) for row in cell),
                       "n_prediction_instances_max": max(int(row["n_rows"]) for row in cell)})
    return output


def compare_tables(actual, expected, keys):
    indexed = {tuple(str(row[key]) for key in keys): row for row in expected}
    if len(actual) != len(indexed):
        raise ValueError(f"Table row count differs: {len(actual)} vs {len(indexed)}")
    for row in actual:
        saved = indexed[tuple(str(row[key]) for key in keys)]
        for field, value in row.items():
            if field in keys:
                continue
            try:
                number = float(value)
            except (ValueError, TypeError):
                if str(value) != str(saved[field]):
                    raise ValueError(f"Different {field}: {value!r} vs {saved[field]!r}")
            else:
                check_close(number, saved[field], field)


def paired_contrast(pooled, spec):
    """Use saved scope, but calculate paired log ratios from individual fits."""
    protocols = dict(part.split("=", 1) for part in spec["task_protocols"].split(";"))
    fraction = float(spec.get("label_fraction", spec.get("primary_label_fraction", 1)))
    view = spec.get("input_view", spec.get("primary_input_view", "valid"))
    subsample = int(spec.get("primary_subsample_id", 0))
    methods = (spec["candidate"], spec["baseline"])
    paired = defaultdict(dict)
    counts = defaultdict(dict)
    for row in pooled:
        if (row["method"] in methods and row["split"] == "test"
                and float(row["label_fraction"]) == fraction
                and int(row["subsample_id"]) == subsample and row["input_view"] == view
                and protocols.get(row["task"]) == row["protocol"]):
            key = (row["representation_seed"], row["head_seed"], row["task"])
            if row["method"] in paired[key]:
                raise ValueError(f"Duplicate paired fit: {key}, {row['method']}")
            paired[key][row["method"]] = float(row["value"])
            counts[key][row["method"]] = int(row["n_rows"])
    if not paired:
        raise ValueError(f"No paired fits for {methods}")
    by_rep = defaultdict(list)
    for key, values in paired.items():
        if set(values) != set(methods) or len(set(counts[key].values())) != 1:
            raise ValueError(f"Unmatched fit metrics: {key}")
        by_rep[key[0]].append(math.log(values[methods[0]] / values[methods[1]]))
    if {key[2] for key in paired} != set(protocols):
        raise ValueError("Missing task in paired contrast")
    return mean(mean(values) for values in by_rep.values()), len(paired)


def interval_display(spec, estimate, scale="log_ratio"):
    """Keep saved interval endpoints and only transform their display units."""
    check_close(estimate, spec["estimate"], "paired estimate")
    if scale == "log_ratio":
        transform, unit = lambda value: 100 * math.expm1(value), "percent change"
    elif scale == "percentage_points":
        transform, unit = lambda value: 100 * value, "percentage points"
    else:
        transform, unit = lambda value: value, "difference"
    return {**spec, "recomputed_estimate": estimate,
            "display_estimate": transform(estimate),
            "display_ci_lower": transform(float(spec["ci_lower"])),
            "display_ci_upper": transform(float(spec["ci_upper"])),
            "display_unit": unit, "interval_source": "saved hierarchical bootstrap; not rerun"}


def geometry_points(points, intervals):
    indexed = groups(points, ("scope", "metric"))
    result = []
    for spec in intervals:
        cell = indexed[(spec["scope"], spec["metric"])]
        estimate = mean(float(row["candidate_value"]) - float(row["baseline_value"]) for row in cell)
        for field in ("candidate_value", "baseline_value"):
            check_close(mean(float(row[field]) for row in cell), spec[field], field)
        scale = "percentage_points" if spec["metric"] in {"recall_at_1", "triplet_accuracy"} else "difference"
        result.append(interval_display(spec, estimate, scale))
    return result


def bridge_points(points, intervals):
    result = []
    for spec in intervals:
        cell = [row for row in points if row["metric"] == spec["metric"]
                and (spec["task"] == "aggregate" or row["task"] == spec["task"])]
        # Heads are nested within representation/task, then representations and
        # fixed tasks are averaged. No ratio of descriptive means is used.
        task_rep = groups(cell, ("task", "representation_seed"))
        by_task = defaultdict(list)
        for (task, _), fits in task_rep.items():
            ratios = [math.log(float(row["candidate_value"]) / float(row["baseline_value"])) for row in fits]
            by_task[task].append(mean(ratios))
        estimate = mean(mean(values) for values in by_task.values())
        result.append(interval_display(spec, estimate))
    return result


def reproduce(results_root, output):
    stats = results_root / "statistics"
    pooled = read_csv(stats / "completed_experiments/tables/pooled_seed_metrics.csv")
    folds = [{"source_group": "frozen_probe", **row}
             for row in read_csv(stats / "formal_frozen_probes/primary_fold_metrics.csv")]
    repooled = pool_folds(folds)
    selected_keys = {tuple(row[key] for key in FIT_KEYS) for row in repooled}
    selected_saved = [row for row in pooled if tuple(row[key] for key in FIT_KEYS) in selected_keys]
    compare_tables(repooled, selected_saved, FIT_KEYS)
    summaries = summarize_replicates(pooled)
    compare_tables(summaries, read_csv(stats / "completed_experiments/tables/all_method_summary.csv"), SUMMARY_KEYS)
    compare_tables(summaries, read_csv(results_root / "si_complete_descriptive_matrix_data.csv"), SUMMARY_KEYS)

    primary = []
    for spec in read_csv(stats / "formal_frozen_probes/primary_aggregate.csv"):
        estimate, _ = paired_contrast(repooled, spec)
        primary.append(interval_display(spec, estimate))
    completed = []
    for name in ("task_bootstrap.csv", "secondary_aggregate.csv"):
        for spec in read_csv(stats / "completed_experiments" / name):
            estimate, _ = paired_contrast(pooled, spec)
            completed.append(interval_display(spec, estimate))
    geometry = geometry_points(read_csv(stats / "draft_closure/geometry/per_seed_metrics.csv"),
                               read_csv(stats / "draft_closure/geometry/intervals.csv"))
    bridge = bridge_points(read_csv(stats / "draft_closure/property_bridge/per_seed_head_task_metrics.csv"),
                           read_csv(stats / "draft_closure/property_bridge/intervals.csv"))

    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("primary_pooled_fit_metrics.csv", repooled),
                       ("all_method_summary.csv", summaries),
                       ("original_primary.csv", primary),
                       ("completed_intervals.csv", completed),
                       ("geometry_intervals.csv", geometry),
                       ("property_bridge_intervals.csv", bridge)):
        write_csv(output / name, rows)
    report = {"execution": "local CPU; Python standard library only",
              "pooled_fit_rows": len(pooled), "primary_fold_rows": len(folds),
              "primary_pooled_fit_rows": len(repooled), "descriptive_summary_rows": len(summaries),
              "point_estimates_recomputed": {"original_primary": len(primary), "completed": len(completed),
                                             "geometry": len(geometry), "property_bridge": len(bridge)},
              "agreement": "all recomputed values agree with released values within floating-point tolerance",
              "confidence_intervals": "saved hierarchical-bootstrap endpoints reused, not resampled",
              "limitation": "Raw prediction, relation/anchor and structure-group rows are not included; the original hierarchical bootstrap cannot be rerun from fit-level metrics."}
    with (output / "reproduction_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reproduced")
    args = parser.parse_args()
    # Do not overwrite the released input tree with generated summaries.
    source, destination = args.results_root.resolve(), args.output.resolve()
    protected_subdirectory = (source in destination.parents
                              and destination.relative_to(source).parts[0] in {"statistics", "metrics"})
    if destination == source or protected_subdirectory:
        parser.error("Choose a separate output folder, such as results/reproduced")
    report = reproduce(args.results_root, args.output)
    print(json.dumps(report, indent=2))
    print(f"Wrote numerical tables to {args.output}")


if __name__ == "__main__":
    main()
