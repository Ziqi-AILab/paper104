"""Prediction-level follow-up intervals without changing the primary tests."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import pandas as pd

from polyfid.analysis.posttraining_metrics import normalize_predictions
from polyfid.analysis.statistics import (
    PRIMARY_COMPARISONS,
    PRIMARY_PREDICTION_COLUMNS,
    pair_method_predictions,
    paired_task_equal_hierarchical_bootstrap,
    select_primary_manifest_scope,
    select_primary_scope,
)
from polyfid.io import atomic_write_csv, atomic_write_text


@dataclass(frozen=True)
class Comparison:
    name: str
    candidate: str
    baseline: str
    analysis_family: str
    tasks: tuple[str, ...] = ()


def declared_comparisons() -> list[Comparison]:
    comparisons = [
        Comparison(name, candidate, baseline, "frozen_task_intervals")
        for name, (candidate, baseline) in PRIMARY_COMPARISONS.items()
    ]
    lora_pairs = {
        **PRIMARY_COMPARISONS,
        "relation_initialization": ("A_eqs", "B_eqs"),
        "generic_contrastive": ("A_gcl", "A_att"),
    }
    comparisons.extend(
        Comparison(name, f"{candidate}_lora", f"{baseline}_lora", "lora", ("EA", "IP", "EGC", "XC"))
        for name, (candidate, baseline) in lora_pairs.items()
    )
    comparisons.append(
        Comparison("full_relation_sensitivity", "A_eqs_full", "A0_full", "full", ("EA", "EGC"))
    )
    comparisons.extend(
        Comparison(f"full_vs_lora_{arm}", f"{arm}_full", f"{arm}_lora", "adaptation_sensitivity", ("EA", "EGC"))
        for arm in ("A0", "A_eqs")
    )
    return comparisons


def normalized_manifest(path: Path, *, adaptation: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame["task"] = frame["task"].astype(str).str.upper()
    if adaptation:
        frame["arm"] = frame["arm"].astype(str) + "_" + frame["method"].astype(str)
        frame["label_fraction"] = 1.0
        frame["subsample_id"] = 0
        frame["input_view"] = "structure_only"
    return frame


def build_work_items(project: Path, output: Path, *, n_resamples: int = 10_000, seed: int = 2026) -> list[dict]:
    manifests = {
        "frozen": project / "runs/manifests/frozen_probes/frozen_probe_manifest.tsv",
        "adaptation": project / "slurm/generated/postpretraining/adaptation/adaptation_manifest.tsv",
    }
    frames = {key: normalized_manifest(path, adaptation=key == "adaptation") for key, path in manifests.items()}
    items: list[dict] = []
    for comparison in declared_comparisons():
        source = "frozen" if comparison.analysis_family == "frozen_task_intervals" else "adaptation"
        selected = select_primary_manifest_scope(frames[source], comparison.candidate, comparison.baseline)
        if comparison.tasks:
            selected = selected.loc[selected["task"].isin(comparison.tasks)]
        tasks = sorted(selected["task"].unique())
        if not tasks or (comparison.tasks and set(tasks) != set(comparison.tasks)):
            raise ValueError(f"missing declared task scope for {comparison.name}: {tasks}")
        scopes = tasks if source == "frozen" else [*tasks, "aggregate"]
        for scope in scopes:
            rows = selected if scope == "aggregate" else selected.loc[selected["task"].eq(scope)]
            present = set(rows["arm"])
            if present != {comparison.candidate, comparison.baseline}:
                raise ValueError(f"missing comparison side for {comparison.name}/{scope}")
            name = f"{comparison.analysis_family}__{comparison.name}__{scope}"
            items.append({
                **asdict(comparison),
                "work_id": name,
                "scope": scope,
                "source_manifest": str(manifests[source]),
                "prediction_files": rows["output"].astype(str).tolist(),
                "n_resamples": n_resamples,
                "bootstrap_seed": seed + len(items),
                "output": str(output / "bootstrap" / f"{name}.json"),
            })
    # Surface the newly completed adaptation evidence first. This changes only
    # execution order: every declared item runs, with its already assigned seed.
    items.sort(key=lambda item: (
        item["analysis_family"] == "frozen_task_intervals",
        item["scope"] != "aggregate",
    ))
    return items


def run_bootstrap_item(item: dict) -> dict:
    """Load only one declared task/comparison at a time; resume by result record."""
    destination = Path(item["output"])
    if destination.is_file():
        existing = json.loads(destination.read_text())
        if existing.get("resamples") == item["n_resamples"] and existing.get("bootstrap_seed") == item["bootstrap_seed"]:
            return existing
    started = time.monotonic()
    print(f"bootstrap_start={item['work_id']} files={len(item['prediction_files'])}", flush=True)
    blocks = [
        normalize_predictions(pd.read_parquet(path, columns=list(PRIMARY_PREDICTION_COLUMNS)))
        for path in item["prediction_files"]
    ]
    predictions = pd.concat(blocks, ignore_index=True)
    del blocks
    paired = pair_method_predictions(predictions, item["candidate"], item["baseline"], require_full_coverage=True)
    del predictions
    scoped = select_primary_scope(paired)
    # Formal arms have five independent initialization repeats and three nested
    # downstream seeds. Baseline sentinel seed 0 never enters these intervals.
    representation_seeds = sorted(scoped["representation_seed"].astype(int).unique().tolist())
    head_seeds = sorted(scoped["head_seed"].astype(int).unique().tolist())
    if representation_seeds != [42, 123, 777, 2025, 2026] or head_seeds != [11, 22, 33]:
        raise ValueError(f"incomplete formal seed scope for {item['work_id']}")
    result = paired_task_equal_hierarchical_bootstrap(scoped, n_resamples=item["n_resamples"], seed=item["bootstrap_seed"])
    protocols = scoped.groupby("task")["protocol"].first().astype(str).to_dict()
    output = {
        "work_id": item["work_id"],
        "comparison": item["name"],
        "candidate": item["candidate"],
        "baseline": item["baseline"],
        "analysis_family": item["analysis_family"],
        "task": item["scope"],
        "protocol": next(iter(protocols.values())) if len(protocols) == 1 else "task_specific_ood",
        "task_protocols": ";".join(f"{task}={value}" for task, value in sorted(protocols.items())),
        "input_view": "valid",
        "label_fraction": 1.0,
        "n_tasks": int(scoped["task"].nunique()),
        "n_representation_seeds": len(representation_seeds),
        "n_head_seeds": len(head_seeds),
        "n_paired_rows": len(scoped),
        "prediction_files": len(item["prediction_files"]),
        "source_manifest": item["source_manifest"],
        "bootstrap_seed": item["bootstrap_seed"],
        "inference_role": "task_specific_interval" if item["analysis_family"] == "frozen_task_intervals" else "secondary_adaptation_comparison",
        "multiplicity": "descriptive interval; no additional primary hypothesis or Holm family",
        **result.as_dict(),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_text(json.dumps(output, indent=2) + "\n", destination)
    print(f"bootstrap_complete={item['work_id']} seconds={output['wall_seconds']:.1f}", flush=True)
    return output


def collect_bootstrap_outputs(items: list[dict], output: Path) -> pd.DataFrame:
    rows = [json.loads(Path(item["output"]).read_text()) for item in items if Path(item["output"]).is_file()]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        atomic_write_csv(frame.loc[~frame["task"].eq("aggregate")], output / "task_bootstrap.csv")
        atomic_write_csv(frame.loc[frame["task"].eq("aggregate")], output / "secondary_aggregate.csv")
    atomic_write_text(json.dumps({"completed": len(rows), "total": len(items), "status": "complete" if len(rows) == len(items) else "running"}, indent=2) + "\n", output / "bootstrap_progress.json")
    return frame


def run_bootstrap_plan(items: list[dict], output: Path, *, workers: int = 2) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_text(json.dumps(items, indent=2) + "\n", output / "bootstrap_plan.json")
    collect_bootstrap_outputs(items, output)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_bootstrap_item, item): item["work_id"] for item in items}
        for future in as_completed(futures):
            future.result()
            collect_bootstrap_outputs(items, output)
    return collect_bootstrap_outputs(items, output)
