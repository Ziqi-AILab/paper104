"""Observed project costs, with allocated resources distinct from recipe timers."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from polyfid.io import atomic_write_csv, atomic_write_text


TASKS = ("egc", "egb", "eea", "ei", "xc", "eps", "nc")
ARMS = ("C", "B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs")
ACCOUNT_FIELDS = ("JobIDRaw", "JobName", "Partition", "QOS", "State", "ExitCode",
                  "Submit", "Start", "End", "ElapsedRaw", "AllocTRES")


def gpu_count(tres: str) -> int:
    values = dict(item.split("=", 1) for item in str(tres).split(",") if "=" in item)
    if "gres/gpu" in values:
        return int(values["gres/gpu"])
    return sum(int(value) for key, value in values.items() if key.startswith("gres/gpu:"))


def resource_family(partition: str, tres: str) -> str:
    text = f"{partition} {tres}".casefold()
    return "A800" if "a800" in text else "RTX4090" if "4090" in text else "unresolved"


def stage_name(text: str) -> str:
    text = text.casefold()
    for words, name in [
        (("cost_recovery", "cost-recovery", "cost_profile", "cost-profile", "workflow_cost", "p104-cost-"), "cost_extension"),
        (("smoke",), "smoke"), (("development", "p104-dev"), "development"),
        (("chemprop",), "chemprop"), (("transpolymer",), "transpolymer"),
        (("periogt",), "periogt"), (("grin",), "grin"),
        (("adapt",), "downstream_adaptation"),
        (("token_efficiency", "token-eff", "tok-eff"), "token_efficiency_evaluation"),
        (("mechanism_control", "mechanism-control"), "mechanism_evaluation"),
        (("controls", "control-"), "mechanism_pretraining"),
        (("postpretraining", "prop-view", "property_views", "a800", "embedding"), "formal_evaluation"),
        (("formal",), "core_pretraining"),
    ]:
        if any(word in text for word in words):
            return name
    return "other_project_job"


def collect_job_sources(project: Path) -> pd.DataFrame:
    jobs: dict[str, dict] = {}
    for base in (project / "slurm", project / "runs", project / "logs"):
        for ledger in sorted(base.rglob("*.accepted.tsv")):
            with ledger.open() as handle:
                for row in csv.reader(handle, delimiter="\t"):
                    if len(row) < 2:
                        continue
                    match = re.search(r"Submitted batch job (\d+)", row[1])
                    if not match:
                        continue
                    job_id = match.group(1)
                    previous = jobs.get(job_id, {})
                    jobs[job_id] = {"job_id": job_id, "script": row[0],
                        "ledger": str(ledger), "source_kind": "accepted_ledger",
                        "log": previous.get("log", "")}
    for base in (project / "logs/slurm",):
        for path in sorted(base.rglob("*")):
            match = re.search(r"-(\d+)\.(?:out|err)$", path.name)
            if match:
                job_id = match.group(1)
                record = jobs.setdefault(job_id, {"job_id": job_id, "script": "", "ledger": "",
                                                  "source_kind": "project_slurm_log"})
                record["log"] = str(path)
    return pd.DataFrame(jobs.values(), columns=["job_id", "script", "ledger", "source_kind", "log"]).sort_values(
        "job_id", key=lambda values: values.astype(int))


def query_allocations(job_ids: list[str]) -> pd.DataFrame:
    rows = []
    for start in range(0, len(job_ids), 500):
        command = ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids[start:start + 500]),
                   "--format=" + ",".join(ACCOUNT_FIELDS)]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        for line in result.stdout.splitlines():
            values = line.split("|")
            if len(values) >= len(ACCOUNT_FIELDS) and values[0] in job_ids:
                rows.append(dict(zip(ACCOUNT_FIELDS, values)))
    return pd.DataFrame(rows, columns=ACCOUNT_FIELDS).drop_duplicates("JobIDRaw", keep="last")


def allocation_table(sources: pd.DataFrame, accounting: pd.DataFrame) -> pd.DataFrame:
    table = sources.merge(accounting, left_on="job_id", right_on="JobIDRaw", how="left", validate="one_to_one")
    table["accounting_available"] = table["JobIDRaw"].notna()
    table["elapsed_seconds"] = pd.to_numeric(table["ElapsedRaw"], errors="coerce")
    table["gpu_count"] = table["AllocTRES"].fillna("").map(gpu_count)
    table["gpu_type"] = [resource_family(str(p), str(t)) for p, t in zip(table["Partition"], table["AllocTRES"])]
    table["allocated_gpu_hours"] = table["elapsed_seconds"] * table["gpu_count"] / 3600
    table["stage"] = [stage_name(f"{a} {b} {c}") for a, b, c in zip(table["script"], table["log"], table["JobName"])]
    submitted = pd.to_datetime(table["Submit"], errors="coerce")
    started = pd.to_datetime(table["Start"], errors="coerce")
    table["queue_wait_seconds"] = (started - submitted).dt.total_seconds().clip(lower=0)
    state = table["State"].fillna("").str.split().str[0].str.rstrip("+")
    table["allocation_outcome"] = np.select(
        [~table["accounting_available"], state.eq("COMPLETED"), state.eq("PENDING"),
         state.isin(["RUNNING", "COMPLETING"]),
         state.isin(["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"])],
        ["accounting_unavailable", "completed", "pending", "running", "failed_or_interrupted"],
        default="other_scheduler_state")
    table["expenditure_interpretation"] = np.where(
        table["State"].eq("COMPLETED"), "completed allocation",
        np.where(table["allocated_gpu_hours"].eq(0), "no GPU allocation time recorded",
                 "occupied resources; saved progress/replayed work not separable from state"))
    return table


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def record_row(path: Path, *, role: str, method: str = "", task: str = "", **extra) -> dict:
    if not path.is_file():
        return {"record_path": str(path), "record_available": False, "role": role,
                "method": method, "task": task, **extra}
    record = read_json(path)
    resources = record.get("resources", record)
    config = record.get("config", {})
    row = {"record_path": str(path), "record_available": True, "role": role,
           "method": method or record.get("method", ""), "task": task,
           "seed": record.get("seed"), "run_id": record.get("run_id", ""),
           "status": record.get("status", "complete"), **extra}
    for field in ("wall_seconds", "training_seconds", "inference_seconds", "total_parameters",
                  "trainable_parameters", "peak_device_memory_bytes", "peak_cuda_memory_mb",
                  "non_padding_tokens", "optimizer_steps", "gpu_hours", "tokens_per_second",
                  "validation_selected_tensor_bytes", "validation_selected_model_bytes", "test_rows",
                  "precision", "max_length", "preprocess_workers", "device"):
        row[field] = resources.get(field, config.get(field))
    if row.get("peak_device_memory_bytes") is None and row.get("peak_cuda_memory_mb") is not None:
        row["peak_device_memory_bytes"] = row["peak_cuda_memory_mb"] * 1024 ** 2
    return row


def recipe_costs(project: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(project.glob("runs/formal_*.json")):
        rows.append(record_row(path, role="core_pretraining_last_invocation",
                               coverage="last invocation; allocation ledger supplies complete occupied time"))
    for arm in ARMS:
        name = "C_fixed" if arm == "C" else f"{arm}_seed-42"
        cache = project / "runs/embeddings/formal" / name / "structure_only.run_record.json"
        rows.append(record_row(cache, role="shared_encoder_cache", method=arm,
                               coverage="all cached tasks, not a seven-task-only measurement"))
        for task in TASKS:
            path = project / "results/predictions/frozen_probe" / name / task / "group_ood/structure_only/fold-0/fraction-1/subsample-0/predictions.run_record.json"
            rows.append(record_row(path, role="frozen_head_fit", method=arm, task=task.upper(),
                                   coverage="three-head bundle (11,22,33); not single-head fit time"))
    for task in TASKS:
        for method in ("chemprop_dpnn", "ecfp_ridge", "rdkit_ridge", "ecfp_xgboost", "rdkit_xgboost", "nearest_neighbor"):
            seed = 0 if method in {"ecfp_ridge", "rdkit_ridge", "nearest_neighbor"} else 11
            path = project / "runs/records/controlled_baselines" / method / task / f"group_ood/structure_only/fold-0/fraction-1/subsample-0/seed-{seed}/run.json"
            rows.append(record_row(path, role="task_fit_workflow", method=method, task=task.upper(),
                                   coverage="observed workflow timer; feature preparation boundary varies by backend"))
        for method in ("transpolymer", "periogt", "grin"):
            directory = project / "runs/external_reference" / method / task / "group_ood/fold-0/seed-11/run"
            path = directory / ("run.json" if method == "transpolymer" else "backend_run.json")
            rows.append(record_row(path, role="external_task_fit", method=method, task=task.upper(), coverage="training and inference separately recorded"))
        rows.append(record_row(project / "runs/external_reference/periogt_cache" / task / "run.json",
                               role="shared_task_preprocessing", method="periogt", task=task.upper(),
                               coverage="CPU preprocessing once per complete task, reused across folds and seeds"))
    return pd.DataFrame(rows)


def cpu_record_descriptors(project: Path) -> pd.DataFrame:
    """Enumerate small timer records from manifests, without reading predictions."""
    rows = []
    manifests = [
        ("frozen_probe", "frozen_probes/frozen_probe_manifest.tsv"),
        ("classical_fit", "controlled_baselines/classical_baseline_manifest.tsv"),
        ("property_head_development", "property_head_development/property_head_development_manifest.tsv"),
        ("property_view_prediction", "property_views/property_view_prediction_manifest.tsv"),
    ]
    for role, name in manifests:
        manifest = project / "runs/manifests" / name
        if not manifest.exists():
            continue
        with manifest.open() as handle:
            for item in csv.DictReader(handle, delimiter="\t"):
                shared = {"role": role, "method": item.get("arm", item.get("method", "")).replace("-", "_"),
                          "task": item["task"], "manifest": str(manifest),
                          "representation_seed": item.get("representation_seed", ""),
                          "head_seeds": item.get("head_seeds", item.get("seed", ""))}
                if role == "property_view_prediction":
                    paths = [Path(item["output_root"]) / f"head-seed-{seed}/run_record.json"
                             for seed in item["head_seeds"].split(",")]
                elif role == "classical_fit":
                    paths = [Path(item["run_record"])]
                else:
                    paths = [Path(item["output"]).with_suffix(".run_record.json")]
                rows.extend({"record_path": str(path), **shared} for path in paths)
    for task in TASKS:
        for phase, base in [("original", "runs/external_reference/periogt_cache"),
                            ("cost_extension", "runs/cost_recovery/periogt_cache")]:
            rows.append({"record_path": str(project / base / task / "run.json"),
                         "role": f"periogt_preprocessing_{phase}", "method": "periogt", "task": task.upper(),
                         "manifest": "task cache run records", "representation_seed": "", "head_seeds": ""})
    return pd.DataFrame(rows).drop_duplicates("record_path")


def research_cpu_records(project: Path, output: Path, *, refresh: bool = False, workers: int = 8) -> pd.DataFrame:
    """Summed per-invocation wall time is not CPU core-time or calendar duration.

    Timer files were overwritten by some resumes/removed in cleanup. This is an
    observed-record account, not a complete history of all process attempts.
    """
    descriptors = cpu_record_descriptors(project)
    cache = output / "research_cpu_records.csv"
    cached = pd.read_csv(cache, keep_default_na=False, low_memory=False) if cache.exists() and not refresh else pd.DataFrame()
    # Refresh missing records, which can arrive while the recovery runs finish.
    existing = (cached.loc[cached["record_available"].eq(True)].set_index("record_path").to_dict("index")
                if not cached.empty else {})

    def read_one(item):
        path = item["record_path"]
        if path in existing:
            return {"record_path": path, **existing[path], **item}
        return record_row(Path(path), **{key: value for key, value in item.items() if key != "record_path"})

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(read_one, descriptors.to_dict("records")), 1):
            rows.append(row)
            if index % 10000 == 0:
                print(f"CPU timer records: {index}/{len(descriptors)}", flush=True)
    table = pd.DataFrame(rows)
    table["wall_seconds"] = pd.to_numeric(table.get("wall_seconds"), errors="coerce")
    table["timer_available"] = table["wall_seconds"].notna()
    table["cpu_process_wall_hours"] = table["wall_seconds"] / 3600
    table["cost_scope"] = "observed saved invocation; CPU process wall, not CPU core-hours"
    atomic_write_csv(table, cache)
    totals = table.groupby(["role", "method"], dropna=False).agg(
        expected_records=("record_path", "size"), available_records=("record_available", "sum"),
        measured_timers=("timer_available", "sum"),
        cpu_process_wall_hours=("cpu_process_wall_hours", lambda values: values.sum(min_count=1)),
    ).reset_index()
    totals["missing_records"] = totals["expected_records"] - totals["available_records"]
    totals["missing_timers"] = totals["expected_records"] - totals["measured_timers"]
    atomic_write_csv(totals, output / "research_cpu_cost.csv")
    atomic_write_csv(pd.DataFrame([
        {"scope": "data/relation construction", "cost_status": "unknown", "reason": "no consolidated complete invocation timers"},
        {"scope": "classical feature cache preparation", "cost_status": "unknown", "reason": "fit timers do not establish a separately timed shared feature preparation cost"},
        {"scope": "bootstrap/plotting/manuscript and historical retries", "cost_status": "unknown", "reason": "historical invocation timers are incomplete; completed analyses do not recover missing attempts"},
        {"scope": "CPU core-hours", "cost_status": "unknown", "reason": "wall time and worker/thread counts do not measure consumed CPU time"},
    ]), output / "cpu_cost_unmeasured_scopes.csv")
    return table


def pretraining_allocations(project: Path, allocations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        if arm == "C":
            rows.append({"method": arm, "representation_seed": 0, "allocated_4090_hours": 0.0,
                         "completed_state_hours": 0.0, "other_state_hours": 0.0,
                         "failed_or_interrupted_hours": 0.0, "running_hours": 0.0, "pending_hours": 0.0,
                         "n_allocations": 0, "scope": "no PI1M training; upstream cost not zero"})
            continue
        for seed in (42, 123, 777, 2025, 2026):
            pattern = re.compile(rf"formal_{re.escape(arm)}_seed-?{seed}(?:\D|$)")
            selected = allocations.loc[allocations["script"].map(lambda text: bool(pattern.search(str(text))))]
            hours = selected["allocated_gpu_hours"]
            rows.append({"method": arm, "representation_seed": seed,
                         "allocated_4090_hours": hours.sum(min_count=1),
                         "completed_state_hours": selected.loc[selected["allocation_outcome"].eq("completed"), "allocated_gpu_hours"].sum(min_count=1),
                         "other_state_hours": selected.loc[selected["allocation_outcome"].ne("completed"), "allocated_gpu_hours"].sum(),
                         **{f"{state}_hours": selected.loc[selected["allocation_outcome"].eq(state), "allocated_gpu_hours"].sum()
                            for state in ("failed_or_interrupted", "running", "pending")},
                         "n_allocations": len(selected), "job_ids": ";".join(selected["job_id"]),
                         "history_class": "multiple allocations, useful/replayed work not separable" if len(selected) > 1 else "single observed allocation",
                         "scope": "all matched allocations, including saved progress in failed runs"})
    return pd.DataFrame(rows)


def common_task_performance(project: Path) -> pd.DataFrame:
    """Existing per-fold metrics only, not a new aggregate statistical test."""
    sources = [project / "results/statistics/formal_frozen_probes/task_metrics.csv"]
    sources += [project / "results/statistics/completed_experiments/metrics" / method / "task_metrics.csv"
                for method in ("chemprop", "classical", "transpolymer", "periogt", "grin")]
    tables = []
    for source in sources:
        if not source.exists():
            continue
        table = pd.read_csv(source)
        table = table.loc[
            table["task"].str.upper().isin([task.upper() for task in TASKS]) &
            table["protocol"].eq("group_ood") & table["split"].eq("test") &
            table["fold"].eq(0) & table["label_fraction"].eq(1) & table["subsample_id"].eq(0) &
            table["input_view"].isin(["valid", "structure_only"]) & table["metric"].eq("rmse")
        ].copy()
        table["method"] = table["method"].str.removeprefix("baseline_").replace({"chemprop_dpnn": "chemprop"})
        expected_rep = table["method"].map(lambda method: 42 if method in ARMS and method != "C" else 0)
        expected_head = table["method"].map(lambda method: 0 if method in ("ecfp_ridge", "rdkit_ridge", "nearest_neighbor") else 11)
        table = table.loc[table["representation_seed"].eq(expected_rep) & table["head_seed"].eq(expected_head)]
        table["metric_source"] = str(source)
        table["task"] = table["task"].str.upper()
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def recipe_components(project: Path, recipes: pd.DataFrame, pretraining: pd.DataFrame,
                      allocations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One adopted representation, seven tasks; shared components stay shared."""
    rows = []
    cache_rows = recipes.loc[recipes["role"].eq("shared_encoder_cache")].set_index("method")
    core_pretrain = pretraining.loc[pretraining["representation_seed"].isin([0, 42])].set_index("method")
    for record in recipes.to_dict("records"):
        role, method = record["role"], record["method"]
        if role not in {"frozen_head_fit", "task_fit_workflow", "external_task_fit"}:
            continue
        method = "chemprop" if method == "chemprop_dpnn" else method
        row = {"method": method, "task": record["task"], "protocol": "group_ood", "fold": 0,
               "head_seed": 0 if method in {"ecfp_ridge", "rdkit_ridge", "nearest_neighbor"} else 11,
               "representation_seed": 42 if method in ARMS and method != "C" else 0,
               "source_record": record["record_path"], "timer_scope": record["coverage"],
               "task_training_seconds": record.get("training_seconds"),
               "task_inference_seconds": record.get("inference_seconds"),
               "task_workflow_seconds": record.get("wall_seconds"),
               "task_fit_allocated_a800_hours": np.nan,
               "task_cpu_wall_seconds": np.nan, "task_three_head_bundle_cpu_seconds": np.nan,
               "single_head_training_seconds": np.nan, "shared_pi1m_allocated_4090_hours": np.nan,
               "shared_embedding_superset_observed_a800_seconds": np.nan,
               "seven_task_only_embedding_a800_seconds": np.nan,
               "upstream_pretraining_gpu_hours": 0.0 if method in {"B0", "B_eqs", "chemprop", "grin", "ecfp_ridge", "rdkit_ridge", "ecfp_xgboost", "rdkit_xgboost", "nearest_neighbor"} else np.nan}
        if method in ARMS:
            row["shared_pi1m_allocated_4090_hours"] = core_pretrain.loc[method, "allocated_4090_hours"]
            row["shared_embedding_superset_observed_a800_seconds"] = cache_rows.loc[method, "wall_seconds"]
            row["task_three_head_bundle_cpu_seconds"] = record.get("wall_seconds")
            row["resource"] = "CPU fit; shared encoder uses A800 and (except C) RTX4090"
        elif method in {"chemprop", "transpolymer", "periogt", "grin"}:
            row["resource"] = "A800 task training"
        else:
            row["resource"] = "CPU fit; shared feature preprocessing not timed separately"
            row["task_cpu_wall_seconds"] = record.get("wall_seconds")
        rows.append(row)
    table = pd.DataFrame(rows)
    manifest = project / "runs/manifests/cost_recovery/baseline_recovery_manifest.tsv"
    if manifest.exists():
        for item in pd.read_csv(manifest, sep="\t").to_dict("records"):
            # Original script identity, not the new restoration job, supplies adoption cost.
            original = allocations.loc[allocations["script"].map(lambda value: Path(str(value)).stem == item["original_job_id"])]
            mask = table["method"].eq(item["method"]) & table["task"].eq(item["task"])
            table.loc[mask, "task_fit_allocated_a800_hours"] = original["allocated_gpu_hours"].sum(min_count=1)
            table.loc[mask, "task_fit_allocation_count"] = len(original)
    # Each shared model/cache appears once in this table, not once per property.
    reuse = []
    for method, part in table.groupby("method", sort=False):
        one = part.iloc[0]
        prep = recipes.loc[recipes["role"].eq("shared_task_preprocessing") & recipes["method"].eq(method)]
        reuse.append({"method": method, "n_tasks": len(part), "protocol": "group_ood", "fold": 0,
                      "representation_seed": one["representation_seed"],
                      "shared_pi1m_allocated_4090_hours": one["shared_pi1m_allocated_4090_hours"],
                      "shared_embedding_superset_observed_a800_seconds": one["shared_embedding_superset_observed_a800_seconds"],
                      "seven_task_only_embedding_a800_seconds": np.nan,
                      "seven_task_fit_allocated_a800_hours": part["task_fit_allocated_a800_hours"].sum(min_count=len(TASKS)),
                      "seven_task_three_head_bundle_cpu_seconds": part["task_three_head_bundle_cpu_seconds"].sum(min_count=len(TASKS)),
                      "seven_task_single_head_training_seconds": np.nan,
                      "seven_task_cpu_fit_seconds": part["task_cpu_wall_seconds"].sum(min_count=len(TASKS)),
                      "seven_task_shared_preprocessing_cpu_seconds": prep["wall_seconds"].sum(min_count=len(TASKS)),
                      "amortization_scope": "one representation (C fixed / seed 42); one task model per baseline; frozen CPU timers are 3-head bundles",
                      "cost_total_status": "not summed across resource types; exact seven-task encoder and single-head fit costs unmeasured"})
    reuse_table = pd.DataFrame(reuse)
    for source, destination in [
        ("shared_pi1m_allocated_4090_hours", "amortized_pi1m_4090_hours_per_task"),
        ("seven_task_fit_allocated_a800_hours", "mean_task_fit_a800_hours_per_task"),
        ("seven_task_three_head_bundle_cpu_seconds", "mean_three_head_bundle_cpu_seconds_per_task"),
        ("seven_task_cpu_fit_seconds", "mean_cpu_fit_seconds_per_task"),
        ("seven_task_shared_preprocessing_cpu_seconds", "mean_shared_preprocessing_cpu_seconds_per_task"),
    ]:
        reuse_table[destination] = reuse_table[source] / len(TASKS)
    reuse_table["per_task_interpretation"] = (
        "arithmetic allocation/mean across the seven tasks, not measured per-task PI1M training; frozen head unit remains three-head bundle")
    return table, reuse_table


def nondominated_mask(table: pd.DataFrame, cost: str, error: str) -> pd.Series:
    """Lower cost/lower error frontier, only within an explicitly shared scope."""
    result = pd.Series(pd.NA, index=table.index, dtype="boolean")
    complete = table[[cost, error]].dropna()
    for index, row in complete.iterrows():
        dominated = ((complete[cost] <= row[cost]) & (complete[error] <= row[error]) &
                     ((complete[cost] < row[cost]) | (complete[error] < row[error]))).any()
        result.loc[index] = not dominated
    return result


def profiling_tables(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join trained-state timing with that state's own test predictions.

    Restored baseline scores are additional single-seed observations, not
    replacements for the completed formal study. Read only these small test files.
    """
    timings, summaries = [], []
    for path in sorted((project / "results/metrics/workflow_cost").glob("*/*/*/summary.json")):
        item = read_json(path)
        if item.get("status") != "complete":
            continue
        measurements = item.pop("measurements", [])
        item["summary_path"] = str(path)
        reference_path = Path(item["reference_predictions"])
        truth = pd.read_parquet(reference_path)
        if "split" in truth:
            truth = truth.loc[truth["split"].eq("test")]
        if "head_seed" in truth:
            truth = truth.loc[truth["head_seed"].eq(item.get("head_seed", 11))]
        truth = truth.set_index("sample_id")["y_true"]
        predicted = pd.read_parquet(path.parent / "restored_predictions.parquet")
        for measured in measurements:
            selected = predicted.loc[predicted["batch_size"].eq(measured["batch_size"]) &
                                     predicted["regime"].eq(measured["regime"])]
            matched_truth = truth.reindex(selected["sample_id"]).to_numpy(dtype=float)
            assert len(selected) == item["test_records"] and np.isfinite(matched_truth).all()
            rmse = float(np.sqrt(np.mean((selected["y_pred"].to_numpy(dtype=float) - matched_truth) ** 2)))
            timings.append({**item, **measured, "rmse": rmse,
                            "performance_source": "original frozen head" if item["method"] in ARMS else "restoration rerun (original recipe)",
                            "peak_allocated_gib": measured["peak_allocated_bytes"] / 1024 ** 3})
        summaries.append(item)
    timing = pd.DataFrame(timings)
    if not timing.empty:
        timing["latency_error_nondominated"] = pd.NA
        timing["memory_error_nondominated"] = pd.NA
        for _, group in timing.groupby(["task", "mode", "batch_size", "regime", "device_name"]):
            timing.loc[group.index, "latency_error_nondominated"] = nondominated_mask(group, "amortized_latency_median_ms_per_record", "rmse")
            if group.iloc[0]["mode"] == "gpu":
                timing.loc[group.index, "memory_error_nondominated"] = nondominated_mask(group, "peak_allocated_gib", "rmse")
    return timing, pd.DataFrame(summaries)


def draw_cost_figures(output: Path, pretraining: pd.DataFrame, reuse: pd.DataFrame,
                      timings: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, method in enumerate(ARMS):
        values = pretraining.loc[pretraining["method"].eq(method), "allocated_4090_hours"].dropna().to_numpy()
        ax.scatter(np.full(len(values), index) + np.linspace(-0.12, 0.12, len(values)), values, s=28)
        if len(values):
            ax.hlines(np.median(values), index - .25, index + .25, color="black", linewidth=1.2)
    ax.set_xticks(range(len(ARMS)), ARMS)
    ax.set(ylabel="Observed occupied RTX 4090 hours / representation", title="Local PI1M training: all original and resumed allocations")
    fig.text(.5, .015, "C has no local PI1M training; its public upstream pretraining cost remains unknown.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1)); fig.savefig(output / "pretraining_allocation_cost.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, column, label in [
        (axes[0], "seven_task_fit_allocated_a800_hours", "Seven single-seed task fits: A800 allocation hours"),
        (axes[1], "seven_task_three_head_bundle_cpu_seconds", "Seven tasks × three heads: CPU process seconds"),
        (axes[2], "seven_task_cpu_fit_seconds", "Seven classical fits: CPU process seconds")]:
        part = reuse.dropna(subset=[column])
        ax.barh(part["method"], part[column]); ax.set_xlabel(label)
    fig.suptitle("Observed task-stage components, not total workflow cost")
    fig.text(.5,.01,"Shared pretraining, encoder cache and preprocessing are separate; CPU and GPU axes are not equivalent.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96)); fig.savefig(output / "seven_task_cost_components.png", dpi=170); plt.close(fig)
    if timings.empty:
        return
    for batch in (1, 32):
        # Cached-head latency belongs in a separate plot, never on this frontier.
        subset = timings.loc[timings["mode"].eq("gpu") & timings["batch_size"].eq(batch) &
                             timings["regime"].eq("prepared_input_end_to_end")]
        if subset.empty:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        methods = list(ARMS) + ["chemprop", "transpolymer", "periogt", "grin"]
        colors = {method: plt.get_cmap("tab20")(index) for index, method in enumerate(methods)}
        for ax, task in zip(axes.flat, TASKS):
            group = subset.loc[subset["task"].str.casefold().eq(task)]
            for row in group.to_dict("records"):
                ax.scatter(row["amortized_latency_median_ms_per_record"], row["rmse"], color=colors[row["method"]], label=row["method"], s=30)
            ax.set(title=task.upper(), xlabel="ms / record (batch-amortized)", ylabel="Test RMSE (task units)")
            if len(group):
                ax.set_xscale("log")
        axes.flat[-1].axis("off")
        handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m], label=m) for m in methods]
        axes.flat[-1].legend(handles=handles, loc="center", ncol=2, fontsize=8)
        fig.suptitle(f"A800 prepared-input prediction timing, batch {batch}; original precision and inputs")
        fig.text(.5, .015, "Fold 0; core rep42/head11 (C fixed); restored baselines seed11. Offline preparation and loading timed separately.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0,.045,1,.96)); fig.savefig(output / f"latency_performance_batch-{batch}.png", dpi=170); plt.close(fig)


def cost_notes() -> str:
    return """# Observed costs for the Paper104 writing closure

These files do not estimate missing upstream costs, energy or total FLOPs.

- `job_allocations.csv`: one row per project scheduler allocation; batch/extern
  steps are excluded. GPU-hours use allocated GPU count times occupied seconds.
  Queue time is separate. A failed allocation can contain useful saved training;
  its full duration is not necessarily wasted computation. RUNNING/PENDING
  allocations are a dated snapshot, not final completed-job costs.
- `research_gpu_cost.csv`: allocated GPU-hours by stage, GPU type and outcome.
  It is a research expenditure account, not the cost of adopting one model.
  Completed, failed/interrupted, running and pending allocations are separate;
  the original scheduler State remains in `job_allocations.csv`. Terminated
  allocations can contain useful checkpointed progress, not only lost work.
- `pretraining_allocations.csv`: initial and resumed formal jobs grouped by
  method and representation seed. Final RunRecord timers cover only the last
  invocation and do not replace this allocation history.
  `other_state_hours` is retained for compatibility; the separate
  `failed_or_interrupted_hours`, `running_hours` and `pending_hours` columns
  distinguish terminated-run cost from current queue state.
- `recipe_cost_records.csv`: source timers for the seven common tasks. Frozen
  head timers cover three heads together. Shared encoding covers the existing
  broader task cache; it is not silently reassigned as exact seven-task cost.
- `upstream_assets.csv`: public checkpoint reuse is treated alike for all
  methods. Missing historical pretraining GPU-hours mean unknown, not zero.
- `research_cpu_records.csv` and `research_cpu_cost.csv`: manifest-indexed
  observed local process-wall timers, including frozen probes, classical fits,
  development, property-view prediction and PerioGT preprocessing. Missing
  records/timers remain explicit. These are neither CPU core-hours nor elapsed
  calendar days; concurrent processes overlap. Overwritten invocation records
  cannot recover the time spent in previous attempts.
  The 55,335 frozen-probe outputs each correspond to one three-head training
  invocation, not three separately counted timers. Duplicate record paths are
  counted once. Completion of all analyses does not fill missing historical
  timers; enclosing-process and child-step durations must not be added together.
- `cpu_cost_unmeasured_scopes.csv`: additional unmeasured work. No complete
  total CPU research cost can be claimed from only the available timers.
- `seven_task_recipe_components.csv`: the same fold-0, 100%-label seven tasks,
  core representation seed 42 (C fixed), head seed 11; deterministic classical
  baselines use seed 0. Original allocation IDs supply baseline task-fit cost.
  Restoration reruns are a separate expenditure, not added to adoption cost.
- `seven_task_reuse_components.csv`: each shared PI1M model and broader encoder
  cache charged once. Task-dependent costs sum the seven actual tasks. Shared
  rows in the per-task table MUST NOT be summed seven times.
  Explicit per-task averages divide each seven-task component by seven. Shared
  PI1M / seven is an accounting allocation, not measured task-specific training.
  CPU frozen-fit / seven still measures one three-head bundle per task, not a
  single head. No exact seven-task encoder total or per-task estimate is filled.
- `seven_task_original_performance.csv`: the selected seed/fold original test
  RMSE, joined with its observed cost components. This descriptive selection is
  not a new significance test and does not replace multi-seed main results.
- `profiling_cost_performance.csv` and `profiling_model_setup.csv`: collect
  completed profiling reports. Profiling uses each restored state's own
  test predictions. Baseline restoration scores do not replace original scores.
  A missing future method does not receive a zero-time entry or a Pareto point.

4090-hours, A800-hours and CPU time are different columns, not a converted total.
Inference profiling is reported separately using restored trained states and
unchanged method-specific precision, length and input processing. Cached-head
latency is not compared directly with uncached structure-to-property latency.

Full scientific-repeat costs, a single recipe, and task amortization are
separate questions. Failure/resume resource use belongs in actual expenditure;
unrecoverable pure useful-compute or replayed-token counts remain unknown.

## Four accounting boundaries

1. **Upstream asset creation.** ChemBERTa, TransPolymer and PerioGT each reuse a
   public pretrained checkpoint; their upstream cost is unknown here. GRIN and
   Chemprop are task-supervised scratch models. Do not describe every baseline
   as scratch or count only ChemBERTa's upstream pretraining as free.
2. **Adopting one fixed recipe.** Use one representation, one selected task
   model and its actual inputs. PI1M cost is a shared once-only component, not
   five representation seeds. Cost to choose the recipe is research expense.
   Existing frozen head fit timers cover a three-head bundle; there is no
   measured single-head training time, and division by three is not a result.
3. **Reusing an encoder across seven tasks.** Pay shared PI1M cost once, then
   actual per-task fits. The existing shared embedding timer covers more tasks
   and records than these seven, so it is a superset provisioning measurement,
   not an exact seven-task encoding cost or a measured break-even threshold.
   Exact amortized workflow savings remain unestablished. New deployment timing
   measures structure-to-prediction latency and does not fill this training gap.
4. **Entire research expenditure.** Include all representations, heads,
   development, controls, failed/resumed allocations, adaptation, baselines and
   the separate restoration/profiling extension. A previous failed allocation
   may have supplied the checkpoint used by its successful resume. Allocation
   expenditure and useful optimization time are not interchangeable.

## What efficiency claims the evidence can support

Molecular checkpoint reuse avoids doing its historical pretraining again in
this project; it does not by itself establish lower total compute than strong
baselines. The observed formal A_eqs allocations are around seven RTX4090 hours
per representation (some resumes higher), versus about 1.5 hours for A0. The
relation objectives add real training work. Downstream frozen heads can be
cheap after a representation is available, but the shared training and encoder
stages still belong in the account. A favorable downstream latency/accuracy
point would support an adoption trade-off, not a claim of cheap pretraining.

The seven-task plots deliberately separate A800 task-fit hours from CPU
three-head-bundle seconds. They are components, not competing end-to-end
bars. The pretraining plot shows all five seeds and every matched allocation.
Neither is a hardware-normalized FLOP comparison.

Latency/error and memory/error Pareto flags are computed within each
task, GPU device name, batch size and timing regime. They are descriptive,
single-seed frontiers without performance or latency uncertainty. The methods
retain their actual lengths and precision (including TransPolymer float64);
this is an as-configured deployment comparison, not matched-precision
architecture efficiency. `prepared_input_end_to_end` starts with fixed prepared
CPU inputs; expensive molecular/graph feature preparation, model loading and
output I/O are separately reported. `compute_only` is measured production
forward wall time, including any host work done inside an official forward.
Cached-head CPU inference excludes the encoder and appears in a separate mode.

TransPolymer uses a batched forward, float64 and fixed length 512, matching
the original workflow. Its observed batch-32/batch-1 throughput ratio is
approximately 1.27–1.29 across the seven tasks. This observation does not
identify the hardware or software bottleneck. Paper104 keeps its original
float32 full MLM forward at maximum length 256; no cheaper encoder-only path
is substituted for timing.

## Refresh

Run locally (no Slurm):

```bash
.conda/env/bin/python scripts/analyze_workflow_costs.py
```

The first run reads manifest-indexed small JSON files. Later runs reuse known
CPU timer rows and revisit missing rows; pass `--refresh-cpu-timers` after a
source RunRecord was replaced. Scheduler history refreshes by default; use
`--reuse-scheduler-snapshot` for tests/replots with no scheduler query. New
profiling reports are picked up automatically. Do not reuse a stale scheduler
snapshot to claim complete costs for newly submitted jobs.
"""


def build_cost_accounting(project: Path, *, refresh_scheduler: bool = True,
                          refresh_cpu_timers: bool = False, cpu_workers: int = 8) -> dict:
    output = project / "results/statistics/cost_analysis"
    output.mkdir(parents=True, exist_ok=True)
    sources = collect_job_sources(project)
    atomic_write_csv(sources, output / "job_sources.csv")
    account_path = output / "scheduler_allocations.csv"
    if refresh_scheduler or not account_path.exists():
        accounting = query_allocations(sources["job_id"].tolist())
        atomic_write_csv(accounting, account_path)
    else:
        accounting = pd.read_csv(account_path, dtype=str, keep_default_na=False)
    allocations = allocation_table(sources, accounting)
    atomic_write_csv(allocations, output / "job_allocations.csv")
    totals = allocations.groupby(["stage", "gpu_type", "allocation_outcome"], dropna=False).agg(
        n_allocations=("job_id", "size"), allocated_gpu_hours=("allocated_gpu_hours", lambda x: x.sum(min_count=1)),
        known_allocations=("accounting_available", "sum"), queue_wait_seconds=("queue_wait_seconds", "sum")
    ).reset_index()
    totals["gpu_hours"] = totals["allocated_gpu_hours"]
    atomic_write_csv(totals, output / "research_gpu_cost.csv")
    pretraining = pretraining_allocations(project, allocations)
    pretraining["gpu_hours"] = pretraining["allocated_4090_hours"]
    atomic_write_csv(pretraining, output / "pretraining_allocations.csv")
    recipes = recipe_costs(project)
    atomic_write_csv(recipes, output / "recipe_cost_records.csv")
    cpu_records = research_cpu_records(project, output, refresh=refresh_cpu_timers, workers=cpu_workers)
    components, reuse = recipe_components(project, recipes, pretraining, allocations)
    atomic_write_csv(components, output / "seven_task_recipe_components.csv")
    atomic_write_csv(reuse, output / "seven_task_reuse_components.csv")
    performance = common_task_performance(project)
    if not performance.empty:
        performance = performance.merge(components, on=["method", "task", "protocol", "fold", "head_seed", "representation_seed"],
                                        how="left", validate="one_to_one")
    atomic_write_csv(performance, output / "seven_task_original_performance.csv")
    timings, setup = profiling_tables(project)
    atomic_write_csv(timings, output / "profiling_cost_performance.csv")
    atomic_write_csv(setup, output / "profiling_model_setup.csv")
    draw_cost_figures(output, pretraining, reuse, timings)
    assets = pd.DataFrame([
        {"method": "C/A0/A_att/A_gcl/A_eqs", "initialization": "public ChemBERTa-100M-MLM", "upstream_corpus": "100M ZINC20 molecules", "upstream_gpu_hours": None,
         "source": "https://huggingface.co/DeepChem/ChemBERTa-100M-MLM"},
        {"method": "B0/B_eqs", "initialization": "random, same architecture", "upstream_corpus": "none", "upstream_gpu_hours": 0, "source": "configs/pretraining/arms.yaml"},
        {"method": "TransPolymer", "initialization": "public pretrained encoder", "upstream_corpus": "external polymer corpus", "upstream_gpu_hours": None, "source": "https://www.nature.com/articles/s41524-023-01016-5"},
        {"method": "PerioGT", "initialization": "public pretrained light checkpoint", "upstream_corpus": "external polymer pretraining", "upstream_gpu_hours": None, "source": "https://www.nature.com/articles/s43588-025-00903-9"},
        {"method": "GRIN/Chemprop", "initialization": "random supervised task model", "upstream_corpus": "none", "upstream_gpu_hours": 0, "source": "src/polyfid/baselines"},
        {"method": "classical", "initialization": "no neural pretraining", "upstream_corpus": "none", "upstream_gpu_hours": 0, "source": "paper/CONTROLLED_BASELINES.md"},
    ])
    atomic_write_csv(assets, output / "upstream_assets.csv")
    summary = {"updated_at": datetime.now().astimezone().isoformat(), "status": "historical_records_collected",
               "scheduler_snapshot_at": datetime.fromtimestamp(account_path.stat().st_mtime).astimezone().isoformat(),
               "project_allocations": len(sources), "accounting_available": int(allocations["accounting_available"].sum()),
               "missing_accounting": int((~allocations["accounting_available"]).sum()),
               "recipe_records": len(recipes), "missing_recipe_records": int((~recipes["record_available"]).sum()),
               "gpu_extension": "28 restoration and 7 profiling jobs, tracked separately",
               "upstream_costs": "unknown where unpublished; no imputation",
               "single_head_training_cost": "existing frozen training timers are three-head bundles",
               "amortization_status": "shared/per-task component tables; exact seven-task encoding and one-head fitting remain unmeasured",
               "cpu_expected_timer_records": len(cpu_records),
               "cpu_available_timer_records": int(cpu_records["timer_available"].sum()),
               "cpu_process_wall_hours_observed": cpu_records["cpu_process_wall_hours"].sum(min_count=1),
               "cpu_account_status": "partial observed saved-invocation timers, not complete research CPU core-hours",
               "seven_task_original_performance_rows": len(performance),
               "completed_profile_method_modes": len(setup), "profile_measurement_rows": len(timings)}
    atomic_write_text(json.dumps(summary, indent=2) + "\n", output / "summary.json")
    atomic_write_text(cost_notes(), output / "README.md")
    return summary
