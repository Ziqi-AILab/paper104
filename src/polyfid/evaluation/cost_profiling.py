"""Measured inference costs for restored, validation-selected Paper104 models.

``prepared_input_end_to_end`` includes collation, device transfer, forward,
return transfer, and inverse target scaling. One-time tokenization/graph
preparation and model loading are measured separately. It is not raw-SMILES
service latency. The production forward and precision are retained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import json
import logging
import os
from pathlib import Path
import shlex
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from polyfid.io import atomic_write_csv, atomic_write_parquet, atomic_write_text


TASKS = ("EEA", "EGB", "EGC", "EI", "EPS", "NC", "XC")
CORE_METHODS = ("C", "B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs")
EXTERNAL_METHODS = ("chemprop", "transpolymer", "periogt", "grin")


@dataclass
class CostBackend:
    method: str
    sample_ids: list[str]
    collate: Callable[[np.ndarray], Any]
    to_device: Callable[[Any], Any]
    forward: Callable[[Any], Any]
    postprocess: Callable[[np.ndarray], np.ndarray]
    metadata: dict[str, Any]
    modules: list[Any] = field(default_factory=list)


def method_order(task: str, *, cached_head: bool = False, seed: int = 10407) -> list[str]:
    """Fixed shuffled order, shared by reruns of a task."""

    task = task.upper()
    methods = list(CORE_METHODS) + ([] if cached_head else list(EXTERNAL_METHODS))
    return np.random.default_rng(seed + TASKS.index(task)).permutation(methods).tolist()


def batch_indices(records: int, batch_size: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + batch_size, records)) for start in range(0, records, batch_size)]


def summarize_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    seconds = frame["seconds"].to_numpy(dtype=float)
    count = frame["batch_records"].to_numpy(dtype=int)
    total = float(seconds.sum())
    return {
        "measured_batches": len(rows),
        "measured_record_instances": int(count.sum()),
        "timed_seconds": total,
        "batch_latency_median_ms": float(np.median(seconds) * 1000),
        "batch_latency_p95_ms": float(np.quantile(seconds, 0.95) * 1000),
        "amortized_latency_median_ms_per_record": float(np.median(seconds / count) * 1000),
        "amortized_latency_p95_ms_per_record": float(np.quantile(seconds / count, 0.95) * 1000),
        "throughput_records_per_second": float(count.sum() / total),
    }


def refresh_amortized_percentiles(output_root: Path) -> dict[str, int]:
    """Enrich saved summaries using measured batches, without repeating timing."""

    updated_reports: dict[Path, dict[str, Any]] = {}
    measurement_count = 0
    field_name = "amortized_latency_p95_ms_per_record"
    for summary_path in sorted(output_root.glob("*/*/*/summary.json")):
        report = _read_json(summary_path)
        if report.get("status") != "complete":
            continue
        timing = pd.read_csv(summary_path.parent / "batch_timings.csv")
        for measurement in report["measurements"]:
            selected = timing.loc[timing["batch_size"].eq(measurement["batch_size"]) & timing["regime"].eq(measurement["regime"])]
            per_record = selected["seconds"].to_numpy(dtype=float) / selected["batch_records"].to_numpy(dtype=int)
            measurement[field_name] = float(np.quantile(per_record, 0.95) * 1000)
            measurement_count += 1
        atomic_write_csv(pd.DataFrame(report["measurements"]), summary_path.parent / "timing_summary.csv")
        atomic_write_text(json.dumps(report, indent=2) + "\n", summary_path)
        updated_reports[summary_path.parent] = report
    for task_path in sorted(output_root.glob("*/*/summary.json")):
        task_report = _read_json(task_path)
        for index, report in enumerate(task_report.get("methods", [])):
            method_path = task_path.parent / report["method"]
            if method_path in updated_reports:
                task_report["methods"][index] = updated_reports[method_path]
        atomic_write_text(json.dumps(task_report, indent=2) + "\n", task_path)
    return {"method_summaries": len(updated_reports), "measurements": measurement_count}


def prediction_agreement(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    expected = np.asarray(expected, dtype=float).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError("restored model prediction count differs from reference")
    maximum = float(np.max(np.abs(actual - expected)))
    matches = bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5))
    if not matches:
        raise ValueError(f"restored model predictions differ from its selected-state reference (max absolute difference {maximum:g})")
    return {"matches": matches, "max_absolute_difference": maximum, "rtol": 1e-4, "atol": 1e-5}


def measure_backend(
    backend: CostBackend,
    *,
    batch_size: int,
    regime: str,
    warmup_batches: int = 20,
    passes: int = 10,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
    to_numpy: Callable[[Any], np.ndarray] | None = None,
    reset_peak_memory: Callable[[], None] = lambda: None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Measure complete passes; warmup records do not enter reported timings."""

    if regime not in {"compute_only", "prepared_input_end_to_end"}:
        raise ValueError(f"unknown timing regime: {regime}")
    if to_numpy is None:
        to_numpy = lambda value: value.detach().cpu().numpy().reshape(-1)
    batches = batch_indices(len(backend.sample_ids), batch_size)
    if not batches:
        raise ValueError("workflow cost requires a nonempty complete test fold")
    for index in range(warmup_batches):
        prepared = backend.to_device(backend.collate(batches[index % len(batches)]))
        output = backend.forward(prepared)
        backend.postprocess(to_numpy(output))
        synchronize()
        del output, prepared
    reset_peak_memory()
    rows: list[dict[str, Any]] = []
    reference_predictions: list[np.ndarray] = []
    for pass_index in range(passes):
        for batch_index, indices in enumerate(batches):
            if regime == "compute_only":
                prepared = backend.to_device(backend.collate(indices))
                synchronize()
                started = clock()
                output = backend.forward(prepared)
                synchronize()
                seconds = clock() - started
                prediction = backend.postprocess(to_numpy(output))
            else:
                synchronize()
                started = clock()
                prepared = backend.to_device(backend.collate(indices))
                output = backend.forward(prepared)
                prediction = backend.postprocess(to_numpy(output))
                synchronize()
                seconds = clock() - started
            rows.append({
                "pass_index": pass_index,
                "batch_index": batch_index,
                "batch_records": len(indices),
                "seconds": float(seconds),
                "regime": regime,
                "batch_size": batch_size,
            })
            if pass_index == 0:
                reference_predictions.append(np.asarray(prediction).reshape(-1).copy())
            del prepared, output, prediction
    return rows, np.concatenate(reference_predictions)


def _probe_row(project: Path, task: str, method: str) -> pd.Series:
    manifest = pd.read_csv(project / "runs/manifests/frozen_probes/frozen_probe_manifest.tsv", sep="\t")
    selected = manifest.loc[
        manifest["task"].str.upper().eq(task.upper())
        & manifest["arm"].eq(method)
        & manifest["representation_seed"].eq(0 if method == "C" else 42)
        & manifest["protocol"].eq("group_ood")
        & manifest["fold"].eq(0)
        & manifest["input_view"].eq("structure_only")
        & manifest["label_fraction"].eq(1.0)
        & manifest["subsample_id"].eq(0)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one original frozen probe for {method}/{task}")
    return selected.iloc[0]


def common_test_frame(project: Path, task: str) -> pd.DataFrame:
    from polyfid.evaluation.adaptation import prepare_downstream_frame

    row = _probe_row(project, task, "C")
    frame = prepare_downstream_frame(
        row.dataset, row.split_manifest, task=task.upper(), protocol="group_ood",
        fold=0, label_column=row.label_column, text_column="structure_only_text",
    )
    return frame.loc[frame["split"].eq("test")].reset_index(drop=True)


def _model_bytes(module: Any) -> int:
    return sum(value.numel() * value.element_size() for value in module.state_dict().values())


def _core_backend(project: Path, model_root: Path, checkpoint_root: Path, task: str, method: str, frame: pd.DataFrame, *, device: str, cached_head: bool) -> CostBackend:
    import torch
    from polyfid.evaluation.embeddings import align_embeddings, load_embedding_table
    from polyfid.evaluation.encoder import load_encoder_checkpoint
    from polyfid.evaluation.probes import _model_from_probe_state, load_probe_state

    row = _probe_row(project, task, method)
    head_path = Path(str(row.probe_state_dir)) / "head_seed-11.npz"
    started = time.perf_counter()
    state = load_probe_state(head_path)
    head = _model_from_probe_state(state, device=device)
    ids = frame["sample_id"].astype(str).tolist()
    checkpoint = model_root / "chemberta_100m_mlm" if method == "C" else checkpoint_root / f"pretraining/{method}/seed-42/token-100.pt"
    if cached_head:
        modules = [head]
        loading_seconds = time.perf_counter() - started
        started = time.perf_counter()
        array = align_embeddings(load_embedding_table(row.embeddings), np.asarray(ids))
        features = torch.as_tensor(array, dtype=torch.float32)
        preprocessing_seconds = time.perf_counter() - started
        collate = lambda indices: features[indices]
        forward = lambda batch: head(batch).reshape(-1)
        processing = "read existing final-hidden embedding cache and align complete test fold; no encoder"
    else:
        model, tokenizer = load_encoder_checkpoint(checkpoint, base_checkpoint=model_root / "chemberta_100m_mlm", device=device)
        modules = [model, head]
        loading_seconds = time.perf_counter() - started
        started = time.perf_counter()
        encoded = tokenizer(frame["model_text"].astype(str).tolist(), padding=False, truncation=True, max_length=256, return_special_tokens_mask=True)
        items = [{key: values[index] for key, values in encoded.items()} for index in range(len(ids))]
        preprocessing_seconds = time.perf_counter() - started

        def collate(indices: np.ndarray) -> Any:
            return tokenizer.pad([items[int(index)] for index in indices], padding=True, return_tensors="pt")

        def forward(batch: dict[str, Any]) -> Any:
            special = batch["special_tokens_mask"]
            inputs = {key: value for key, value in batch.items() if key != "special_tokens_mask"}
            output = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden = output.hidden_states[-1]
            weights = (inputs["attention_mask"].bool() & ~special.bool()).to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            return head(pooled.float()).reshape(-1)

        processing = "tokenize complete test fold once, max_length=256; dynamic batch padding remains timed"
    transfer = (lambda value: {key: item.to(device) for key, item in value.items()}) if not cached_head else (lambda value: value.to(device))
    return CostBackend(method, ids, collate, transfer, forward,
        lambda values: values.reshape(-1) * state.target_scale + state.target_center,
        {"backend": "cached_frozen_head" if cached_head else "AutoModelForMaskedLM_hidden_states_plus_frozen_head", "dtype": "float32", "max_length": None if cached_head else 256,
         "checkpoint": str(checkpoint), "head_path": str(head_path), "reference_predictions": str(row.output), "head_seed": 11, "representation_seed": int(row.representation_seed),
         "model_loading_seconds": loading_seconds, "input_preprocessing_seconds": preprocessing_seconds, "input_preprocessing_records": len(ids), "preprocessing_scope": processing,
         "parameter_count": sum(sum(value.numel() for value in model.parameters()) for model in modules), "resident_state_bytes": sum(_model_bytes(model) for model in modules)}, modules)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _external_backend(model_root: Path, method: str, row: pd.Series, frame: pd.DataFrame, *, device: str) -> CostBackend:
    import torch
    from polyfid.baselines.external import external_model_text

    ids = frame["sample_id"].astype(str).tolist()
    selected = Path(str(row.selected_model_path))
    config = _read_json(row.backend_config_path)
    config = config.get("config", config)
    scaler = _read_json(row.scaler_path)
    started = time.perf_counter()
    if method == "transpolymer":
        from polyfid.baselines.transpolymer import _regressor, load_transpolymer_encoder, load_transpolymer_tokenizer
        tokenizer = load_transpolymer_tokenizer(model_root, max_length=int(config["max_length"]))
        model = _regressor(load_transpolymer_encoder(model_root), float(config["dropout"]))
        if config["use_float64"]:
            model = model.double()
        payload = torch.load(selected, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        model.to(device).eval()
        loading = time.perf_counter() - started
        started = time.perf_counter()
        texts = external_model_text(frame, method, str(row.task)).tolist()
        encoded = tokenizer(texts, add_special_tokens=True, max_length=int(config["max_length"]), return_token_type_ids=False, padding="max_length", truncation=True, return_attention_mask=True, return_tensors="pt")
        preprocessing = time.perf_counter() - started
        collate = lambda indices: {key: value[indices] for key, value in encoded.items()}
        transfer = lambda batch: {key: value.to(device) for key, value in batch.items()}
        forward = lambda batch: model(batch["input_ids"], batch["attention_mask"]).reshape(-1)
        scale = float(np.asarray(scaler["std"]).reshape(-1)[0])
        center = float(np.asarray(scaler["mean"]).reshape(-1)[0])
        postprocess = lambda values: values.reshape(-1) * scale + center
        prep_scope = "test-only official tokenizer, fixed padding to original max_length=512"
    elif method == "periogt":
        import argparse
        from polyfid.baselines.periogt import _CachedPerioGTDataset, _portable_torch_load, _upstream_context
        with _upstream_context(model_root) as (_, initial), _portable_torch_load():
            from data.collator_light import CollatorFinetune
            from models.get_model_finetune import get_model
            args = argparse.Namespace(config="base", backbone="light", mode="finetune", model_path=str(initial), dropout=float(config["dropout"]), device=device, max_prompt=int(config["maximum_prompt"]), use_prompt=True)
            model = get_model(args)
            model.load_state_dict(torch.load(selected, map_location=device, weights_only=False))
            model.to(device).eval()
            original_collator = CollatorFinetune(int(config["maximum_prompt"]))
        loading = time.perf_counter() - started
        started = time.perf_counter()
        dataset = _CachedPerioGTDataset(Path(str(row.cache_dir)), ids)
        preprocessing = time.perf_counter() - started

        def collate(indices: np.ndarray) -> Any:
            _, graph, fp, md, _ = original_collator([dataset[int(index)] for index in indices])
            return graph, fp, md

        transfer = lambda batch: tuple(value.to(device) for value in batch)
        forward = lambda batch: model.forward_tune(*batch).reshape(-1)
        scale, center = float(scaler["std"]), float(scaler["mean"])
        postprocess = lambda values: values.reshape(-1) * scale + center
        prep_scope = "load full-task prebuilt PerioGT cache and select test IDs; offline descriptor/periodic-prompt generation is separate full-task CPU cost, not test-only preprocessing"
    elif method == "grin":
        from polyfid.baselines.grin import GRINRunConfig, _predictor
        from torch_geometric.data import Batch
        predictor = _predictor(model_root, GRINRunConfig(**config), device=device)
        predictor.load_from_local(str(selected))
        model = predictor.model.to(device).eval()
        loading = time.perf_counter() - started
        started = time.perf_counter()
        texts = external_model_text(frame, method, str(row.task)).tolist()
        molecules, _ = predictor._validate_inputs(texts)
        dataset = predictor._convert_to_pytorch_data(molecules)
        preprocessing = time.perf_counter() - started
        collate = lambda indices: Batch.from_data_list([dataset[int(index)] for index in indices])
        transfer = lambda batch: batch.to(device)
        forward = lambda batch: model(batch)["prediction"].reshape(-1)
        postprocess = lambda values: values.reshape(-1)
        prep_scope = "test-only official validation and PyG conversion; original test_time_augmentation=False"
    elif method == "chemprop":
        from polyfid.baselines.chemprop import apply_pandas_rdkit_compat, apply_sklearn_metrics_compat, chemprop_torch_load_compat
        from polyfid.baselines.classical import structure_smiles
        apply_pandas_rdkit_compat()
        apply_sklearn_metrics_compat()
        from chemprop.features import BatchMolGraph, MolGraph
        from chemprop.utils import load_checkpoint, load_scalers
        logger = logging.getLogger("polyfid.workflow_cost.chemprop")
        logger.setLevel(logging.WARNING)
        with chemprop_torch_load_compat():
            model = load_checkpoint(str(selected), device=torch.device(device), logger=logger).eval()
            target_scaler = load_scalers(str(selected))[0]
        loading = time.perf_counter() - started
        started = time.perf_counter()
        dataset = [MolGraph(text) for text in structure_smiles(frame)]
        preprocessing = time.perf_counter() - started
        collate = lambda indices: [BatchMolGraph([dataset[int(index)] for index in indices])]

        def transfer(batch: list[Any]) -> list[Any]:
            graph = batch[0]
            for attribute in ("f_atoms", "f_bonds", "a2b", "b2a", "b2revb", "a2a", "b2b"):
                value = getattr(graph, attribute, None)
                if torch.is_tensor(value):
                    setattr(graph, attribute, value.to(device))
            return batch

        forward = lambda batch: model(batch).reshape(-1)
        postprocess = lambda values: target_scaler.inverse_transform(values.reshape(-1, 1)).reshape(-1)
        prep_scope = "test-only official MolGraph features; official BatchMolGraph collation remains timed"
    else:
        raise ValueError(method)
    return CostBackend(method, ids, collate, transfer, forward, postprocess,
        {"backend": method, "dtype": str(next(model.parameters()).dtype).replace("torch.", ""), "max_length": config.get("max_length"),
         "checkpoint": str(selected), "backend_config_path": str(row.backend_config_path), "scaler_path": str(row.scaler_path), "reference_predictions": str(row.output),
         "model_loading_seconds": loading, "input_preprocessing_seconds": preprocessing, "input_preprocessing_records": len(ids), "preprocessing_scope": prep_scope,
         "parameter_count": sum(value.numel() for value in model.parameters()), "resident_state_bytes": _model_bytes(model)}, [model])


def _reference_values(backend: CostBackend) -> np.ndarray:
    frame = pd.read_parquet(backend.metadata["reference_predictions"])
    if "head_seed" in frame:
        frame = frame.loc[frame["head_seed"].eq(11)]
    if "split" in frame:
        frame = frame.loc[frame["split"].eq("test")]
    if set(frame["sample_id"].astype(str)) != set(backend.sample_ids):
        raise ValueError(f"{backend.method}: selected-state reference is not the complete common test fold")
    return frame.set_index(frame["sample_id"].astype(str)).loc[backend.sample_ids, "y_pred"].to_numpy(dtype=float)


def run_task_profile(project: Path, *, task: str, recovery_manifest: Path, output_root: Path, mode: str = "gpu", warmup_batches: int = 20, passes: int = 10) -> dict[str, Any]:
    import torch
    from polyfid.paths import load_paths

    task = task.upper()
    cached_head = mode == "cached-head"
    device = "cpu" if cached_head else "cuda"
    if not cached_head and (not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or "A800" not in torch.cuda.get_device_name(0).upper()):
        raise RuntimeError("formal workflow profiling runs in a single-A800 Slurm job")
    if cached_head and os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("cached-head CPU profiling runs locally, not through Slurm")
    paths = load_paths(project / "configs/paths.yaml")
    started = time.perf_counter()
    frame = common_test_frame(project, task)
    data_loading_seconds = time.perf_counter() - started
    recovery = None if cached_head else pd.read_csv(recovery_manifest, sep="\t")
    if recovery is not None:
        recovery["method"] = recovery["method"].astype(str).str.lower().replace({"chemprop_dpnn": "chemprop"})
    destination = output_root / mode / task.lower()
    destination.mkdir(parents=True, exist_ok=True)
    order = method_order(task, cached_head=cached_head)
    reports = []
    for method in order:
        method_root = destination / method
        completed_path = method_root / "summary.json"
        if completed_path.is_file():
            completed = _read_json(completed_path)
            measurements = completed.get("measurements", [])
            if (completed.get("status") == "complete" and completed.get("test_records") == len(frame)
                and len(measurements) == 4 and all(item.get("passes") == passes and item.get("warmup_batches") == warmup_batches for item in measurements)):
                reports.append(completed)
                print(f"{mode} {task} {method}: reuse completed timing", flush=True)
                continue
        if method in CORE_METHODS:
            backend = _core_backend(project, paths.shared_model_root, paths.checkpoint_root, task, method, frame, device=device, cached_head=cached_head)
        else:
            selected = recovery.loc[recovery["task"].str.upper().eq(task) & recovery["method"].eq(method)]
            if len(selected) != 1:
                raise ValueError(f"expected one restored {method}/{task} model")
            backend = _external_backend(paths.shared_model_root, method, selected.iloc[0], frame, device=device)
        expected = _reference_values(backend)
        rows = []
        summaries = []
        predictions = []
        for batch_size in (1, 32):
            for regime in ("compute_only", "prepared_input_end_to_end"):
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
                with torch.no_grad():
                    measurements, prediction = measure_backend(backend, batch_size=batch_size, regime=regime, warmup_batches=warmup_batches, passes=passes, synchronize=synchronize,
                        reset_peak_memory=torch.cuda.reset_peak_memory_stats if device == "cuda" else lambda: None)
                agreement = prediction_agreement(prediction, expected)
                summary = {"method": method, "task": task, "mode": mode, "batch_size": batch_size, "regime": regime, "test_records": len(frame), "warmup_batches": warmup_batches, "passes": passes,
                    **summarize_measurements(measurements), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device == "cuda" else 0, **agreement}
                summaries.append(summary)
                rows.extend({"method": method, "task": task, **row} for row in measurements)
                predictions.append(pd.DataFrame({"sample_id": backend.sample_ids, "task": task, "method": method, "batch_size": batch_size, "regime": regime, "y_pred": prediction, "reference_y_pred": expected}))
        writing_started = time.perf_counter()
        atomic_write_csv(pd.DataFrame(rows), method_root / "batch_timings.csv")
        atomic_write_csv(pd.DataFrame(summaries), method_root / "timing_summary.csv")
        atomic_write_parquet(pd.concat(predictions, ignore_index=True), method_root / "restored_predictions.parquet")
        report = {"status": "complete", "method": method, "task": task, "mode": mode, "device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "local CPU",
            "cpu_threads": torch.get_num_threads(), "test_records": len(frame), "fold": 0, "protocol": "group_ood", "head_seed": 11,
            "data_loading_seconds": data_loading_seconds, "output_writing_seconds": time.perf_counter() - writing_started,
            "timing_scope": "prepared CPU records to prediction; model loading, fixed input preprocessing and output writing excluded and separately reported",
            "compute_only_scope": "synchronized production-forward wall time on a prepared device batch; includes model-internal host work/transfers where the official forward performs them",
            **backend.metadata, "measurements": summaries}
        atomic_write_text(json.dumps(report, indent=2) + "\n", method_root / "summary.json")
        reports.append(report)
        print(f"{mode} {task} {method}: complete full-test timing at batch 1/32", flush=True)
        del backend
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    result = {"status": "complete", "task": task, "mode": mode, "method_order": order, "test_records": len(frame), "methods": reports}
    atomic_write_text(json.dumps(result, indent=2) + "\n", destination / "summary.json")
    return result


def generate_timing_jobs(project: Path, recovery_manifest: Path, output_root: Path) -> dict[str, Any]:
    """Write seven A800 jobs; submission belongs to the existing submitter."""

    directory = project / "slurm/generated/workflow_cost"
    paths = []
    for task in TASKS:
        path = directory / f"workflow_cost_{task.lower()}.sbatch"
        command = [str(project / ".conda/env/bin/python"), "scripts/profile_workflow_cost.py", "--mode", "gpu", "--task", task, "--recovery-manifest", str(recovery_manifest), "--output-root", str(output_root)]
        script = "\n".join([
            "#!/usr/bin/env bash", f"#SBATCH --job-name=p104-cost-{task.lower()}", "#SBATCH --partition=gpua800", "#SBATCH --gres=gpu:a800:1", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=80G", "#SBATCH --time=01:00:00", f"#SBATCH --output={project}/logs/slurm/%x-%j.out", f"#SBATCH --error={project}/logs/slurm/%x-%j.err", "", "set -euo pipefail", f"cd {shlex.quote(str(project))}", "export OMP_NUM_THREADS=8", "export MKL_NUM_THREADS=8", shlex.join(command), "",
        ])
        atomic_write_text(script, path)
        paths.append(str(path))
    job_list = directory / "workflow_cost_job_list.txt"
    atomic_write_text("\n".join(paths) + "\n", job_list)
    return {"jobs": len(paths), "gpus_per_job": 1, "gpu": "A800", "wall_time": "01:00:00", "job_list": str(job_list), "execution_path": "[GPU-A800 → sbatch]"}
