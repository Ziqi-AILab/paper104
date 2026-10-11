"""Same-split adapter for the public PerioGT light checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from polyfid.records import RunRecord
from polyfid.baselines.restoration import (
    verify_prediction_roundtrip,
    write_retained_description,
)


@dataclass(frozen=True)
class PerioGTRunConfig:
    epochs: int = 50
    patience: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    dropout: float = 0.05
    maximum_prompt: int = 10
    num_workers: int = 2


def _asset_paths(model_root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(model_root) / "external_baselines/periogt"
    source = root / "source/PerioGT_common"
    checkpoint = root / "checkpoint/pretrained_ckpt/pretrained/light/base.pth"
    scaler = source / "datasets/pretrain/scaler_all.pkl"
    return source, checkpoint, scaler


@contextmanager
def _upstream_context(model_root: str | Path) -> Iterator[tuple[Path, Path]]:
    source, checkpoint, _ = _asset_paths(model_root)
    scripts = source / "scripts"
    previous = Path.cwd()
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.chdir(scripts)
    try:
        yield source, checkpoint
    finally:
        os.chdir(previous)


@contextmanager
def _portable_torch_load(*, force_cpu: bool = False) -> Iterator[None]:
    """Allow the CUDA-trained checkpoint to be inspected on a CPU host."""

    import torch

    original = torch.load

    def load(*args: Any, **kwargs: Any) -> Any:
        if force_cpu or not torch.cuda.is_available():
            kwargs.setdefault("map_location", "cpu")
        return original(*args, **kwargs)

    torch.load = load
    try:
        yield
    finally:
        torch.load = original


def _seed_everything(seed: int) -> None:
    import dgl
    import torch

    random.seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe_periogt_load(model_root: str | Path) -> dict[str, Any]:
    """Load the checkpoint and construct one real periodic-prompt graph on CPU."""

    import pickle

    import dgl
    import torch
    from mordred import Calculator, descriptors

    source, checkpoint, scaler_path = _asset_paths(model_root)
    with _upstream_context(model_root):
        from data.smiles2g_light import PolyGraphBuilderFinetune
        from models.get_model_finetune import get_model
        from utils.features import precompute_features

        args = argparse.Namespace(
            config="base",
            backbone="light",
            mode="finetune",
            model_path=str(checkpoint),
            dropout=0.05,
            device="cpu",
            maximum_prompt=10,
            max_prompt=10,
            use_prompt=True,
        )
        with _portable_torch_load(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator")
            model = get_model(args)
            with scaler_path.open("rb") as handle:
                scaler = pickle.load(handle)
            feature_cache = precompute_features(["*CC*"], units=(3,), workers=1)
            graph = PolyGraphBuilderFinetune(args, scaler).build("*CC*", feature_cache)
    fingerprint, descriptor = feature_cache[("*CC*", 3)]
    return {
        "checkpoint_loaded": True,
        "checkpoint_parameters": int(sum(value.numel() for value in model.parameters())),
        "dgl_version": str(dgl.__version__),
        "descriptor_count": int(len(Calculator(descriptors, ignore_3D=True).descriptors)),
        "representative_fingerprint_size": int(len(fingerprint)),
        "representative_descriptor_size": int(len(descriptor)),
        "representative_nodes": int(graph.num_nodes()),
        "representative_edges": int(graph.num_edges()),
        "representative_prompt_shape": [int(value) for value in graph.ndata["prompt"].shape],
        "scaler_path": str(source / "datasets/pretrain/scaler_all.pkl"),
    }


def prepare_periogt_cache(
    frame: pd.DataFrame,
    cache_dir: str | Path,
    *,
    model_root: str | Path,
    task: str,
    preprocess_workers: int = 8,
) -> Path:
    """Prepare PerioGT descriptors, graphs, and periodic prompts once per task."""
    destination = Path(cache_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source, checkpoint, scaler = _asset_paths(model_root)
    ordered = frame.loc[:, ["sample_id", "smiles", "value"]].copy()
    ordered["sample_id"] = ordered["sample_id"].astype(str)
    ordered = ordered.sort_values("sample_id", kind="stable").reset_index(drop=True)
    ordered.to_csv(destination / "row_index.csv", index=False)
    ordered.loc[:, ["smiles", "value"]].to_csv(
        destination / f"{destination.name}.csv", index=False
    )
    command = [
        sys.executable,
        "-m",
        "polyfid.baselines.periogt_prepare_bridge",
        str(source / "scripts/prepare_ft_dataset.py"),
        "--dataset",
        destination.name,
        "--device",
        "cpu",
        "--config",
        "base",
        "--backbone",
        "light",
        "--model_path",
        str(checkpoint),
        "--data_path",
        str(destination.parent),
        "--n_jobs",
        str(int(preprocess_workers)),
        "--use_prompt",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source), environment.get("PYTHONPATH", "")) if value
    )
    started = time.perf_counter()
    subprocess.run(command, cwd=source / "scripts", env=environment, check=True)
    seconds = time.perf_counter() - started
    record = RunRecord(
        run_id=f"external_periogt_prepare_{task.casefold()}",
        stage="external_model_preprocessing",
        method="periogt",
        seed=0,
        config_path="configs/downstream/external_baselines.yaml",
        command=tuple(command),
        input_paths=(str(destination / "row_index.csv"), str(checkpoint), str(scaler)),
        output_path=str(destination),
        status="complete",
        metrics={"rows": float(len(ordered))},
        resources={
            "device": "cpu",
            "wall_seconds": float(seconds),
            "preprocess_workers": int(preprocess_workers),
        },
    )
    (destination / "run.json").write_text(
        json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


class _CachedPerioGTDataset:
    def __new__(cls, cache_dir: Path, sample_ids: list[str]) -> Any:
        import scipy.sparse as sparse
        import torch
        from dgl.data.utils import load_graphs
        from torch.utils.data import Dataset

        class DatasetImpl(Dataset):
            def __init__(self) -> None:
                rows = pd.read_csv(cache_dir / "row_index.csv")
                rows["sample_id"] = rows["sample_id"].astype(str)
                positions = pd.Series(rows.index, index=rows["sample_id"])
                self.indices = positions.loc[sample_ids].to_numpy(dtype=int)
                graphs, _ = load_graphs(str(cache_dir / "graphs.pkl"))
                self.graphs = graphs
                self.fingerprints = torch.from_numpy(
                    sparse.load_npz(cache_dir / "maccs_ecfp.npz")
                    .todense()
                    .astype(np.float32)
                )
                descriptors = np.load(cache_dir / "polymer_descriptors.npz")["md"].astype(
                    np.float32
                )
                self.descriptors = torch.from_numpy(np.nan_to_num(descriptors))
                self.rows = rows

            def __len__(self) -> int:
                return int(len(self.indices))

            def __getitem__(self, item: int) -> tuple[Any, ...]:
                position = int(self.indices[item])
                row = self.rows.iloc[position]
                return (
                    str(row["smiles"]),
                    self.graphs[position],
                    self.fingerprints[position],
                    self.descriptors[position],
                    torch.tensor([float(row["value"])], dtype=torch.float32),
                )

        return DatasetImpl()


class _PolynomialDecay:
    """Current-PyTorch form of the release's polynomial scheduler."""

    def __init__(
        self,
        optimizer: Any,
        *,
        warmup_updates: int,
        total_updates: int,
        learning_rate: float,
        ending_rate: float = 1e-9,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_updates = max(1, int(warmup_updates))
        self.total_updates = max(self.warmup_updates + 1, int(total_updates))
        self.learning_rate = float(learning_rate)
        self.ending_rate = float(ending_rate)
        # The released scheduler performs its first warmup update during
        # construction, before the first optimizer step.
        self.updates = 1
        initial_rate = self.learning_rate / self.warmup_updates
        for group in self.optimizer.param_groups:
            group["lr"] = initial_rate

    def step(self) -> None:
        self.updates += 1
        if self.updates <= self.warmup_updates:
            rate = self.learning_rate * self.updates / self.warmup_updates
        elif self.updates >= self.total_updates:
            rate = self.ending_rate
        else:
            remaining = 1 - (self.updates - self.warmup_updates) / (
                self.total_updates - self.warmup_updates
            )
            rate = (
                self.learning_rate - self.ending_rate
            ) * remaining + self.ending_rate
        for group in self.optimizer.param_groups:
            group["lr"] = rate


def _predict(model: Any, loader: Any, device: Any, mean: float, std: float) -> np.ndarray:
    import torch

    outputs: list[Any] = []
    model.eval()
    with torch.no_grad():
        for _, graph, fingerprint, descriptor, _ in loader:
            prediction = model.forward_tune(
                graph.to(device), fingerprint.to(device), descriptor.to(device)
            )
            outputs.append(prediction.detach().cpu())
    scaled = torch.cat(outputs, dim=0).numpy().reshape(-1)
    return scaled * std + mean


def run_periogt_fold(
    dataset_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    model_root: str | Path,
    seed: int,
    config: PerioGTRunConfig = PerioGTRunConfig(),
    checkpoint_dir: str | Path | None = None,
    checkpoint_retention: str = "trainable",
    require_a800: bool = True,
) -> Path:
    """Fine-tune PerioGT on one fixed Paper104 fold using a task-level cache."""

    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    if require_a800 and not torch.cuda.is_available():
        raise RuntimeError("formal PerioGT fine-tuning requires an A800 GPU job")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_a800 and "A800" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("formal PerioGT fine-tuning must run on an A800")

    _seed_everything(int(seed))
    source_tables = Path(dataset_dir)
    cache = Path(cache_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_destination = Path(checkpoint_dir or destination)
    checkpoint_destination.mkdir(parents=True, exist_ok=True)
    tables = {
        split: pd.read_csv(source_tables / f"{split}.csv")
        for split in ("train", "validation", "test")
    }

    with _upstream_context(model_root) as (_, checkpoint), _portable_torch_load():
        from data.collator_light import CollatorFinetune
        from models.get_model_finetune import get_model

        model_args = argparse.Namespace(
            config="base",
            backbone="light",
            mode="finetune",
            model_path=str(checkpoint),
            dropout=float(config.dropout),
            device=str(device),
            max_prompt=int(config.maximum_prompt),
            use_prompt=True,
        )
        model = get_model(model_args)
        collator = CollatorFinetune(int(config.maximum_prompt))

    datasets = {
        split: _CachedPerioGTDataset(
            cache, tables[split]["sample_id"].astype(str).tolist()
        )
        for split in tables
    }
    generator = torch.Generator().manual_seed(int(seed))
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(config.batch_size),
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=int(config.num_workers),
            drop_last=split == "train",
            collate_fn=collator,
        )
        for split, dataset in datasets.items()
    }
    node_attention = {id(value) for value in model.node_attn.parameters()}
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    value for value in model.parameters() if id(value) not in node_attention
                ]
            },
            {
                "params": model.node_attn.parameters(),
                "lr": float(config.learning_rate) * 5,
            },
        ],
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    total_updates = max(1, int(config.epochs) * len(loaders["train"]))
    scheduler = _PolynomialDecay(
        optimizer,
        warmup_updates=max(1, total_updates // 10),
        total_updates=total_updates,
        learning_rate=float(config.learning_rate),
    )
    loss_function = nn.MSELoss()
    train_values = tables["train"]["value"].to_numpy(dtype=float)
    mean = float(train_values.mean())
    std = max(float(train_values.std()), 1e-12)
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        for _, graph, fingerprint, descriptor, label in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model.forward_tune(
                graph.to(device), fingerprint.to(device), descriptor.to(device)
            ).reshape(-1)
            target = (label.to(device).reshape(-1) - mean) / std
            loss = loss_function(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
        validation = _predict(model, loaders["validation"], device, mean, std)
        validation_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        validation
                        - tables["validation"]["value"].to_numpy(dtype=float)
                    )
                )
            )
        )
        if validation_rmse < best_validation:
            best_validation = validation_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= int(config.patience):
            break
    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("PerioGT did not produce a validation-selected state")
    model.load_state_dict(best_state)
    if checkpoint_retention not in {"trainable", "metadata"}:
        raise ValueError("checkpoint_retention must be trainable or metadata")
    selected_tensor_bytes = int(
        sum(value.numel() * value.element_size() for value in best_state.values())
    )
    if checkpoint_retention == "trainable":
        model_artifact = checkpoint_destination / "periogt_best.pth"
        torch.save(best_state, model_artifact)
        write_retained_description(
            checkpoint_destination, config=asdict(config),
            target_scaler={"kind": "standard", "mean": mean, "std": std},
        )
    else:
        model_artifact = checkpoint_destination / "model_metadata.json"
        model_artifact.write_text(
            json.dumps(
                {
                    "artifact_type": "periogt_model_metadata",
                    "checkpoint_retention": "metadata",
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_rmse": float(best_validation),
                    "validation_selected_tensor_bytes": selected_tensor_bytes,
                    "source_checkpoint": str(checkpoint),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    model_output_bytes = int(model_artifact.stat().st_size)
    inference_started = time.perf_counter()
    outputs = {
        split: _predict(model, loaders[split], device, mean, std)
        for split in ("validation", "test")
    }
    inference_seconds = time.perf_counter() - inference_started
    if checkpoint_retention == "trainable":
        model.load_state_dict(torch.load(model_artifact, map_location="cpu", weights_only=False))
        restored_scaler = json.loads((checkpoint_destination / "target_scaler.json").read_text())
        verify_prediction_roundtrip(
            outputs["test"],
            _predict(model, loaders["test"], device, restored_scaler["mean"], restored_scaler["std"]),
            checkpoint_destination / "reload_check.json",
        )
    prediction_path = destination / "backend_predictions.csv"
    pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": tables[split]["sample_id"].astype(str),
                    "split": split,
                    "prediction": outputs[split],
                }
            )
            for split in ("validation", "test")
        ],
        ignore_index=True,
    ).to_csv(prediction_path, index=False)

    resources = {
        "device": str(device),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "trainable_parameters": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "peak_device_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
    }
    report = {
        "backend": "public PerioGT light checkpoint, periodic prompt, and LiGhT predictor",
        "protocol": "Paper104 fixed split; validation-only model selection",
        "seed": int(seed),
        "config": asdict(config),
        "best_epoch": int(best_epoch),
        "best_validation_rmse": float(best_validation),
        "cache": str(cache),
        "checkpoint_retention": checkpoint_retention,
        "validation_selected_tensor_bytes": selected_tensor_bytes,
        "model_output_bytes": model_output_bytes,
        "model_artifact": str(model_artifact),
        "prediction_path": str(prediction_path),
        **resources,
    }
    (destination / "backend_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    record = RunRecord(
        run_id=f"external_periogt_seed-{seed}",
        stage="same_split_external_reference",
        method="periogt",
        seed=int(seed),
        config_path="configs/downstream/external_baselines.yaml",
        command=tuple(sys.argv),
        input_paths=tuple(
            [str(source_tables / f"{split}.csv") for split in tables] + [str(cache)]
        ),
        output_path=str(prediction_path),
        status="complete",
        metrics={
            "best_validation_rmse": float(best_validation),
            "test_rows": float(len(tables["test"])),
        },
        resources=resources,
    )
    (destination / "run.json").write_text(
        json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return prediction_path


__all__ = [
    "PerioGTRunConfig",
    "prepare_periogt_cache",
    "probe_periogt_load",
    "run_periogt_fold",
]
