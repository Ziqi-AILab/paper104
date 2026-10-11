"""Same-split runner for the public torch-molecule GRIN implementation."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polyfid.records import RunRecord
from polyfid.baselines.restoration import (
    verify_prediction_roundtrip,
    write_retained_description,
)


@dataclass(frozen=True)
class GRINRunConfig:
    epochs: int = 500
    patience: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_layers: int = 5
    hidden_size: int = 300
    dropout: float = 0.5
    l1_penalty: float = 1e-3
    epochs_to_penalize: int = 100
    repetition_augmentation: bool = True
    test_time_augmentation: bool = False


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _native_max_aggregate(_module, inputs, index, dim_size=None):
    """GRIN's max aggregation using native PyTorch reductions.

    The compiled torch-scatter wheel for the pinned PyTorch build needs a newer
    system C library than this cluster provides.  This is the same channel-wise
    max operation used by the released GRIN layers and works on both CPU and
    CUDA tensors.
    """

    import torch

    output_size = (
        int(index.max().item()) + 1
        if dim_size is None and int(index.numel()) > 0
        else int(dim_size or 0)
    )
    shape = list(inputs.shape)
    shape[0] = output_size
    expanded_index = index.reshape((-1,) + (1,) * (inputs.ndim - 1)).expand_as(inputs)
    output = torch.full(
        shape,
        float("-inf"),
        dtype=inputs.dtype,
        device=inputs.device,
    ).scatter_reduce(0, expanded_index, inputs, reduce="amax", include_self=True)
    return torch.where(torch.isinf(output), torch.zeros_like(output), output)


def _load_grin(model_root: str | Path) -> tuple[type, Any]:
    # Let PyG initialize first without an optional compiled scatter extension.
    import torch_geometric  # noqa: F401

    source = Path(model_root) / "external_baselines/grin/source"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from torch_molecule import GRINMolecularPredictor
    from torch_molecule.nn import gnn
    from torch_molecule.utils import root_mean_squared_error

    # Only the two GRIN max-aggregation classes need torch-scatter.  Replacing
    # those methods leaves the released architecture, parameters, and objective
    # unchanged while using torch.scatter_reduce on the cluster runtime.
    _native_max_aggregate.__name__ = "aggregate"
    gnn.GINConv_GRIN.aggregate = _native_max_aggregate
    gnn.GCNConv_GRIN.aggregate = _native_max_aggregate
    return GRINMolecularPredictor, root_mean_squared_error


def _predictor(
    model_root: str | Path,
    config: GRINRunConfig,
    *,
    device: str,
) -> Any:
    predictor_class, rmse = _load_grin(model_root)
    metric = partial(rmse, average=True, sample_weight=None)
    model = predictor_class(
        num_task=1,
        task_type="regression",
        repetition_augmentation=bool(config.repetition_augmentation),
        l1_penalty=float(config.l1_penalty),
        epochs_to_penalize=int(config.epochs_to_penalize),
        num_layer=int(config.num_layers),
        hidden_size=int(config.hidden_size),
        gnn_type="gin-virtual",
        drop_ratio=float(config.dropout),
        norm_layer="batch_norm",
        graph_pooling="sum",
        augmented_feature=None,
        batch_size=int(config.batch_size),
        epochs=int(config.epochs),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        patience=int(config.patience),
        evaluate_criterion=metric,
        evaluate_higher_better=False,
        verbose="none",
        device=device,
    )
    model.evaluate_name = "rmse"
    return model


def probe_grin_load(model_root: str | Path) -> dict[str, Any]:
    """Exercise the real source, repeat augmentation, graph conversion, and model."""

    import torch_geometric

    model = _predictor(
        model_root,
        GRINRunConfig(epochs=1, patience=1, num_layers=2, hidden_size=32),
        device="cpu",
    )
    model._initialize_model(model.model_class)
    model.model.initialize_parameters(seed=0)
    repeated, _ = __import__(
        "torch_molecule.predictor.grin.utils", fromlist=["SmilesRepeat"]
    ).SmilesRepeat(2).repeat(["*CC*"])
    graph = model._convert_to_pytorch_data(["*CC*"])[0]
    return {
        "source_loaded": True,
        "torch_geometric_version": str(torch_geometric.__version__),
        "reduction_backend": "torch.scatter_reduce",
        "representative_repeat_count": int(len(repeated)),
        "representative_nodes": int(graph.num_nodes),
        "representative_edges": int(graph.num_edges),
        "model_parameters": int(sum(value.numel() for value in model.model.parameters())),
    }


def run_grin_fold(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    model_root: str | Path,
    seed: int,
    config: GRINRunConfig = GRINRunConfig(),
    checkpoint_dir: str | Path | None = None,
    checkpoint_retention: str = "trainable",
    require_a800: bool = True,
) -> Path:
    """Train released GRIN on one fixed Paper104 fold."""

    import torch

    if not torch.cuda.is_available() and require_a800:
        raise RuntimeError("formal GRIN training requires an A800 GPU job")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if require_a800 and "A800" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("formal GRIN training must run on an A800")

    _seed_everything(int(seed))
    source = Path(dataset_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_destination = Path(checkpoint_dir or destination)
    checkpoint_destination.mkdir(parents=True, exist_ok=True)
    tables = {
        split: pd.read_csv(source / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    model = _predictor(model_root, config, device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    model.fit(
        tables["train"]["smiles"].astype(str).tolist(),
        tables["train"]["value"].to_numpy(dtype=float).reshape(-1, 1),
        X_val=tables["validation"]["smiles"].astype(str).tolist(),
        y_val=tables["validation"]["value"].to_numpy(dtype=float).reshape(-1, 1),
    )
    training_seconds = time.perf_counter() - started

    if checkpoint_retention not in {"trainable", "metadata"}:
        raise ValueError("checkpoint_retention must be trainable or metadata")
    selected_tensor_bytes = int(
        sum(
            value.numel() * value.element_size()
            for value in model.model.state_dict().values()
        )
    )
    if checkpoint_retention == "trainable":
        model_artifact = checkpoint_destination / "grin.pt"
        model.save_to_local(str(model_artifact))
        write_retained_description(
            checkpoint_destination, config=asdict(config),
            target_scaler={"kind": "identity", "mean": 0.0, "std": 1.0},
        )
    else:
        model_artifact = checkpoint_destination / "model_metadata.json"
        model_artifact.write_text(
            json.dumps(
                {
                    "artifact_type": "grin_model_metadata",
                    "checkpoint_retention": "metadata",
                    "seed": int(seed),
                    "validation_selected_tensor_bytes": selected_tensor_bytes,
                    "config": asdict(config),
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
        split: np.asarray(
            model.predict(
                tables[split]["smiles"].astype(str).tolist(),
                test_time_augmentation=bool(config.test_time_augmentation),
            )["prediction"],
            dtype=float,
        ).reshape(-1)
        for split in ("validation", "test")
    }
    inference_seconds = time.perf_counter() - inference_started
    peak_device_memory_bytes = int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0
    if checkpoint_retention == "trainable":
        restored = _predictor(model_root, config, device=device)
        restored.load_from_local(str(model_artifact))
        reloaded_predictions = restored.predict(
            tables["test"]["smiles"].astype(str).tolist(),
            test_time_augmentation=bool(config.test_time_augmentation),
        )["prediction"]
        verify_prediction_roundtrip(
            outputs["test"], reloaded_predictions,
            checkpoint_destination / "reload_check.json",
        )
        del restored
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
        "device": device,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "trainable_parameters": int(
            sum(value.numel() for value in model.model.parameters() if value.requires_grad)
        ),
        "peak_device_memory_bytes": peak_device_memory_bytes,
        "checkpoint_retention": checkpoint_retention,
        "validation_selected_tensor_bytes": selected_tensor_bytes,
        "model_output_bytes": model_output_bytes,
    }
    report = {
        "backend": "public torch-molecule GRINMolecularPredictor",
        "protocol": "Paper104 fixed split; validation-only model selection",
        "seed": int(seed),
        "config": asdict(config),
        "model_artifact": str(model_artifact),
        "prediction_path": str(prediction_path),
        **resources,
    }
    (destination / "backend_run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    run_record = RunRecord(
        run_id=f"external_grin_seed-{seed}",
        stage="same_split_external_reference",
        method="grin",
        seed=int(seed),
        config_path="configs/downstream/external_baselines.yaml",
        command=tuple(sys.argv),
        input_paths=tuple(str(source / f"{split}.csv") for split in tables),
        output_path=str(prediction_path),
        status="complete",
        metrics={"validation_rows": float(len(tables["validation"])), "test_rows": float(len(tables["test"]))},
        resources=resources,
    )
    (destination / "run.json").write_text(
        json.dumps(run_record.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return prediction_path


__all__ = ["GRINRunConfig", "probe_grin_load", "run_grin_fold"]
