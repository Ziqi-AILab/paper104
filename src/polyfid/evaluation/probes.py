"""Common frozen two-layer property head and its reusable inference state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from polyfid.io import atomic_save_npz


TaskType = Literal["regression", "binary"]


@dataclass(frozen=True)
class ProbeState:
    """Minimal state needed to apply one fitted property head.

    Regression labels are standardized with statistics fitted on the training
    split.  The stored center and scale undo that transformation at inference;
    validation and test labels never determine the transformation.
    """

    task_type: TaskType
    input_dim: int
    hidden_dim: int
    dropout: float
    target_center: float
    target_scale: float
    input_weight: np.ndarray
    input_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def predict(
        self,
        features: np.ndarray,
        *,
        batch_size: int = 4096,
        device: str = "cpu",
    ) -> np.ndarray:
        return predict_with_probe(self, features, batch_size=batch_size, device=device)


@dataclass(frozen=True)
class ProbeResult:
    predictions: np.ndarray
    validation_loss: float
    epochs: int
    learning_rate: float
    state: ProbeState


def _make_model(input_dim: int, hidden_dim: int, dropout: float):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


def _probe_state_from_model(
    model,
    *,
    task_type: TaskType,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    target_center: float,
    target_scale: float,
) -> ProbeState:
    parameters = model.state_dict()
    return ProbeState(
        task_type=task_type,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        target_center=float(target_center),
        target_scale=float(target_scale),
        input_weight=parameters["0.weight"].detach().cpu().numpy().copy(),
        input_bias=parameters["0.bias"].detach().cpu().numpy().copy(),
        output_weight=parameters["3.weight"].detach().cpu().numpy().copy(),
        output_bias=parameters["3.bias"].detach().cpu().numpy().copy(),
    )


def _model_from_probe_state(state: ProbeState, *, device: str):
    import torch

    model = _make_model(state.input_dim, state.hidden_dim, state.dropout).to(device)
    model.load_state_dict(
        {
            "0.weight": torch.as_tensor(state.input_weight),
            "0.bias": torch.as_tensor(state.input_bias),
            "3.weight": torch.as_tensor(state.output_weight),
            "3.bias": torch.as_tensor(state.output_bias),
        }
    )
    model.eval()
    return model


def predict_with_probe(
    state: ProbeState,
    features: np.ndarray,
    *,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Apply an already fitted probe to any cache from the same encoder."""

    import torch

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != state.input_dim:
        raise ValueError(
            f"probe expects [rows, {state.input_dim}] features, got {values.shape}"
        )
    if not len(values):
        return np.empty(0, dtype=np.float32)
    model = _model_from_probe_state(state, device=device)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            raw = model(
                torch.as_tensor(
                    values[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
            ).squeeze(-1)
            if state.task_type == "binary":
                prediction = torch.sigmoid(raw)
            else:
                prediction = raw * state.target_scale + state.target_center
            outputs.append(prediction.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def save_probe_state(state: ProbeState, path: str | Path) -> Path:
    """Persist a fitted head and its train-fitted target transformation."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("probe state path must end in .npz")
    return atomic_save_npz(
        destination,
        {
            "task_type": np.asarray(state.task_type),
            "input_dim": np.asarray(state.input_dim, dtype=np.int64),
            "hidden_dim": np.asarray(state.hidden_dim, dtype=np.int64),
            "dropout": np.asarray(state.dropout, dtype=np.float64),
            "target_center": np.asarray(state.target_center, dtype=np.float64),
            "target_scale": np.asarray(state.target_scale, dtype=np.float64),
            "input_weight": np.asarray(state.input_weight, dtype=np.float32),
            "input_bias": np.asarray(state.input_bias, dtype=np.float32),
            "output_weight": np.asarray(state.output_weight, dtype=np.float32),
            "output_bias": np.asarray(state.output_bias, dtype=np.float32),
        },
    )


def load_probe_state(path: str | Path) -> ProbeState:
    """Load a head saved by :func:`save_probe_state`."""

    with np.load(path, allow_pickle=False) as values:
        return ProbeState(
            task_type=str(values["task_type"].item()),  # type: ignore[arg-type]
            input_dim=int(values["input_dim"].item()),
            hidden_dim=int(values["hidden_dim"].item()),
            dropout=float(values["dropout"].item()),
            target_center=float(values["target_center"].item()),
            target_scale=float(values["target_scale"].item()),
            input_weight=values["input_weight"].astype(np.float32),
            input_bias=values["input_bias"].astype(np.float32),
            output_weight=values["output_weight"].astype(np.float32),
            output_bias=values["output_bias"].astype(np.float32),
        )


def train_frozen_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    *,
    seed: int,
    learning_rate: float,
    task_type: TaskType = "regression",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 256,
    device: str = "cpu",
) -> ProbeResult:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    input_dim = int(train_x.shape[1])
    model = _make_model(input_dim, hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if task_type == "regression":
        center = float(np.mean(train_y))
        scale = float(max(np.std(train_y), 1e-8))
        train_target = (np.asarray(train_y) - center) / scale
        validation_target = (np.asarray(validation_y) - center) / scale
        loss_function = nn.MSELoss()
    else:
        center, scale = 0.0, 1.0
        train_target = np.asarray(train_y)
        validation_target = np.asarray(validation_y)
        loss_function = nn.BCEWithLogitsLoss()

    x_train = torch.as_tensor(train_x, dtype=torch.float32)
    y_train = torch.as_tensor(train_target, dtype=torch.float32).reshape(-1, 1)
    x_validation = torch.as_tensor(validation_x, dtype=torch.float32, device=device)
    y_validation = torch.as_tensor(validation_target, dtype=torch.float32, device=device).reshape(-1, 1)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    completed = 0
    for epoch in range(max_epochs):
        model.train()
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(loss_function(model(x_validation), y_validation).cpu())
        completed = epoch + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    state = _probe_state_from_model(
        model,
        task_type=task_type,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        target_center=center,
        target_scale=scale,
    )
    prediction = predict_with_probe(state, test_x, batch_size=batch_size, device=device)
    return ProbeResult(
        predictions=prediction,
        validation_loss=best_loss,
        epochs=completed,
        learning_rate=learning_rate,
        state=state,
    )


__all__ = [
    "ProbeResult",
    "ProbeState",
    "TaskType",
    "load_probe_state",
    "predict_with_probe",
    "save_probe_state",
    "train_frozen_probe",
]
