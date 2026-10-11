"""Restorable training state without experiment-management machinery."""

from __future__ import annotations

import fcntl
import os
import random
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass
class TrainingState:
    arm: str
    seed: int
    epoch: int = 0
    global_step: int = 0
    non_padding_tokens: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "TrainingState":
        return cls(
            arm=str(values["arm"]),
            seed=int(values["seed"]),
            epoch=int(values["epoch"]),
            global_step=int(values["global_step"]),
            non_padding_tokens=int(values["non_padding_tokens"]),
        )


@contextmanager
def checkpoint_write_lock(path: str | Path | None) -> Iterator[None]:
    """Serialize large writes that share the Paper104 checkpoint filesystem."""

    if path is None:
        yield
        return
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def save_training_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    state: TrainingState,
    scheduler: Any | None = None,
    projection: Any | None = None,
    scaler: Any | None = None,
    checkpoint_schedule: Any | None = None,
    write_lock_path: str | Path | None = None,
) -> None:
    """Save model, optimizer, schedule, RNG and token progress for resume."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - integration environment
        raise RuntimeError("saving training checkpoints requires torch") from exc

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "training_state": state.to_dict(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_states"] = torch.cuda.get_rng_state_all()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if projection is not None:
        payload["projection"] = projection.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if checkpoint_schedule is not None:
        payload["checkpoint_schedule"] = checkpoint_schedule.to_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    staging_directory = os.environ.get("SLURM_TMPDIR")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f"polyfid-{target.stem}-",
        suffix=".pt",
        dir=staging_directory,
    )
    os.close(descriptor)
    staging = Path(staging_name)
    try:
        torch.save(payload, staging)
        with checkpoint_write_lock(write_lock_path):
            shutil.copyfile(staging, temporary)
            temporary.replace(target)
    except (OSError, RuntimeError):
        temporary.unlink(missing_ok=True)
        raise
    finally:
        staging.unlink(missing_ok=True)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    projection: Any | None = None,
    scaler: Any | None = None,
    map_location: str = "cpu",
) -> tuple[TrainingState, dict[str, object] | None]:
    """Restore a checkpoint and return training and checkpoint-schedule state."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - integration environment
        raise RuntimeError("loading training checkpoints requires torch") from exc

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if projection is not None and "projection" in payload:
        projection.load_state_dict(payload["projection"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    if torch.cuda.is_available() and "cuda_random_states" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_states"])
    return (
        TrainingState.from_dict(payload["training_state"]),
        payload.get("checkpoint_schedule"),
    )


__all__ = [
    "TrainingState",
    "checkpoint_write_lock",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
