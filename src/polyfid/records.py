"""Small shared records used across data, training, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from polyfid.io import atomic_write_text


@dataclass(frozen=True)
class PolymerRecord:
    """A curated linear, dual-attachment repeat-unit record."""

    sample_id: str
    raw_psmiles: str
    canonical_psmiles: str
    closed_smiles: str
    repeat_equivalence_id: str
    group_id: str
    split: str
    attachment_atom_indices: tuple[int, int]
    atom_count: int
    formula: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionRecord:
    """One observed prediction; all formal statistics start from these rows."""

    sample_id: str
    task: str
    split: str
    group_id: str
    y_true: float
    y_pred: float
    method: str
    representation_seed: int
    head_seed: int
    protocol: str | None = None
    family_id: str | None = None
    cluster_id: str | None = None
    fold: str | None = None
    label_fraction: float = 1.0
    subsample_id: int = 0
    input_view: str = "valid"
    representation_path: str | None = None
    structural_fidelity_eligible: bool | None = None
    scope_class: str | None = None
    checkpoint_fraction: float | None = None
    checkpoint_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunRecord:
    """Readable provenance for one run, without embedding large artifacts."""

    run_id: str
    stage: str
    method: str
    seed: int
    config_path: str
    command: tuple[str, ...]
    input_paths: tuple[str, ...]
    output_path: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    resources: dict[str, float | int | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measured_resources(
    started_at: float,
    *,
    device: str,
    **extra: float | int | str,
) -> dict[str, float | int | str]:
    """Return the small common resource record available inside a run."""

    wall_seconds = float(time.time() - started_at)
    values: dict[str, float | int | str] = {
        "device": str(device),
        "wall_seconds": wall_seconds,
        **extra,
    }
    if str(device).startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            values["peak_cuda_memory_mb"] = float(
                torch.cuda.max_memory_allocated() / (1024**2)
            )
            values["gpu_hours"] = wall_seconds / 3600.0
    return values


def write_run_record(record: RunRecord, path: str | Path) -> Path:
    destination = Path(path)
    atomic_write_text(
        json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
        destination,
    )
    return destination


__all__ = [
    "PolymerRecord",
    "PredictionRecord",
    "RunRecord",
    "measured_resources",
    "write_run_record",
]
