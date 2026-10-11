"""The common optimizer and schedule used by the six trained arms."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


WARM_START_LEARNING_RATES = (1e-5, 3e-5, 1e-4)
SCRATCH_LEARNING_RATES = (3e-5, 1e-4, 3e-4)


@dataclass(frozen=True)
class PretrainingRecipe:
    epochs: int = 3
    max_length: int = 256
    effective_anchor_batch_size: int = 256
    micro_batch_size: int = 32
    gradient_accumulation: int = 8
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    precision: str = "bf16"


def learning_rate_candidates(initialization: str) -> tuple[float, ...]:
    if initialization == "warm":
        return WARM_START_LEARNING_RATES
    if initialization == "scratch":
        return SCRATCH_LEARNING_RATES
    raise ValueError("initialization must be 'warm' or 'scratch'")


def build_optimizer_and_scheduler(
    model: Any,
    *,
    learning_rate: float,
    total_optimizer_steps: int,
    projection: Any | None = None,
    weight_decay: float = 0.01,
    warmup_fraction: float = 0.05,
    optimizer_name: str = "adamw",
    scheduler_name: str = "linear",
) -> tuple[Any, Any]:
    """Build AdamW and the fixed warmup-plus-linear-decay schedule."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - integration environment
        raise RuntimeError("optimizer construction requires torch") from exc

    if str(optimizer_name).casefold() != "adamw":
        raise ValueError("Paper104 pretraining currently supports optimizer=adamw")
    if str(scheduler_name).casefold() != "linear":
        raise ValueError("Paper104 pretraining currently supports scheduler=linear")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if projection is not None:
        parameters.extend(
            parameter for parameter in projection.parameters() if parameter.requires_grad
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    total_steps = int(total_optimizer_steps)
    warmup_steps = math.ceil(total_steps * float(warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    return optimizer, scheduler


__all__ = [
    "PretrainingRecipe",
    "SCRATCH_LEARNING_RATES",
    "WARM_START_LEARNING_RATES",
    "build_optimizer_and_scheduler",
    "learning_rate_candidates",
]
