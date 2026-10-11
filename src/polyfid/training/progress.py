"""Progress accounting in actual non-padding tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


CHECKPOINT_PERCENTAGES = (1, 5, 10, 25, 50, 100)


def count_non_padding(attention_mask: Any) -> int:
    """Count valid tokens from tensor-like or nested Python masks."""

    if hasattr(attention_mask, "sum"):
        value = attention_mask.sum()
        return int(value.item() if hasattr(value, "item") else value)
    return sum(int(value) for row in attention_mask for value in row)


@dataclass
class TokenProgress:
    non_padding_tokens: int = 0
    optimizer_steps: int = 0

    def update(self, attention_mask: Any, *, optimizer_step: bool = False) -> int:
        added = count_non_padding(attention_mask)
        self.non_padding_tokens += added
        self.optimizer_steps += int(optimizer_step)
        return added

    def to_dict(self) -> dict[str, int]:
        return {
            "non_padding_tokens": int(self.non_padding_tokens),
            "optimizer_steps": int(self.optimizer_steps),
        }

    @classmethod
    def from_dict(cls, values: dict[str, int]) -> "TokenProgress":
        return cls(
            non_padding_tokens=int(values["non_padding_tokens"]),
            optimizer_steps=int(values["optimizer_steps"]),
        )


class CheckpointSchedule:
    """Emit the 1/5/10/25/50/100% checkpoints exactly once."""

    def __init__(
        self,
        total_non_padding_tokens: int,
        percentages: Iterable[int] = CHECKPOINT_PERCENTAGES,
        *,
        emitted: Iterable[int] = (),
    ) -> None:
        if int(total_non_padding_tokens) <= 0:
            raise ValueError("total_non_padding_tokens must be positive")
        self.total_non_padding_tokens = int(total_non_padding_tokens)
        self.percentages = tuple(int(value) for value in percentages)
        if not self.percentages or any(value <= 0 or value > 100 for value in self.percentages):
            raise ValueError("checkpoint percentages must lie in (0, 100]")
        if tuple(sorted(set(self.percentages))) != self.percentages:
            raise ValueError("checkpoint percentages must be unique and increasing")
        self.emitted = set(int(value) for value in emitted)

    def threshold(self, percentage: int) -> int:
        return math.ceil(self.total_non_padding_tokens * int(percentage) / 100)

    def newly_reached(self, non_padding_tokens: int) -> tuple[int, ...]:
        reached = tuple(
            value
            for value in self.percentages
            if value not in self.emitted
            and int(non_padding_tokens) >= self.threshold(value)
        )
        self.emitted.update(reached)
        return reached

    def to_dict(self) -> dict[str, object]:
        return {
            "total_non_padding_tokens": self.total_non_padding_tokens,
            "percentages": list(self.percentages),
            "emitted": sorted(self.emitted),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "CheckpointSchedule":
        return cls(
            int(values["total_non_padding_tokens"]),
            values["percentages"],  # type: ignore[arg-type]
            emitted=values["emitted"],  # type: ignore[arg-type]
        )


__all__ = [
    "CHECKPOINT_PERCENTAGES",
    "CheckpointSchedule",
    "TokenProgress",
    "count_non_padding",
]
