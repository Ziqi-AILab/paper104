"""Data structures for whole-atom token masking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AtomTokenGroup:
    """All tokenizer positions that touch one real atom.

    ``boundary_distance`` is the shortest graph distance to either attachment
    point.  A value of ``None`` is used for molecular records without polymer
    attachment points.  Attachment marker tokens are represented with
    ``eligible=False`` and never become primary MLM targets.
    """

    atom_index: int
    token_indices: tuple[int, ...]
    boundary_distance: int | None = None
    eligible: bool = True

    def __post_init__(self) -> None:
        if not self.token_indices:
            raise ValueError("an atom-token group needs at least one token")
        if len(set(self.token_indices)) != len(self.token_indices):
            raise ValueError("token positions within an atom group must be unique")


@dataclass(frozen=True)
class MaskPlan:
    """A reproducible selection of complete atom groups."""

    strategy: str
    selected_group_indices: tuple[int, ...]
    atom_budget: int
    selection_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.selection_roles and len(self.selection_roles) != len(
            self.selected_group_indices
        ):
            raise ValueError("mask selection roles must align with selected groups")

    def token_indices(self, groups: Sequence[AtomTokenGroup]) -> tuple[int, ...]:
        positions = {
            token
            for group_index in self.selected_group_indices
            for token in groups[group_index].token_indices
        }
        return tuple(sorted(positions))


def apply_mask_plan(
    input_ids: Sequence[int],
    groups: Sequence[AtomTokenGroup],
    plan: MaskPlan,
    *,
    mask_token_id: int,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    """Replace every token of every selected atom and construct MLM labels."""

    corrupted = list(input_ids)
    labels = [int(ignore_index)] * len(corrupted)
    for token_index in plan.token_indices(groups):
        if token_index < 0 or token_index >= len(corrupted):
            raise IndexError("atom-token position is outside input_ids")
        labels[token_index] = corrupted[token_index]
        corrupted[token_index] = int(mask_token_id)
    return corrupted, labels
