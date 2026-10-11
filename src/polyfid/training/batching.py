"""Batch clean relation views through one shared encoder call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass
class RelationViewInputs:
    anchors: Mapping[str, torch.Tensor]
    positives: Mapping[str, torch.Tensor]
    hard_negatives: Mapping[str, torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return int(self.anchors["input_ids"].shape[0])


@dataclass
class RelationRepresentations:
    anchors: torch.Tensor
    positives: torch.Tensor
    hard_negatives: torch.Tensor | None


def merge_relation_inputs(
    views: RelationViewInputs,
) -> tuple[dict[str, torch.Tensor], tuple[slice, slice, slice | None]]:
    """Concatenate matched views so all use the same encoder forward."""

    mappings = [views.anchors, views.positives]
    if views.hard_negatives is not None:
        mappings.append(views.hard_negatives)
    common_fields = set.intersection(*(set(values) for values in mappings))
    merged = {
        name: torch.cat([values[name] for values in mappings], dim=0)
        for name in common_fields
    }
    batch_size = views.batch_size
    anchor_slice = slice(0, batch_size)
    positive_slice = slice(batch_size, batch_size * 2)
    negative_slice = (
        None if views.hard_negatives is None else slice(batch_size * 2, batch_size * 3)
    )
    return merged, (anchor_slice, positive_slice, negative_slice)


def split_relation_representations(
    projected: torch.Tensor,
    slices: tuple[slice, slice, slice | None],
) -> RelationRepresentations:
    anchor_slice, positive_slice, negative_slice = slices
    return RelationRepresentations(
        anchors=projected[anchor_slice],
        positives=projected[positive_slice],
        hard_negatives=None if negative_slice is None else projected[negative_slice],
    )


__all__ = [
    "RelationRepresentations",
    "RelationViewInputs",
    "merge_relation_inputs",
    "split_relation_representations",
]
