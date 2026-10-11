"""Shared model-forward path for all Paper104 pretraining arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from polyfid.models.arms import ArmSpec
from polyfid.models.projection import mean_pool

from .batching import (
    RelationViewInputs,
    merge_relation_inputs,
    split_relation_representations,
)
from .objectives import PretrainingLoss, paper104_objective


_MODEL_INPUT_FIELDS = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
}


@dataclass
class PretrainingBatch:
    mlm_inputs: Mapping[str, torch.Tensor]
    labels: torch.Tensor
    relation_views: RelationViewInputs | None = None
    masking_diagnostics: Mapping[str, int] | None = None


def _field(output: Any, name: str) -> Any:
    value = getattr(output, name, None)
    if value is None and isinstance(output, Mapping):
        value = output.get(name)
    return value


def _last_hidden(output: Any) -> torch.Tensor:
    hidden_states = _field(output, "hidden_states")
    if hidden_states is None:
        raise ValueError("relation arms require output_hidden_states=True")
    return hidden_states[-1]


def _model_inputs(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in values.items() if name in _MODEL_INPUT_FIELDS}


def forward_pretraining_batch(
    model: Any,
    projection: Any | None,
    batch: PretrainingBatch,
    arm: ArmSpec,
) -> PretrainingLoss:
    """Run full MLM plus the arm's matched clean-view relation objective."""

    mlm_output = model(
        **_model_inputs(batch.mlm_inputs),
        output_hidden_states=False,
        return_dict=True,
    )
    logits = _field(mlm_output, "logits")
    if logits is None:
        raise TypeError("masked-language model output must expose logits")

    relation = None
    if arm.needs_relation_views:
        if projection is None or batch.relation_views is None:
            raise ValueError(f"{arm.name} requires a projection head and relation views")
        merged, slices = merge_relation_inputs(batch.relation_views)
        relation_output = model(
            **_model_inputs(merged), output_hidden_states=True, return_dict=True
        )
        pooled = mean_pool(
            _last_hidden(relation_output),
            merged["attention_mask"],
            merged.get("special_tokens_mask"),
        )
        relation = split_relation_representations(projection(pooled), slices)

    return paper104_objective(
        logits,
        batch.labels,
        anchor_projections=None if relation is None else relation.anchors,
        positive_projections=None if relation is None else relation.positives,
        hard_negative_projections=None if relation is None else relation.hard_negatives,
        relation_weight=arm.relation_weight,
        temperature=arm.temperature,
    )


__all__ = ["PretrainingBatch", "forward_pretraining_batch"]
