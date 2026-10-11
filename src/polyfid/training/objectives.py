"""Objective-matched MLM and relation losses for Paper104."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


@dataclass
class PretrainingLoss:
    total: torch.Tensor
    mlm: torch.Tensor
    relation: torch.Tensor
    relation_count: int

    def detached_metrics(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach().cpu()),
            "mlm_loss": float(self.mlm.detach().cpu()),
            "relation_loss": float(self.relation.detach().cpu()),
            "relation_count": float(self.relation_count),
        }


def masked_language_model_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if logits.shape[:-1] != labels.shape:
        raise ValueError("labels must align with token logits")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.to(device=logits.device).reshape(-1),
        ignore_index=-100,
    )


def relation_infonce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    *,
    hard_negatives: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """In-batch InfoNCE with an optional paired topology near-negative."""

    if anchors.ndim != 2 or positives.shape != anchors.shape:
        raise ValueError("anchors and positives must be matching rank-two tensors")
    if hard_negatives is not None and hard_negatives.shape != anchors.shape:
        raise ValueError("hard negatives must align with anchors")
    if anchors.shape[0] == 0:
        return anchors.sum() * 0.0
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)
    logits = anchors @ positives.transpose(0, 1)
    if hard_negatives is not None:
        negatives = F.normalize(hard_negatives, dim=-1)
        paired_logits = (anchors * negatives).sum(dim=-1, keepdim=True)
        logits = torch.cat((logits, paired_logits), dim=1)
    targets = torch.arange(anchors.shape[0], device=anchors.device)
    return F.cross_entropy(logits / float(temperature), targets)


def paper104_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    anchor_projections: torch.Tensor | None = None,
    positive_projections: torch.Tensor | None = None,
    hard_negative_projections: torch.Tensor | None = None,
    relation_weight: float = 0.1,
    temperature: float = 0.1,
) -> PretrainingLoss:
    """Compute MLM or the predeclared ``MLM + 0.1 relation`` objective."""

    mlm = masked_language_model_loss(logits, labels)
    if anchor_projections is None:
        relation = mlm.new_zeros(())
        relation_count = 0
    else:
        if positive_projections is None:
            raise ValueError("relation anchors require positive projections")
        relation = relation_infonce_loss(
            anchor_projections,
            positive_projections,
            hard_negatives=hard_negative_projections,
            temperature=temperature,
        )
        relation_count = int(anchor_projections.shape[0])
    return PretrainingLoss(
        total=mlm + float(relation_weight) * relation,
        mlm=mlm,
        relation=relation,
        relation_count=relation_count,
    )


__all__ = [
    "PretrainingLoss",
    "masked_language_model_loss",
    "paper104_objective",
    "relation_infonce_loss",
]
