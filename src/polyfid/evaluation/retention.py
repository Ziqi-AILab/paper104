"""Fixed-mask molecular MLM retention evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from polyfid.chemistry import (
    StructureError,
    TokenAlignmentError,
    TokenLengthError,
    align_with_fast_tokenizer,
)
from polyfid.data.mlm import row_seed
from polyfid.masking import MaskPlanner, apply_mask_plan


@dataclass(frozen=True)
class _RetentionCase:
    sample_id: str
    features: dict[str, list[int]]
    labels: tuple[int, ...]
    token_to_group: dict[int, int]


_DETAIL_COLUMNS = (
    "sample_id",
    "atom_group_id",
    "token_position",
    "target_token_id",
    "predicted_token_id",
    "nll",
    "correct",
)


def _build_cases(
    sample_ids: Sequence[str],
    texts: Sequence[str],
    tokenizer: Any,
    *,
    seed: int,
    mask_probability: float,
    max_length: int,
    skipped: Counter[str],
) -> list[_RetentionCase]:
    planner = MaskPlanner(mask_probability=mask_probability)
    cases: list[_RetentionCase] = []
    for sample_id, text in zip(sample_ids, texts, strict=True):
        try:
            encoded, alignment = align_with_fast_tokenizer(
                str(text), tokenizer, max_length=max_length
            )
        except TokenLengthError:
            skipped["over_max_length"] += 1
            continue
        except (StructureError, TokenAlignmentError):
            skipped["structure_or_alignment"] += 1
            continue
        groups = alignment.masking_groups((None,) * len(alignment.lexemes))
        plan = planner.plan(
            groups,
            strategy="uniform",
            seed=row_seed(seed, 0, str(sample_id)),
        )
        if not plan.selected_group_indices:
            skipped["no_maskable_atom"] += 1
            continue
        corrupted, labels = apply_mask_plan(
            encoded["input_ids"],
            groups,
            plan,
            mask_token_id=int(tokenizer.mask_token_id),
        )
        token_to_group = {
            int(token): int(group_index)
            for group_index in plan.selected_group_indices
            for token in groups[group_index].token_indices
        }
        features = {
            name: [int(value) for value in encoded[name]]
            for name in ("input_ids", "attention_mask", "token_type_ids")
            if name in encoded
        }
        features["input_ids"] = corrupted
        cases.append(
            _RetentionCase(
                sample_id=str(sample_id),
                features=features,
                labels=tuple(int(value) for value in labels),
                token_to_group=token_to_group,
            )
        )
    return cases


def evaluate_uniform_atom_group_mlm(
    model: Any,
    tokenizer: Any,
    sample_ids: Sequence[str],
    texts: Sequence[str],
    *,
    seed: int = 10_407,
    mask_probability: float = 0.15,
    max_length: int = 256,
    batch_size: int = 128,
    device: str = "cuda",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate one deterministic uniform whole-atom mask per molecule."""

    import torch

    if len(sample_ids) != len(texts):
        raise ValueError("sample IDs and texts must align")
    if tokenizer.mask_token_id is None:
        raise ValueError("the tokenizer must define a mask token")

    skipped: Counter[str] = Counter()
    cases = _build_cases(
        sample_ids,
        texts,
        tokenizer,
        seed=seed,
        mask_probability=mask_probability,
        max_length=max_length,
        skipped=skipped,
    )
    rows: list[dict[str, object]] = []
    model.eval()
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        encoded = tokenizer.pad(
            [case.features for case in batch], padding=True, return_tensors="pt"
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, return_dict=True)
            logits = output["logits"] if isinstance(output, dict) else output.logits
        for batch_index, case in enumerate(batch):
            positions = sorted(case.token_to_group)
            position_tensor = torch.as_tensor(
                positions, dtype=torch.long, device=logits.device
            )
            targets = torch.as_tensor(
                [case.labels[position] for position in positions],
                dtype=torch.long,
                device=logits.device,
            )
            selected = logits[batch_index, position_tensor]
            losses = -torch.log_softmax(selected.float(), dim=-1).gather(
                1, targets[:, None]
            )[:, 0]
            predictions = selected.argmax(dim=-1)
            for item, position in enumerate(positions):
                predicted = int(predictions[item].item())
                target = int(targets[item].item())
                rows.append(
                    {
                        "sample_id": case.sample_id,
                        "atom_group_id": (
                            f"{case.sample_id}:atom_group:"
                            f"{case.token_to_group[position]}"
                        ),
                        "token_position": int(position),
                        "target_token_id": target,
                        "predicted_token_id": predicted,
                        "nll": float(losses[item].item()),
                        "correct": predicted == target,
                    }
                )

    details = pd.DataFrame(rows, columns=_DETAIL_COLUMNS)
    group_scores = (
        details.groupby("atom_group_id", sort=False)
        .agg(nll=("nll", "mean"), correct=("correct", "all"))
        .reset_index()
        if not details.empty
        else pd.DataFrame(columns=("atom_group_id", "nll", "correct"))
    )
    summary: dict[str, object] = {
        "records_requested": int(len(texts)),
        "records_scored": int(details["sample_id"].nunique()),
        "records_skipped": int(sum(skipped.values())),
        "skip_reasons": dict(sorted(skipped.items())),
        "mask_probability": float(mask_probability),
        "mask_seed": int(seed),
        "masked_atom_groups": int(len(group_scores)),
        "supervised_tokens": int(len(details)),
        "token_mean_nll": (
            float(details["nll"].mean()) if not details.empty else np.nan
        ),
        "token_accuracy": (
            float(details["correct"].mean()) if not details.empty else np.nan
        ),
        "atom_group_mean_nll": (
            float(group_scores["nll"].mean()) if not group_scores.empty else np.nan
        ),
        "atom_group_accuracy": (
            float(group_scores["correct"].mean()) if not group_scores.empty else np.nan
        ),
    }
    return details, summary


def evaluate_molecular_mlm_retention(*args: Any, **kwargs: Any):
    """Backward-compatible name for molecular retention evaluation."""

    return evaluate_uniform_atom_group_mlm(*args, **kwargs)


__all__ = [
    "evaluate_molecular_mlm_retention",
    "evaluate_uniform_atom_group_mlm",
]
