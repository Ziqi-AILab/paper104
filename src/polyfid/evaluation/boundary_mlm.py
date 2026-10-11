"""Atom-group MLM evaluation by attachment distance."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from polyfid.chemistry import (
    StructureError,
    TokenAlignmentError,
    TokenLengthError,
    align_with_fast_tokenizer,
    attachment_distances_in_text_order,
)


@dataclass(frozen=True)
class _MaskCase:
    sample_id: str
    group_id: str
    features: dict[str, list[int]]
    token_positions: tuple[int, ...]
    target_ids: tuple[int, ...]
    boundary_distance: int | None
    boundary_stratum: str
    interior_region: bool


_TOKEN_COLUMNS = [
    "sample_id",
    "group_id",
    "group_token_count",
    "token_position",
    "target_token_id",
    "predicted_token_id",
    "boundary_distance",
    "boundary_stratum",
    "interior_region",
    "nll",
    "correct",
]


def _stratum(distance: int | None, scope: str) -> tuple[str, bool]:
    if scope == "molecule":
        return "interior", True
    if distance is None:
        return "interior", True
    if distance <= 2:
        return f"distance_{distance}", False
    return "distance_3_plus", True


def _mask_cases(
    sample_ids: Sequence[str],
    texts: Sequence[str],
    tokenizer: Any,
    *,
    scope: str,
    max_length: int,
    skipped: Counter[str],
) -> Iterator[_MaskCase]:
    allowed_features = {"input_ids", "attention_mask", "token_type_ids"}
    for sample_id, text in zip(sample_ids, texts):
        try:
            encoded, alignment = align_with_fast_tokenizer(
                str(text), tokenizer, max_length=max_length
            )
            if scope == "repeat":
                distances = attachment_distances_in_text_order(str(text))
            else:
                distances = (None,) * len(alignment.lexemes)
            groups = alignment.masking_groups(distances)
        except TokenLengthError:
            skipped["over_max_length"] += 1
            continue
        except (StructureError, TokenAlignmentError):
            skipped["structure_or_alignment"] += 1
            continue

        clean_ids = [int(value) for value in encoded["input_ids"]]
        base = {
            key: [int(value) for value in encoded[key]]
            for key in allowed_features
            if key in encoded
        }
        for group_index, group in enumerate(groups):
            if not group.eligible:
                continue
            positions = tuple(int(value) for value in group.token_indices)
            corrupted = clean_ids.copy()
            for position in positions:
                corrupted[position] = int(tokenizer.mask_token_id)
            features = {key: values.copy() for key, values in base.items()}
            features["input_ids"] = corrupted
            boundary_stratum, interior = _stratum(group.boundary_distance, scope)
            yield _MaskCase(
                sample_id=str(sample_id),
                group_id=f"{sample_id}:atom_group:{group_index}",
                features=features,
                token_positions=positions,
                target_ids=tuple(clean_ids[position] for position in positions),
                boundary_distance=group.boundary_distance,
                boundary_stratum=boundary_stratum,
                interior_region=interior,
            )


def summarise_boundary_tokens(
    details: pd.DataFrame, *, scope: str = "repeat"
) -> pd.DataFrame:
    """Summarise local MLM with token- and atom-group-weighted estimates.

    For repeat units, ``distance_3_plus`` and ``interior`` select the same atom
    groups.  The former is retained for compatibility but is labelled as an
    alias, so it is not interpreted as a second independent result.
    """

    rows: list[dict[str, object]] = []
    for stratum in (
        "distance_0",
        "distance_1",
        "distance_2",
        "distance_3_plus",
        "interior",
    ):
        if stratum == "interior":
            cell = details.loc[details["interior_region"]]
        else:
            cell = details.loc[details["boundary_stratum"].eq(stratum)]
        atom_groups = (
            cell.groupby("group_id", sort=False)
            .agg(group_mean_nll=("nll", "mean"), group_accuracy=("correct", "mean"))
            .reset_index(drop=True)
        )
        is_alias = scope == "repeat" and stratum == "distance_3_plus"
        rows.append(
            {
                "stratum": stratum,
                "scope": scope,
                "alias_of": "interior" if is_alias else None,
                "independent_evidence": not is_alias,
                "n_tokens": int(len(cell)),
                "n_groups": int(cell["group_id"].nunique()),
                "n_samples": int(cell["sample_id"].nunique()),
                "mean_nll": float(cell["nll"].mean()) if len(cell) else np.nan,
                "accuracy": float(cell["correct"].mean()) if len(cell) else np.nan,
                "atom_group_mean_nll": (
                    float(atom_groups["group_mean_nll"].mean())
                    if len(atom_groups)
                    else np.nan
                ),
                "atom_group_accuracy": (
                    float(atom_groups["group_accuracy"].mean())
                    if len(atom_groups)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_atom_group_mlm(
    model: Any,
    tokenizer: Any,
    sample_ids: Sequence[str],
    texts: Sequence[str],
    *,
    scope: str = "repeat",
    max_length: int = 256,
    batch_size: int = 128,
    device: str = "cuda",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Mask each real atom group once and score every target token.

    Attachment markers are ineligible masking groups.  A bracket atom split
    across tokenizer pieces is therefore corrupted and scored as one group.
    """

    import torch

    if scope not in {"repeat", "molecule"}:
        raise ValueError("scope must be 'repeat' or 'molecule'")
    if len(sample_ids) != len(texts):
        raise ValueError("sample IDs and texts must align")
    if tokenizer.mask_token_id is None:
        raise ValueError("the tokenizer must define a mask token")
    skipped: Counter[str] = Counter()
    cases = _mask_cases(
        sample_ids,
        texts,
        tokenizer,
        scope=scope,
        max_length=max_length,
        skipped=skipped,
    )
    rows: list[dict[str, object]] = []
    batch: list[_MaskCase] = []
    model.eval()

    def score_batch(values: list[_MaskCase]) -> None:
        if not values:
            return
        encoded = tokenizer.pad(
            [case.features for case in values], padding=True, return_tensors="pt"
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded, return_dict=True).logits
            for batch_index, case in enumerate(values):
                positions = torch.as_tensor(
                    case.token_positions, dtype=torch.long, device=logits.device
                )
                targets = torch.as_tensor(
                    case.target_ids, dtype=torch.long, device=logits.device
                )
                selected = logits[batch_index, positions]
                losses = -torch.log_softmax(selected.float(), dim=-1).gather(
                    1, targets[:, None]
                )[:, 0]
                predictions = selected.argmax(dim=-1)
                for token_index, position in enumerate(case.token_positions):
                    predicted = int(predictions[token_index].item())
                    target = int(case.target_ids[token_index])
                    rows.append(
                        {
                            "sample_id": case.sample_id,
                            "group_id": case.group_id,
                            "group_token_count": len(case.token_positions),
                            "token_position": int(position),
                            "target_token_id": target,
                            "predicted_token_id": predicted,
                            "boundary_distance": case.boundary_distance,
                            "boundary_stratum": case.boundary_stratum,
                            "interior_region": case.interior_region,
                            "nll": float(losses[token_index].item()),
                            "correct": predicted == target,
                        }
                    )

    for case in cases:
        batch.append(case)
        if len(batch) == batch_size:
            score_batch(batch)
            batch = []
    score_batch(batch)

    details = pd.DataFrame(rows, columns=_TOKEN_COLUMNS)
    summary = summarise_boundary_tokens(details, scope=scope)
    metadata: dict[str, object] = {
        "scope": scope,
        "records_requested": int(len(texts)),
        "records_skipped": int(sum(skipped.values())),
        "skip_reasons": dict(sorted(skipped.items())),
        "masked_atom_groups": int(details["group_id"].nunique()),
        "target_tokens": int(len(details)),
        "marker_is_prediction_target": False,
        "interior_definition": (
            "all molecular atoms"
            if scope == "molecule"
            else "attachment distance >= 3; the interior row therefore overlaps distance_3_plus"
        ),
        "independent_strata": (
            ["interior"]
            if scope == "molecule"
            else ["distance_0", "distance_1", "distance_2", "interior"]
        ),
        "stratum_aliases": (
            {"distance_3_plus": "interior"} if scope == "repeat" else {}
        ),
        "summary_weighting": {
            "mean_nll_and_accuracy": "token weighted",
            "atom_group_mean_nll_and_accuracy": "equal weight per masked atom group",
        },
    }
    return details, summary, metadata


__all__ = ["evaluate_atom_group_mlm", "summarise_boundary_tokens"]
