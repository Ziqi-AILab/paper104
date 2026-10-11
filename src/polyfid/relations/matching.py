"""Observed-structure near-negative matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .schema import StructureView, stereo_agnostic_graphs_differ


DEFAULT_PROFILE_WEIGHTS: Mapping[str, float] = {
    "atom_count": 1.0,
    "hetero_atom_count": 1.0,
    "ring_count": 0.75,
    "molecular_weight": 1.0,
    "attachment_span": 0.75,
    "formula_mismatch": 0.5,
}


@dataclass(frozen=True)
class NearNegativeMatch:
    """A real corpus structure matched to an anchor by simple descriptors."""

    view: StructureView
    distance: float
    component_differences: Mapping[str, float]


def profile_differences(
    anchor: StructureView, candidate: StructureView
) -> dict[str, float]:
    """Return interpretable, scale-normalized profile differences."""

    if anchor.profile is None or candidate.profile is None:
        raise ValueError("both structures need an eligible descriptor profile")
    left, right = anchor.profile, candidate.profile

    def relative(a: float, b: float) -> float:
        return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1.0)

    return {
        "atom_count": relative(left.atom_count, right.atom_count),
        "hetero_atom_count": relative(
            left.hetero_atom_count, right.hetero_atom_count
        ),
        "ring_count": relative(left.ring_count, right.ring_count),
        "molecular_weight": relative(left.molecular_weight, right.molecular_weight),
        "attachment_span": relative(left.attachment_span, right.attachment_span),
        "formula_mismatch": float(left.formula != right.formula),
    }


def profile_distance(
    anchor: StructureView,
    candidate: StructureView,
    *,
    weights: Mapping[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    differences = profile_differences(anchor, candidate)
    active_weights = dict(DEFAULT_PROFILE_WEIGHTS if weights is None else weights)
    total_weight = sum(abs(float(value)) for value in active_weights.values()) or 1.0
    distance = sum(
        float(active_weights.get(name, 0.0)) * value
        for name, value in differences.items()
    ) / total_weight
    return float(distance), differences


def match_true_structure_negatives(
    anchor: StructureView,
    candidates: Sequence[StructureView],
    *,
    max_matches: int = 1,
    require_same_atom_count: bool = True,
    require_same_formula: bool = False,
    max_token_length_difference: int | None = None,
    token_length: Callable[[StructureView], int] | None = None,
    candidate_filter: Callable[[StructureView, StructureView], bool] | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[NearNegativeMatch, ...]:
    """Select descriptor-near negatives from actual corpus records.

    Synthetic endpoint edits are intentionally excluded here.  Callers can
    constrain family, source, or split through ``candidate_filter``.
    """

    if max_matches < 0:
        raise ValueError("max_matches must be non-negative")
    if max_token_length_difference is not None and token_length is None:
        raise ValueError(
            "token_length is required when a token-length difference is requested"
        )
    if not anchor.eligible or anchor.profile is None:
        return ()
    scored: list[tuple[float, str, str, StructureView, dict[str, float]]] = []
    for candidate in candidates:
        if not candidate.eligible or candidate.profile is None:
            continue
        if candidate.source != "observed":
            continue
        if candidate.sample_id == anchor.sample_id:
            continue
        if candidate.canonical_identity == anchor.canonical_identity:
            continue
        if not stereo_agnostic_graphs_differ(
            anchor.canonical_identity, candidate.canonical_identity
        ):
            continue
        if (
            require_same_atom_count
            and candidate.profile.atom_count != anchor.profile.atom_count
        ):
            continue
        if require_same_formula and candidate.profile.formula != anchor.profile.formula:
            continue
        if max_token_length_difference is not None and token_length is not None:
            if abs(token_length(anchor) - token_length(candidate)) > int(
                max_token_length_difference
            ):
                continue
        if candidate_filter is not None and not candidate_filter(anchor, candidate):
            continue
        distance, differences = profile_distance(
            anchor, candidate, weights=weights
        )
        scored.append(
            (
                distance,
                candidate.sample_id,
                candidate.psmiles,
                candidate,
                differences,
            )
        )
    scored.sort(key=lambda item: item[:3])
    matches: list[NearNegativeMatch] = []
    for distance, _, _, candidate, differences in scored[:max_matches]:
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "matched_anchor_id": anchor.sample_id,
                "profile_distance": distance,
                "component_differences": differences,
            }
        )
        view = candidate.as_relation(
            "matched_real_periodic_topology", source="observed", metadata=metadata
        )
        matches.append(
            NearNegativeMatch(
                view=view,
                distance=distance,
                component_differences=differences,
            )
        )
    return tuple(matches)


def match_true_structure_negative(
    anchor: StructureView,
    candidates: Sequence[StructureView],
    **kwargs: object,
) -> NearNegativeMatch | None:
    """Return the nearest observed negative, or ``None`` when none is available."""

    matches = match_true_structure_negatives(
        anchor, candidates, max_matches=1, **kwargs
    )
    return matches[0] if matches else None


__all__ = [
    "DEFAULT_PROFILE_WEIGHTS",
    "NearNegativeMatch",
    "match_true_structure_negative",
    "match_true_structure_negatives",
    "profile_differences",
    "profile_distance",
]
