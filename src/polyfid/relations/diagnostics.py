"""Eligibility and composition statistics for relation datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .schema import (
    RelationTriplet,
    StructureView,
    stereo_agnostic_graphs_differ,
)


@dataclass(frozen=True)
class RelationEligibility:
    sample_id: str
    structure_eligible: bool
    main_relation_eligible: bool
    diagnostic: str
    positive_count: int
    true_negative_count: int
    endpoint_relocation_count: int
    closure_rewiring_count: int

    @property
    def supplemental_relation_eligible(self) -> bool:
        return self.structure_eligible and self.positive_count > 0 and (
            self.endpoint_relocation_count + self.closure_rewiring_count > 0
        )

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EligibilitySummary:
    total_samples: int
    structure_eligible_samples: int
    main_relation_eligible_samples: int
    supplemental_relation_eligible_samples: int
    structure_eligible_fraction: float
    main_relation_eligible_fraction: float
    diagnostic_counts: dict[str, int]
    positive_views: int
    true_structure_negatives: int
    endpoint_relocations: int
    closure_rewirings: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TripletSummary:
    total_triplets: int
    eligible_triplets: int
    negative_relation_counts: dict[str, int]
    diagnostic_counts: dict[str, int]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_relation_sample(
    anchor: StructureView,
    positives: Sequence[StructureView],
    negatives: Sequence[StructureView],
) -> RelationEligibility:
    """Report whether a sample supports the main observed-negative objective."""

    valid_positives = [
        view
        for view in positives
        if view.eligible and view.canonical_identity == anchor.canonical_identity
    ]
    true_negatives = [
        view
        for view in negatives
        if view.eligible
        and view.relation == "matched_real_periodic_topology"
        and stereo_agnostic_graphs_differ(
            anchor.canonical_identity, view.canonical_identity
        )
    ]
    endpoint_relocations = [
        view
        for view in negatives
        if view.eligible
        and view.relation == "endpoint_relocation"
        and stereo_agnostic_graphs_differ(
            anchor.canonical_identity, view.canonical_identity
        )
    ]
    closure_rewirings = [
        view
        for view in negatives
        if view.eligible
        and view.relation == "closure_rewiring"
        and stereo_agnostic_graphs_differ(
            anchor.canonical_identity, view.canonical_identity
        )
    ]

    if not anchor.eligible:
        diagnostic = anchor.diagnostic
    elif not valid_positives:
        diagnostic = "no_verified_positive"
    elif not true_negatives:
        diagnostic = "no_true_structure_negative"
    else:
        diagnostic = "eligible"
    return RelationEligibility(
        sample_id=anchor.sample_id,
        structure_eligible=anchor.eligible,
        main_relation_eligible=diagnostic == "eligible",
        diagnostic=diagnostic,
        positive_count=len(valid_positives),
        true_negative_count=len(true_negatives),
        endpoint_relocation_count=len(endpoint_relocations),
        closure_rewiring_count=len(closure_rewirings),
    )


def summarize_eligibility(
    records: Iterable[RelationEligibility],
) -> EligibilitySummary:
    rows = tuple(records)
    total = len(rows)
    denominator = max(total, 1)
    structure_eligible = sum(row.structure_eligible for row in rows)
    main_eligible = sum(row.main_relation_eligible for row in rows)
    supplemental_eligible = sum(row.supplemental_relation_eligible for row in rows)
    return EligibilitySummary(
        total_samples=total,
        structure_eligible_samples=structure_eligible,
        main_relation_eligible_samples=main_eligible,
        supplemental_relation_eligible_samples=supplemental_eligible,
        structure_eligible_fraction=structure_eligible / denominator,
        main_relation_eligible_fraction=main_eligible / denominator,
        diagnostic_counts=dict(sorted(Counter(row.diagnostic for row in rows).items())),
        positive_views=sum(row.positive_count for row in rows),
        true_structure_negatives=sum(row.true_negative_count for row in rows),
        endpoint_relocations=sum(row.endpoint_relocation_count for row in rows),
        closure_rewirings=sum(row.closure_rewiring_count for row in rows),
    )


def summarize_triplets(triplets: Iterable[RelationTriplet]) -> TripletSummary:
    rows = tuple(triplets)
    diagnostics: Counter[str] = Counter()
    negative_relations: Counter[str] = Counter()
    eligible = 0
    for triplet in rows:
        assessment = triplet.assess()
        if assessment.eligible:
            eligible += 1
            negative_relations[triplet.negative.relation] += 1
        else:
            diagnostics.update(assessment.diagnostics)
    return TripletSummary(
        total_triplets=len(rows),
        eligible_triplets=eligible,
        negative_relation_counts=dict(sorted(negative_relations.items())),
        diagnostic_counts=dict(sorted(diagnostics.items())),
    )


__all__ = [
    "EligibilitySummary",
    "RelationEligibility",
    "TripletSummary",
    "diagnose_relation_sample",
    "summarize_eligibility",
    "summarize_triplets",
]
