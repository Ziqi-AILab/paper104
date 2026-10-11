"""Verified relation construction for PolyFiD pretraining."""

from .builders import (
    assemble_triplets,
    build_positive_views,
    build_supplemental_negative_views,
    make_anchor_view,
)
from .diagnostics import (
    EligibilitySummary,
    RelationEligibility,
    TripletSummary,
    diagnose_relation_sample,
    summarize_eligibility,
    summarize_triplets,
)
from .matching import (
    DEFAULT_PROFILE_WEIGHTS,
    NearNegativeMatch,
    match_true_structure_negative,
    match_true_structure_negatives,
    profile_differences,
    profile_distance,
)
from .schema import (
    PropertyTarget,
    RelationClass,
    RelationKind,
    RelationTriplet,
    RepresentationTarget,
    StructureView,
    TwoCellIdentityRelation,
    TypedRelationRecord,
    TripletEligibility,
    ViewSource,
    stereo_agnostic_graphs_differ,
    stereo_agnostic_two_cell_identity,
)

__all__ = [
    "DEFAULT_PROFILE_WEIGHTS",
    "EligibilitySummary",
    "NearNegativeMatch",
    "PropertyTarget",
    "RelationEligibility",
    "RelationClass",
    "RelationKind",
    "RelationTriplet",
    "RepresentationTarget",
    "StructureView",
    "TwoCellIdentityRelation",
    "TypedRelationRecord",
    "TripletEligibility",
    "TripletSummary",
    "ViewSource",
    "assemble_triplets",
    "build_positive_views",
    "build_supplemental_negative_views",
    "diagnose_relation_sample",
    "make_anchor_view",
    "match_true_structure_negative",
    "match_true_structure_negatives",
    "profile_differences",
    "profile_distance",
    "summarize_eligibility",
    "summarize_triplets",
    "stereo_agnostic_graphs_differ",
    "stereo_agnostic_two_cell_identity",
]
