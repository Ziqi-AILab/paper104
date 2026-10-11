"""Data structures for verified polymer relation examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from typing import Any, Literal, Mapping

from rdkit import Chem, rdBase

from polyfid.chemistry import (
    StructureProfile,
    StructureError,
    parse_repeat_unit,
    repeat_profile,
)


RelationKind = Literal[
    "anchor",
    "randomized_traversal",
    "alternative_periodic_cut",
    "reverse_when_non_noop",
    "matched_real_periodic_topology",
    "endpoint_relocation",
    "closure_rewiring",
]
ViewSource = Literal["observed", "verified_transformation", "synthetic_control"]
RelationClass = Literal[
    "equivalence_preserving",
    "topology_changing",
    "polymer_semantics_inapplicable",
    "unknown_or_ineligible",
]
TwoCellIdentityRelation = Literal["same", "different", "unknown"]
RepresentationTarget = Literal["close", "separate", "descriptive_only", "none"]
PropertyTarget = Literal["stable_structure_only", "unconstrained", "none"]


@lru_cache(maxsize=500_000)
def stereo_agnostic_two_cell_identity(identity: str | None) -> str | None:
    """Return a canonical two-cell graph identity without stereochemistry.

    PI1M contains records that differ only by specified versus unspecified
    alkene stereochemistry.  Those records are useful stereochemical variants,
    but they are not evidence of a different two-cell structural graph for the
    topology objective used in Paper104.
    """

    if identity is None or not str(identity).strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(identity))
    if molecule is None:
        # Some PI1M two-cell aromatic serializations cannot be kekulized even
        # though their remaining chemistry checks are valid.  Retry without
        # kekulization, but retain every other RDKit sanitization operation.
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(str(identity), sanitize=False)
            if molecule is not None:
                try:
                    Chem.SanitizeMol(
                        molecule,
                        sanitizeOps=(
                            Chem.SanitizeFlags.SANITIZE_ALL
                            ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                        ),
                    )
                except (ValueError, RuntimeError):
                    molecule = None
    if molecule is None:
        return None
    Chem.RemoveStereochemistry(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def stereo_agnostic_graphs_differ(
    left_identity: str | None,
    right_identity: str | None,
) -> bool:
    """Whether two parseable identities remain different without stereo."""

    left = stereo_agnostic_two_cell_identity(left_identity)
    right = stereo_agnostic_two_cell_identity(right_identity)
    return left is not None and right is not None and left != right


@dataclass(frozen=True)
class TypedRelationRecord:
    """Analysis-side relation contract; not an input to pretraining."""

    relation_id: str
    anchor_id: str
    paired_id: str | None
    split: str
    split_group: str
    relation_family: str
    relation_class: RelationClass
    two_cell_identity_relation: TwoCellIdentityRelation
    changed_fields: tuple[str, ...]
    parse_status: str
    polymer_semantic_status: str
    representation_target: RepresentationTarget
    property_target: PropertyTarget
    training_eligible: bool
    evaluation_eligible: bool
    evidence_rule: str
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        if self.paired_id is None and self.relation_class != "unknown_or_ineligible":
            raise ValueError("only unknown or ineligible relations may omit paired_id")
        if self.relation_class == "equivalence_preserving":
            if self.two_cell_identity_relation != "same":
                raise ValueError("equivalence-preserving relations require the same identity")
        elif self.relation_class == "topology_changing":
            if self.two_cell_identity_relation != "different":
                raise ValueError("topology-changing relations require different graph identities")
        elif self.training_eligible:
            raise ValueError(
                "semantic applicability controls and ineligible rows cannot enter training"
            )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["changed_fields"] = list(self.changed_fields)
        return record


@dataclass(frozen=True)
class StructureView:
    """One observed or transformed serialization of a polymer structure."""

    sample_id: str
    psmiles: str
    relation: RelationKind
    source: ViewSource
    # Two-cell periodic-supercell identity used for all relation decisions.
    canonical_identity: str | None
    # Explicit fields keep the scientific identity separate from the legacy
    # one-cell closure retained for readable backward compatibility.
    periodic_supercell_smiles: str | None
    closed_smiles: str | None
    eligible: bool
    diagnostic: str
    profile: StructureProfile | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_psmiles(
        cls,
        sample_id: str,
        psmiles: str,
        *,
        relation: RelationKind = "anchor",
        source: ViewSource = "observed",
        metadata: Mapping[str, Any] | None = None,
        include_profile: bool = True,
    ) -> "StructureView":
        try:
            unit = parse_repeat_unit(psmiles)
        except StructureError as exc:
            return cls(
                sample_id=str(sample_id),
                psmiles=str(psmiles).strip(),
                relation=relation,
                source=source,
                canonical_identity=None,
                periodic_supercell_smiles=None,
                closed_smiles=None,
                eligible=False,
                diagnostic=exc.code,
                metadata=dict(metadata or {}),
            )
        return cls(
            sample_id=str(sample_id),
            psmiles=str(psmiles).strip(),
            relation=relation,
            source=source,
            canonical_identity=unit.canonical_identity,
            periodic_supercell_smiles=unit.periodic_supercell_smiles,
            closed_smiles=unit.closed_smiles,
            eligible=True,
            diagnostic="eligible",
            profile=repeat_profile(unit) if include_profile else None,
            metadata={
                "canonical_psmiles": unit.canonical_psmiles,
                "periodic_supercell_smiles": unit.periodic_supercell_smiles,
                "closed_smiles": unit.closed_smiles,
                **dict(metadata or {}),
            },
        )

    @property
    def repeat_equivalence_id(self) -> str | None:
        return self.canonical_identity

    @property
    def periodic_identity(self) -> str | None:
        return self.canonical_identity

    @property
    def view_type(self) -> RelationKind:
        return self.relation

    def as_relation(
        self,
        relation: RelationKind,
        *,
        source: ViewSource | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "StructureView":
        return replace(
            self,
            relation=relation,
            source=source or self.source,
            metadata=dict(self.metadata if metadata is None else metadata),
        )

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["metadata"] = dict(self.metadata)
        return record


@dataclass(frozen=True)
class TripletEligibility:
    eligible: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class RelationTriplet:
    """Anchor, equivalent positive, and structurally distinct negative."""

    anchor: StructureView
    positive: StructureView
    negative: StructureView

    def assess(self) -> TripletEligibility:
        diagnostics: list[str] = []
        for role, view in (
            ("anchor", self.anchor),
            ("positive", self.positive),
            ("negative", self.negative),
        ):
            if not view.eligible:
                diagnostics.append(f"{role}:{view.diagnostic}")
        if not diagnostics:
            if (
                self.positive.repeat_equivalence_id
                != self.anchor.repeat_equivalence_id
            ):
                diagnostics.append("positive_identity_mismatch")
            if (
                self.negative.repeat_equivalence_id
                == self.anchor.repeat_equivalence_id
            ):
                diagnostics.append("negative_identity_match")
            elif not stereo_agnostic_graphs_differ(
                self.anchor.repeat_equivalence_id,
                self.negative.repeat_equivalence_id,
            ):
                diagnostics.append("negative_stereo_agnostic_identity_match")
        return TripletEligibility(eligible=not diagnostics, diagnostics=tuple(diagnostics))

    @property
    def eligible(self) -> bool:
        return self.assess().eligible

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return self.assess().diagnostics

    @property
    def triplet_id(self) -> str:
        return "::".join(
            (
                self.anchor.sample_id,
                self.positive.sample_id,
                self.negative.sample_id,
            )
        )

    def to_record(self) -> dict[str, Any]:
        assessment = self.assess()
        return {
            "triplet_id": self.triplet_id,
            "anchor": self.anchor.to_record(),
            "positive": self.positive.to_record(),
            "negative": self.negative.to_record(),
            "eligible": assessment.eligible,
            "diagnostics": assessment.diagnostics,
        }


__all__ = [
    "PropertyTarget",
    "RelationClass",
    "RelationKind",
    "RelationTriplet",
    "RepresentationTarget",
    "StructureView",
    "TwoCellIdentityRelation",
    "TypedRelationRecord",
    "TripletEligibility",
    "ViewSource",
    "stereo_agnostic_graphs_differ",
    "stereo_agnostic_two_cell_identity",
]
