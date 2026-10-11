"""Chemical structure primitives for linear polymer repeat units.

PolyFiD defines periodic identity with a two-cell supercell.  Each supercell
contains two copies of the attachment-free repeat core and connects them with
the two translated periodic bonds (R0--L1 and R1--L0).  Its canonical isomeric
SMILES is a readable identity and, unlike a one-cell closure, can represent a
periodic bond even when the two attachment atoms are already bonded inside the
reported repeat window (for example ``*CC*``).

The older one-cell closed SMILES remains available as a compatibility and
display field, but it does not define repeat equivalence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ModuleNotFoundError:  # pragma: no cover - analysis-only environments
    Chem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]


class ChemistryDependencyError(RuntimeError):
    """RDKit is required for a graph operation."""


class StructureError(ValueError):
    """A structure does not satisfy the linear-repeat contract."""

    def __init__(self, message: str, *, code: str = "invalid_structure") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RepeatEligibility:
    """Result of assessing whether a string is a linear dual-attachment unit."""

    psmiles: str
    eligible: bool
    reason: str
    dummy_atom_count: int | None = None
    atom_count: int | None = None


@dataclass(frozen=True)
class RepeatUnit:
    """Canonical representation of an eligible linear repeat unit."""

    original_psmiles: str
    canonical_psmiles: str
    canonical_identity: str
    periodic_supercell_smiles: str
    closed_smiles: str
    formula: str
    atom_count: int
    dummy_atom_indices: tuple[int, int]
    attachment_atom_indices: tuple[int, int]
    attachment_bond_type: str
    attachment_distances: tuple[int | None, ...]
    boundary_classes: tuple[str, ...]

    @property
    def canonical_structure_id(self) -> str:
        """Serialization identity, retained separately from periodic identity."""

        return self.canonical_psmiles

    @property
    def repeat_equivalence_id(self) -> str:
        return self.canonical_identity

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClosedRepeat:
    """Closed-repeat graph plus the mapping from the open repeat unit."""

    canonical_smiles: str
    canonical_identity: str
    attachment_atom_indices: tuple[int, int]
    attachment_bond_type: str
    closure_added: bool
    source_to_closed: tuple[tuple[int, int], ...]
    molecule: Any = field(repr=False, compare=False)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("molecule", None)
        return record


@dataclass(frozen=True)
class PeriodicSupercell:
    """Two translated repeat cells joined by paired periodic bonds.

    Atom indices in ``cell_atom_indices`` and ``cross_cell_edges`` refer to the
    stored RDKit molecule.  ``bond_orbits`` contains the pair of translated
    bonds associated with every possible one-cell boundary: internal bonds in
    cell 0/1 followed by the periodic cross-bond pair.
    """

    canonical_smiles: str
    canonical_identity: str
    unit_atom_count: int
    cell_atom_indices: tuple[tuple[int, ...], tuple[int, ...]]
    attachment_atom_indices: tuple[tuple[int, int], tuple[int, int]]
    cross_cell_edges: tuple[tuple[int, int], tuple[int, int]]
    bond_orbits: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    attachment_bond_type: str
    molecule: Any = field(repr=False, compare=False)

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record.pop("molecule", None)
        return record


@dataclass(frozen=True)
class StructureProfile:
    """Small interpretable descriptor set used for near-negative matching."""

    formula: str
    atom_count: int
    heavy_atom_count: int
    hetero_atom_count: int
    ring_count: int
    molecular_weight: float
    attachment_span: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def rdkit_available() -> bool:
    return Chem is not None


def require_rdkit() -> None:
    if Chem is None:
        raise ChemistryDependencyError("RDKit is required for chemistry operations")


def _normalise_molecule(molecule: Any) -> Any:
    require_rdkit()
    normalised = Chem.Mol(molecule)
    for atom in normalised.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return normalised


def _parse_smiles(text: str) -> Any:
    require_rdkit()
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise StructureError("RDKit could not parse the structure", code="invalid_smiles")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise StructureError(
            "repeat unit must contain one connected component", code="disconnected"
        )
    return _normalise_molecule(molecule)


def canonicalize_smiles(smiles: str) -> str:
    """Return canonical isomeric SMILES with atom maps removed."""

    value = str(smiles).strip()
    if not value:
        raise StructureError("empty SMILES", code="empty")
    molecule = _parse_smiles(value)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def dummy_and_attachment_indices(
    molecule: Any,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Locate the two terminal dummy atoms and their real neighbours."""

    dummies = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        raise StructureError(
            f"expected two attachment markers, found {len(dummies)}",
            code="attachment_count",
        )
    if any(atom.GetDegree() != 1 for atom in dummies):
        raise StructureError(
            "both attachment markers must be terminal", code="nonterminal_attachment"
        )
    dummy_indices = tuple(sorted(atom.GetIdx() for atom in dummies))
    attachments = tuple(
        molecule.GetAtomWithIdx(index).GetNeighbors()[0].GetIdx()
        for index in dummy_indices
    )
    if any(molecule.GetAtomWithIdx(index).GetAtomicNum() == 0 for index in attachments):
        raise StructureError(
            "attachment markers must connect to real atoms", code="dummy_connection"
        )
    if attachments[0] == attachments[1]:
        raise StructureError(
            "attachment markers must connect to distinct atoms",
            code="shared_attachment_atom",
        )
    return dummy_indices, attachments


def _close_repeat_with_mapping(
    molecule: Any,
) -> tuple[Any, dict[int, int], tuple[int, int], Any, bool]:
    source = _normalise_molecule(molecule)
    dummies, attachments = dummy_and_attachment_indices(source)
    terminal_bonds = [
        source.GetBondBetweenAtoms(dummy, attachment)
        for dummy, attachment in zip(dummies, attachments)
    ]
    terminal_types = [bond.GetBondType() for bond in terminal_bonds if bond is not None]
    if len(terminal_types) != 2 or terminal_types[0] != terminal_types[1]:
        raise StructureError(
            "the two attachment bonds must have the same bond type",
            code="attachment_bond_mismatch",
        )

    removed = set(dummies)
    source_to_closed: dict[int, int] = {}
    next_index = 0
    for old_index in range(source.GetNumAtoms()):
        if old_index not in removed:
            source_to_closed[old_index] = next_index
            next_index += 1

    editable = Chem.RWMol(source)
    for dummy_index in sorted(dummies, reverse=True):
        editable.RemoveAtom(dummy_index)
    closed_attachments = tuple(source_to_closed[index] for index in attachments)
    closure_added = editable.GetBondBetweenAtoms(*closed_attachments) is None
    if closure_added:
        editable.AddBond(*closed_attachments, terminal_types[0])
    closed = editable.GetMol()
    try:
        Chem.SanitizeMol(closed)
    except (ValueError, RuntimeError) as exc:
        raise StructureError(
            "closed-repeat graph is chemically invalid", code="closure_invalid"
        ) from exc
    Chem.AssignStereochemistry(closed, cleanIt=True, force=True)
    return (
        closed,
        source_to_closed,
        closed_attachments,
        terminal_types[0],
        closure_added,
    )


def _build_periodic_supercell_from_source(molecule: Any) -> PeriodicSupercell:
    """Construct the explicit two-cell periodic graph from one open repeat.

    The two dummy atoms are removed before the core is duplicated.  If the
    canonical open unit has attachment neighbours ``L`` and ``R``, periodic
    translation is represented by the two bonds ``R0--L1`` and ``R1--L0``.
    This construction never relies on adding a second edge to a one-cell
    simple graph.
    """

    source = _normalise_molecule(molecule)
    dummies, attachments = dummy_and_attachment_indices(source)
    terminal_bonds = [
        source.GetBondBetweenAtoms(dummy, attachment)
        for dummy, attachment in zip(dummies, attachments)
    ]
    if any(bond is None for bond in terminal_bonds):
        raise StructureError(
            "attachment marker is missing its terminal bond",
            code="missing_attachment_bond",
        )
    terminal_types = [bond.GetBondType() for bond in terminal_bonds]
    if terminal_types[0] != terminal_types[1]:
        raise StructureError(
            "the two attachment bonds must have the same bond type",
            code="attachment_bond_mismatch",
        )

    removed = set(dummies)
    source_to_core: dict[int, int] = {}
    next_index = 0
    for old_index in range(source.GetNumAtoms()):
        if old_index not in removed:
            source_to_core[old_index] = next_index
            next_index += 1

    editable_core = Chem.RWMol(source)
    for dummy_index in sorted(dummies, reverse=True):
        editable_core.RemoveAtom(dummy_index)
    core = editable_core.GetMol()
    # The isolated core can be an intentionally open aromatic fragment whose
    # valence is restored only after the translated periodic bonds are added.
    # Sanitizing it here would reject valid imide and conjugated repeat units;
    # the completed two-cell graph is sanitized below.
    unit_atom_count = core.GetNumAtoms()
    if unit_atom_count == 0:
        raise StructureError("repeat core is empty", code="empty_core")
    left, right = (source_to_core[index] for index in attachments)
    left_1, right_1 = left + unit_atom_count, right + unit_atom_count

    combined = Chem.RWMol(Chem.CombineMols(core, core))
    bond_type = terminal_types[0]
    combined.AddBond(right, left_1, bond_type)
    combined.AddBond(right_1, left, bond_type)
    supercell = combined.GetMol()
    try:
        Chem.SanitizeMol(supercell)
    except (ValueError, RuntimeError) as exc:
        raise StructureError(
            "two-cell periodic supercell is chemically invalid",
            code="periodic_supercell_invalid",
        ) from exc
    Chem.AssignStereochemistry(supercell, cleanIt=True, force=True)

    internal_orbits = []
    for bond in core.GetBonds():
        edge_0 = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        edge_1 = tuple(
            sorted(
                (
                    bond.GetBeginAtomIdx() + unit_atom_count,
                    bond.GetEndAtomIdx() + unit_atom_count,
                )
            )
        )
        internal_orbits.append((edge_0, edge_1))
    internal_orbits.sort(key=lambda pair: pair[0])
    cross_edges = (
        tuple(sorted((right, left_1))),
        tuple(sorted((right_1, left))),
    )
    canonical = Chem.MolToSmiles(
        supercell, canonical=True, isomericSmiles=True
    )
    return PeriodicSupercell(
        canonical_smiles=canonical,
        canonical_identity=canonical,
        unit_atom_count=unit_atom_count,
        cell_atom_indices=(
            tuple(range(unit_atom_count)),
            tuple(range(unit_atom_count, 2 * unit_atom_count)),
        ),
        attachment_atom_indices=((left, right), (left_1, right_1)),
        cross_cell_edges=cross_edges,
        bond_orbits=tuple([*internal_orbits, cross_edges]),
        attachment_bond_type=str(bond_type),
        molecule=supercell,
    )


def close_repeat_unit_with_mapping(
    molecule: Any,
) -> tuple[Any, dict[int, int], tuple[int, int], Any]:
    """Remove attachment markers and add the periodic closure bond."""

    closed, mapping, endpoints, bond_type, _ = _close_repeat_with_mapping(molecule)
    return closed, mapping, endpoints, bond_type


def close_repeat_unit(molecule: Any) -> Any:
    return close_repeat_unit_with_mapping(molecule)[0]


def build_closed_repeat(unit_or_text: RepeatUnit | str) -> ClosedRepeat:
    """Construct the legacy one-cell closed graph for display/compatibility."""

    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    source = _parse_smiles(unit.canonical_psmiles)
    closed, mapping, endpoints, bond_type, added = _close_repeat_with_mapping(source)
    identity = Chem.MolToSmiles(closed, canonical=True, isomericSmiles=True)
    return ClosedRepeat(
        canonical_smiles=identity,
        canonical_identity=identity,
        attachment_atom_indices=endpoints,
        attachment_bond_type=str(bond_type),
        closure_added=added,
        source_to_closed=tuple(sorted(mapping.items())),
        molecule=closed,
    )


def build_periodic_supercell(
    unit_or_text: RepeatUnit | str,
) -> PeriodicSupercell:
    """Build the explicit two-cell graph that defines periodic identity."""

    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    source = _parse_smiles(unit.canonical_psmiles)
    return _build_periodic_supercell_from_source(source)


def _attachment_context(
    molecule: Any,
    dummies: tuple[int, int],
    attachments: tuple[int, int],
) -> tuple[tuple[int | None, ...], tuple[str, ...]]:
    distance_matrix = Chem.GetDistanceMatrix(molecule)
    distances: list[int | None] = []
    classes: list[str] = []
    for atom in molecule.GetAtoms():
        index = atom.GetIdx()
        if index in dummies:
            distances.append(None)
            classes.append("attachment_marker")
            continue
        distance = min(int(distance_matrix[index, anchor]) for anchor in attachments)
        distances.append(distance)
        classes.append(f"boundary_{distance}" if distance <= 2 else "interior")
    return tuple(distances), tuple(classes)


def parse_repeat_unit(psmiles: str) -> RepeatUnit:
    """Parse and canonicalize a connected linear dual-attachment PSMILES."""

    value = str(psmiles).strip()
    if not value:
        raise StructureError("empty PSMILES", code="empty")
    source = _parse_smiles(value)
    dummy_and_attachment_indices(source)
    canonical = Chem.MolToSmiles(source, canonical=True, isomericSmiles=True)
    canonical_molecule = _parse_smiles(canonical)
    dummies, attachments = dummy_and_attachment_indices(canonical_molecule)
    closed, _, _, terminal_type, _ = _close_repeat_with_mapping(canonical_molecule)
    closed_smiles = Chem.MolToSmiles(closed, canonical=True, isomericSmiles=True)
    periodic_supercell = _build_periodic_supercell_from_source(canonical_molecule)
    distances, boundary_classes = _attachment_context(
        canonical_molecule, dummies, attachments
    )
    return RepeatUnit(
        original_psmiles=value,
        canonical_psmiles=canonical,
        canonical_identity=periodic_supercell.canonical_identity,
        periodic_supercell_smiles=periodic_supercell.canonical_smiles,
        closed_smiles=closed_smiles,
        formula=rdMolDescriptors.CalcMolFormula(closed),
        atom_count=closed.GetNumAtoms(),
        dummy_atom_indices=dummies,
        attachment_atom_indices=attachments,
        attachment_bond_type=str(terminal_type),
        attachment_distances=distances,
        boundary_classes=boundary_classes,
    )


def canonical_periodic_identity(psmiles: str) -> str:
    return parse_repeat_unit(psmiles).canonical_identity


def attachment_distances(psmiles: str) -> tuple[int | None, ...]:
    return parse_repeat_unit(psmiles).attachment_distances


def assess_repeat_eligibility(psmiles: str) -> RepeatEligibility:
    """Assess corpus eligibility without turning parse failures into exceptions."""

    value = str(psmiles).strip()
    require_rdkit()
    raw = Chem.MolFromSmiles(value) if value else None
    dummy_count = (
        sum(atom.GetAtomicNum() == 0 for atom in raw.GetAtoms()) if raw is not None else None
    )
    try:
        unit = parse_repeat_unit(value)
    except StructureError as exc:
        return RepeatEligibility(
            psmiles=value,
            eligible=False,
            reason=exc.code,
            dummy_atom_count=dummy_count,
            atom_count=None,
        )
    return RepeatEligibility(
        psmiles=value,
        eligible=True,
        reason="eligible",
        dummy_atom_count=2,
        atom_count=unit.atom_count,
    )


def repeat_profile(unit_or_text: RepeatUnit | str) -> StructureProfile:
    """Compute the descriptor profile used by observed-structure matching."""

    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    closed = build_closed_repeat(unit).molecule
    source = _parse_smiles(unit.canonical_psmiles)
    attachment_span = len(Chem.GetShortestPath(source, *unit.attachment_atom_indices)) - 1
    heavy_atoms = [atom for atom in closed.GetAtoms() if atom.GetAtomicNum() > 1]
    return StructureProfile(
        formula=unit.formula,
        atom_count=closed.GetNumAtoms(),
        heavy_atom_count=len(heavy_atoms),
        hetero_atom_count=sum(atom.GetAtomicNum() not in (1, 6) for atom in heavy_atoms),
        ring_count=int(rdMolDescriptors.CalcNumRings(closed)),
        molecular_weight=float(Descriptors.MolWt(closed)),
        attachment_span=int(attachment_span),
    )


__all__ = [
    "ChemistryDependencyError",
    "ClosedRepeat",
    "PeriodicSupercell",
    "RepeatEligibility",
    "RepeatUnit",
    "StructureError",
    "StructureProfile",
    "assess_repeat_eligibility",
    "attachment_distances",
    "build_closed_repeat",
    "build_periodic_supercell",
    "canonical_periodic_identity",
    "canonicalize_smiles",
    "close_repeat_unit",
    "close_repeat_unit_with_mapping",
    "dummy_and_attachment_indices",
    "parse_repeat_unit",
    "rdkit_available",
    "repeat_profile",
    "require_rdkit",
]
