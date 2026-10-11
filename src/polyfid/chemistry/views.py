"""Verified structure views for periodic-consistency experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, TypeVar

from .structure import (
    PeriodicSupercell,
    RepeatUnit,
    StructureError,
    build_closed_repeat,
    build_periodic_supercell,
    canonicalize_smiles,
    dummy_and_attachment_indices,
    parse_repeat_unit,
    require_rdkit,
)

try:
    from rdkit import Chem, rdBase
except ModuleNotFoundError:  # pragma: no cover - analysis-only environments
    Chem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PeriodicView:
    psmiles: str
    canonical_identity: str
    cut_atom_indices: tuple[int, int] | None
    cut_bond_type: str
    transformation: str = "equivalent_periodic_cut"
    paired_cut_edges: tuple[tuple[int, int], tuple[int, int]] | None = None
    periodic_identity_status: str = "verified_alternative_cut"

    @property
    def repeat_equivalence_id(self) -> str:
        return self.canonical_identity


@dataclass(frozen=True)
class EndpointRelocation:
    psmiles: str
    canonical_identity: str
    endpoint_index: int
    old_attachment_index: int
    new_attachment_index: int
    transformation: str = "endpoint_relocation"


@dataclass(frozen=True)
class ClosureRewiring:
    psmiles: str
    canonical_identity: str
    removed_edges: tuple[tuple[int, int], tuple[int, int]]
    added_edges: tuple[tuple[int, int], tuple[int, int]]
    transformation: str = "closure_rewiring"


@dataclass(frozen=True)
class TraversalReversal:
    psmiles: str
    canonical_identity: str
    transformation: str = "reverse_when_non_noop"


T = TypeVar("T")


def _molecule(smiles: str) -> Any:
    require_rdkit()
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        raise StructureError("RDKit could not parse the structure", code="invalid_smiles")
    return molecule


def _limit(items: list[T], count: int, seed: int, *, keep_first: bool = False) -> tuple[T, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(items) <= count:
        return tuple(items)
    if count == 0:
        return ()
    chooser = random.Random(int(seed))
    if keep_first:
        selected = [0, *sorted(chooser.sample(range(1, len(items)), count - 1))]
    else:
        selected = sorted(chooser.sample(range(len(items)), count))
    return tuple(items[index] for index in selected)


def generate_randomized_smiles(
    smiles: str,
    *,
    count: int = 10,
    seed: int = 42,
) -> tuple[str, ...]:
    """Generate a fixed number of graph-equivalent SMILES traversals."""

    if count < 0:
        raise ValueError("count must be non-negative")
    molecule = _molecule(smiles)
    reference = canonicalize_smiles(smiles)
    views: list[str] = []
    for view_index in range(count):
        order = list(range(molecule.GetNumAtoms()))
        random.Random(int(seed) + 104729 * view_index).shuffle(order)
        candidate = Chem.MolToSmiles(
            Chem.RenumberAtoms(molecule, order),
            canonical=False,
            isomericSmiles=True,
        )
        if canonicalize_smiles(candidate) != reference:
            raise StructureError(
                "randomized traversal changed molecular identity",
                code="randomization_changed_identity",
            )
        views.append(candidate)
    return tuple(views)


def generate_reversed_repeat_view(
    unit_or_text: RepeatUnit | str,
) -> TraversalReversal | None:
    """Return the opposite attachment-rooted traversal when it is nontrivial."""

    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    source = _molecule(unit.canonical_psmiles)
    dummies, _ = dummy_and_attachment_indices(source)
    forward = Chem.MolToSmiles(
        source,
        rootedAtAtom=dummies[0],
        canonical=False,
        isomericSmiles=True,
    )
    reverse = Chem.MolToSmiles(
        source,
        rootedAtAtom=dummies[1],
        canonical=False,
        isomericSmiles=True,
    )
    if reverse == forward:
        return None
    candidate = parse_repeat_unit(reverse)
    if candidate.canonical_identity != unit.canonical_identity:
        return None
    return TraversalReversal(
        psmiles=reverse,
        canonical_identity=candidate.canonical_identity,
    )


def _repeat_cycle_edges(
    closed: Any, closure_edge: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Select the attachment-defined cycle rather than unrelated side rings."""

    left, right = sorted(closure_edge)
    closure_bond = closed.GetBondBetweenAtoms(left, right)
    if closure_bond is None:
        return ()
    ranks = tuple(
        int(value)
        for value in Chem.CanonicalRankAtoms(
            closed,
            breakTies=True,
            includeChirality=True,
            includeIsotopes=True,
        )
    )
    cycles: list[
        tuple[int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]
    ] = []
    for bond_ring in closed.GetRingInfo().BondRings():
        if closure_bond.GetIdx() not in bond_ring:
            continue
        edges = tuple(
            sorted(
                tuple(
                    sorted(
                        (
                            closed.GetBondWithIdx(int(index)).GetBeginAtomIdx(),
                            closed.GetBondWithIdx(int(index)).GetEndAtomIdx(),
                        )
                    )
                )
                for index in bond_ring
            )
        )
        signature = tuple(
            sorted(tuple(sorted((ranks[a], ranks[b]))) for a, b in edges)
        )
        cycles.append((len(edges), signature, edges))
    if cycles:
        return min(cycles, key=lambda item: (-item[0], item[1]))[2]

    editable = Chem.RWMol(closed)
    editable.RemoveBond(left, right)
    opened = editable.GetMol()
    if len(Chem.GetMolFrags(opened)) != 1:
        return ((left, right),)
    path = tuple(Chem.GetShortestPath(opened, left, right))
    edges = {(left, right)}
    edges.update(tuple(sorted(pair)) for pair in zip(path, path[1:]))
    return tuple(sorted(edges))


def _cut_to_repeat(closed: Any, edge: tuple[int, int]) -> str | None:
    left, right = edge
    bond = closed.GetBondBetweenAtoms(left, right)
    if bond is None or bond.GetBondType() not in (
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
    ):
        return None
    bond_type = bond.GetBondType()
    editable = Chem.RWMol(closed)
    editable.RemoveBond(left, right)
    left_dummy = editable.AddAtom(Chem.Atom(0))
    right_dummy = editable.AddAtom(Chem.Atom(0))
    editable.AddBond(left, left_dummy, bond_type)
    editable.AddBond(right, right_dummy, bond_type)
    candidate = editable.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(candidate)
    except (ValueError, RuntimeError):
        return None
    if len(Chem.GetMolFrags(candidate)) != 1:
        return None
    return Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)


def _paired_cut_to_repeats(
    periodic: PeriodicSupercell,
    paired_edges: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[str, str], ...]:
    """Cut one translated bond pair into verified one-cell repeat windows.

    The returned tuples contain a canonical PSMILES and the cut bond type.
    A useful periodic cut must split the two-cell graph into exactly two
    components, each containing one cell's number of real atoms.  For each
    component, the two severed bonds are capped with dummy attachment atoms.
    """

    supercell = periodic.molecule
    snapshots: list[tuple[tuple[int, int], Any, Any]] = []
    for edge in paired_edges:
        bond = supercell.GetBondBetweenAtoms(*edge)
        if bond is None:
            return ()
        snapshots.append((edge, bond.GetBondType(), bond.GetBondDir()))
    if snapshots[0][1] != snapshots[1][1]:
        return ()

    editable = Chem.RWMol(supercell)
    for edge, _, _ in snapshots:
        editable.RemoveBond(*edge)
    opened = editable.GetMol()
    components = tuple(
        tuple(sorted(int(index) for index in component))
        for component in Chem.GetMolFrags(opened, asMols=False, sanitizeFrags=False)
    )
    components = tuple(sorted(components))
    if len(components) != 2 or any(
        len(component) != periodic.unit_atom_count for component in components
    ):
        return ()

    candidates: dict[str, str] = {}
    all_indices = set(range(opened.GetNumAtoms()))
    for component in components:
        component_set = set(component)
        dangling: list[tuple[int, Any, Any]] = []
        for (left, right), bond_type, bond_direction in snapshots:
            left_inside = left in component_set
            right_inside = right in component_set
            if left_inside == right_inside:
                dangling = []
                break
            dangling.append(
                (left if left_inside else right, bond_type, bond_direction)
            )
        if len(dangling) != 2:
            continue

        old_to_new = {
            old_index: new_index
            for new_index, old_index in enumerate(sorted(component_set))
        }
        component_molecule = Chem.RWMol(opened)
        for atom_index in sorted(all_indices - component_set, reverse=True):
            component_molecule.RemoveAtom(atom_index)
        for old_atom_index, bond_type, bond_direction in dangling:
            dummy_index = component_molecule.AddAtom(Chem.Atom(0))
            component_molecule.AddBond(
                old_to_new[old_atom_index], dummy_index, bond_type
            )
            added = component_molecule.GetBondBetweenAtoms(
                old_to_new[old_atom_index], dummy_index
            )
            if bond_direction != Chem.BondDir.NONE:
                added.SetBondDir(bond_direction)
        candidate = component_molecule.GetMol()
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(candidate)
                Chem.AssignStereochemistry(candidate, cleanIt=True, force=True)
        except (ValueError, RuntimeError):
            continue
        if len(Chem.GetMolFrags(candidate)) != 1:
            continue
        text = Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)
        try:
            canonical = parse_repeat_unit(text).canonical_psmiles
        except StructureError:
            continue
        candidates[canonical] = str(snapshots[0][1])
    return tuple((text, candidates[text]) for text in sorted(candidates))


def generate_equivalent_periodic_cuts(
    unit_or_text: RepeatUnit | str,
    *,
    max_views: int = 10,
    seed: int = 42,
) -> tuple[PeriodicView, ...]:
    """Generate repeat-window translations verified by two-cell identity.

    Every candidate is obtained by removing a translated pair of equivalent
    bonds from the two-cell graph.  The two resulting equal-sized components
    are converted back to one-cell PSMILES and independently rebuilt as
    supercells before being accepted.  The reference is always first, results
    are unique by canonical PSMILES, and at most ten views are returned.
    """

    if max_views < 0:
        raise ValueError("max_views must be non-negative")
    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    periodic = build_periodic_supercell(unit)
    alternatives: dict[str, PeriodicView] = {}
    for paired_edges in periodic.bond_orbits:
        for text, bond_type in _paired_cut_to_repeats(periodic, paired_edges):
            try:
                candidate = parse_repeat_unit(text)
            except StructureError:
                continue
            if candidate.canonical_identity != unit.canonical_identity:
                continue
            if candidate.atom_count != unit.atom_count:
                continue
            if candidate.canonical_psmiles == unit.canonical_psmiles:
                continue
            alternatives.setdefault(
                candidate.canonical_psmiles,
                PeriodicView(
                    psmiles=candidate.canonical_psmiles,
                    canonical_identity=candidate.canonical_identity,
                    cut_atom_indices=paired_edges[0],
                    cut_bond_type=bond_type,
                    paired_cut_edges=paired_edges,
                    periodic_identity_status="verified_alternative_cut",
                ),
            )

    has_alternative = bool(alternatives)
    reference = PeriodicView(
        psmiles=unit.canonical_psmiles,
        canonical_identity=unit.canonical_identity,
        cut_atom_indices=None,
        cut_bond_type=unit.attachment_bond_type,
        transformation="reference",
        paired_cut_edges=None,
        periodic_identity_status=(
            "alternative_cuts_available"
            if has_alternative
            else "identity_representable_no_nontrivial_cut"
        ),
    )
    ordered = [reference]
    ordered.extend(alternatives[key] for key in sorted(alternatives))
    return _limit(ordered, min(max_views, 10), seed, keep_first=True)


def generate_periodic_views(
    unit_or_text: RepeatUnit | str,
    *,
    max_views: int = 10,
    seed: int = 42,
) -> tuple[PeriodicView, ...]:
    """Compatibility alias for equivalent periodic cuts."""

    return generate_equivalent_periodic_cuts(
        unit_or_text, max_views=max_views, seed=seed
    )


def generate_endpoint_relocations(
    unit_or_text: RepeatUnit | str,
    *,
    max_views: int = 10,
    seed: int = 42,
) -> tuple[EndpointRelocation, ...]:
    """Move one attachment marker and retain only topology-changing controls."""

    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    source = _molecule(unit.canonical_psmiles)
    dummies, attachments = dummy_and_attachment_indices(source)
    real_atoms = [
        atom.GetIdx() for atom in source.GetAtoms() if atom.GetAtomicNum() != 0
    ]
    candidates: dict[str, EndpointRelocation] = {}
    for endpoint_index, (dummy, old_attachment) in enumerate(zip(dummies, attachments)):
        terminal_bond = source.GetBondBetweenAtoms(dummy, old_attachment)
        other_attachment = attachments[1 - endpoint_index]
        for new_attachment in real_atoms:
            if new_attachment in (old_attachment, other_attachment):
                continue
            editable = Chem.RWMol(source)
            editable.RemoveBond(dummy, old_attachment)
            editable.AddBond(dummy, new_attachment, terminal_bond.GetBondType())
            candidate_molecule = editable.GetMol()
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(candidate_molecule)
                    text = Chem.MolToSmiles(
                        candidate_molecule, canonical=True, isomericSmiles=True
                    )
                    candidate = parse_repeat_unit(text)
            except (StructureError, ValueError, RuntimeError):
                continue
            if candidate.canonical_identity == unit.canonical_identity:
                continue
            if candidate.formula != unit.formula or candidate.atom_count != unit.atom_count:
                continue
            candidates[candidate.canonical_psmiles] = EndpointRelocation(
                psmiles=candidate.canonical_psmiles,
                canonical_identity=candidate.canonical_identity,
                endpoint_index=endpoint_index,
                old_attachment_index=old_attachment,
                new_attachment_index=new_attachment,
            )
    ordered = [candidates[key] for key in sorted(candidates)]
    return _limit(ordered, max_views, seed)


def generate_closure_rewirings(
    unit_or_text: RepeatUnit | str,
    *,
    max_views: int = 10,
    seed: int = 42,
) -> tuple[ClosureRewiring, ...]:
    """Create degree-preserving two-edge rewires as supplemental controls."""

    unit = parse_repeat_unit(unit_or_text) if isinstance(unit_or_text, str) else unit_or_text
    closed_repeat = build_closed_repeat(unit)
    if not closed_repeat.closure_added:
        return ()
    closed = closed_repeat.molecule
    left, right = closed_repeat.attachment_atom_indices
    closure_edge = tuple(sorted((left, right)))
    closure_bond = closed.GetBondBetweenAtoms(left, right)
    closure_type = closure_bond.GetBondType()
    candidates: dict[str, ClosureRewiring] = {}

    for partner_bond in closed.GetBonds():
        partner = tuple(
            sorted((partner_bond.GetBeginAtomIdx(), partner_bond.GetEndAtomIdx()))
        )
        if partner == closure_edge or partner_bond.GetBondType() != closure_type:
            continue
        first, second = partner
        if len({left, right, first, second}) != 4:
            continue
        pairings = (
            ((left, first), (right, second)),
            ((left, second), (right, first)),
        )
        for raw_added in pairings:
            added = tuple(tuple(sorted(edge)) for edge in raw_added)
            if any(
                closed.GetBondBetweenAtoms(*edge) is not None
                and edge not in (closure_edge, partner)
                for edge in added
            ):
                continue
            editable = Chem.RWMol(closed)
            editable.RemoveBond(*closure_edge)
            editable.RemoveBond(*partner)
            editable.AddBond(*added[0], closure_type)
            editable.AddBond(*added[1], closure_type)
            rewired = editable.GetMol()
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(rewired)
            except (ValueError, RuntimeError):
                continue
            if len(Chem.GetMolFrags(rewired)) != 1:
                continue
            rewired_closed_smiles = Chem.MolToSmiles(
                rewired, canonical=True, isomericSmiles=True
            )
            if rewired_closed_smiles == unit.closed_smiles:
                continue
            text = _cut_to_repeat(rewired, added[0])
            if text is None:
                continue
            try:
                with rdBase.BlockLogs():
                    candidate = parse_repeat_unit(text)
            except StructureError:
                continue
            if candidate.closed_smiles != rewired_closed_smiles:
                continue
            if candidate.canonical_identity == unit.canonical_identity:
                continue
            if candidate.formula != unit.formula or candidate.atom_count != unit.atom_count:
                continue
            candidates[candidate.canonical_psmiles] = ClosureRewiring(
                psmiles=candidate.canonical_psmiles,
                canonical_identity=candidate.canonical_identity,
                removed_edges=(closure_edge, partner),
                added_edges=added,
            )
    ordered = [candidates[key] for key in sorted(candidates)]
    return _limit(ordered, max_views, seed)


def renumber_atoms(molecule: Any, order: list[int]) -> Any:
    require_rdkit()
    if sorted(order) != list(range(molecule.GetNumAtoms())):
        raise ValueError("order must be a permutation of atom indices")
    return Chem.RenumberAtoms(molecule, order)


__all__ = [
    "ClosureRewiring",
    "EndpointRelocation",
    "PeriodicView",
    "TraversalReversal",
    "generate_closure_rewirings",
    "generate_endpoint_relocations",
    "generate_equivalent_periodic_cuts",
    "generate_periodic_views",
    "generate_randomized_smiles",
    "generate_reversed_repeat_view",
    "renumber_atoms",
]
