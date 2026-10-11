from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem

from polyfid.chemistry import (
    StructureError,
    assess_repeat_eligibility,
    build_closed_repeat,
    build_periodic_supercell,
    canonical_periodic_identity,
    generate_closure_rewirings,
    generate_endpoint_relocations,
    generate_equivalent_periodic_cuts,
    generate_randomized_smiles,
    generate_reversed_repeat_view,
    parse_repeat_unit,
    repeat_profile,
)


def test_linear_repeat_parsing_and_readable_periodic_identity() -> None:
    unit = parse_repeat_unit("[*:1]CCO[*:2]")
    assert unit.canonical_psmiles == "*CCO*"
    assert unit.closed_smiles == "C1CO1"
    assert unit.canonical_identity == unit.periodic_supercell_smiles == "C1COCCO1"
    assert unit.canonical_identity != unit.closed_smiles
    assert unit.repeat_equivalence_id == unit.canonical_identity
    assert unit.formula == "C2H4O"
    assert unit.atom_count == 3


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("CCO", "attachment_count"),
        ("*C(*)C", "shared_attachment_atom"),
        ("**", "dummy_connection"),
        ("*CC.*", "disconnected"),
    ],
)
def test_linear_repeat_eligibility_reports_real_corpus_failures(
    text: str, reason: str
) -> None:
    assessment = assess_repeat_eligibility(text)
    assert not assessment.eligible
    assert assessment.reason == reason
    with pytest.raises(StructureError):
        parse_repeat_unit(text)


def test_closed_repeat_and_attachment_distances_use_real_attachment_neighbours() -> None:
    unit = parse_repeat_unit("*CCCO*")
    assert unit.attachment_distances == (None, 0, 1, 1, 0, None)
    assert unit.boundary_classes == (
        "attachment_marker",
        "boundary_0",
        "boundary_1",
        "boundary_1",
        "boundary_0",
        "attachment_marker",
    )
    closed = build_closed_repeat(unit)
    assert closed.closure_added
    assert closed.canonical_identity == "C1COC1"
    assert closed.molecule.GetBondBetweenAtoms(*closed.attachment_atom_indices) is not None


def test_two_cell_supercell_represents_parallel_periodic_bond_case() -> None:
    unit = parse_repeat_unit("*CC*")
    legacy = build_closed_repeat(unit)
    periodic = build_periodic_supercell(unit)

    assert not legacy.closure_added
    assert legacy.canonical_smiles == "CC"
    assert periodic.unit_atom_count == 2
    assert periodic.molecule.GetNumAtoms() == 4
    assert periodic.canonical_identity == unit.canonical_identity == "C1CCC1"
    (left_0, right_0), (left_1, right_1) = periodic.attachment_atom_indices
    expected_cross_edges = {
        tuple(sorted((right_0, left_1))),
        tuple(sorted((right_1, left_0))),
    }
    assert set(periodic.cross_cell_edges) == expected_cross_edges
    assert all(
        periodic.molecule.GetBondBetweenAtoms(*edge) is not None
        for edge in periodic.cross_cell_edges
    )


def test_randomized_smiles_preserves_open_and_periodic_identity() -> None:
    reference = "*C(=O)NCCCO*"
    expected_open = Chem.MolToSmiles(Chem.MolFromSmiles(reference), canonical=True)
    expected_periodic = canonical_periodic_identity(reference)
    views = generate_randomized_smiles(reference, count=10, seed=19)
    assert len(views) == 10
    for view in views:
        assert Chem.MolToSmiles(Chem.MolFromSmiles(view), canonical=True) == expected_open
        assert canonical_periodic_identity(view) == expected_periodic


def test_reverse_view_preserves_two_cell_periodic_identity() -> None:
    unit = parse_repeat_unit("*CC(c1ccccc1)CCO*")
    reversal = generate_reversed_repeat_view(unit)
    assert reversal is not None
    assert reversal.psmiles != unit.canonical_psmiles
    assert canonical_periodic_identity(reversal.psmiles) == unit.canonical_identity


def test_equivalent_periodic_cuts_are_reverified_against_two_cell_graph() -> None:
    unit = parse_repeat_unit("*CC(c1ccccc1)CCO*")
    views = generate_equivalent_periodic_cuts(unit, max_views=10, seed=23)
    assert views[0].transformation == "reference"
    assert views[0].periodic_identity_status == "alternative_cuts_available"
    assert len(views) >= 2
    assert len(views) <= 10
    assert len({view.psmiles for view in views}) == len(views)
    assert all(view.paired_cut_edges is not None for view in views[1:])
    assert all(parse_repeat_unit(view.psmiles).atom_count == unit.atom_count for view in views)
    assert all(view.canonical_identity == unit.canonical_identity for view in views)
    assert all(
        canonical_periodic_identity(view.psmiles) == unit.canonical_identity
        for view in views
    )


def test_short_repeat_has_periodic_identity_without_nontrivial_cut() -> None:
    views = generate_equivalent_periodic_cuts("*CC*", max_views=10)
    assert len(views) == 1
    assert views[0].transformation == "reference"
    assert (
        views[0].periodic_identity_status
        == "identity_representable_no_nontrivial_cut"
    )
    assert views[0].canonical_identity == canonical_periodic_identity("*CC*")


def test_periodic_bonds_restore_open_aromatic_core_before_sanitization() -> None:
    unit = parse_repeat_unit(
        "*c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2C(C)(C)C)cc1"
    )
    assert unit.periodic_supercell_smiles
    assert unit.repeat_equivalence_id == unit.periodic_supercell_smiles


def test_relation_identity_is_not_defined_by_legacy_one_cell_closure() -> None:
    translated_single_double = parse_repeat_unit("*C=C*")
    translated_double_single = parse_repeat_unit("*=CC=*")
    changed_periodic_bonding = parse_repeat_unit("*=C=C=*")

    # Moving the repeat boundary swaps which bond is internal, but represents
    # the same alternating infinite chain.
    assert (
        translated_single_double.canonical_identity
        == translated_double_single.canonical_identity
    )
    assert (
        translated_single_double.closed_smiles
        != translated_double_single.closed_smiles
    )

    # Conversely, the legacy one-cell closure can look identical while the
    # explicit periodic bond pattern is different.
    assert (
        translated_single_double.closed_smiles
        == changed_periodic_bonding.closed_smiles
    )
    assert (
        translated_single_double.canonical_identity
        != changed_periodic_bonding.canonical_identity
    )


def test_endpoint_relocations_preserve_composition_but_change_periodic_topology() -> None:
    unit = parse_repeat_unit("*CCCO*")
    controls = generate_endpoint_relocations(unit, max_views=20, seed=7)
    assert controls
    for control in controls:
        candidate = parse_repeat_unit(control.psmiles)
        assert candidate.formula == unit.formula
        assert candidate.atom_count == unit.atom_count
        assert candidate.canonical_identity != unit.canonical_identity


def test_closure_rewiring_is_a_verified_supplemental_control() -> None:
    unit = parse_repeat_unit("*CC(C)CCC*")
    controls = generate_closure_rewirings(unit, max_views=10, seed=7)
    assert controls
    reference_profile = repeat_profile(unit)
    for control in controls:
        candidate = parse_repeat_unit(control.psmiles)
        candidate_profile = repeat_profile(candidate)
        assert candidate.canonical_identity == control.canonical_identity
        assert candidate.canonical_identity != unit.canonical_identity
        assert candidate_profile.formula == reference_profile.formula
        assert candidate_profile.atom_count == reference_profile.atom_count


def test_genuine_topology_change_is_distinct_under_two_cell_identity() -> None:
    reference = parse_repeat_unit("*CC(C)CCC*")
    changed = parse_repeat_unit("*CC(C)C(*)CC")
    assert changed.formula == reference.formula
    assert changed.atom_count == reference.atom_count
    assert changed.closed_smiles != reference.closed_smiles
    assert changed.canonical_identity != reference.canonical_identity
