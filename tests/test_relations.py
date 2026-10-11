from __future__ import annotations

import random

import pytest
import pandas as pd

pytest.importorskip("rdkit")

from polyfid.relations import (
    RelationTriplet,
    StructureView,
    TypedRelationRecord,
    assemble_triplets,
    build_positive_views,
    build_supplemental_negative_views,
    diagnose_relation_sample,
    make_anchor_view,
    match_true_structure_negative,
    profile_distance,
    summarize_eligibility,
    summarize_triplets,
    stereo_agnostic_graphs_differ,
    stereo_agnostic_two_cell_identity,
)
from polyfid.workflows.relations import (
    _build_retrieval_decoy_bank,
    _formula_distance,
    _manual_review_sample,
    _matched_negative,
    _qualify_relation_ids,
    _semantic_applicability_controls,
    _typed_relation_registry,
)


def test_formula_distance_supports_same_formula_then_near_formula_matching() -> None:
    assert _formula_distance("C8H10O2", "C8H10O2") == 0
    assert _formula_distance("C8H10O2", "C8H10N2") == 4
    assert _formula_distance("C8H10Cl", "C8H9Br") == 3


def test_stereo_only_variants_are_not_topology_negatives() -> None:
    specified = stereo_agnostic_two_cell_identity("C/C=C/C")
    unspecified = stereo_agnostic_two_cell_identity("CC=CC")
    changed_graph = stereo_agnostic_two_cell_identity("CC(C)C")
    assert specified == unspecified
    assert changed_graph != specified
    assert not stereo_agnostic_graphs_differ("C/C=C/C", "CC=CC")
    assert stereo_agnostic_graphs_differ("C/C=C/C", "CC(C)C")
    aromatic_two_cell = (
        "c1=n/c2ccccc2/n=n/n-c2ccc(cc2)/c=n/c2ccccc2/n=n/n-c2ccc/1cc2"
    )
    assert stereo_agnostic_two_cell_identity(aromatic_two_cell) is not None
    assert stereo_agnostic_two_cell_identity("C(C)(C)(C)(C)C") is None
    assert stereo_agnostic_two_cell_identity("[CH5]") is None


def test_primary_matcher_skips_stereo_only_candidate() -> None:
    anchor = pd.Series(
        {
            "sample_id": "a",
            "periodic_identity": "C/C=C/C",
            "stereo_agnostic_periodic_identity": "CC=CC",
            "token_length": 8,
            "closed_smiles": "C/C=C/C",
            "formula": "C4H8",
            "canonical_psmiles": "*/C=C/C*",
        }
    )
    candidates = pd.DataFrame(
        [
            {
                "sample_id": "stereo-only",
                "periodic_identity": "CC=CC",
                "stereo_agnostic_periodic_identity": "CC=CC",
                "token_length": 8,
                "closed_smiles": "CC=CC",
                "formula": "C4H8",
                "canonical_psmiles": "*CC=CC*",
            },
            {
                "sample_id": "graph-change",
                "periodic_identity": "CC(C)C",
                "stereo_agnostic_periodic_identity": "CC(C)C",
                "token_length": 8,
                "closed_smiles": "CC(C)C",
                "formula": "C4H10",
                "canonical_psmiles": "*CC(C)C*",
            },
        ]
    )
    matched = _matched_negative(
        anchor,
        candidates,
        rng=random.Random(42),
        similarity_minimum=0.0,
    )
    assert matched is not None
    assert matched[0]["sample_id"] == "graph-change"


def test_structure_view_preserves_serialization_and_assigns_periodic_identity() -> None:
    text = "C(C*)CO*"
    view = StructureView.from_psmiles("polymer-1", text)
    assert view.eligible
    assert view.psmiles == text
    assert view.canonical_identity is not None
    assert view.periodic_supercell_smiles == view.canonical_identity
    assert view.periodic_identity == view.repeat_equivalence_id
    assert view.closed_smiles is not None
    assert view.metadata["canonical_psmiles"] == "*CCCO*"
    assert view.metadata["closed_smiles"] == view.closed_smiles
    assert view.profile is not None


def test_positive_views_share_identity_with_anchor() -> None:
    anchor = make_anchor_view("polymer-1", "*CC(c1ccccc1)CCO*")
    positives = build_positive_views(
        anchor, randomized_count=10, periodic_max_views=10, seed=17
    )
    assert len(
        [view for view in positives if view.relation == "randomized_traversal"]
    ) == 10
    assert any(view.relation == "alternative_periodic_cut" for view in positives)
    assert any(view.relation == "reverse_when_non_noop" for view in positives)
    assert all(view.source == "verified_transformation" for view in positives)
    assert all(view.canonical_identity == anchor.canonical_identity for view in positives)
    periodic = [
        view for view in positives if view.relation == "alternative_periodic_cut"
    ]
    assert all(view.metadata["paired_cut_edges"] is not None for view in periodic)
    assert all(
        view.metadata["periodic_identity_status"] == "verified_alternative_cut"
        for view in periodic
    )


def test_true_structure_matching_uses_observed_distinct_candidates() -> None:
    anchor = make_anchor_view("anchor", "*CCCO*")
    close_candidate = make_anchor_view("close", "*CCCC*")
    far_candidate = make_anchor_view("far", "*NCCCCCCN*")
    equivalent_record = make_anchor_view("same-topology", "*CCOC*")
    synthetic = StructureView.from_psmiles(
        "synthetic",
        "*CC(*)CO",
        relation="endpoint_relocation",
        source="synthetic_control",
    )
    match = match_true_structure_negative(
        anchor,
        [far_candidate, equivalent_record, synthetic, close_candidate],
    )
    assert match is not None
    expected_distance, _ = profile_distance(anchor, close_candidate)
    far_distance, _ = profile_distance(anchor, far_candidate)
    assert expected_distance < far_distance
    assert match.view.sample_id == "close"
    assert match.view.relation == "matched_real_periodic_topology"
    assert match.view.source == "observed"
    assert match.view.canonical_identity != anchor.canonical_identity


def test_matching_interface_can_constrain_dataset_groups() -> None:
    anchor = make_anchor_view("a", "*CCCO*").as_relation(
        "anchor", metadata={"family": "A"}
    )
    family_a = make_anchor_view("a-neg", "*CCCC*").as_relation(
        "anchor", metadata={"family": "A"}
    )
    family_b = make_anchor_view("b-neg", "*CCNCC*").as_relation(
        "anchor", metadata={"family": "B"}
    )
    match = match_true_structure_negative(
        anchor,
        [family_b, family_a],
        candidate_filter=lambda left, right: left.metadata["family"]
        == right.metadata["family"],
    )
    assert match is not None
    assert match.view.sample_id == "a-neg"


def test_synthetic_controls_are_kept_separate_from_true_structure_negatives() -> None:
    anchor = make_anchor_view("polymer-1", "*CC(C)CCC*")
    controls = build_supplemental_negative_views(
        anchor,
        max_endpoint_relocations=20,
        max_closure_rewirings=20,
        seed=31,
    )
    assert controls
    assert any(view.relation == "endpoint_relocation" for view in controls)
    assert any(view.relation == "closure_rewiring" for view in controls)
    assert all(view.source == "synthetic_control" for view in controls)
    assert all(view.canonical_identity != anchor.canonical_identity for view in controls)


def test_relation_triplet_checks_only_the_scientific_relation() -> None:
    anchor = make_anchor_view("a", "*CCCO*")
    positive = build_positive_views(
        anchor, randomized_count=1, periodic_max_views=0, seed=3
    )[0]
    negative = make_anchor_view("n", "*CCCC*").as_relation(
        "matched_real_periodic_topology"
    )
    triplet = RelationTriplet(anchor, positive, negative)
    assert triplet.eligible
    assert triplet.diagnostics == ()

    wrong_negative = positive.as_relation("endpoint_relocation")
    invalid = RelationTriplet(anchor, positive, wrong_negative)
    assert not invalid.eligible
    assert invalid.diagnostics == ("negative_identity_match",)

    stereo_anchor = make_anchor_view(
        "stereo-a", "*/C=C/c1ccc(OCCCOc2ccc(*)cc2)cc1"
    )
    stereo_positive = build_positive_views(
        stereo_anchor, randomized_count=1, periodic_max_views=0, seed=7
    )[0]
    stereo_only = make_anchor_view(
        "stereo-n", "*CCCOc1ccc(/C=C/c2ccc(O*)cc2)cc1"
    ).as_relation("matched_real_periodic_topology")
    stereo_triplet = RelationTriplet(stereo_anchor, stereo_positive, stereo_only)
    assert stereo_anchor.canonical_identity != stereo_only.canonical_identity
    assert not stereo_triplet.eligible
    assert stereo_triplet.diagnostics == (
        "negative_stereo_agnostic_identity_match",
    )


def test_triplet_uses_periodic_supercell_not_legacy_closed_smiles() -> None:
    anchor = make_anchor_view("a", "*C=C*")
    positive = make_anchor_view("p", "*=CC=*").as_relation(
        "alternative_periodic_cut", source="verified_transformation"
    )
    negative = make_anchor_view("n", "*=C=C=*").as_relation(
        "matched_real_periodic_topology"
    )

    assert anchor.closed_smiles != positive.closed_smiles
    assert anchor.repeat_equivalence_id == positive.repeat_equivalence_id
    assert anchor.closed_smiles == negative.closed_smiles
    assert anchor.repeat_equivalence_id != negative.repeat_equivalence_id
    assert RelationTriplet(anchor, positive, negative).eligible


def test_eligibility_and_triplet_statistics_report_main_and_supplemental_use() -> None:
    anchor = make_anchor_view("a", "*CC(C)CCC*")
    positive = build_positive_views(
        anchor, randomized_count=1, periodic_max_views=0, seed=5
    )[0]
    observed = make_anchor_view("n", "*CCOCCC*").as_relation(
        "matched_real_periodic_topology"
    )
    synthetic = build_supplemental_negative_views(
        anchor,
        max_endpoint_relocations=1,
        max_closure_rewirings=1,
        seed=5,
    )
    eligible = diagnose_relation_sample(
        anchor, [positive], [observed, *synthetic]
    )
    missing_negative = diagnose_relation_sample(anchor, [positive], list(synthetic))
    summary = summarize_eligibility([eligible, missing_negative])
    assert eligible.main_relation_eligible
    assert eligible.supplemental_relation_eligible
    assert missing_negative.diagnostic == "no_true_structure_negative"
    assert summary.total_samples == 2
    assert summary.main_relation_eligible_samples == 1
    assert summary.diagnostic_counts == {
        "eligible": 1,
        "no_true_structure_negative": 1,
    }

    triplets = assemble_triplets(anchor, [positive], [observed, *synthetic])
    triplet_summary = summarize_triplets(triplets)
    assert triplet_summary.total_triplets == len(triplets)
    assert triplet_summary.eligible_triplets == len(triplets)
    assert triplet_summary.negative_relation_counts["matched_real_periodic_topology"] == 1


def test_retrieval_bank_uses_unique_partition_candidates_and_matched_composition() -> None:
    index = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e", "unavailable"],
            "canonical_psmiles": [
                "*CC*", "*CO*", "*CN*", "*CCC*", "*CCO*", "invalid"
            ],
            "closed_smiles": ["CC", "CO", "CN", "CCC", "CCO", "C"],
            "periodic_identity": ["CC", "CO", "CN", "CCC", "CCO", "not_smiles"],
            "periodic_supercell_smiles": [
                "CC", "CO", "CN", "CCC", "CCO", "not_smiles"
            ],
            "formula": ["C2H6", "C2H6", "CH5N", "C3H8", "C3H8", "C2H6"],
            "atom_count": [2, 2, 2, 3, 3, 2],
            "token_length": [7, 8, 7, 9, 8, 7],
            "split": ["train"] * 6,
        }
    )
    relations = pd.DataFrame(
        {
            "anchor_sample_id": ["a", "d"],
            "anchor_identity": ["CC", "CCC"],
            "anchor_formula": ["C2H6", "C3H8"],
            "anchor_atom_count": [2, 3],
            "anchor_token_length": [7, 9],
        }
    )

    bank, decoys, summary = _build_retrieval_decoy_bank(index, relations, count=2)

    assert len(decoys) == len(relations)
    assert all(len(values) == 2 for values in decoys)
    assert all(len(set(values)) == 2 for values in decoys)
    identities = bank.set_index("decoy_bank_index")["decoy_identity"]
    formulas = bank.set_index("decoy_bank_index")["formula"]
    for position, values in enumerate(decoys):
        assert relations.iloc[position]["anchor_identity"] not in {
            identities.loc[value] for value in values
        }
        assert formulas.loc[values[0]] == relations.iloc[position]["anchor_formula"]
    assert summary["decoys_per_anchor"] == 2
    assert summary["bank_size"] == len(bank)
    assert bank["decoy_stereo_agnostic_identity"].notna().all()
    assert 0.0 <= summary["same_atom_count_fraction"] <= 1.0
    assert set(summary["token_delta"]) == {"mean", "median", "p90", "maximum"}


def test_typed_registry_separates_training_relations_from_scope_controls() -> None:
    anchor = make_anchor_view("a", "*CCO*")
    relations = pd.DataFrame(
        {
            "triplet_id": ["triplet:0"],
            "split": ["train"],
            "anchor_sample_id": ["a"],
            "anchor_psmiles": [anchor.psmiles],
            "anchor_identity": [anchor.canonical_identity],
            "positive_sample_id": ["p"],
            "positive_relation": ["alternative_periodic_cut"],
            "negative_sample_id": ["n"],
            "negative_relation": ["matched_real_periodic_topology"],
        }
    )
    supplemental = pd.DataFrame(
        {
            "anchor_sample_id": ["a"],
            "anchor_identity": [anchor.canonical_identity],
            "negative_sample_id": ["a::closure_rewiring:0"],
            "negative_relation": ["closure_rewiring"],
            "split": ["train"],
        }
    )
    semantic = _semantic_applicability_controls(
        relations,
        partition="train",
        seed=42,
    )
    assert set(semantic["control_family"]) == {"h_capped", "marker_substituted"}
    registry = _typed_relation_registry(
        relations,
        supplemental,
        [
            {
                "anchor_id": "missing",
                "split": "train",
                "split_group": "CO",
                "exclusion_reason": "no_verified_positive",
            }
        ],
        semantic_controls=semantic,
    )

    assert set(registry["relation_class"]) == {
        "equivalence_preserving",
        "topology_changing",
        "polymer_semantics_inapplicable",
        "unknown_or_ineligible",
    }
    blocked = registry["relation_class"].isin(
        ["polymer_semantics_inapplicable", "unknown_or_ineligible"]
    )
    assert not registry.loc[blocked, "training_eligible"].any()
    assert registry.loc[
        registry["relation_family"].eq("closure_rewiring"), "training_eligible"
    ].eq(False).all()
    assert registry.loc[
        registry["relation_family"].eq("matched_real_periodic_topology"),
        "property_target",
    ].eq("unconstrained").all()
    assert registry["relation_id"].str.startswith("train::").all()

    legacy_test = registry.assign(
        split="test",
        relation_id=registry["relation_id"].str.removeprefix("train::"),
    )
    qualified_test = _qualify_relation_ids(legacy_test)
    combined = pd.concat((registry, qualified_test), ignore_index=True)
    assert combined["relation_id"].is_unique


def test_typed_record_rejects_semantic_control_as_training_input() -> None:
    with pytest.raises(ValueError, match="cannot enter training"):
        TypedRelationRecord(
            relation_id="semantic:1",
            anchor_id="a",
            paired_id="control",
            split="test",
            split_group="group",
            relation_family="h_capped",
            relation_class="polymer_semantics_inapplicable",
            two_cell_identity_relation="unknown",
            changed_fields=("attachment_markers",),
            parse_status="parseable",
            polymer_semantic_status="inapplicable_as_repeat_unit",
            representation_target="descriptive_only",
            property_target="none",
            training_eligible=True,
            evaluation_eligible=True,
            evidence_rule="applicability_control_only",
        )


def test_manual_review_sheet_has_exact_stratified_rows_and_blank_verdicts() -> None:
    families = (
        ["randomized_traversal"] * 130
        + ["alternative_periodic_cut"] * 110
        + ["reverse_when_non_noop"] * 60
    )
    frame = pd.DataFrame(
        {
            "triplet_id": [f"t{index}" for index in range(len(families))],
            "positive_relation": families,
        }
    )
    review = _manual_review_sample(frame, count=200, seed=42)
    assert len(review) == 200
    assert review["triplet_id"].nunique() == 200
    assert set(review["relation_family"]) == set(families)
    assert review["positive_equivalent_verdict"].eq("").all()
    assert review["negative_topology_changed_verdict"].eq("").all()
