from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from polyfid.analysis.mechanism_control_statistics import (
    BOUNDARY_COMPARISONS,
    FORMAL_REPRESENTATION_SEEDS,
    MECHANISM_CONTROL_SEEDS,
    pair_boundary_cells,
    write_mechanism_control_analysis_outputs,
)


BOUNDARY_METHODS = {
    "A0": (2.0, 0.50, FORMAL_REPRESENTATION_SEEDS),
    "A_att": (1.0, 0.75, FORMAL_REPRESENTATION_SEEDS),
    "matched_random_masking": (1.5, 0.625, MECHANISM_CONTROL_SEEDS),
    "permuted_boundary_masking": (2.5, 0.375, MECHANISM_CONTROL_SEEDS),
}
RELATION_METHODS = {
    "A_eqs": (0.80, 0.70, 0.65, 0.20, 0.75),
    "equivalence_only": (0.90, 0.80, 0.75, 0.10, 0.60),
    "topology_negative_only": (0.70, 0.55, 0.50, 0.35, 0.85),
}


def _boundary_frame(method: str, seed: int, nll: float, accuracy: float) -> pd.DataFrame:
    rows = []
    source_strata = (
        ("distance_0", False),
        ("distance_1", False),
        ("distance_2", False),
        ("distance_3_plus", True),
    )
    for sample_index in range(3):
        for stratum_index, (stratum, interior) in enumerate(source_strata):
            for token_index in range(2):
                rows.append(
                    {
                        "sample_id": f"sample-{sample_index}",
                        "group_id": f"sample-{sample_index}:group-{stratum_index}",
                        "token_position": 10 * stratum_index + token_index,
                        "target_token_id": 100 + stratum_index + token_index,
                        "boundary_stratum": stratum,
                        "interior_region": interior,
                        "nll": nll + 0.01 * sample_index,
                        "correct": token_index < round(2 * accuracy),
                        "method": method,
                        "representation_seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def _relation_frame(method: str, seed: int, values: tuple[float, ...]) -> pd.DataFrame:
    cosine, recall, mrr, margin, accuracy = values
    families = (
        "alternative_periodic_cut",
        "randomized_traversal",
        "reverse_direction",
    )
    rows = []
    for index in range(6):
        rows.append(
            {
                "triplet_id": f"triplet:{index}",
                "anchor_sample_id": f"anchor:{index}",
                "positive_sample_id": f"positive:{index}",
                "negative_sample_id": f"negative:{index}",
                "positive_relation": families[index % len(families)],
                "negative_relation": "matched_real_periodic_topology",
                "anchor_positive_cosine": cosine + 0.001 * index,
                "retrieval_correct_at_1": recall,
                "reciprocal_rank": mrr + 0.001 * index,
                "topology_margin": margin + 0.001 * index,
                "triplet_correct": accuracy,
                "method": method,
                "representation_seed": seed,
            }
        )
    return pd.DataFrame(rows)


def _write_inputs(boundary_root: Path, relation_root: Path) -> None:
    for method, (nll, accuracy, seeds) in BOUNDARY_METHODS.items():
        for seed in seeds:
            destination = boundary_root / f"{method}_seed-{seed}"
            destination.mkdir(parents=True)
            _boundary_frame(method, seed, nll, accuracy).to_parquet(
                destination / "token_metrics.parquet", index=False
            )
    for method, values in RELATION_METHODS.items():
        for seed in MECHANISM_CONTROL_SEEDS:
            destination = relation_root / f"{method}_seed-{seed}"
            destination.mkdir(parents=True)
            _relation_frame(method, seed, values).to_parquet(
                destination / "triplet_metrics.parquet", index=False
            )


def test_mechanism_control_writer_keeps_main_and_three_seed_controls_separate(
    tmp_path: Path,
) -> None:
    boundary_root = tmp_path / "boundary"
    relation_root = tmp_path / "relations"
    output = tmp_path / "statistics"
    _write_inputs(boundary_root, relation_root)

    summary = write_mechanism_control_analysis_outputs(
        boundary_root,
        relation_root,
        output,
        n_resamples=40,
        bootstrap_seed=7,
        bootstrap_chunk_size=8,
    )

    inventory = pd.read_csv(output / "result_inventory.csv")
    assert len(inventory) == 25
    main = pd.read_csv(output / "boundary_main_comparison.csv")
    controls = pd.read_csv(output / "boundary_control_comparisons.csv")
    assert set(main["comparison"]) == {"boundary_supervision"}
    assert set(main["n_representation_seeds"]) == {5}
    assert set(main["stratum"]) == {
        "distance_0",
        "distance_1",
        "distance_2",
        "interior",
    }
    assert len(main) == 4
    assert (main["relative_nll_percent"] < 0.0).all()
    assert (main["accuracy_difference_pp"] > 0.0).all()
    assert set(controls["n_representation_seeds"]) == {3}
    assert len(controls) == 8
    assert (
        controls.loc[
            controls["candidate"].eq("matched_random_masking"),
            "relative_nll_percent",
        ]
        < 0.0
    ).all()
    assert (
        controls.loc[
            controls["candidate"].eq("permuted_boundary_masking"),
            "relative_nll_percent",
        ]
        > 0.0
    ).all()

    relation = pd.read_csv(output / "relation_control_comparisons.csv")
    assert len(relation) == 40
    assert set(relation["analysis_scope"]) == {"overall", "relation_family"}
    assert set(relation["metric"]) == {
        "equivalence_cosine",
        "recall_at_1",
        "mrr",
        "topology_margin",
        "triplet_accuracy",
    }
    assert set(relation["n_representation_seeds"]) == {3}
    equivalence_overall = relation.loc[
        relation["candidate"].eq("equivalence_only")
        & relation["analysis_scope"].eq("overall")
    ].set_index("metric")
    assert equivalence_overall.loc[
        "equivalence_cosine", "candidate_minus_full_objective"
    ] > 0.0
    assert equivalence_overall.loc[
        "topology_margin", "candidate_minus_full_objective"
    ] < 0.0
    topology_overall = relation.loc[
        relation["candidate"].eq("topology_negative_only")
        & relation["analysis_scope"].eq("overall")
    ].set_index("metric")
    assert topology_overall.loc[
        "topology_margin", "candidate_minus_full_objective"
    ] > 0.0
    assert topology_overall.loc[
        "equivalence_cosine", "candidate_minus_full_objective"
    ] < 0.0

    stored_summary = json.loads((output / "summary.json").read_text())
    assert stored_summary == summary
    assert stored_summary["combined_score"] == "not computed"
    assert "three-seed controls are supplementary" in stored_summary["claim_scope"]
    readout = (output / "supplementary_readout.md").read_text()
    assert "five-seed main comparison" in readout
    assert "three shared seeds" in readout


def test_boundary_pairing_detects_changed_fixed_mask_targets() -> None:
    candidate = _boundary_frame("A_att", 42, 1.0, 0.75)
    reference = _boundary_frame("A0", 42, 2.0, 0.50).iloc[:-1].copy()
    with pytest.raises(ValueError, match="same fixed atom-token targets"):
        pair_boundary_cells(
            candidate,
            reference,
            comparison=BOUNDARY_COMPARISONS[0],
            seed=42,
        )
