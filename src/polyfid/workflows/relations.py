"""Scalable construction of the verified PI1M relation set."""

from __future__ import annotations

import json
import random
import re
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

from polyfid.paths import ProjectPaths
from polyfid.chemistry import (
    StructureError,
    generate_equivalent_periodic_cuts,
    generate_randomized_smiles,
)
from polyfid.relations import (
    TypedRelationRecord,
    build_positive_views,
    build_supplemental_negative_views,
    make_anchor_view,
    stereo_agnostic_graphs_differ,
    stereo_agnostic_two_cell_identity,
)
from polyfid.workflows.periodic_data import prepare_periodic_pi1m


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=True
)


def _stereo_agnostic_identity_batch(values: tuple[str, ...]) -> list[str | None]:
    return [stereo_agnostic_two_cell_identity(value) for value in values]


def _stereo_agnostic_identities(
    values: list[str], *, workers: int, chunk_size: int = 4096
) -> list[str | None]:
    """Canonicalize a relation index in bounded CPU batches."""

    chunks = [
        tuple(values[start : start + chunk_size])
        for start in range(0, len(values), chunk_size)
    ]
    if int(workers) <= 1:
        return [value for chunk in chunks for value in _stereo_agnostic_identity_batch(chunk)]
    output: list[str | None] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        for block in executor.map(_stereo_agnostic_identity_batch, chunks, chunksize=1):
            output.extend(block)
    return output


def build_relation_index(
    paths: ProjectPaths,
    *,
    partition: str = "train",
    workers: int = 12,
    max_rows: int | None = None,
    seed: int = 42,
) -> Path:
    periodic_data = prepare_periodic_pi1m(paths, workers=workers)
    source = Path(str(periodic_data["output_path"]))
    columns = [
        "sample_id",
        "canonical_psmiles",
        "closed_smiles",
        "periodic_supercell_smiles",
        "periodic_identity",
        "repeat_equivalence_id",
        "source_repeat_equivalence_id",
        "formula",
        "atom_count",
        "split",
    ]
    frame = pd.read_parquet(source, columns=columns, filters=[("split", "==", partition)])
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=seed).sort_values("sample_id").reset_index(drop=True)
    frame["stereo_agnostic_periodic_identity"] = _stereo_agnostic_identities(
        frame["periodic_identity"].astype(str).tolist(), workers=workers
    )
    tokenizer = AutoTokenizer.from_pretrained(
        paths.shared_model_root / "chemberta_100m_mlm",
        local_files_only=True,
        use_fast=True,
    )
    lengths: list[int] = []
    for start in range(0, len(frame), 8192):
        encoded = tokenizer(
            frame["canonical_psmiles"].iloc[start : start + 8192].astype(str).tolist(),
            add_special_tokens=True,
            truncation=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    frame["token_length"] = lengths
    suffix = "" if max_rows is None else f"_smoke_{max_rows}"
    destination = paths.project_root / f"data/indices/pi1m_relation_index_{partition}{suffix}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return destination


@lru_cache(maxsize=300_000)
def _morgan_bits(smiles: str) -> Any | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(molecule)


@lru_cache(maxsize=10_000)
def _formula_counts(formula: str) -> tuple[tuple[str, int], ...]:
    """Parse an RDKit molecular formula into deterministic element counts."""

    return tuple(
        (element, int(count) if count else 1)
        for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula))
    )


def _formula_distance(left: str, right: str) -> int:
    """Element-count distance used after exact-formula matching is exhausted."""

    left_counts = dict(_formula_counts(str(left)))
    right_counts = dict(_formula_counts(str(right)))
    elements = left_counts.keys() | right_counts.keys()
    return int(
        sum(abs(left_counts.get(element, 0) - right_counts.get(element, 0)) for element in elements)
    )


def _matched_negative(
    anchor: pd.Series,
    candidates: pd.DataFrame,
    *,
    rng: random.Random,
    similarity_minimum: float,
    max_candidates: int = 256,
) -> tuple[pd.Series, float] | None:
    pool = candidates.loc[
        candidates["sample_id"].ne(anchor["sample_id"])
        & candidates["periodic_identity"].ne(anchor["periodic_identity"])
        & candidates["stereo_agnostic_periodic_identity"].notna()
        & candidates["stereo_agnostic_periodic_identity"].ne(
            anchor["stereo_agnostic_periodic_identity"]
        )
        & (candidates["token_length"] - int(anchor["token_length"])).abs().le(5)
    ]
    if pool.empty:
        return None
    positions = list(range(len(pool)))
    if len(positions) > max_candidates:
        positions = sorted(rng.sample(positions, max_candidates))
    pool = pool.iloc[positions]
    anchor_fp = _morgan_bits(str(anchor["closed_smiles"]))
    if anchor_fp is None:
        return None
    candidate_positions: list[int] = []
    candidate_fps: list[Any] = []
    for row_index, closed_smiles in enumerate(pool["closed_smiles"].astype(str)):
        candidate_fp = _morgan_bits(closed_smiles)
        if candidate_fp is not None:
            candidate_positions.append(row_index)
            candidate_fps.append(candidate_fp)
    if not candidate_fps:
        return None
    scored = list(
        zip(
            DataStructs.BulkTanimotoSimilarity(anchor_fp, candidate_fps),
            candidate_positions,
            strict=True,
        )
    )
    eligible = [item for item in scored if item[0] >= similarity_minimum]
    if not eligible:
        return None
    similarity, position = min(
        eligible,
        key=lambda item: (
            _formula_distance(
                str(anchor["formula"]), str(pool.iloc[item[1]]["formula"])
            ),
            abs(int(pool.iloc[item[1]]["token_length"]) - int(anchor["token_length"])),
            abs(
                len(str(pool.iloc[item[1]]["canonical_psmiles"]))
                - len(str(anchor["canonical_psmiles"]))
            ),
            -item[0],
            item[1],
        ),
    )
    return pool.iloc[position], similarity


def _sorted_position_buckets(
    index: pd.DataFrame,
    columns: str | list[str],
) -> dict[Any, np.ndarray]:
    """Index partition rows by chemistry strata, ordered by token length."""

    ordered = index.sort_values(["token_length", "sample_id"], kind="stable")
    grouped = ordered.groupby(columns, sort=False, dropna=False)
    return {
        key: cell.index.to_numpy(dtype=int)
        for key, cell in grouped
    }


def _near_token_positions(
    positions: np.ndarray,
    token_lengths: np.ndarray,
    *,
    target_length: int,
    limit: int,
) -> list[int]:
    """Return a bounded token-length neighbourhood from a pre-sorted bucket."""

    if positions.size <= limit:
        return [int(value) for value in positions]
    values = token_lengths[positions]
    centre = bisect_left(values, int(target_length))
    half_width = max(1, int(limit) // 2)
    start = max(0, centre - half_width)
    stop = min(len(positions), start + int(limit))
    start = max(0, stop - int(limit))
    return [int(value) for value in positions[start:stop]]


def _candidate_neighbourhood(
    index: pd.DataFrame,
    *,
    anchor_formula: str,
    anchor_atom_count: int,
    anchor_token_length: int,
    exact_buckets: dict[Any, np.ndarray],
    atom_buckets: dict[Any, np.ndarray],
    global_positions: np.ndarray,
    limit_per_tier: int,
) -> list[int]:
    token_lengths = index["token_length"].to_numpy(dtype=int)
    tiers = (
        exact_buckets.get(
            (anchor_formula, anchor_atom_count), np.asarray([], dtype=int)
        ),
        atom_buckets.get(anchor_atom_count, np.asarray([], dtype=int)),
        global_positions,
    )
    values: list[int] = []
    seen: set[int] = set()
    for tier in tiers:
        for position in _near_token_positions(
            tier,
            token_lengths,
            target_length=anchor_token_length,
            limit=limit_per_tier,
        ):
            if position not in seen:
                values.append(position)
                seen.add(position)
    return values


def _build_retrieval_decoy_bank(
    index: pd.DataFrame,
    relations: pd.DataFrame,
    *,
    count: int = 99,
    candidate_limit_per_tier: int = 2048,
) -> tuple[pd.DataFrame, list[list[int]], dict[str, Any]]:
    """Build one fixed bank from the full partition index and matched bank lookups."""

    if len(index) <= int(count):
        raise ValueError("relation partition is too small for the requested decoy count")
    if "stereo_agnostic_periodic_identity" not in index:
        index = index.copy()
        index["stereo_agnostic_periodic_identity"] = [
            stereo_agnostic_two_cell_identity(value)
            for value in index["periodic_identity"].astype(str)
        ]
    exact_buckets = _sorted_position_buckets(index, ["formula", "atom_count"])
    atom_buckets = _sorted_position_buckets(index, "atom_count")
    global_positions = index.sort_values(
        ["token_length", "sample_id"], kind="stable"
    ).index.to_numpy(dtype=int)
    requested = int(count)
    formulas = index["formula"].astype(str).to_numpy()
    atom_counts = index["atom_count"].to_numpy(dtype=int)
    token_lengths = index["token_length"].to_numpy(dtype=int)
    sample_ids = index["sample_id"].astype(str).to_numpy()
    identities = index["periodic_identity"].astype(str).to_numpy()
    graph_identities = index["stereo_agnostic_periodic_identity"].to_numpy(
        dtype=object
    )
    source_selections: list[list[int]] = []
    exact_formula: list[bool] = []
    same_atom_count: list[bool] = []
    token_deltas: list[int] = []
    for _, anchor in relations.iterrows():
        candidates = _candidate_neighbourhood(
            index,
            anchor_formula=str(anchor["anchor_formula"]),
            anchor_atom_count=int(anchor["anchor_atom_count"]),
            anchor_token_length=int(anchor["anchor_token_length"]),
            exact_buckets=exact_buckets,
            atom_buckets=atom_buckets,
            global_positions=global_positions,
            limit_per_tier=max(candidate_limit_per_tier, requested * 4),
        )
        candidate_array = np.asarray(candidates, dtype=int)
        ordering = np.lexsort(
            (
                sample_ids[candidate_array],
                np.abs(token_lengths[candidate_array] - int(anchor["anchor_token_length"])),
                np.abs(atom_counts[candidate_array] - int(anchor["anchor_atom_count"])),
                formulas[candidate_array] != str(anchor["anchor_formula"]),
            )
        )
        eligible: list[int] = []
        anchor_graph_identity = str(
            anchor.get(
                "anchor_stereo_agnostic_identity",
                stereo_agnostic_two_cell_identity(str(anchor["anchor_identity"])),
            )
        )
        seen_identities = {str(anchor["anchor_identity"])}
        seen_graph_identities = {anchor_graph_identity}
        for position in candidate_array[ordering]:
            position = int(position)
            identity = identities[position]
            if pd.isna(graph_identities[position]):
                continue
            graph_identity = str(graph_identities[position])
            if identity in seen_identities or graph_identity in seen_graph_identities:
                continue
            eligible.append(position)
            seen_identities.add(identity)
            seen_graph_identities.add(graph_identity)
            if len(eligible) == requested:
                break
        if len(eligible) != requested:
            raise ValueError(
                f"only {len(eligible)} distinct retrieval decoys are available for "
                f"anchor {anchor['anchor_sample_id']}; requested {requested}"
            )
        source_selections.append(eligible)
        for position in eligible:
            exact_formula.append(
                formulas[position] == str(anchor["anchor_formula"])
            )
            same_atom_count.append(
                atom_counts[position] == int(anchor["anchor_atom_count"])
            )
            token_deltas.append(
                abs(
                    token_lengths[position]
                    - int(anchor["anchor_token_length"])
                )
            )

    bank_source_positions = sorted(
        {position for selections in source_selections for position in selections}
    )
    bank_lookup = {
        source_position: bank_index
        for bank_index, source_position in enumerate(bank_source_positions)
    }
    bank_indices = [
        [bank_lookup[position] for position in selections]
        for selections in source_selections
    ]
    bank = index.iloc[bank_source_positions][
        [
            "sample_id",
            "canonical_psmiles",
            "closed_smiles",
            "periodic_supercell_smiles",
            "periodic_identity",
            "stereo_agnostic_periodic_identity",
            "formula",
            "atom_count",
            "token_length",
            "split",
        ]
    ].copy()
    bank.insert(0, "decoy_bank_index", np.arange(len(bank), dtype=int))
    bank = bank.rename(
        columns={
            "canonical_psmiles": "decoy_psmiles",
            "periodic_identity": "decoy_identity",
            "stereo_agnostic_periodic_identity": "decoy_stereo_agnostic_identity",
        }
    )
    delta_array = np.asarray(token_deltas, dtype=float)
    summary: dict[str, Any] = {
        "decoys_per_anchor": requested,
        "distinct_decoy_identities_per_anchor": requested,
        "bank_size": int(len(bank)),
        "source_partition_rows": int(len(index)),
        "exact_formula_fraction": float(np.mean(exact_formula)),
        "same_atom_count_fraction": float(np.mean(same_atom_count)),
        "token_delta": {
            "mean": float(np.mean(delta_array)),
            "median": float(np.median(delta_array)),
            "p90": float(np.quantile(delta_array, 0.90)),
            "maximum": int(np.max(delta_array)),
        },
    }
    return bank, bank_indices, summary


def _supplemental_controls(
    arguments: tuple[str, str, str, int],
) -> list[dict[str, Any]]:
    sample_id, psmiles, split, local_seed = arguments
    anchor = make_anchor_view(sample_id, psmiles, include_profile=False)
    controls = build_supplemental_negative_views(
        anchor,
        max_endpoint_relocations=1,
        max_closure_rewirings=1,
        seed=local_seed,
    )
    return [
        {
            "anchor_sample_id": anchor.sample_id,
            "anchor_psmiles": anchor.psmiles,
            "anchor_identity": anchor.canonical_identity,
            "negative_sample_id": control.sample_id,
            "negative_psmiles": control.psmiles,
            "negative_identity": control.canonical_identity,
            "negative_relation": control.relation,
            "split": split,
        }
        for control in controls
    ]


_TYPED_RELATION_COLUMNS = [
    "relation_id",
    "anchor_id",
    "paired_id",
    "split",
    "split_group",
    "relation_family",
    "relation_class",
    "two_cell_identity_relation",
    "changed_fields",
    "parse_status",
    "polymer_semantic_status",
    "representation_target",
    "property_target",
    "training_eligible",
    "evaluation_eligible",
    "evidence_rule",
    "exclusion_reason",
]


def _qualify_relation_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Prefix legacy sidecar IDs with split so partitions concatenate cleanly."""

    output = frame.copy()
    output["relation_id"] = [
        relation_id
        if str(relation_id).startswith(f"{split}::")
        else f"{split}::{relation_id}"
        for relation_id, split in zip(
            output["relation_id"].astype(str),
            output["split"].astype(str),
            strict=True,
        )
    ]
    if output["relation_id"].duplicated().any():
        raise ValueError("typed relation IDs must be unique within a partition")
    return output


def _equivalence_changed_fields(family: str) -> tuple[str, ...]:
    return {
        "randomized_traversal": ("serialization",),
        "generic_randomized_traversal": ("serialization",),
        "alternative_periodic_cut": ("repeat_window_cut",),
        "reverse_when_non_noop": ("repeat_direction",),
        "atom_order_permutation": ("atom_order",),
    }.get(str(family), ("serialization",))


def _typed_relation_registry(
    relation_frame: pd.DataFrame,
    supplemental_frame: pd.DataFrame,
    exclusion_rows: list[dict[str, Any]],
    *,
    semantic_controls: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create an analysis sidecar without changing the triplet training table."""

    records: list[TypedRelationRecord] = []
    for row in relation_frame.to_dict(orient="records"):
        triplet_id = str(row["triplet_id"])
        anchor_id = str(row["anchor_sample_id"])
        split = str(row["split"])
        split_group = str(row["anchor_identity"])
        positive_family = str(row["positive_relation"])
        records.append(
            TypedRelationRecord(
                relation_id=f"{split}::{triplet_id}::positive",
                anchor_id=anchor_id,
                paired_id=str(row["positive_sample_id"]),
                split=split,
                split_group=split_group,
                relation_family=positive_family,
                relation_class="equivalence_preserving",
                two_cell_identity_relation="same",
                changed_fields=_equivalence_changed_fields(positive_family),
                parse_status="parseable",
                polymer_semantic_status="valid_repeat_unit",
                representation_target="close",
                property_target="stable_structure_only",
                training_eligible=True,
                evaluation_eligible=True,
                evidence_rule="same_two_cell_periodic_identity",
            )
        )
        records.append(
            TypedRelationRecord(
                relation_id=f"{split}::{triplet_id}::generic_positive",
                anchor_id=anchor_id,
                paired_id=f"{anchor_id}::generic_randomized_traversal",
                split=split,
                split_group=split_group,
                relation_family="generic_randomized_traversal",
                relation_class="equivalence_preserving",
                two_cell_identity_relation="same",
                changed_fields=("serialization",),
                parse_status="parseable",
                polymer_semantic_status="valid_repeat_unit",
                representation_target="close",
                property_target="stable_structure_only",
                training_eligible=True,
                evaluation_eligible=True,
                evidence_rule="identity_preserving_randomized_traversal",
            )
        )
        records.append(
            TypedRelationRecord(
                relation_id=f"{split}::{triplet_id}::negative",
                anchor_id=anchor_id,
                paired_id=str(row["negative_sample_id"]),
                split=split,
                split_group=split_group,
                relation_family=str(row["negative_relation"]),
                relation_class="topology_changing",
                two_cell_identity_relation="different",
                changed_fields=("two_cell_structural_graph",),
                parse_status="parseable",
                polymer_semantic_status="valid_repeat_unit",
                representation_target="separate",
                property_target="unconstrained",
                training_eligible=True,
                evaluation_eligible=True,
                evidence_rule="distinct_stereo_agnostic_two_cell_graph",
            )
        )

    for row in supplemental_frame.to_dict(orient="records"):
        family = str(row["negative_relation"])
        records.append(
            TypedRelationRecord(
                relation_id=f"{row['split']}::supplemental::{row['negative_sample_id']}",
                anchor_id=str(row["anchor_sample_id"]),
                paired_id=str(row["negative_sample_id"]),
                split=str(row["split"]),
                split_group=str(row["anchor_identity"]),
                relation_family=family,
                relation_class="topology_changing",
                two_cell_identity_relation="different",
                changed_fields=(
                    ("attachment_endpoint",)
                    if family == "endpoint_relocation"
                    else ("periodic_closure",)
                ),
                parse_status="parseable",
                polymer_semantic_status="valid_repeat_unit",
                representation_target="separate",
                property_target="unconstrained",
                training_eligible=False,
                evaluation_eligible=True,
                evidence_rule="verified_graph_edit_with_distinct_stereo_agnostic_identity",
            )
        )

    if semantic_controls is not None:
        for row in semantic_controls.to_dict(orient="records"):
            if str(row.get("control_family")) == "valid":
                continue
            records.append(
                TypedRelationRecord(
                    relation_id=f"{row.get('split', 'test')}::{row['control_id']}",
                    anchor_id=str(row["sample_id"]),
                    paired_id=str(row["control_id"]),
                    split=str(row.get("split", "test")),
                    split_group=str(row.get("split_group", row["sample_id"])),
                    relation_family=str(row["control_family"]),
                    relation_class="polymer_semantics_inapplicable",
                    two_cell_identity_relation="unknown",
                    changed_fields=("attachment_markers", "polymer_semantics"),
                    parse_status=str(row.get("parse_status", "parseable")),
                    polymer_semantic_status="inapplicable_as_repeat_unit",
                    representation_target="descriptive_only",
                    property_target="none",
                    training_eligible=False,
                    evaluation_eligible=True,
                    evidence_rule="applicability_control_only",
                )
            )

    records.extend(
        TypedRelationRecord(
            relation_id=f"{row['split']}::ineligible::{row['anchor_id']}",
            anchor_id=str(row["anchor_id"]),
            paired_id=None,
            split=str(row["split"]),
            split_group=str(row["split_group"]),
            relation_family="eligibility_exclusion",
            relation_class="unknown_or_ineligible",
            two_cell_identity_relation="unknown",
            changed_fields=(),
            parse_status="parseable",
            polymer_semantic_status="valid_repeat_unit",
            representation_target="none",
            property_target="none",
            training_eligible=False,
            evaluation_eligible=False,
            evidence_rule="relation_builder_eligibility_diagnostic",
            exclusion_reason=str(row["exclusion_reason"]),
        )
        for row in exclusion_rows
    )
    return _qualify_relation_ids(
        pd.DataFrame(
            [record.to_record() for record in records],
            columns=_TYPED_RELATION_COLUMNS,
        )
    )


def _semantic_applicability_controls(
    relation_frame: pd.DataFrame,
    *,
    partition: str,
    seed: int,
) -> pd.DataFrame:
    """Reuse the fidelity view builder for scope-only semantic controls."""

    from polyfid.evaluation.invariance import build_invariance_views

    anchors = relation_frame[
        ["anchor_sample_id", "anchor_psmiles", "anchor_identity"]
    ].rename(
        columns={
            "anchor_sample_id": "sample_id",
            "anchor_psmiles": "canonical_psmiles",
        }
    )
    _, controls, _ = build_invariance_views(
        anchors,
        sample_id_column="sample_id",
        text_column="canonical_psmiles",
        randomized_count=0,
        periodic_count=0,
        atom_permutation_count=0,
        seed=seed,
    )
    controls = controls.loc[controls["control_family"].ne("valid")].copy()
    controls["split"] = partition
    split_groups = anchors.set_index("sample_id")["anchor_identity"].astype(str)
    controls["split_group"] = controls["sample_id"].astype(str).map(split_groups)
    controls["parse_status"] = "parseable"
    controls["polymer_semantic_status"] = "inapplicable_as_repeat_unit"
    return controls.reset_index(drop=True)


def _shortcut_diagnostic(frame: pd.DataFrame, *, seed: int) -> dict[str, float | int | bool]:
    positive = pd.DataFrame(
        {
            "anchor_id": frame["anchor_sample_id"],
            "label": 1,
            "character_difference": (frame["anchor_psmiles"].str.len() - frame["positive_psmiles"].str.len()).abs(),
            "token_difference": (frame["anchor_token_length"] - frame["positive_token_length"]).abs(),
            "formula_mismatch": 0,
        }
    )
    negative = pd.DataFrame(
        {
            "anchor_id": frame["anchor_sample_id"],
            "label": 0,
            "character_difference": (frame["anchor_psmiles"].str.len() - frame["negative_psmiles"].str.len()).abs(),
            "token_difference": (frame["anchor_token_length"] - frame["negative_token_length"]).abs(),
            "formula_mismatch": (frame["anchor_formula"] != frame["negative_formula"]).astype(int),
        }
    )
    pairs = pd.concat((positive, negative), ignore_index=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, test = next(splitter.split(pairs, groups=pairs["anchor_id"]))
    feature_columns = ["character_difference", "token_difference", "formula_mismatch"]
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(pairs.iloc[train][feature_columns], pairs.iloc[train]["label"])
    probability = model.predict_proba(pairs.iloc[test][feature_columns])[:, 1]
    auc = float(roc_auc_score(pairs.iloc[test]["label"], probability))
    return {
        "heldout_auc": auc,
        "rows": int(len(pairs)),
        "rebalance_recommended": auc >= 0.80,
    }


def _manual_review_sample(
    frame: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> pd.DataFrame:
    """Draw one deterministic relation-family-stratified review sheet."""

    requested = min(int(count), len(frame))
    families = sorted(frame["positive_relation"].astype(str).unique())
    selected_indices: list[int] = []
    if requested and families:
        base, remainder = divmod(requested, len(families))
        for family_index, family in enumerate(families):
            cell = frame.loc[frame["positive_relation"].astype(str).eq(family)]
            take = min(len(cell), base + int(family_index < remainder))
            if take:
                selected_indices.extend(
                    cell.sample(n=take, random_state=seed + family_index).index.tolist()
                )
        if len(selected_indices) < requested:
            remaining = frame.drop(index=selected_indices)
            selected_indices.extend(
                remaining.sample(
                    n=requested - len(selected_indices), random_state=seed + 10_000
                ).index.tolist()
            )
    review = frame.loc[selected_indices].copy().reset_index(drop=True)
    review.insert(0, "review_id", [f"review:{index}" for index in range(len(review))])
    review["relation_family"] = review["positive_relation"].astype(str)
    review["positive_equivalent_verdict"] = ""
    review["negative_topology_changed_verdict"] = ""
    review["reviewer_notes"] = ""
    return review


def build_relation_set(
    paths: ProjectPaths,
    *,
    partition: str = "train",
    target_triplets: int = 10_000,
    seed: int = 42,
    max_index_rows: int | None = None,
    workers: int = 12,
    similarity_minimum: float = 0.60,
    rebuild_index: bool = False,
    allow_near_formula_fallback: bool = False,
) -> dict[str, Any]:
    suffix = "" if max_index_rows is None else f"_smoke_{max_index_rows}"
    index_path = paths.project_root / f"data/indices/pi1m_relation_index_{partition}{suffix}.parquet"
    should_rebuild_index = bool(rebuild_index) or not index_path.is_file()
    if not should_rebuild_index:
        import pyarrow.parquet as pq

        should_rebuild_index = (
            not {
                "periodic_identity",
                "stereo_agnostic_periodic_identity",
            }.issubset(pq.read_schema(index_path).names)
        )
    if should_rebuild_index:
        index_path = build_relation_index(
            paths,
            partition=partition,
            workers=workers,
            max_rows=max_index_rows,
            seed=seed,
        )
    index = pd.read_parquet(index_path)
    index = index.dropna(subset=["formula"]).reset_index(drop=True)
    exact_candidate_buckets = _sorted_position_buckets(
        index, ["formula", "atom_count"]
    )
    atom_candidate_buckets = _sorted_position_buckets(index, "atom_count")
    order = list(range(len(index)))
    random.Random(seed).shuffle(order)
    family_counts: Counter[str] = Counter()
    eligibility_counts: Counter[str] = Counter()
    exclusion_rows: list[dict[str, Any]] = []
    periodic_view_counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    supplemental: list[dict[str, Any]] = []
    anchors_examined = 0
    primary_match_counts: Counter[str] = Counter()
    tokenizer = AutoTokenizer.from_pretrained(
        paths.shared_model_root / "chemberta_100m_mlm",
        local_files_only=True,
        use_fast=True,
    )

    def record_exclusion(anchor_row: pd.Series, reason: str) -> None:
        eligibility_counts[reason] += 1
        exclusion_rows.append(
            {
                "anchor_id": str(anchor_row["sample_id"]),
                "split": partition,
                "split_group": str(anchor_row["periodic_identity"]),
                "exclusion_reason": reason,
            }
        )

    for order_index in order:
        if len(rows) >= target_triplets:
            break
        anchors_examined += 1
        anchor_row = index.iloc[order_index]
        rng = random.Random(seed + order_index * 104729)
        exact_positions = exact_candidate_buckets.get(
            (anchor_row["formula"], anchor_row["atom_count"]),
            np.asarray([], dtype=int),
        )
        exact_neighbours = _near_token_positions(
            exact_positions,
            index["token_length"].to_numpy(dtype=int),
            target_length=int(anchor_row["token_length"]),
            limit=2048,
        )
        matched = (
            _matched_negative(
                anchor_row,
                index.iloc[exact_neighbours],
                rng=rng,
                similarity_minimum=similarity_minimum,
                max_candidates=2048,
            )
            if exact_neighbours
            else None
        )
        primary_match_stratum = "same_formula_atom_count"
        if matched is None:
            if not allow_near_formula_fallback:
                record_exclusion(
                    anchor_row,
                    "no_same_formula_negative_at_similarity_threshold",
                )
                continue
            atom_positions = atom_candidate_buckets.get(
                int(anchor_row["atom_count"]), np.asarray([], dtype=int)
            )
            atom_neighbours = _near_token_positions(
                atom_positions,
                index["token_length"].to_numpy(dtype=int),
                target_length=int(anchor_row["token_length"]),
                limit=4096,
            )
            if len(atom_neighbours) <= 1:
                record_exclusion(anchor_row, "no_same_atom_count_candidate")
                continue
            matched = _matched_negative(
                anchor_row,
                index.iloc[atom_neighbours],
                rng=rng,
                similarity_minimum=similarity_minimum,
                max_candidates=4096,
            )
            primary_match_stratum = "nearest_formula_same_atom_count"
        if matched is None:
            record_exclusion(
                anchor_row, "no_observed_negative_at_similarity_threshold"
            )
            continue
        negative_row, similarity = matched
        if not stereo_agnostic_graphs_differ(
            str(anchor_row["periodic_identity"]),
            str(negative_row["periodic_identity"]),
        ):
            record_exclusion(anchor_row, "no_distinct_stereo_agnostic_graph_negative")
            continue
        if str(negative_row["formula"]) == str(anchor_row["formula"]):
            primary_match_stratum = "same_formula_atom_count"
        primary_formula_distance = _formula_distance(
            str(anchor_row["formula"]), str(negative_row["formula"])
        )
        # A_gcl and A_eqs share the identical observed negative.  A_gcl treats
        # it as an ordinary instance negative, while A_eqs pairs it with a
        # verified equivalence view.  Sharing the row removes negative
        # composition and difficulty as a source of the H3 difference.
        generic_row = negative_row
        generic_similarity = similarity
        generic_match_stratum = "shared_paired_observed_negative"
        anchor = make_anchor_view(
            str(anchor_row["sample_id"]),
            str(anchor_row["canonical_psmiles"]),
            include_profile=False,
        )
        try:
            generic_candidates = generate_randomized_smiles(
                anchor.psmiles, count=10, seed=seed + order_index
            )
        except StructureError as error:
            record_exclusion(
                anchor_row,
                f"generic_traversal_generation_failed:{error.code}",
            )
            continue
        generic_positive = next(
            (
                str(candidate)
                for candidate in generic_candidates
                if str(candidate).strip() != anchor.psmiles.strip()
            ),
            None,
        )
        if generic_positive is None:
            record_exclusion(anchor_row, "no_nontrivial_generic_traversal")
            continue
        positives = build_positive_views(
            anchor,
            randomized_count=1,
            periodic_max_views=10,
            include_reverse=True,
            seed=seed + order_index,
        )
        positives = tuple(
            view
            for view in positives
            if view.psmiles.strip() != anchor.psmiles.strip()
        )
        if not positives:
            record_exclusion(anchor_row, "no_nontrivial_positive")
            continue
        anchor_periodic_identity_representable = anchor.periodic_identity is not None
        anchor_nontrivial_periodic_cut_available = any(
            view.relation == "alternative_periodic_cut" for view in positives
        )
        negative_periodic_views = generate_equivalent_periodic_cuts(
            str(negative_row["canonical_psmiles"]),
            max_views=10,
            seed=seed + order_index,
        )
        negative_periodic_identity_representable = bool(negative_periodic_views)
        negative_nontrivial_periodic_cut_available = any(
            view.transformation != "reference" for view in negative_periodic_views
        )
        periodic_view_counts[
            "anchor_nontrivial_periodic_cut_available"
            if anchor_nontrivial_periodic_cut_available
            else "anchor_identity_representable_no_nontrivial_cut"
        ] += 1
        periodic_view_counts[
            "negative_nontrivial_periodic_cut_available"
            if negative_nontrivial_periodic_cut_available
            else "negative_identity_representable_no_nontrivial_cut"
        ] += 1
        priorities = sorted(
            positives,
            key=lambda view: (family_counts[view.relation], view.relation, view.sample_id),
        )
        positive = priorities[0]
        positive_tokens = len(tokenizer.encode(positive.psmiles))
        row = {
            "triplet_id": f"triplet:{len(rows)}",
            "split": partition,
            "anchor_sample_id": anchor.sample_id,
            "anchor_psmiles": anchor.psmiles,
            "anchor_identity": anchor.canonical_identity,
            "anchor_stereo_agnostic_identity": str(
                anchor_row["stereo_agnostic_periodic_identity"]
            ),
            "anchor_formula": str(anchor_row["formula"]),
            "anchor_atom_count": int(anchor_row["atom_count"]),
            "anchor_token_length": int(anchor_row["token_length"]),
            "anchor_periodic_identity_representable": bool(
                anchor_periodic_identity_representable
            ),
            "anchor_nontrivial_periodic_cut_available": bool(
                anchor_nontrivial_periodic_cut_available
            ),
            "positive_sample_id": positive.sample_id,
            "positive_psmiles": positive.psmiles,
            "positive_identity": positive.canonical_identity,
            "positive_relation": positive.relation,
            "positive_token_length": positive_tokens,
            "negative_sample_id": str(negative_row["sample_id"]),
            "negative_psmiles": str(negative_row["canonical_psmiles"]),
            "negative_identity": str(negative_row["periodic_identity"]),
            "negative_stereo_agnostic_identity": str(
                negative_row["stereo_agnostic_periodic_identity"]
            ),
            "negative_relation": "matched_real_periodic_topology",
            "negative_formula": str(negative_row["formula"]),
            "negative_formula_distance": int(primary_formula_distance),
            "negative_match_stratum": primary_match_stratum,
            "negative_atom_count": int(negative_row["atom_count"]),
            "negative_token_length": int(negative_row["token_length"]),
            "negative_periodic_identity_representable": bool(
                negative_periodic_identity_representable
            ),
            "negative_nontrivial_periodic_cut_available": bool(
                negative_nontrivial_periodic_cut_available
            ),
            "morgan_similarity": float(similarity),
            "generic_negative_sample_id": str(generic_row["sample_id"]),
            "generic_positive_psmiles": generic_positive,
            "generic_positive_relation": "randomized_traversal",
            "generic_negative_psmiles": str(generic_row["canonical_psmiles"]),
            "generic_negative_identity": str(generic_row["periodic_identity"]),
            "generic_negative_stereo_agnostic_identity": str(
                generic_row["stereo_agnostic_periodic_identity"]
            ),
            "generic_negative_source": "observed",
            "generic_negative_formula": str(generic_row["formula"]),
            "generic_negative_atom_count": int(generic_row["atom_count"]),
            "generic_negative_token_length": int(generic_row["token_length"]),
            "generic_negative_morgan_similarity": float(generic_similarity),
            "generic_negative_match_stratum": generic_match_stratum,
            "generic_negative_exact_formula": bool(
                str(generic_row["formula"]) == str(anchor_row["formula"])
            ),
            "generic_negative_same_atom_count": bool(
                int(generic_row["atom_count"]) == int(anchor_row["atom_count"])
            ),
            "generic_negative_token_delta_gap_from_paired": abs(
                abs(
                    int(generic_row["token_length"])
                    - int(anchor_row["token_length"])
                )
                - abs(
                    int(negative_row["token_length"])
                    - int(anchor_row["token_length"])
                )
            ),
            "generic_negative_morgan_similarity_gap_from_paired": abs(
                float(generic_similarity) - float(similarity)
            ),
        }
        rows.append(row)
        primary_match_counts[primary_match_stratum] += 1
        family_counts[positive.relation] += 1

    relation_frame = pd.DataFrame(rows)
    supplemental_inputs = [
        (
            str(row["anchor_sample_id"]),
            str(row["anchor_psmiles"]),
            partition,
            seed + position,
        )
        for position, row in relation_frame.iterrows()
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for controls in executor.map(
            _supplemental_controls, supplemental_inputs, chunksize=16
        ):
            supplemental.extend(controls)
    output_root = paths.project_root / "data/relations" / partition
    if max_index_rows is not None:
        output_root = output_root / f"smoke_{max_index_rows}"
    output_root.mkdir(parents=True, exist_ok=True)
    decoy_bank, decoy_rows, decoy_summary = _build_retrieval_decoy_bank(
        index, relation_frame, count=99
    )
    relation_frame["retrieval_decoy_bank_indices"] = decoy_rows
    decoy_bank_path = output_root / "retrieval_decoy_bank.parquet"
    decoy_bank.to_parquet(decoy_bank_path, index=False)
    relation_path = output_root / "pi1m_triplets.parquet"
    relation_frame.to_parquet(relation_path, index=False)
    supplemental_path = output_root / "supplemental_topology_controls.parquet"
    supplemental_frame = pd.DataFrame(supplemental)
    supplemental_frame.to_parquet(supplemental_path, index=False)
    semantic_controls = _semantic_applicability_controls(
        relation_frame,
        partition=partition,
        seed=seed,
    )
    semantic_controls_path = output_root / "semantic_applicability_controls.parquet"
    semantic_controls.to_parquet(semantic_controls_path, index=False)
    typed_registry = _typed_relation_registry(
        relation_frame,
        supplemental_frame,
        exclusion_rows,
        semantic_controls=semantic_controls,
    )
    typed_registry_path = output_root / "typed_relation_registry.parquet"
    typed_registry.to_parquet(typed_registry_path, index=False)
    review_count = min(200, len(relation_frame))
    review_source = relation_frame.drop(columns="retrieval_decoy_bank_indices")
    review = _manual_review_sample(review_source, count=review_count, seed=seed)
    review.to_csv(output_root / "manual_review_sample.csv", index=False)
    diagnostic = _shortcut_diagnostic(relation_frame, seed=seed) if len(relation_frame) >= 20 else {}
    paired_token_delta = (
        relation_frame["negative_token_length"]
        - relation_frame["anchor_token_length"]
    ).abs()
    generic_token_delta = (
        relation_frame["generic_negative_token_length"]
        - relation_frame["anchor_token_length"]
    ).abs()
    similarity_gap = (
        relation_frame["generic_negative_morgan_similarity"]
        - relation_frame["morgan_similarity"]
    ).abs()
    generic_summary = {
        "negative_policy": "shared_paired_observed_negative",
        "comparison_isolated_to": "positive_relation",
        "match_stratum_counts": {
            str(key): int(value)
            for key, value in relation_frame["generic_negative_match_stratum"]
            .value_counts()
            .sort_index()
            .items()
        },
        "exact_formula_fraction": float(
            (
                relation_frame["generic_negative_formula"]
                == relation_frame["anchor_formula"]
            ).mean()
        ),
        "same_atom_count_fraction": float(
            (
                relation_frame["generic_negative_atom_count"]
                == relation_frame["anchor_atom_count"]
            ).mean()
        ),
        "mean_absolute_token_delta_gap_from_paired": float(
            (generic_token_delta - paired_token_delta).abs().mean()
        ),
        "mean_absolute_morgan_similarity_gap_from_paired": float(
            similarity_gap.mean()
        ),
        "morgan_similarity_gap_from_paired": {
            "median": float(similarity_gap.median()),
            "p90": float(similarity_gap.quantile(0.90)),
            "maximum": float(similarity_gap.max()),
        },
        "by_match_stratum": {
            str(stratum): {
                "rows": int(len(cell)),
                "mean_absolute_morgan_similarity_gap_from_paired": float(
                    cell["generic_negative_morgan_similarity_gap_from_paired"].mean()
                ),
                "mean_absolute_token_delta_gap_from_paired": float(
                    cell["generic_negative_token_delta_gap_from_paired"].mean()
                ),
            }
            for stratum, cell in relation_frame.groupby(
                "generic_negative_match_stratum", sort=True
            )
        },
    }
    formula_distances = relation_frame["negative_formula_distance"].to_numpy(dtype=float)
    primary_summary = {
        "matching_order": (
            "same_formula_then_nearest_formula_at_same_atom_count"
            if allow_near_formula_fallback
            else "same_formula_atom_count_only"
        ),
        "near_formula_fallback_enabled": bool(allow_near_formula_fallback),
        "match_stratum_counts": dict(sorted(primary_match_counts.items())),
        "formula_distance": {
            "mean": float(np.mean(formula_distances)),
            "median": float(np.median(formula_distances)),
            "p90": float(np.quantile(formula_distances, 0.90)),
            "maximum": int(np.max(formula_distances)),
        },
    }
    summary = {
        "relation_path": str(relation_path),
        "supplemental_path": str(supplemental_path),
        "semantic_applicability_controls_path": str(semantic_controls_path),
        "typed_relation_registry_path": str(typed_registry_path),
        "triplets": int(len(relation_frame)),
        "target_triplets": int(target_triplets),
        "partition": partition,
        "source_partition_rows": int(len(index)),
        "anchors_examined": int(anchors_examined),
        "positive_relation_counts": dict(sorted(family_counts.items())),
        "eligibility_exclusions": dict(sorted(eligibility_counts.items())),
        "eligibility_denominators": {
            "full_partition_rows": int(len(index)),
            "candidate_search_rows_examined": int(anchors_examined),
            "eligible_triplets_retained": int(len(relation_frame)),
        },
        "typed_relation_registry": {
            "rows": int(len(typed_registry)),
            "relation_class_counts": {
                str(key): int(value)
                for key, value in typed_registry["relation_class"]
                .value_counts()
                .sort_index()
                .items()
            },
            "training_eligible_rows": int(typed_registry["training_eligible"].sum()),
            "evaluation_eligible_rows": int(
                typed_registry["evaluation_eligible"].sum()
            ),
            "partition_qualified_relation_ids": bool(
                typed_registry["relation_id"].is_unique
                and typed_registry["relation_id"].str.startswith(
                    f"{partition}::"
                ).all()
            ),
        },
        "semantic_applicability_controls": {
            "rows": int(len(semantic_controls)),
            "family_counts": {
                str(key): int(value)
                for key, value in semantic_controls["control_family"]
                .value_counts()
                .sort_index()
                .items()
            },
            "training_eligible_rows": 0,
        },
        "primary_negative_graph_rule": (
            "distinct_two_cell_graph_after_removing_stereochemistry"
        ),
        "stereo_only_primary_negative_pairs": int(
            relation_frame["anchor_stereo_agnostic_identity"]
            .eq(relation_frame["negative_stereo_agnostic_identity"])
            .sum()
        ),
        "periodic_view_eligibility": dict(sorted(periodic_view_counts.items())),
        "manual_review": {
            "requested_rows": 200,
            "prepared_rows": int(len(review)),
            "review_complete": False,
            "path": str(output_root / "manual_review_sample.csv"),
        },
        "morgan_similarity_minimum": similarity_minimum,
        "primary_negative_matching": primary_summary,
        "shortcut_diagnostic": diagnostic,
        "generic_negative_matching": generic_summary,
        "retrieval_decoys": {
            **decoy_summary,
            "bank_path": str(decoy_bank_path),
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["build_relation_index", "build_relation_set"]
