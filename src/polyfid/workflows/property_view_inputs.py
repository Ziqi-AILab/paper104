"""Build split-aware equivalent-view inputs for downstream property analysis.

The formal property-view experiment is limited to test records from the
declared downstream protocols.  Randomized SMILES and explicit atom-order
permutations are two generators for one traversal axis; alternative repeat
windows form the separate periodic-cut axis.  Semantic controls are not built
here because they are applicability analyses, not equivalent representations.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from polyfid.chemistry import (
    StructureError,
    assess_repeat_eligibility,
    generate_equivalent_periodic_cuts,
    generate_randomized_smiles,
    parse_repeat_unit,
)
from polyfid.evaluation.adaptation import prepare_downstream_frame
from polyfid.evaluation.invariance import generate_atom_order_permutations

try:
    from rdkit import rdBase
except ModuleNotFoundError:  # pragma: no cover - declared project dependency
    rdBase = None  # type: ignore[assignment]


VIEW_COLUMNS = (
    "sample_id",
    "view_id",
    "partition_id",
    "input_view",
    "view_family",
    "view_generator",
    "view_index",
    "reference_text",
    "view_text",
    "canonical_identity",
    "analysis_role",
    "structural_fidelity_eligible",
    "scope_class",
)

SCOPE_COLUMNS = (
    "sample_id",
    "partition_id",
    "input_view",
    "structural_fidelity_eligible",
    "scope_class",
    "eligibility_reason",
    "randomized_traversal_views",
    "atom_permutation_views",
    "traversal_views",
    "periodic_cut_views",
    "generation_failure",
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def property_view_partition_id(dataset: str | Path, input_view: str) -> str:
    """Return the stable, readable partition name shared by all matrices."""

    return f"{_slug(Path(dataset).stem)}__{_slug(input_view)}"


def _non_noop_unique(
    values: list[tuple[str, str]], reference: str
) -> list[tuple[str, str]]:
    seen = {str(reference)}
    output: list[tuple[str, str]] = []
    for generator, value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            output.append((generator, text))
    return output


def build_verified_property_views(
    records: pd.DataFrame,
    *,
    partition_id: str,
    input_view: str,
    randomized_count: int = 10,
    atom_permutation_count: int = 10,
    periodic_count: int = 10,
    seed: int = 10407,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct legal views for one unique downstream test-record partition.

    ``records`` contains the actual model reference in ``reference_text`` and
    the repeat-unit portion in ``structure_text``.  For PE-I metadata inputs,
    the unchanged metadata suffix is appended to every transformed structure.
    """

    views: list[dict[str, object]] = []
    scope: list[dict[str, object]] = []
    for row_index, row in records.reset_index(drop=True).iterrows():
        sample_id = str(row["sample_id"])
        structure_text = str(row["structure_text"])
        reference_text = str(row["reference_text"])
        assessment = assess_repeat_eligibility(structure_text)
        if not assessment.eligible:
            scope.append(
                {
                    "sample_id": sample_id,
                    "partition_id": partition_id,
                    "input_view": input_view,
                    "structural_fidelity_eligible": False,
                    "scope_class": "outside_article_a_scope",
                    "eligibility_reason": assessment.reason,
                    "randomized_traversal_views": 0,
                    "atom_permutation_views": 0,
                    "traversal_views": 0,
                    "periodic_cut_views": 0,
                    "generation_failure": None,
                }
            )
            continue

        unit = parse_repeat_unit(structure_text)
        view_seed = int(seed) + row_index * 104729
        generation_failure: str | None = None
        try:
            randomized_values = generate_randomized_smiles(
                structure_text, count=randomized_count, seed=view_seed
            )
        except StructureError as exc:
            if exc.code != "randomization_changed_identity":
                raise
            randomized_values = ()
            generation_failure = "randomized_smiles:randomization_changed_identity"
        traversal_candidates = [
            ("randomized_smiles", value) for value in randomized_values
        ]
        # Atom renumbering followed by a non-canonical serialization is the
        # same scientific traversal axis.  Reuse the explicit implementation
        # from the representation-level invariance experiment so the generator
        # label describes the operation actually performed.
        traversal_candidates.extend(
            ("atom_order_permutation", value)
            for value in generate_atom_order_permutations(
                structure_text,
                count=atom_permutation_count,
                seed=view_seed + 15485863,
            )
        )
        traversal = _non_noop_unique(traversal_candidates, structure_text)
        periodic = _non_noop_unique(
            [
                ("alternative_periodic_cut", view.psmiles)
                for view in generate_equivalent_periodic_cuts(
                    unit, max_views=periodic_count, seed=view_seed
                )
                if view.transformation != "reference"
            ],
            structure_text,
        )

        if input_view == "structure_plus_metadata":
            if not reference_text.startswith(structure_text):
                raise ValueError(
                    "structure+metadata input does not begin with its structure text"
                )
            suffix = reference_text[len(structure_text) :]
        else:
            suffix = ""

        family_counts = {"traversal": len(traversal), "periodic_cut": len(periodic)}
        randomized_view_count = sum(
            generator == "randomized_smiles" for generator, _ in traversal
        )
        atom_permutation_view_count = sum(
            generator == "atom_order_permutation" for generator, _ in traversal
        )
        for family, generated in (("traversal", traversal), ("periodic_cut", periodic)):
            for view_index, (generator, transformed_structure) in enumerate(generated):
                view_id = (
                    f"{sample_id}:{_slug(input_view)}:{family}:{view_index}"
                )
                views.append(
                    {
                        "sample_id": sample_id,
                        "view_id": view_id,
                        "partition_id": partition_id,
                        "input_view": input_view,
                        "view_family": family,
                        "view_generator": generator,
                        "view_index": view_index,
                        "reference_text": reference_text,
                        "view_text": transformed_structure + suffix,
                        "canonical_identity": unit.canonical_identity,
                        "analysis_role": "equivalence_consistency",
                        "structural_fidelity_eligible": True,
                        "scope_class": "linear_dual_attachment_repeat",
                    }
                )
        scope.append(
            {
                "sample_id": sample_id,
                "partition_id": partition_id,
                "input_view": input_view,
                "structural_fidelity_eligible": True,
                "scope_class": "linear_dual_attachment_repeat",
                "eligibility_reason": "eligible",
                "randomized_traversal_views": randomized_view_count,
                "atom_permutation_views": atom_permutation_view_count,
                "traversal_views": family_counts["traversal"],
                "periodic_cut_views": family_counts["periodic_cut"],
                "generation_failure": generation_failure,
            }
        )
    return (
        pd.DataFrame(views, columns=VIEW_COLUMNS),
        pd.DataFrame(scope, columns=SCOPE_COLUMNS),
    )


def _formal_units(probe_manifest: pd.DataFrame) -> pd.DataFrame:
    frame = probe_manifest.copy()
    frame = frame.loc[
        frame["label_fraction"].astype(float).eq(1.0)
        & frame["subsample_id"].astype(int).eq(0)
        & ~frame["protocol_role"].astype(str).eq("molecular_retention")
    ].copy()
    columns = [
        "dataset",
        "split_manifest",
        "task",
        "protocol",
        "input_view",
        "fold",
        "label_column",
    ]
    return frame.loc[:, columns].drop_duplicates().reset_index(drop=True)


def _text_columns(dataset: pd.DataFrame, input_view: str) -> tuple[str, str]:
    if input_view == "structure_plus_metadata":
        return "structure_with_metadata_text", "structure_only_text"
    for candidate in ("structure_only_text", "canonical_text", "canonical_psmiles"):
        if candidate in dataset:
            return candidate, candidate
    raise ValueError("downstream dataset lacks a structure text column")


def collect_formal_test_records(probe_manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collect each unique test structure once after all formal split choices."""

    partition_rows: dict[str, list[pd.DataFrame]] = {}
    dataset_cache: dict[str, pd.DataFrame] = {}
    split_cache: dict[str, pd.DataFrame] = {}
    for unit in _formal_units(probe_manifest).itertuples(index=False):
        if unit.dataset not in dataset_cache:
            dataset_cache[unit.dataset] = pd.read_parquet(unit.dataset)
        dataset = dataset_cache[unit.dataset]
        if unit.split_manifest not in split_cache:
            split_cache[unit.split_manifest] = pd.read_parquet(unit.split_manifest)
        split_manifest = split_cache[unit.split_manifest]
        model_column, structure_column = _text_columns(dataset, unit.input_view)
        joined = prepare_downstream_frame(
            dataset,
            split_manifest,
            task=unit.task,
            protocol=unit.protocol,
            fold=int(unit.fold),
            label_column=unit.label_column,
            text_column=model_column,
        )
        test = joined.loc[joined["split"].eq("test")].copy()
        partition_id = property_view_partition_id(unit.dataset, unit.input_view)
        rows = pd.DataFrame(
            {
                "sample_id": test["sample_id"].astype(str),
                "reference_text": test[model_column].astype(str),
                "structure_text": test[structure_column].astype(str),
            }
        )
        partition_rows.setdefault(partition_id, []).append(rows)

    output: dict[str, pd.DataFrame] = {}
    for partition_id, pieces in partition_rows.items():
        combined = pd.concat(pieces, ignore_index=True)
        conflicts = combined.groupby("sample_id").agg(
            reference_texts=("reference_text", "nunique"),
            structure_texts=("structure_text", "nunique"),
        )
        if conflicts[["reference_texts", "structure_texts"]].gt(1).any(axis=None):
            raise ValueError(f"sample IDs map to conflicting texts in {partition_id}")
        output[partition_id] = combined.drop_duplicates("sample_id").reset_index(drop=True)
    return output


def build_task_scope_tables(
    probe_manifest: pd.DataFrame,
    scope_by_partition: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize scope failures and view coverage for every formal test unit."""

    dataset_cache: dict[str, pd.DataFrame] = {}
    split_cache: dict[str, pd.DataFrame] = {}
    task_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for unit in _formal_units(probe_manifest).itertuples(index=False):
        if unit.dataset not in dataset_cache:
            dataset_cache[unit.dataset] = pd.read_parquet(unit.dataset)
        dataset = dataset_cache[unit.dataset]
        if unit.split_manifest not in split_cache:
            split_cache[unit.split_manifest] = pd.read_parquet(unit.split_manifest)
        split_manifest = split_cache[unit.split_manifest]
        model_column, _ = _text_columns(dataset, unit.input_view)
        joined = prepare_downstream_frame(
            dataset,
            split_manifest,
            task=unit.task,
            protocol=unit.protocol,
            fold=int(unit.fold),
            label_column=unit.label_column,
            text_column=model_column,
        )
        test_ids = set(
            joined.loc[joined["split"].eq("test"), "sample_id"].astype(str)
        )
        partition_id = property_view_partition_id(unit.dataset, unit.input_view)
        selected = scope_by_partition[partition_id]
        selected = selected.loc[
            selected["sample_id"].astype(str).isin(test_ids)
        ].copy()
        common = {
            "task": str(unit.task),
            "protocol": str(unit.protocol),
            "fold": int(unit.fold),
            "input_view": str(unit.input_view),
            "partition_id": partition_id,
        }
        task_rows.append(
            {
                **common,
                "test_records": int(len(test_ids)),
                "eligible_records": int(
                    selected["structural_fidelity_eligible"].sum()
                ),
                "ineligible_records": int(
                    (~selected["structural_fidelity_eligible"].astype(bool)).sum()
                ),
                "generation_failure_records": int(
                    selected["generation_failure"].notna().sum()
                ),
                "records_with_randomized_traversal": int(
                    selected["randomized_traversal_views"].gt(0).sum()
                ),
                "randomized_traversal_views": int(
                    selected["randomized_traversal_views"].sum()
                ),
                "records_with_atom_permutation": int(
                    selected["atom_permutation_views"].gt(0).sum()
                ),
                "atom_permutation_views": int(
                    selected["atom_permutation_views"].sum()
                ),
                "records_with_periodic_cut": int(
                    selected["periodic_cut_views"].gt(0).sum()
                ),
                "periodic_cut_views": int(selected["periodic_cut_views"].sum()),
            }
        )
        scope_failures = selected.loc[
            ~selected["structural_fidelity_eligible"].astype(bool),
            "eligibility_reason",
        ].value_counts()
        for reason, count in scope_failures.items():
            failure_rows.append(
                {
                    **common,
                    "failure_domain": "article_scope",
                    "failure_reason": str(reason),
                    "records": int(count),
                }
            )
        generation_failures = selected["generation_failure"].dropna().value_counts()
        for reason, count in generation_failures.items():
            failure_rows.append(
                {
                    **common,
                    "failure_domain": "view_generation",
                    "failure_reason": str(reason),
                    "records": int(count),
                }
            )
    failure_columns = [
        "task",
        "protocol",
        "fold",
        "input_view",
        "partition_id",
        "failure_domain",
        "failure_reason",
        "records",
    ]
    return pd.DataFrame(task_rows), pd.DataFrame(failure_rows, columns=failure_columns)


def build_property_view_inputs(
    probe_manifest_path: str | Path,
    output_root: str | Path,
    *,
    include_openpoly: bool = False,
    randomized_count: int = 10,
    atom_permutation_count: int = 10,
    periodic_count: int = 10,
    seed: int = 10407,
) -> dict[str, Any]:
    """Write partitioned formal view rows, scope rows, and an encoder registry."""

    manifest = pd.read_csv(probe_manifest_path, sep="\t")
    if not include_openpoly:
        manifest = manifest.loc[
            ~manifest["task"].astype(str).str.startswith("OpenPoly:")
        ].copy()
    partitions = collect_formal_test_records(manifest)
    destination = Path(output_root)
    view_root = destination / "partitions"
    scope_root = destination / "scope"
    view_root.mkdir(parents=True, exist_ok=True)
    scope_root.mkdir(parents=True, exist_ok=True)
    registry_rows: list[dict[str, object]] = []
    scope_by_partition: dict[str, pd.DataFrame] = {}
    if rdBase is None:
        raise RuntimeError("RDKit is required to build property views")
    with rdBase.BlockLogs():
        for partition_id, records in sorted(partitions.items()):
            input_view = partition_id.rsplit("__", 1)[-1]
            input_view = (
                "structure_plus_metadata"
                if input_view == "structure_plus_metadata"
                else "structure_only"
            )
            try:
                view_rows, scope_rows = build_verified_property_views(
                    records,
                    partition_id=partition_id,
                    input_view=input_view,
                    randomized_count=randomized_count,
                    atom_permutation_count=atom_permutation_count,
                    periodic_count=periodic_count,
                    seed=seed,
                )
            except StructureError as exc:
                raise RuntimeError(f"failed to build {partition_id}: {exc}") from exc
            view_path = view_root / f"{partition_id}.parquet"
            scope_path = scope_root / f"{partition_id}.parquet"
            view_rows.to_parquet(view_path, index=False)
            scope_rows.to_parquet(scope_path, index=False)
            scope_by_partition[partition_id] = scope_rows
            registry_rows.append(
                {
                    "partition_id": partition_id,
                    "input_view": input_view,
                    "view_rows": str(view_path),
                    "scope_rows": str(scope_path),
                    "test_records": int(len(records)),
                    "eligible_records": int(
                        scope_rows["structural_fidelity_eligible"].sum()
                    ),
                    "views": int(len(view_rows)),
                    "traversal_views": int(view_rows["view_family"].eq("traversal").sum()),
                    "randomized_traversal_views": int(
                        scope_rows["randomized_traversal_views"].sum()
                    ),
                    "atom_permutation_views": int(
                        scope_rows["atom_permutation_views"].sum()
                    ),
                    "periodic_cut_views": int(
                        view_rows["view_family"].eq("periodic_cut").sum()
                    ),
                    "generation_failure_records": int(
                        scope_rows["generation_failure"].notna().sum()
                    ),
                }
            )
    registry = pd.DataFrame(registry_rows).sort_values("partition_id")
    registry_path = destination / "registry.tsv"
    registry.to_csv(registry_path, sep="\t", index=False)
    task_scope, task_failures = build_task_scope_tables(
        manifest, scope_by_partition
    )
    task_scope_path = destination / "task_scope_census.tsv"
    task_failure_path = destination / "task_failure_reasons.tsv"
    task_scope.to_csv(task_scope_path, sep="\t", index=False)
    task_failures.to_csv(task_failure_path, sep="\t", index=False)
    report = {
        "probe_manifest": str(probe_manifest_path),
        "openpoly_included": bool(include_openpoly),
        "seed": int(seed),
        "randomized_traversals_per_record": int(randomized_count),
        "atom_permutations_per_record": int(atom_permutation_count),
        "maximum_periodic_cuts_per_record": int(periodic_count),
        "partitions": int(len(registry)),
        "test_records": int(registry["test_records"].sum()),
        "eligible_records": int(registry["eligible_records"].sum()),
        "views": int(registry["views"].sum()),
        "traversal_views": int(registry["traversal_views"].sum()),
        "randomized_traversal_views": int(
            registry["randomized_traversal_views"].sum()
        ),
        "atom_permutation_views": int(
            registry["atom_permutation_views"].sum()
        ),
        "periodic_cut_views": int(registry["periodic_cut_views"].sum()),
        "generation_failure_records": int(
            registry["generation_failure_records"].sum()
        ),
        "registry": str(registry_path),
        "task_scope_census": str(task_scope_path),
        "task_failure_reasons": str(task_failure_path),
        "formal_task_units": int(len(task_scope)),
    }
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


__all__ = [
    "SCOPE_COLUMNS",
    "VIEW_COLUMNS",
    "build_property_view_inputs",
    "build_task_scope_tables",
    "build_verified_property_views",
    "collect_formal_test_records",
    "property_view_partition_id",
]
