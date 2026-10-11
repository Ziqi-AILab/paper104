"""Collect held-out polymer repeat units for PI1M overlap removal.

The protected set is defined from the Paper104 downstream split manifests,
then expressed with the same explicit two-cell periodic identity used for
PI1M.  Molecular retention datasets are intentionally outside this workflow.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import rdBase

from polyfid.chemistry import StructureError, parse_repeat_unit
from polyfid.paths import ProjectPaths


_HELDOUT_ROLES = frozenset({"validation", "test"})
_SCOPE_FAILURES = frozenset(
    {
        "attachment_count",
        "disconnected",
        "dummy_connection",
        "nonterminal_attachment",
        "shared_attachment_atom",
    }
)


def _flatten_values(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_values(item)


def _repeat_components(value: Any) -> tuple[str, ...]:
    """Return structural components that may contain one linear repeat unit."""

    components: list[str] = []
    for section in _flatten_values(value):
        # ``structure_only_text`` separates mixtures with ``|`` or ``.``.
        # Metadata-bearing text is not used for protection.
        for pipe_piece in str(section).split("|"):
            for component in pipe_piece.split("."):
                component = component.strip()
                if component:
                    components.append(component)
    return tuple(components)


def _heldout_sample_ids(manifests: Iterable[Path]) -> tuple[set[str], list[str]]:
    sample_ids: set[str] = set()
    used: list[str] = []
    for path in sorted(manifests):
        if not path.is_file():
            continue
        roles = pd.read_parquet(path, columns=["sample_id", "split"])
        heldout = roles.loc[roles["split"].isin(_HELDOUT_ROLES), "sample_id"]
        sample_ids.update(heldout.astype(str))
        used.append(path.name)
    return sample_ids, used


def _candidate_rows(
    frame: pd.DataFrame,
    *,
    source_dataset: str,
    heldout_sample_ids: set[str],
    structure_column: str,
) -> list[dict[str, str]]:
    selected = frame.loc[
        frame["sample_id"].astype(str).isin(heldout_sample_ids),
        ["sample_id", structure_column],
    ]
    rows: list[dict[str, str]] = []
    for sample_id, value in selected.itertuples(index=False, name=None):
        for component in _repeat_components(value):
            rows.append(
                {
                    "source_dataset": source_dataset,
                    "sample_id": str(sample_id),
                    "candidate_psmiles": component,
                }
            )
    return rows


def collect_downstream_repeat_candidates(
    paths: ProjectPaths,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read project manifests and collect their held-out polymer components."""

    shared = paths.shared_data_root / "standardized/v1"
    splits = paths.project_root / "data/splits"
    rows: list[dict[str, str]] = []
    sources: dict[str, Any] = {}

    aldeghi_manifests = tuple(splits.glob("aldeghi_*.parquet"))
    aldeghi_ids, used = _heldout_sample_ids(aldeghi_manifests)
    if used:
        table = pd.read_parquet(shared / "aldeghi_vipea/vipea.parquet")
        source_rows = _candidate_rows(
            table,
            source_dataset="aldeghi_vipea",
            heldout_sample_ids=aldeghi_ids,
            structure_column="component_canonical_psmiles",
        )
        rows.extend(source_rows)
        sources["aldeghi_vipea"] = {
            "manifests": used,
            "heldout_records": len(aldeghi_ids),
            "candidate_components": len(source_rows),
        }

    for manifest in sorted(splits.glob("transpolymer_*.parquet")):
        stem = manifest.stem.removeprefix("transpolymer_")
        if stem in {"pe_i", "pe_i_official_group_disjoint"}:
            continue
        table_path = shared / f"transpolymer/{stem}.parquet"
        if not table_path.is_file():
            continue
        heldout_ids, used = _heldout_sample_ids((manifest,))
        table = pd.read_parquet(table_path)
        source_name = f"transpolymer_{stem}"
        source_rows = _candidate_rows(
            table,
            source_dataset=source_name,
            heldout_sample_ids=heldout_ids,
            structure_column="structure_only_text",
        )
        rows.extend(source_rows)
        sources[source_name] = {
            "manifests": used,
            "heldout_records": len(heldout_ids),
            "candidate_components": len(source_rows),
        }

    official_manifest = (
        splits / "transpolymer_pe_i_official_group_disjoint.parquet"
    )
    official_table = paths.project_root / "data/derived/transpolymer_pe_i_official.parquet"
    if official_manifest.is_file() and official_table.is_file():
        heldout_ids, used = _heldout_sample_ids((official_manifest,))
        table = pd.read_parquet(official_table)
        source_rows = _candidate_rows(
            table,
            source_dataset="transpolymer_pe_i_official",
            heldout_sample_ids=heldout_ids,
            structure_column="structure_only_text",
        )
        rows.extend(source_rows)
        sources["transpolymer_pe_i_official"] = {
            "manifests": used,
            "heldout_records": len(heldout_ids),
            "candidate_components": len(source_rows),
        }

    openpoly_manifest = splits / "openpoly.parquet"
    openpoly_table = shared / "openpoly/openpoly_live_2026_08_08.parquet"
    if openpoly_manifest.is_file() and openpoly_table.is_file():
        heldout_ids, used = _heldout_sample_ids((openpoly_manifest,))
        table = pd.read_parquet(openpoly_table)
        source_rows = _candidate_rows(
            table,
            source_dataset="openpoly",
            heldout_sample_ids=heldout_ids,
            structure_column="canonical_psmiles",
        )
        rows.extend(source_rows)
        sources["openpoly"] = {
            "manifests": used,
            "heldout_records": len(heldout_ids),
            "candidate_components": len(source_rows),
        }

    candidates = pd.DataFrame(
        rows,
        columns=["source_dataset", "sample_id", "candidate_psmiles"],
    ).drop_duplicates(ignore_index=True)
    summary = {
        "source_counts": sources,
        "heldout_record_components": int(len(candidates)),
        "unique_candidate_psmiles": int(candidates["candidate_psmiles"].nunique()),
    }
    return candidates, summary


def _candidate_periodic_fields(psmiles: str) -> dict[str, Any]:
    try:
        with rdBase.BlockLogs():
            unit = parse_repeat_unit(psmiles)
    except StructureError as exc:
        scope = "non_applicable" if exc.code in _SCOPE_FAILURES else "invalid"
        return {
            "periodic_identity": None,
            "periodic_identity_status": exc.code,
            "scope_status": scope,
        }
    return {
        "periodic_identity": unit.canonical_identity,
        "periodic_identity_status": "representable",
        "scope_status": "representable",
    }


def build_downstream_periodic_protection(
    paths: ProjectPaths,
) -> tuple[set[str], dict[str, Any]]:
    """Build and persist the held-out downstream two-cell identity table."""

    candidates, summary = collect_downstream_repeat_candidates(paths)
    unique = candidates[["candidate_psmiles"]].drop_duplicates().copy()
    fields = [_candidate_periodic_fields(value) for value in unique["candidate_psmiles"]]
    unique = pd.concat([unique.reset_index(drop=True), pd.DataFrame(fields)], axis=1)
    detailed = candidates.merge(unique, on="candidate_psmiles", how="left")
    protected = set(
        detailed.loc[
            detailed["scope_status"].eq("representable"), "periodic_identity"
        ].astype(str)
    )

    output = paths.project_root / "data/splits/downstream_periodic_protection.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_parquet(output, index=False)
    status_counts = detailed[["candidate_psmiles", "scope_status"]].drop_duplicates()[
        "scope_status"
    ].value_counts()
    summary.update(
        {
            "periodic_identity_definition": "explicit_two_cell_periodic_supercell",
            "unique_periodic_identities": len(protected),
            "unique_candidate_status_counts": {
                str(key): int(value) for key, value in status_counts.sort_index().items()
            },
            "details_path": str(output),
        }
    )
    if not paths.formal_openpoly:
        # Exact exclusions retained by the completed study; no property labels.
        frozen = paths.exclusion_path("protected_repeat_groups_periodic.txt")
        if not frozen.is_file():
            raise FileNotFoundError(
                f"Frozen periodic exclusion list required: {frozen}. "
                "Configure --exclusion-root with configure_local_paths.py; see docs/data.md."
            )
        summary["reconstructed_formal_periodic_identities"] = len(protected)
        protected = set(frozen.read_text().splitlines())
        summary["unique_periodic_identities"] = len(protected)
        summary["retained_exclusion_source"] = str(frozen)
        summary["retained_exclusion_note"] = "includes historical conditional OpenPoly exclusions"
    return protected, summary


__all__ = [
    "build_downstream_periodic_protection",
    "collect_downstream_repeat_candidates",
]
