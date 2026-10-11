"""Project-local PI1M identities based on explicit two-cell periodic graphs."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import rdBase

from polyfid.chemistry import StructureError, parse_repeat_unit
from polyfid.paths import ProjectPaths
from polyfid.workflows.downstream_protection import (
    build_downstream_periodic_protection,
)


_DOWNSTREAM_PROTECTION_BASIS = "project_downstream_manifests_two_cell_unique_v2"


def _upstream_protected_count(paths: ProjectPaths) -> int:
    """Return the downstream-overlap rows removed during PI1M standardization."""

    metadata_path = (
        paths.shared_data_root / "standardized/v1/pi1m/pi1m_v2.metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("rows", {}).get("protected", 0))


def _periodic_fields(value: tuple[str, str]) -> dict[str, object]:
    sample_id, psmiles = value
    try:
        with rdBase.BlockLogs():
            unit = parse_repeat_unit(psmiles)
    except StructureError as exc:
        return {
            "sample_id": sample_id,
            "periodic_identity": None,
            "periodic_supercell_smiles": None,
            "closed_smiles_v2": None,
            "formula_v2": None,
            "atom_count_v2": None,
            "periodic_identity_status": exc.code,
        }
    return {
        "sample_id": sample_id,
        "periodic_identity": unit.canonical_identity,
        "periodic_supercell_smiles": unit.periodic_supercell_smiles,
        "closed_smiles_v2": unit.closed_smiles,
        "formula_v2": unit.formula,
        "atom_count_v2": unit.atom_count,
        "periodic_identity_status": "representable",
    }


def _retained_roles(frame: pd.DataFrame) -> pd.Series:
    priority = frame["source_split"].map({"train": 0, "validation": 1, "test": 2})
    group_priority = priority.groupby(frame["periodic_identity"], sort=False).transform(
        "max"
    )
    return frame["periodic_identity"].notna() & priority.eq(group_priority)


def _paper104_retention_flags(
    frame: pd.DataFrame,
    protected_periodic: set[str],
) -> dict[str, pd.Series]:
    split_consistent = _retained_roles(frame)
    representative = pd.Series(False, index=frame.index)
    candidates = frame.loc[split_consistent].copy()
    candidates["_row_index"] = candidates.index
    candidates = candidates.sort_values(
        ["periodic_identity", "source_row_index", "sample_id"],
        kind="stable",
    )
    representative_indices = candidates.drop_duplicates(
        "periodic_identity", keep="first"
    )["_row_index"]
    representative.loc[representative_indices] = True

    representable = frame["periodic_identity"].notna()
    identity_split_counts = (
        frame.loc[representable]
        .groupby("periodic_identity", sort=False)["source_split"]
        .transform("nunique")
    )
    cross_split_duplicate = pd.Series(False, index=frame.index)
    cross_split_duplicate.loc[representable] = (
        identity_split_counts.gt(1) & ~split_consistent.loc[representable]
    )
    same_split_duplicate = split_consistent & ~representative
    downstream_overlap = (
        frame["source_split"].eq("train")
        & frame["periodic_identity"].astype(str).isin(protected_periodic)
    )
    retained = representative & ~downstream_overlap
    return {
        "split_consistent": split_consistent,
        "representative": representative,
        "invalid": ~representable,
        "cross_split_duplicate": cross_split_duplicate,
        "same_split_duplicate": same_split_duplicate,
        "downstream_overlap": downstream_overlap,
        "retained": retained,
    }


def prepare_periodic_pi1m(
    paths: ProjectPaths,
    *,
    workers: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    """Build the PI1M table used by Paper104 training and relation sampling.

    Source split labels are preserved.  If representations that are equivalent
    under the two-cell identity occur in more than one source role, rows from
    the lower-priority role are excluded (test before validation before train)
    instead of being relabelled.
    """

    output = paths.project_root / "data/derived/pi1m_periodic_identity.parquet"
    summary_path = paths.project_root / "data/splits/pi1m_periodic_summary.json"
    if output.is_file() and summary_path.is_file() and not force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("downstream_protection_basis") == _DOWNSTREAM_PROTECTION_BASIS:
            return summary

    source = paths.shared_data_root / "standardized/v1/pi1m/pi1m_v2.parquet"
    if output.is_file() and not force:
        # A two-cell PI1M table can be upgraded with the downstream protected
        # set without repeating the expensive graph construction.
        frame = pd.read_parquet(output)
    else:
        frame = pd.read_parquet(source)
        frame = frame.rename(
            columns={
                "repeat_equivalence_id": "source_repeat_equivalence_id",
                "closed_smiles": "source_closed_smiles",
                "atom_count": "source_atom_count",
                "split": "source_split",
            }
        )
        arguments = zip(
            frame["sample_id"].astype(str),
            frame["canonical_psmiles"].astype(str),
            strict=True,
        )
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            fields = list(executor.map(_periodic_fields, arguments, chunksize=512))
        periodic = pd.DataFrame(fields).set_index("sample_id")
        frame = frame.set_index("sample_id").join(periodic, how="left").reset_index()

    protected_periodic, protection_summary = (
        build_downstream_periodic_protection(paths)
    )
    flags = _paper104_retention_flags(frame, protected_periodic)
    frame["retained_for_split_consistency"] = flags["split_consistent"]
    frame["retained_as_identity_representative"] = flags["representative"]
    frame["excluded_invalid_periodic_identity"] = flags["invalid"]
    frame["excluded_cross_split_duplicate"] = flags["cross_split_duplicate"]
    frame["excluded_same_split_duplicate"] = flags["same_split_duplicate"]
    frame["excluded_for_downstream_overlap"] = flags["downstream_overlap"]
    frame["retained_for_paper104"] = flags["retained"]
    frame["exclusion_reason"] = "retained"
    frame.loc[flags["invalid"], "exclusion_reason"] = "invalid_periodic_identity"
    frame.loc[
        flags["cross_split_duplicate"], "exclusion_reason"
    ] = "cross_split_periodic_identity"
    frame.loc[
        flags["same_split_duplicate"], "exclusion_reason"
    ] = "same_identity_duplicate"
    frame.loc[
        flags["representative"] & flags["downstream_overlap"], "exclusion_reason"
    ] = "downstream_overlap"
    frame["split"] = frame["source_split"].where(
        frame["retained_for_paper104"], "excluded"
    )
    frame["repeat_equivalence_id"] = frame["periodic_identity"]
    frame["closed_smiles"] = frame["closed_smiles_v2"]
    frame["atom_count"] = frame["atom_count_v2"].astype("Int64")
    frame["formula"] = frame["formula_v2"]

    split_counts = (
        frame.loc[frame["periodic_identity"].notna()]
        .groupby("periodic_identity")["source_split"]
        .nunique()
    )
    crossing_identities = set(split_counts.loc[split_counts.gt(1)].index.astype(str))
    excluded = frame.loc[~frame["retained_for_paper104"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.parent.parent.joinpath("splits").mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    excluded_path = paths.project_root / "data/splits/pi1m_periodic_excluded_rows.parquet"
    excluded.loc[
        :,
        [
            "sample_id",
            "source_split",
            "source_repeat_equivalence_id",
            "periodic_identity",
            "periodic_identity_status",
            "retained_for_split_consistency",
            "retained_as_identity_representative",
            "excluded_invalid_periodic_identity",
            "excluded_cross_split_duplicate",
            "excluded_same_split_duplicate",
            "excluded_for_downstream_overlap",
            "exclusion_reason",
        ],
    ].to_parquet(excluded_path, index=False)
    protected_path = (
        paths.project_root / "data/splits/protected_repeat_groups_periodic.txt"
    )
    protected_path.write_text(
        "\n".join(sorted(protected_periodic))
        + ("\n" if protected_periodic else ""),
        encoding="utf-8",
    )

    retained = frame.loc[frame["retained_for_paper104"]]
    remaining_crossings = int(
        retained.groupby("periodic_identity")["split"].nunique().gt(1).sum()
    )
    summary: dict[str, Any] = {
        "source_path": str(source),
        "output_path": str(output),
        "source_rows": int(len(frame)),
        "periodic_identity_representable_rows": int(
            frame["periodic_identity"].notna().sum()
        ),
        "source_cross_split_periodic_identities": int(len(crossing_identities)),
        "excluded_rows": int(len(excluded)),
        "invalid_periodic_identity_rows": int(flags["invalid"].sum()),
        "cross_split_duplicate_rows": int(flags["cross_split_duplicate"].sum()),
        "same_split_duplicate_rows": int(flags["same_split_duplicate"].sum()),
        "downstream_overlap_rows": int(
            (flags["representative"] & flags["downstream_overlap"]).sum()
        ),
        "retained_rows_by_split": {
            str(key): int(value)
            for key, value in retained["split"].value_counts().sort_index().items()
        },
        "remaining_cross_split_periodic_identities": remaining_crossings,
        "protected_periodic_identities": int(len(protected_periodic)),
        "pi1m_train_downstream_overlap_rows": int(
            frame["excluded_for_downstream_overlap"].sum()
        ),
        "pi1m_train_downstream_overlap_identities": int(
            frame.loc[
                frame["excluded_for_downstream_overlap"], "periodic_identity"
            ].nunique()
        ),
        "additional_overlap_rows_after_two_cell_remap": int(
            (flags["representative"] & flags["downstream_overlap"]).sum()
        ),
        "upstream_prestandardization_protected_rows": _upstream_protected_count(
            paths
        ),
        "downstream_protection_basis": _DOWNSTREAM_PROTECTION_BASIS,
        "downstream_protection": protection_summary,
        "protected_periodic_path": str(protected_path),
        "excluded_rows_path": str(excluded_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


__all__ = ["prepare_periodic_pi1m"]
