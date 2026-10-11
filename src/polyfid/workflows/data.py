"""Build the project-local split manifests from shared standardized tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from polyfid.paths import ProjectPaths
from polyfid.workflows.periodic_data import prepare_periodic_pi1m


def _write_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {"path": str(path), "rows": int(len(frame))}


def _retain_declared_roles(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    """Retain held-out source rows and drop lower-priority group overlaps."""

    priority = {"train": 0, "validation": 1, "test": 2}
    role_priority = frame["split"].map(priority)
    if role_priority.isna().any():
        raise ValueError("unsupported split role")
    selected = role_priority.groupby(
        [frame[column] for column in group_columns], sort=False
    ).transform("max")
    return frame.loc[role_priority.eq(selected)].copy()


def _paper104_pe_i_records(frame: pd.DataFrame) -> pd.DataFrame:
    """Give official PE-I rows identities unique across source partitions."""

    output = frame.copy()
    output["source_sample_id"] = output["sample_id"].astype(str)
    output["sample_id"] = (
        "transpolymer:PE-I:"
        + output["official_partition"].astype(str)
        + ":"
        + output["source_row_index"].astype(str)
    )
    if output.duplicated(["sample_id", "task"]).any():
        raise ValueError("PE-I official row identity is not unique")
    return output


def _make_group_disjoint(
    roles: pd.DataFrame,
    group_by_sample: pd.Series,
) -> pd.DataFrame:
    """Drop lower-priority rows when a polymer group crosses source roles.

    The published family role of a row is part of the evaluation protocol.  It
    must not be changed merely to make a group-disjoint manifest.  If one
    monomer-pair group occurs in more than one source role, retain the rows in
    the held-out role (test before validation before train) and remove the
    lower-priority occurrences from that fold.
    """

    frame = roles.merge(
        group_by_sample.rename("monomer_pair_group_id"),
        left_on="sample_id",
        right_index=True,
        how="left",
    )
    return _retain_declared_roles(
        frame,
        group_columns=["fold", "monomer_pair_group_id"],
    )


def _random_group_roles(records: pd.DataFrame, *, folds: int = 10) -> pd.DataFrame:
    groups = np.asarray(sorted(records["monomer_pair_group_id"].astype(str).unique()))
    rows: list[pd.DataFrame] = []
    for fold in range(folds):
        rng = np.random.default_rng(10_407 + fold)
        ordered = groups[rng.permutation(len(groups))]
        test_count = round(0.10 * len(ordered))
        validation_count = round(0.10 * len(ordered))
        test_groups = set(ordered[:test_count])
        validation_groups = set(
            ordered[test_count : test_count + validation_count]
        )
        cell = records.loc[:, ["sample_id", "source_row_index", "monomer_pair_group_id"]].copy()
        cell["fold"] = fold
        cell["split"] = "train"
        cell.loc[cell["monomer_pair_group_id"].isin(validation_groups), "split"] = "validation"
        cell.loc[cell["monomer_pair_group_id"].isin(test_groups), "split"] = "test"
        rows.append(cell)
    return pd.concat(rows, ignore_index=True)


def _aldeghi_manifests(paths: ProjectPaths, output_root: Path) -> dict[str, Any]:
    source = paths.shared_data_root / "standardized/v1/aldeghi_vipea"
    records = pd.read_parquet(source / "vipea.parquet")
    outputs: dict[str, Any] = {}
    pair_id = records.loc[:, ["sample_id", "monomer_pair_group_id", "split"]].copy()
    pair_id["fold"] = 0
    pair_id["protocol"] = "pair_id"
    outputs["pair_id"] = _write_parquet(pair_id, output_root / "aldeghi_pair_id.parquet")
    group_by_sample = records.set_index("sample_id")["monomer_pair_group_id"]
    for protocol, filename in (
        ("family_ood", "vipea_cv9_roles.parquet"),
        ("butina_cluster", "vipea_butina_roles.parquet"),
        ("random", "vipea_random_roles.parquet"),
    ):
        if protocol == "random":
            roles = _random_group_roles(records)
        else:
            roles = pd.read_parquet(source / filename)
        if "monomer_pair_group_id" not in roles:
            roles = _make_group_disjoint(roles, group_by_sample)
        roles = roles.merge(
            records.loc[:, ["sample_id", "cv9_test_fold"]],
            on="sample_id",
            how="left",
        )
        roles["family_id"] = roles["cv9_test_fold"].map(
            lambda value: None if pd.isna(value) else f"family_{int(value)}"
        )
        roles = roles.drop(columns="cv9_test_fold")
        roles["protocol"] = protocol
        outputs[protocol] = _write_parquet(roles, output_root / f"aldeghi_{protocol}.parquet")
    return outputs


def _transpolymer_manifests(paths: ProjectPaths, output_root: Path) -> dict[str, Any]:
    source = paths.shared_data_root / "standardized/v1/transpolymer"
    outputs: dict[str, Any] = {}
    for table_path in sorted(source.glob("*.parquet")):
        if ".rejected." in table_path.name:
            continue
        frame = pd.read_parquet(table_path)
        if table_path.stem == "pe_i_official":
            # The official train and test files restart their row numbering,
            # so 146 source ``sample_id`` values name two different records.
            # Materialize a Paper104-local table with an unambiguous readable
            # row identity before any cache or split join is performed.
            frame = _paper104_pe_i_records(frame)
            derived_path = (
                paths.project_root
                / "data/derived/transpolymer_pe_i_official.parquet"
            )
            _write_parquet(frame, derived_path)
            roles = frame.loc[:, ["sample_id", "task", "group_id", "split"]].copy()
            roles["fold"] = 0
            roles["evaluation_fold"] = 0
            # Preserve the official role of every retained record.  For a
            # structure group seen in multiple official partitions, exclude
            # the lower-priority occurrence instead of relabelling it.
            roles = _retain_declared_roles(roles, group_columns=["group_id"])
            outputs[table_path.stem] = _write_parquet(
                roles,
                output_root / "transpolymer_pe_i_official_group_disjoint.parquet",
            )
            outputs[table_path.stem]["dataset_path"] = str(derived_path)
            continue
        if "fold" not in frame or "sample_id" not in frame:
            continue
        pieces = []
        for test_fold in range(5):
            validation_fold = (test_fold + 1) % 5
            roles = frame.loc[:, ["sample_id", "task", "group_id", "fold"]].copy()
            roles["evaluation_fold"] = test_fold
            roles["split"] = "train"
            roles.loc[roles["fold"].eq(validation_fold), "split"] = "validation"
            roles.loc[roles["fold"].eq(test_fold), "split"] = "test"
            pieces.append(roles)
        manifest = pd.concat(pieces, ignore_index=True)
        outputs[table_path.stem] = _write_parquet(
            manifest, output_root / f"transpolymer_{table_path.stem}.parquet"
        )
    return outputs


def _direct_manifests(paths: ProjectPaths, output_root: Path) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    direct = {
        "esol": paths.shared_data_root / "standardized/v1/molecular_retention/esol.parquet",
        "freesolv": paths.shared_data_root / "standardized/v1/molecular_retention/freesolv.parquet",
        "lipophilicity": paths.shared_data_root / "standardized/v1/molecular_retention/lipophilicity.parquet",
    }
    if paths.formal_openpoly:
        direct["openpoly"] = (
            paths.shared_data_root
            / "standardized/v1/openpoly/openpoly_live_2026_08_08.parquet"
        )
    for name, source in direct.items():
        frame = pd.read_parquet(source)
        columns = [column for column in ("sample_id", "task", "group_id", "split") if column in frame]
        outputs[name] = _write_parquet(frame.loc[:, columns], output_root / f"{name}.parquet")
    return outputs


def prepare_project_data(paths: ProjectPaths) -> dict[str, Any]:
    """Materialize only Paper104-specific split and overlap records."""

    output_root = paths.project_root / "data/splits"
    output_root.mkdir(parents=True, exist_ok=True)
    # Downstream manifests define the held-out repeat units that must be
    # removed from PI1M train, so they are materialized before periodic PI1M.
    aldeghi = _aldeghi_manifests(paths, output_root)
    transpolymer = _transpolymer_manifests(paths, output_root)
    direct = _direct_manifests(paths, output_root)
    periodic_summary = prepare_periodic_pi1m(paths)
    pi1m_source = Path(str(periodic_summary["output_path"]))
    pi1m = pd.read_parquet(pi1m_source)
    pi1m_manifest = pi1m.loc[
        pi1m["retained_for_paper104"],
        [
            "sample_id",
            "repeat_equivalence_id",
            "periodic_supercell_smiles",
            "group_id",
            "split",
        ],
    ]
    registry: dict[str, Any] = {
        "pi1m": _write_parquet(pi1m_manifest, output_root / "pi1m.parquet"),
        "pi1m_periodic_identity": periodic_summary,
        "aldeghi": aldeghi,
        "transpolymer": transpolymer,
        "direct": direct,
        "formal_openpoly": paths.formal_openpoly,
    }

    protected_source = Path(str(periodic_summary["protected_periodic_path"]))
    protected_destination = output_root / "protected_repeat_groups.txt"
    protected_text = protected_source.read_text(encoding="utf-8")
    protected_destination.write_text(protected_text, encoding="utf-8")
    protected = {
        line.strip()
        for line in protected_text.splitlines()
        if line.strip()
    }
    overlap = pi1m.loc[
        pi1m["split"].eq("train")
        & pi1m["repeat_equivalence_id"].astype(str).isin(protected)
    ]
    overlap_summary = {
        "current_table_protected_groups": len(protected),
        "upstream_prestandardization_protected_rows": int(
            periodic_summary["upstream_prestandardization_protected_rows"]
        ),
        "pi1m_train_rows": int(pi1m["split"].eq("train").sum()),
        "pi1m_train_protected_overlap_rows": int(len(overlap)),
    }
    (output_root / "overlap_summary.json").write_text(
        json.dumps(overlap_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    registry["overlap"] = overlap_summary
    registry_path = output_root / "registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return registry


__all__ = ["prepare_project_data"]
