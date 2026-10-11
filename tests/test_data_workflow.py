from __future__ import annotations

import pandas as pd

from polyfid.workflows.data import (
    _make_group_disjoint,
    _paper104_pe_i_records,
    _retain_declared_roles,
)


def test_aldeghi_group_resolution_drops_rows_without_relabelling_family_roles() -> None:
    roles = pd.DataFrame(
        {
            "sample_id": ["test-family", "other-family", "train-only"],
            "fold": [0, 0, 0],
            "split": ["test", "train", "train"],
        }
    )
    groups = pd.Series(
        {"test-family": "shared-pair", "other-family": "shared-pair", "train-only": "train-pair"}
    )

    resolved = _make_group_disjoint(roles, groups)

    assert resolved.set_index("sample_id")["split"].to_dict() == {
        "test-family": "test",
        "train-only": "train",
    }


def test_declared_role_resolution_is_fold_local() -> None:
    roles = pd.DataFrame(
        {
            "sample_id": ["a", "b", "a", "b"],
            "fold": [0, 0, 1, 1],
            "group_id": ["g", "g", "g", "g"],
            "split": ["test", "train", "train", "validation"],
        }
    )

    resolved = _retain_declared_roles(roles, group_columns=["fold", "group_id"])

    assert list(zip(resolved["fold"], resolved["split"], strict=True)) == [
        (0, "test"),
        (1, "validation"),
    ]


def test_pe_i_record_identity_distinguishes_partition_local_row_numbers() -> None:
    records = pd.DataFrame(
        {
            "sample_id": ["transpolymer:PE-I:0", "transpolymer:PE-I:0"],
            "task": ["PE-I", "PE-I"],
            "official_partition": ["train", "test"],
            "source_row_index": [0, 0],
        }
    )

    prepared = _paper104_pe_i_records(records)

    assert prepared["sample_id"].tolist() == [
        "transpolymer:PE-I:train:0",
        "transpolymer:PE-I:test:0",
    ]
    assert prepared["source_sample_id"].tolist() == [
        "transpolymer:PE-I:0",
        "transpolymer:PE-I:0",
    ]

