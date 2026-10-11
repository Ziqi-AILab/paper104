"""Build project-local encoder inputs reused across formal evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from polyfid.paths import ProjectPaths


_TRANSPOLYMER_TASKS = (
    "eea",
    "egb",
    "egc",
    "ei",
    "eps",
    "nc",
    "opv",
    "pe_ii",
    "xc",
)


def _input_rows(
    path: Path,
    *,
    source_dataset: str,
    text_candidates: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    text_column = next(column for column in text_candidates if column in frame)
    columns = ["sample_id", text_column]
    for optional in ("task",):
        if optional in frame:
            columns.append(optional)
    rows = frame.loc[:, columns].copy().rename(columns={text_column: "canonical_text"})
    rows["source_dataset"] = source_dataset
    rows = rows.loc[
        rows["sample_id"].notna() & rows["canonical_text"].notna()
    ].copy()
    rows["sample_id"] = rows["sample_id"].astype(str)
    rows["canonical_text"] = rows["canonical_text"].astype(str)
    return rows.loc[rows["canonical_text"].ne("")]


def _unique_encoder_rows(frame: pd.DataFrame) -> pd.DataFrame:
    text_counts = frame.groupby("sample_id", sort=False)["canonical_text"].nunique()
    conflicting = text_counts.loc[text_counts.gt(1)]
    if not conflicting.empty:
        examples = ", ".join(conflicting.index.astype(str)[:5])
        raise ValueError(f"sample IDs map to different model inputs: {examples}")
    return frame.drop_duplicates("sample_id", keep="first").reset_index(drop=True)


def build_evaluation_inputs(paths: ProjectPaths) -> dict[str, Any]:
    """Materialize the two formal input universes and fidelity anchor table."""

    shared = paths.shared_data_root / "standardized/v1"
    project = paths.project_root
    pieces = [
        _input_rows(
            shared / "aldeghi_vipea/vipea.parquet",
            source_dataset="aldeghi_vipea",
            text_candidates=("canonical_text",),
        )
    ]
    for task in _TRANSPOLYMER_TASKS:
        pieces.append(
            _input_rows(
                shared / f"transpolymer/{task}.parquet",
                source_dataset=f"transpolymer_{task}",
                text_candidates=("structure_only_text", "canonical_text"),
            )
        )
    pe_i_path = project / "data/derived/transpolymer_pe_i_official.parquet"
    pieces.append(
        _input_rows(
            pe_i_path,
            source_dataset="transpolymer_pe_i_official",
            text_candidates=("structure_only_text",),
        )
    )
    if paths.formal_openpoly:
        pieces.append(
            _input_rows(
                shared / "openpoly/openpoly_live_2026_08_08.parquet",
                source_dataset="openpoly",
                text_candidates=("canonical_text",),
            )
        )
    for task in ("esol", "freesolv", "lipophilicity"):
        pieces.append(
            _input_rows(
                shared / f"molecular_retention/{task}.parquet",
                source_dataset=task,
                text_candidates=("canonical_text",),
            )
        )

    structure_only = _unique_encoder_rows(pd.concat(pieces, ignore_index=True))
    metadata = _unique_encoder_rows(
        _input_rows(
            pe_i_path,
            source_dataset="transpolymer_pe_i_official",
            text_candidates=("structure_with_metadata_text",),
        )
    )

    output_root = project / "data/derived/evaluation_inputs"
    output_root.mkdir(parents=True, exist_ok=True)
    structure_path = output_root / "downstream_structure_only.parquet"
    metadata_path = output_root / "pe_i_structure_plus_metadata.parquet"
    structure_only.to_parquet(structure_path, index=False)
    metadata.to_parquet(metadata_path, index=False)

    relation_path = project / "data/relations/test/pi1m_triplets.parquet"
    fidelity_path = output_root / "fidelity_test_records.parquet"
    fidelity_rows = 0
    if relation_path.is_file():
        relations = pd.read_parquet(
            relation_path,
            columns=["anchor_sample_id", "anchor_psmiles"],
        ).drop_duplicates("anchor_sample_id")
        relations = relations.rename(
            columns={
                "anchor_sample_id": "sample_id",
                "anchor_psmiles": "canonical_psmiles",
            }
        )
        relations.to_parquet(fidelity_path, index=False)
        fidelity_rows = int(len(relations))

    report = {
        "structure_only": {
            "path": str(structure_path),
            "rows": int(len(structure_only)),
            "source_counts": {
                str(key): int(value)
                for key, value in structure_only["source_dataset"]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "pe_i_structure_plus_metadata": {
            "path": str(metadata_path),
            "rows": int(len(metadata)),
        },
        "fidelity_test": {
            "path": str(fidelity_path),
            "rows": fidelity_rows,
            "available": relation_path.is_file(),
        },
        "openpoly": {
            "formal_retained": paths.formal_openpoly,
            "included_in_structure_only": paths.formal_openpoly,
        },
    }
    (output_root / "registry.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["build_evaluation_inputs"]
