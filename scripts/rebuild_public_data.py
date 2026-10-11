#!/usr/bin/env python
"""Rebuild Paper104 shared standardized tables from the retained public raw files."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import pickle
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from polyfid.data.public_standardization import (
    DatasetProvenance,
    PreparedRecords,
    aldeghi_cv9_partitions,
    iter_csv_records,
    iter_zinc20_jsonl,
    load_aldeghi_cv9,
    prepare_aldeghi_records,
    prepare_moleculenet_directory,
    prepare_openpoly_excel,
    prepare_pi1m_records,
    prepare_transpolymer_directory,
    prepare_zinc20_records,
    write_prepared_records,
)
from polyfid.paths import DEFAULT_PATH_CONFIG, ProjectPaths


ACQUIRED = "2026-08-08"


def _provenance(
    source: str,
    use_note: str,
    version: str,
    paths: Sequence[Path],
) -> DatasetProvenance:
    return DatasetProvenance(
        source=source,
        date=ACQUIRED,
        license=use_note,
        version=version,
        paths=tuple(str(path.resolve()) for path in paths),
    )


def _write_many(
    tables: Mapping[str, PreparedRecords],
    output_dir: Path,
    provenance: DatasetProvenance,
) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for name, table in sorted(tables.items()):
        slug = name.casefold().replace("-", "_").replace(" ", "_")
        output[name] = write_prepared_records(
            table, output_dir, name=slug, provenance=provenance
        )
    return output


def _write_rows(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def _load_index_folds(path: Path) -> tuple[dict[str, tuple[int, ...]], ...]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    output = []
    for fold in payload:
        output.append(
            {
                "train": tuple(int(value) for value in fold["train_idx"]),
                "validation": tuple(int(value) for value in fold["val_idx"]),
                "test": tuple(int(value) for value in fold["test_idx"]),
            }
        )
    return tuple(output)


def _random_roles(
    records: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Sequence[int]]],
) -> tuple[dict[str, Any], ...]:
    sample_by_row = {
        int(row["source_row_index"]): str(row["sample_id"]) for row in records
    }
    output = []
    for fold_index, fold in enumerate(folds):
        for split in ("train", "validation", "test"):
            for source_index in fold[split]:
                sample_id = sample_by_row.get(int(source_index))
                if sample_id is not None:
                    output.append(
                        {
                            "sample_id": sample_id,
                            "source_row_index": int(source_index),
                            "fold": fold_index,
                            "split": split,
                        }
                    )
    return tuple(output)


def _pair_morgan_vector(components: Sequence[str], *, bits: int = 2_048) -> Any:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    combined = DataStructs.ExplicitBitVect(2 * bits)
    for index, component in enumerate(sorted(str(value) for value in components)):
        molecule = Chem.MolFromSmiles(component)
        if molecule is None:
            raise ValueError(f"unparseable closed repeat: {component}")
        morgan_vector = AllChem.GetMorganFingerprintAsBitVect(
            molecule, 2, nBits=bits, useChirality=True
        )
        for bit in morgan_vector.GetOnBits():
            combined.SetBit(index * bits + int(bit))
    return combined


def _butina_roles(
    records: Sequence[Mapping[str, Any]], *, seed: int = 42
) -> tuple[dict[str, Any], ...]:
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    by_group: dict[str, dict[str, Any]] = {}
    for row in records:
        group = str(row["monomer_pair_group_id"])
        by_group.setdefault(
            group,
            {
                "components": tuple(row["component_repeat_equivalence_ids"]),
                "rows": 0,
            },
        )["rows"] += 1
    groups = tuple(sorted(by_group))
    morgan_vectors = [
        _pair_morgan_vector(by_group[group]["components"]) for group in groups
    ]
    distances: list[float] = []
    for index in range(1, len(morgan_vectors)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            morgan_vectors[index], morgan_vectors[:index]
        )
        distances.extend(1.0 - float(value) for value in similarities)
    clusters = list(
        Butina.ClusterData(
            distances, len(morgan_vectors), 0.40, isDistData=True, reordering=True
        )
    )
    clusters.sort(key=lambda cluster: min(groups[int(index)] for index in cluster))
    group_cluster = {
        groups[int(member)]: cluster_index
        for cluster_index, cluster in enumerate(clusters)
        for member in cluster
    }
    cluster_rows = {
        index: sum(by_group[groups[int(member)]]["rows"] for member in cluster)
        for index, cluster in enumerate(clusters)
    }
    order = list(range(len(clusters)))
    random.Random(seed).shuffle(order)
    tie_rank = {cluster: rank for rank, cluster in enumerate(order)}
    order.sort(key=lambda cluster: (-cluster_rows[cluster], tie_rank[cluster]))
    fold_loads = [0] * 5
    cluster_fold: dict[int, int] = {}
    for cluster in order:
        fold = min(range(5), key=lambda value: (fold_loads[value], value))
        cluster_fold[cluster] = fold
        fold_loads[fold] += cluster_rows[cluster]
    output = []
    for evaluation_fold in range(5):
        validation_fold = (evaluation_fold + 1) % 5
        for row in records:
            group = str(row["monomer_pair_group_id"])
            cluster = group_cluster[group]
            assigned = cluster_fold[cluster]
            split = (
                "test"
                if assigned == evaluation_fold
                else "validation"
                if assigned == validation_fold
                else "train"
            )
            output.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "source_row_index": int(row["source_row_index"]),
                    "monomer_pair_group_id": group,
                    "cluster_id": f"butina-{cluster:05d}",
                    "cluster_fold": assigned,
                    "fold": evaluation_fold,
                    "split": split,
                }
            )
    return tuple(output)


def _repeat_values(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for field in ("repeat_equivalence_ids", "component_repeat_equivalence_ids"):
        supplied = row.get(field, ())
        if isinstance(supplied, str):
            supplied = (supplied,)
        values.update(str(value) for value in supplied or () if str(value).strip())
    direct = row.get("repeat_equivalence_id")
    if direct is not None and bool(row.get("linear_repeat_eligible", True)):
        values.add(str(direct))
    return values


def rebuild_downstream(
    paths: ProjectPaths, output_root: Path, *, seed: int
) -> dict[str, Any]:
    raw = paths.shared_data_root / "raw"
    summary: dict[str, Any] = {}

    trans_root = raw / "transpolymer/exports"
    trans_paths = sorted(trans_root.glob("*.csv"))
    trans = prepare_transpolymer_directory(trans_root, seed=seed)
    summary["transpolymer"] = _write_many(
        trans,
        output_root / "transpolymer",
        _provenance(
            "https://github.com/ChangwenXu98/TransPolymer",
            "MIT repository; endpoint data retain their cited source terms",
            "official repository exports acquired 2026-08-08",
            trans_paths,
        ),
    )

    vipea_root = raw / "aldeghi_vipea/datasets/vipea"
    vipea_csv = vipea_root / "chemprop_inputs/dataset-poly_chemprop.csv"
    cv9_path = vipea_root / "cv9-monomer_split.pkl"
    random_path = vipea_root / "cv10-random_split.pkl"
    cv9 = load_aldeghi_cv9(cv9_path)
    vipea = prepare_aldeghi_records(iter_csv_records(vipea_csv), folds=cv9, seed=seed)
    provenance = _provenance(
        "https://github.com/coleygroup/polymer-chemprop-data",
        "MIT repository",
        "VIPEA table and supplied split files",
        (vipea_csv, cv9_path, random_path),
    )
    summary["aldeghi_vipea"] = write_prepared_records(
        vipea, output_root / "aldeghi_vipea", name="vipea", provenance=provenance
    )
    _write_rows(
        aldeghi_cv9_partitions(vipea.records, cv9),
        output_root / "aldeghi_vipea/vipea_cv9_roles.parquet",
    )
    _write_rows(
        _random_roles(vipea.records, _load_index_folds(random_path)),
        output_root / "aldeghi_vipea/vipea_random_roles.parquet",
    )
    _write_rows(
        _butina_roles(vipea.records, seed=seed),
        output_root / "aldeghi_vipea/vipea_butina_roles.parquet",
    )

    retention_root = raw / "molecular_retention"
    retention = prepare_moleculenet_directory(retention_root, seed=seed)
    summary["molecular_retention"] = _write_many(
        retention,
        output_root / "molecular_retention",
        _provenance(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/",
            "upstream dataset terms",
            "public ESOL, FreeSolv, and Lipophilicity CSV exports",
            sorted(retention_root.glob("*.csv")),
        ),
    )

    if paths.formal_openpoly:
        openpoly_path = raw / "openpoly/experiment_polymer_data.xlsx"
        audit_path = paths.project_root / "configs/data/openpoly_endpoints.yaml"
        audit = yaml.safe_load(audit_path.read_text(encoding="utf-8"))
        openpoly = prepare_openpoly_excel(
            openpoly_path,
            min_endpoint_groups=int(audit["minimum_independent_structure_groups"]),
            endpoint_audit=audit["endpoints"],
            seed=seed,
        )
        summary["openpoly"] = write_prepared_records(
            openpoly,
            output_root / "openpoly",
            name="openpoly_live_2026_08_08",
            provenance=_provenance(
                "https://cleanenergymaterials.cn/polymer/polymer_database/experiment_polymer_database",
                "public snapshot; acquisition route and reuse terms require confirmation before formal use",
                "website export captured 2026-08-08; not the GitHub OpenPoly dataset",
                (openpoly_path, audit_path),
            ),
        )

    protected: set[str] = set()
    for name, table in trans.items():
        for row in table.records:
            if name != "PE-I-official" or row.get("split") in {"validation", "test"}:
                protected.update(_repeat_values(row))
    for row in vipea.records:
        protected.update(_repeat_values(row))
    if paths.formal_openpoly:
        for row in openpoly.records:
            if row.get("split") in {"validation", "test"}:
                protected.update(_repeat_values(row))
    else:
        # Keep the original exclusions, including the historical conditional
        # snapshot, without requiring that excluded dataset's raw file.
        frozen = paths.exclusion_path("protected_repeat_groups.txt")
        if not frozen.is_file():
            raise FileNotFoundError(
                f"Frozen exclusion list required: {frozen}. "
                "Configure --exclusion-root with configure_local_paths.py; see docs/data.md."
            )
        protected = set(frozen.read_text().splitlines())
    protected_path = output_root / "protected_repeat_groups.txt"
    protected_path.write_text("\n".join(sorted(protected)) + "\n", encoding="utf-8")
    summary["protected_repeat_groups"] = {
        "path": str(protected_path),
        "count": len(protected),
    }
    return summary


def rebuild_pi1m(
    paths: ProjectPaths,
    output_root: Path,
    *,
    seed: int,
    max_source_rows: int | None,
) -> dict[str, Any]:
    protected_path = output_root / "protected_repeat_groups.txt"
    if not protected_path.is_file():
        raise FileNotFoundError("run the downstream stage first to build protected groups")
    protected = tuple(
        line.strip()
        for line in protected_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    source = paths.shared_data_root / "raw/pi1m/PI1M_v2.csv"
    table = prepare_pi1m_records(
        iter_csv_records(source),
        protected_repeat_groups=protected,
        seed=seed,
        split_ratios=(0.9, 0.05, 0.05),
        max_source_rows=max_source_rows,
    )
    name = "pi1m_v2" if max_source_rows is None else f"pi1m_v2_smoke_{max_source_rows}"
    return write_prepared_records(
        table,
        output_root / "pi1m",
        name=name,
        provenance=_provenance(
            "https://github.com/RUIMINMA1996/PI1M",
            "MIT repository license; README states academic purpose only",
            "PI1M v2",
            (source,),
        ),
    )


def rebuild_zinc20(
    paths: ProjectPaths,
    output_root: Path,
    *,
    seed: int,
    smoke_count: int | None,
) -> dict[str, Any]:
    source = (
        paths.shared_data_root
        / "raw/zinc20/zinc_processed/smiles_all_00_clean.jsonl.gz"
    )
    if smoke_count is None:
        train_count, validation_count, retention_count = 1_000_000, 50_000, 50_000
        rows = iter_zinc20_jsonl(source)
        name = "zinc20_1m"
    else:
        train_count = int(smoke_count)
        validation_count = max(1, train_count // 10)
        retention_count = max(1, train_count // 10)
        rows = islice(iter_zinc20_jsonl(source), max(10_000, train_count * 20))
        name = f"zinc20_smoke_{train_count}"
    table = prepare_zinc20_records(
        rows,
        train_count=train_count,
        validation_count=validation_count,
        retention_count=retention_count,
        seed=seed,
    )
    return write_prepared_records(
        table,
        output_root / "zinc20",
        name=name,
        provenance=_provenance(
            "https://huggingface.co/datasets/zpn/zinc20",
            "ODbL as stated on the source dataset card",
            "smiles_all_00_clean public source shard",
            (source,),
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("downstream", "pi1m", "zinc20", "all"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pi1m-max-source-rows", type=int)
    parser.add_argument("--zinc-smoke-count", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    paths = ProjectPaths.from_yaml(args.paths_config)
    output_root = args.output_root.expanduser().resolve()
    frozen_root = (paths.shared_data_root / "standardized/v1").resolve()
    if output_root == frozen_root or frozen_root in output_root.parents:
        raise SystemExit(
            "--output-root must be outside the frozen shared standardized/v1 directory"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if args.stage in {"downstream", "all"}:
        summary["downstream"] = rebuild_downstream(paths, output_root, seed=args.seed)
    if args.stage in {"pi1m", "all"}:
        summary["pi1m"] = rebuild_pi1m(
            paths,
            output_root,
            seed=args.seed,
            max_source_rows=args.pi1m_max_source_rows,
        )
    if args.stage in {"zinc20", "all"}:
        summary["zinc20"] = rebuild_zinc20(
            paths,
            output_root,
            seed=args.seed,
            smoke_count=args.zinc_smoke_count,
        )
    summary_path = output_root / "public_data_rebuild.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": str(summary_path), "stages": list(summary)}, indent=2))


if __name__ == "__main__":
    main()
