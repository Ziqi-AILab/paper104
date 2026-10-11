#!/usr/bin/env python3
"""Generate formal downstream property-view GPU and local-CPU matrices.

This script only writes manifests and one-A800 sbatch files.  It never submits
them.  Equivalent-view rows must first be built locally with
``scripts/build_property_view_inputs.py``; 100%-data frozen probe states must
also exist before the local prediction batches are run.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from polyfid.paths import ProjectPaths, load_paths
from polyfid.workflows.property_view_inputs import property_view_partition_id


FORMAL_ARMS = ("B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs")
FORMAL_SEEDS = (42, 123, 777, 2025, 2026)


@dataclass(frozen=True)
class PropertyViewRepresentation:
    arm: str
    representation_seed: int
    representation_label: str
    replicate_role: str
    checkpoint: str
    base_checkpoint: str


@dataclass(frozen=True)
class PropertyViewEmbeddingJob:
    job_id: str
    arm: str
    representation_seed: int
    representation_label: str
    replicate_role: str
    checkpoint: str
    base_checkpoint: str
    input_registry: str
    output_directory: str
    run_record: str
    script: str
    command: str
    resource: str = "GPU-A800"
    execution_path: str = "[GPU-A800 → sbatch]"


@dataclass(frozen=True)
class PropertyViewPredictionUnit:
    row_index: int
    representation_label: str
    arm: str
    representation_seed: int
    replicate_role: str
    dataset: str
    split_manifest: str
    task: str
    protocol: str
    protocol_role: str
    fold: int
    label_column: str
    input_view: str
    partition_id: str
    reference_embeddings: str
    view_rows: str
    scope_rows: str
    view_embeddings: str
    probe_state_dir: str
    head_seeds: str
    label_fraction: float
    subsample_id: int
    output_root: str
    batch_summary: str
    resource: str = "CPU"
    execution_path: str = "[CPU → local, no sbatch]"


def build_representations(paths: ProjectPaths) -> list[PropertyViewRepresentation]:
    base = paths.shared_model_root / "chemberta_100m_mlm"
    output = [
        PropertyViewRepresentation(
            arm="C",
            representation_seed=0,
            representation_label="C_fixed",
            replicate_role="fixed_reference",
            checkpoint=str(base),
            base_checkpoint=str(base),
        )
    ]
    output.extend(
        PropertyViewRepresentation(
            arm=arm,
            representation_seed=seed,
            representation_label=f"{arm}_seed-{seed}",
            replicate_role="independent_pretraining",
            checkpoint=str(
                paths.checkpoint_root
                / "pretraining"
                / arm
                / f"seed-{seed}"
                / "token-100.pt"
            ),
            base_checkpoint=str(base),
        )
        for arm in FORMAL_ARMS
        for seed in FORMAL_SEEDS
    )
    return output


def _python_command(arguments: list[str]) -> str:
    return '"${PYTHON_BIN}" ' + shlex.join(arguments)


def _render_a800_script(job_name: str, command: str) -> str:
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:a800:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-.}}"
PYTHON_BIN="${{PROJECT_DIR}}/.conda/env/bin/python"
cd "${{PROJECT_DIR}}"
mkdir -p "${{PROJECT_DIR}}/logs/slurm"

{command}
'''


def build_embedding_jobs(
    paths: ProjectPaths, output_dir: Path
) -> list[PropertyViewEmbeddingJob]:
    registry = paths.project_root / "data/derived/property_views/registry.tsv"
    if not paths.formal_openpoly and registry.is_file():
        rows = pd.read_csv(registry, sep="\t")
        rows = rows.loc[
            ~rows["partition_id"].astype(str).str.startswith("openpoly_")
        ].copy()
        registry = output_dir / "formal_property_view_registry.tsv"
        registry.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(registry, sep="\t", index=False)
    jobs: list[PropertyViewEmbeddingJob] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for representation in build_representations(paths):
        output = (
            paths.project_root
            / "runs/embeddings/formal"
            / representation.representation_label
            / "property_views"
        )
        job_id = f"property_views_{representation.representation_label}"
        script = output_dir / f"{job_id}.sbatch"
        command = _python_command(
            [
                "scripts/cache_property_view_embeddings.py",
                "--registry",
                str(registry),
                "--checkpoint",
                representation.checkpoint,
                "--base-checkpoint",
                representation.base_checkpoint,
                "--method",
                representation.arm,
                "--seed",
                str(representation.representation_seed),
                "--batch-size",
                "128",
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--output",
                str(output),
                "--run-record",
                str(
                    paths.project_root
                    / "runs/property_views"
                    / f"{representation.representation_label}.json"
                ),
            ]
        )
        job = PropertyViewEmbeddingJob(
            job_id=job_id,
            arm=representation.arm,
            representation_seed=representation.representation_seed,
            representation_label=representation.representation_label,
            replicate_role=representation.replicate_role,
            checkpoint=representation.checkpoint,
            base_checkpoint=representation.base_checkpoint,
            input_registry=str(registry),
            output_directory=str(output),
            run_record=str(
                paths.project_root
                / "runs/property_views"
                / f"{representation.representation_label}.json"
            ),
            script=str(script),
            command=command,
        )
        jobs.append(job)
        script.write_text(
            _render_a800_script(
                f"p104-prop-view-{representation.representation_label}".replace(
                    "_", "-"
                ),
                command,
            ),
            encoding="utf-8",
        )
        script.chmod(0o755)
    return jobs


def _representation_label(row: pd.Series) -> str:
    if str(row["replicate_role"]) == "fixed_reference":
        return "C_fixed"
    return f"{row['arm']}_seed-{int(row['representation_seed'])}"


def build_prediction_units(
    paths: ProjectPaths, probe_manifest: str | Path
) -> list[PropertyViewPredictionUnit]:
    probes = pd.read_csv(probe_manifest, sep="\t")
    probes = probes.loc[
        probes["label_fraction"].astype(float).eq(1.0)
        & probes["subsample_id"].astype(int).eq(0)
        & ~probes["protocol_role"].astype(str).eq("molecular_retention")
    ].copy()
    if not paths.formal_openpoly:
        probes = probes.loc[
            ~probes["task"].astype(str).str.startswith("OpenPoly:")
        ].copy()
    probes["representation_label"] = probes.apply(_representation_label, axis=1)
    units: list[PropertyViewPredictionUnit] = []
    for row_index, row in enumerate(probes.itertuples(index=False)):
        partition_id = property_view_partition_id(row.dataset, row.input_view)
        label = row.representation_label
        if row.input_view == "structure_plus_metadata":
            reference_name = "pe_i_structure_plus_metadata.npz"
        else:
            reference_name = "structure_only.npz"
        output = (
            paths.project_root
            / "results/predictions/property_views"
            / label
            / str(row.task).casefold().replace(":", "_").replace("-", "_")
            / str(row.protocol)
            / str(row.input_view)
            / f"fold-{int(row.fold)}"
        )
        units.append(
            PropertyViewPredictionUnit(
                row_index=row_index,
                representation_label=label,
                arm=str(row.arm),
                representation_seed=int(row.representation_seed),
                replicate_role=str(row.replicate_role),
                dataset=str(row.dataset),
                split_manifest=str(row.split_manifest),
                task=str(row.task),
                protocol=str(row.protocol),
                protocol_role=str(row.protocol_role),
                fold=int(row.fold),
                label_column=str(row.label_column),
                input_view=str(row.input_view),
                partition_id=partition_id,
                reference_embeddings=str(
                    paths.project_root
                    / "runs/embeddings/formal"
                    / label
                    / reference_name
                ),
                view_rows=str(
                    paths.project_root
                    / "data/derived/property_views/partitions"
                    / f"{partition_id}.parquet"
                ),
                scope_rows=str(
                    paths.project_root
                    / "data/derived/property_views/scope"
                    / f"{partition_id}.parquet"
                ),
                view_embeddings=str(
                    paths.project_root
                    / "runs/embeddings/formal"
                    / label
                    / "property_views"
                    / f"{partition_id}.npz"
                ),
                probe_state_dir=str(row.probe_state_dir),
                head_seeds="11,22,33",
                label_fraction=1.0,
                subsample_id=0,
                output_root=str(output),
                batch_summary=str(
                    paths.project_root
                    / "results/statistics/property_views"
                    / f"{label}_family_summary.csv"
                ),
            )
        )
    return units


def _write_tsv(rows: list[object], path: Path) -> None:
    materialized = [asdict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(materialized[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(materialized)


def write_matrix(
    paths: ProjectPaths,
    *,
    probe_manifest: Path,
    output_root: Path,
    prediction_root: Path,
) -> tuple[list[PropertyViewEmbeddingJob], list[PropertyViewPredictionUnit]]:
    embedding_jobs = build_embedding_jobs(paths, output_root / "embeddings")
    _write_tsv(embedding_jobs, output_root / "embedding_manifest.tsv")
    (output_root / "embedding_job_list.txt").write_text(
        "\n".join(job.script for job in embedding_jobs) + "\n", encoding="utf-8"
    )

    prediction_units = build_prediction_units(paths, probe_manifest)
    prediction_manifest = prediction_root / "property_view_prediction_manifest.tsv"
    _write_tsv(prediction_units, prediction_manifest)
    python = paths.project_root / ".conda/env/bin/python"
    runner = paths.project_root / "scripts/run_local_property_view_manifest.py"
    labels = list(dict.fromkeys(unit.representation_label for unit in prediction_units))
    commands = [
        shlex.join(
            [
                str(python),
                str(runner),
                "--manifest",
                str(prediction_manifest),
                "--representation-label",
                label,
            ]
        )
        for label in labels
    ]
    (prediction_root / "property_view_local_command_list.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    return embedding_jobs, prediction_units


def main() -> None:
    paths = load_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=paths.project_root / "runs/manifests/frozen_probes/frozen_probe_manifest.tsv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=paths.project_root / "slurm/generated/property_views",
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=paths.project_root / "runs/manifests/property_views",
    )
    args = parser.parse_args()
    embedding_jobs, units = write_matrix(
        paths,
        probe_manifest=args.probe_manifest,
        output_root=args.output_root,
        prediction_root=args.prediction_root,
    )
    labels = {unit.representation_label for unit in units}
    print(
        f"embedding_jobs={len(embedding_jobs)}, prediction_units={len(units)}, "
        f"local_representation_batches={len(labels)}, head_state_evaluations={len(units) * 3}; "
        "no jobs were submitted"
    )


if __name__ == "__main__":
    main()
