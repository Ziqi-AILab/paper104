#!/usr/bin/env python3
"""Generate PerioGT and GRIN same-split execution matrices.

PerioGT graph, descriptor, and periodic-prompt preparation is a separate first
local-CPU phase with one cache per task.  The generated A800 training list is
intended to be submitted only after those seven caches complete.  This script
never executes or submits jobs.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from polyfid.paths import DEFAULT_PATH_CONFIG, ProjectPaths, load_paths


BASELINE_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs/downstream/external_baselines.yaml"
)


@dataclass(frozen=True)
class GraphExternalJob:
    row_index: int
    phase: str
    job_id: str
    method: str
    reporting_table: str
    task: str
    protocol: str
    fold: int
    seed: int
    dataset: str
    split_manifest: str
    cache_dir: str
    work_dir: str
    checkpoint_dir: str
    checkpoint_retention: str
    output: str
    script: str
    command: str
    resource: str = "GPU-A800"
    gpus: int = 1
    execution_path: str = "[GPU-A800 → sbatch]"


def _slug(value: str) -> str:
    return value.casefold().replace("-", "_")


def _folds(path: Path) -> tuple[int, ...]:
    columns = set(pq.read_schema(path).names)
    fold_column = "evaluation_fold" if "evaluation_fold" in columns else "fold"
    values = pq.read_table(path, columns=[fold_column]).column(0).to_pylist()
    return tuple(sorted({int(value) for value in values}))


def _command(arguments: list[str]) -> str:
    return '"${PYTHON_BIN}" ' + shlex.join(arguments)


def _local_command(paths: ProjectPaths, arguments: list[str]) -> str:
    python = paths.project_root / ".conda/env/bin/python"
    return shlex.join([str(python), *arguments])


def _render(job: GraphExternalJob) -> str:
    log_name = f"external_{job.method.casefold()}"
    hours = "1-00:00:00" if job.phase == "prepare" else "3-00:00:00"
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=p104-{_slug(job.method)}-{_slug(job.task)}-f{job.fold}-s{job.seed}
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:a800:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time={hours}
#SBATCH --output=logs/slurm/{log_name}/%x-%j.out
#SBATCH --error=logs/slurm/{log_name}/%x-%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-.}}"
PYTHON_BIN="${{PROJECT_DIR}}/.conda/env/bin/python"
export DGLBACKEND=pytorch
cd "${{PROJECT_DIR}}"
mkdir -p "${{PROJECT_DIR}}/logs/slurm/{log_name}"

{job.command}
'''


def _configs(path: Path) -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: dict(payload[name]) for name in ("periogt", "grin")}


def build_jobs(
    paths: ProjectPaths,
    output_root: Path,
    *,
    baseline_config: Path = BASELINE_CONFIG,
) -> tuple[list[GraphExternalJob], list[GraphExternalJob], list[GraphExternalJob]]:
    configs = _configs(baseline_config)
    standardized = paths.shared_data_root / "standardized/v1/transpolymer"
    prepare: list[GraphExternalJob] = []
    periogt: list[GraphExternalJob] = []
    grin: list[GraphExternalJob] = []

    for method in ("periogt", "grin"):
        config = configs[method]
        protocol = str(config["protocol"])
        method_dir = output_root / method
        for task in (str(value) for value in config["tasks"]):
            task_slug = _slug(task)
            dataset = standardized / f"{task_slug}.parquet"
            split_manifest = paths.project_root / f"data/splits/transpolymer_{task_slug}.parquet"
            folds = _folds(split_manifest)
            cache_dir = (
                paths.project_root / "runs/external_reference/periogt_cache" / task_slug
                if method == "periogt"
                else Path("")
            )
            if method == "periogt":
                prep_id = f"periogt_prepare_{task_slug}"
                prep_args = [
                    "-m", "polyfid.cli", "downstream", "external-prepare",
                    "--dataset", str(dataset),
                    "--split-manifest", str(split_manifest),
                    "--task", task,
                    "--protocol", protocol,
                    "--fold", str(folds[0]),
                    "--cache-dir", str(cache_dir),
                    "--preprocess-workers", str(config["preprocess_workers"]),
                ]
                prepare.append(
                    GraphExternalJob(
                        row_index=len(prepare),
                        phase="prepare",
                        job_id=prep_id,
                        method="PerioGT",
                        reporting_table=str(config["reporting_table"]),
                        task=task,
                        protocol=protocol,
                        fold=folds[0],
                        seed=0,
                        dataset=str(dataset),
                        split_manifest=str(split_manifest),
                        cache_dir=str(cache_dir),
                        work_dir=str(cache_dir),
                        checkpoint_dir="",
                        checkpoint_retention="not_applicable",
                        output=str(cache_dir),
                        script="",
                        command=_local_command(paths, prep_args),
                        resource="CPU",
                        gpus=0,
                        execution_path="[CPU → local, no sbatch]",
                    )
                )

            for fold in folds:
                for seed in (int(value) for value in config["downstream_seeds"]):
                    job_id = f"{method}_{task_slug}_{protocol}_fold-{fold}_seed-{seed}"
                    work_dir = (
                        paths.project_root / f"runs/external_reference/{method}"
                        / task_slug / protocol / f"fold-{fold}" / f"seed-{seed}"
                    )
                    checkpoint_dir = (
                        paths.checkpoint_root / f"downstream/external_reference/{method}"
                        / task_slug / protocol / f"fold-{fold}" / f"seed-{seed}"
                    )
                    output = (
                        paths.project_root / f"results/predictions/external_reference/{method}"
                        / task_slug / protocol / f"fold-{fold}" / f"seed-{seed}"
                        / "predictions.parquet"
                    )
                    script = method_dir / "train" / f"{job_id}.sbatch"
                    arguments = [
                        "-m", "polyfid.cli", "downstream", "external-baseline",
                        "--baseline", method,
                        "--dataset", str(dataset),
                        "--split-manifest", str(split_manifest),
                        "--task", task,
                        "--protocol", protocol,
                        "--fold", str(fold),
                        "--seed", str(seed),
                        "--epochs", str(config["epochs"]),
                        "--patience", str(config["patience"]),
                        "--batch-size", str(config["batch_size"]),
                        "--learning-rate", str(config["learning_rate"]),
                        "--work-dir", str(work_dir),
                        "--checkpoint-dir", str(checkpoint_dir),
                        "--checkpoint-retention", "metadata",
                        "--output", str(output),
                    ]
                    if method == "periogt":
                        arguments.extend(
                            [
                                "--cache-dir", str(cache_dir),
                                "--num-workers", "2",
                            ]
                        )
                    job = GraphExternalJob(
                        row_index=len(periogt if method == "periogt" else grin),
                        phase="train_and_evaluate",
                        job_id=job_id,
                        method="PerioGT" if method == "periogt" else "GRIN",
                        reporting_table=str(config["reporting_table"]),
                        task=task,
                        protocol=protocol,
                        fold=fold,
                        seed=seed,
                        dataset=str(dataset),
                        split_manifest=str(split_manifest),
                        cache_dir=str(cache_dir),
                        work_dir=str(work_dir),
                        checkpoint_dir=str(checkpoint_dir),
                        checkpoint_retention="metadata",
                        output=str(output),
                        script=str(script),
                        command=_command(arguments),
                    )
                    (periogt if method == "periogt" else grin).append(job)
    return prepare, periogt, grin


def _write_group(name: str, jobs: list[GraphExternalJob], output_root: Path) -> None:
    if not jobs:
        return
    for job in jobs:
        script = Path(job.script)
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(_render(job), encoding="utf-8")
        script.chmod(0o755)
    rows = [{**asdict(job), "row_index": index} for index, job in enumerate(jobs)]
    manifest = output_root / f"{name}_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / f"{name}_job_list.txt").write_text(
        "\n".join(job.script for job in jobs) + "\n", encoding="utf-8"
    )


def _write_prepare_group(jobs: list[GraphExternalJob], output_root: Path) -> None:
    rows = [{**asdict(job), "row_index": index} for index, job in enumerate(jobs)]
    manifest = output_root / "periogt_prepare_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "periogt_prepare_command_list.txt").write_text(
        "\n".join(job.command for job in jobs) + "\n", encoding="utf-8"
    )

    # These files came from the earlier GPU-scheduled preparation plan.
    (output_root / "periogt_prepare_job_list.txt").unlink(missing_ok=True)
    prepare_script_dir = output_root / "periogt/prepare"
    if prepare_script_dir.is_dir():
        for script in prepare_script_dir.glob("*.sbatch"):
            script.unlink()


def write_matrices(
    paths: ProjectPaths,
    output_root: Path,
    *,
    baseline_config: Path = BASELINE_CONFIG,
) -> tuple[list[GraphExternalJob], list[GraphExternalJob], list[GraphExternalJob]]:
    output_root.mkdir(parents=True, exist_ok=True)
    prepare, periogt, grin = build_jobs(
        paths, output_root, baseline_config=baseline_config
    )
    _write_prepare_group(prepare, output_root)
    _write_group("periogt_external_reference", periogt, output_root)
    _write_group("grin_external_reference", grin, output_root)
    _write_group("graph_external_reference_training", periogt + grin, output_root)
    return prepare, periogt, grin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument("--baseline-config", type=Path, default=BASELINE_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = load_paths(args.paths_config)
    output = args.output_dir or (
        paths.project_root / "slurm/generated/external_reference/graph_models"
    )
    prepare, periogt, grin = write_matrices(
        paths, output.resolve(), baseline_config=args.baseline_config.resolve()
    )
    print(
        f"periogt_prepare={len(prepare)}; periogt={len(periogt)}; "
        f"grin={len(grin)}; no jobs were submitted"
    )


if __name__ == "__main__":
    main()
