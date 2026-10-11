#!/usr/bin/env python3
"""Generate the runnable TransPolymer same-split external-reference matrix.

The generator follows the folds present in each Paper104 split manifest and
writes one single-A800 script per task, fold, and downstream seed.  It only
creates execution artifacts; it never submits jobs.
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


DEFAULT_BASELINE_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/downstream/external_baselines.yaml"
)


@dataclass(frozen=True)
class ExternalReferenceJob:
    row_index: int
    job_id: str
    method: str
    reporting_table: str
    task: str
    protocol: str
    fold: int
    seed: int
    dataset: str
    split_manifest: str
    work_dir: str
    checkpoint_dir: str
    checkpoint_retention: str
    output: str
    epochs: int
    patience: int
    batch_size: int
    max_length: int
    script: str
    command: str
    resource: str = "GPU-A800"
    execution_path: str = "[GPU-A800 → sbatch]"


def _slug(task: str) -> str:
    return task.casefold().replace("-", "_")


def _folds(split_manifest: Path) -> tuple[int, ...]:
    columns = set(pq.read_schema(split_manifest).names)
    fold_column = "evaluation_fold" if "evaluation_fold" in columns else "fold"
    values = pq.read_table(split_manifest, columns=[fold_column]).column(0).to_pylist()
    return tuple(sorted({int(value) for value in values}))


def _python_command(arguments: list[str]) -> str:
    return '"${PYTHON_BIN}" ' + shlex.join(arguments)


def _render_script(job: ExternalReferenceJob) -> str:
    job_name = f"p104-tp-{_slug(job.task)}-f{job.fold}-s{job.seed}"
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:a800:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/external_transpolymer/%x-%j.out
#SBATCH --error=logs/slurm/external_transpolymer/%x-%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-.}}"
PYTHON_BIN="${{PROJECT_DIR}}/.conda/env/bin/python"
cd "${{PROJECT_DIR}}"
mkdir -p "${{PROJECT_DIR}}/logs/slurm/external_transpolymer"

{job.command}
'''


def _load_transpolymer_config(path: Path) -> dict[str, object]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(values["transpolymer"])


def build_jobs(
    paths: ProjectPaths,
    output_dir: Path,
    *,
    baseline_config: Path = DEFAULT_BASELINE_CONFIG,
) -> list[ExternalReferenceJob]:
    config = _load_transpolymer_config(baseline_config)
    protocol = str(config["protocol"])
    jobs: list[ExternalReferenceJob] = []
    standardized = paths.shared_data_root / "standardized/v1/transpolymer"

    for task in tuple(str(value) for value in config["tasks"]):
        task_slug = _slug(task)
        dataset = standardized / f"{task_slug}.parquet"
        split_manifest = (
            paths.project_root / f"data/splits/transpolymer_{task_slug}.parquet"
        )
        for fold in _folds(split_manifest):
            for seed in tuple(int(value) for value in config["downstream_seeds"]):
                job_id = (
                    f"transpolymer_{task_slug}_{protocol}_fold-{fold}_seed-{seed}"
                )
                work_dir = (
                    paths.project_root
                    / "runs/external_reference/transpolymer"
                    / task_slug
                    / protocol
                    / f"fold-{fold}"
                    / f"seed-{seed}"
                )
                checkpoint_dir = (
                    paths.checkpoint_root
                    / "downstream/external_reference/transpolymer"
                    / task_slug
                    / protocol
                    / f"fold-{fold}"
                    / f"seed-{seed}"
                )
                output = (
                    paths.project_root
                    / "results/predictions/external_reference/transpolymer"
                    / task_slug
                    / protocol
                    / f"fold-{fold}"
                    / f"seed-{seed}"
                    / "predictions.parquet"
                )
                script = output_dir / f"{job_id}.sbatch"
                arguments = [
                    "-m",
                    "polyfid.cli",
                    "downstream",
                    "external-baseline",
                    "--dataset",
                    str(dataset),
                    "--split-manifest",
                    str(split_manifest),
                    "--task",
                    task,
                    "--protocol",
                    protocol,
                    "--fold",
                    str(fold),
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(config["epochs"]),
                    "--patience",
                    str(config["patience"]),
                    "--batch-size",
                    str(config["batch_size"]),
                    "--max-length",
                    str(config["max_length"]),
                    "--num-workers",
                    "0",
                    "--work-dir",
                    str(work_dir),
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--checkpoint-retention",
                    "metadata",
                    "--output",
                    str(output),
                ]
                provisional = ExternalReferenceJob(
                    row_index=len(jobs),
                    job_id=job_id,
                    method="TransPolymer",
                    reporting_table=str(config["reporting_table"]),
                    task=task,
                    protocol=protocol,
                    fold=fold,
                    seed=seed,
                    dataset=str(dataset),
                    split_manifest=str(split_manifest),
                    work_dir=str(work_dir),
                    checkpoint_dir=str(checkpoint_dir),
                    checkpoint_retention="metadata",
                    output=str(output),
                    epochs=int(config["epochs"]),
                    patience=int(config["patience"]),
                    batch_size=int(config["batch_size"]),
                    max_length=int(config["max_length"]),
                    script=str(script),
                    command=_python_command(arguments),
                )
                jobs.append(provisional)
    return jobs


def write_matrix(
    paths: ProjectPaths,
    output_dir: Path,
    *,
    baseline_config: Path = DEFAULT_BASELINE_CONFIG,
) -> list[ExternalReferenceJob]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(paths, output_dir, baseline_config=baseline_config)
    for job in jobs:
        script = Path(job.script)
        script.write_text(_render_script(job), encoding="utf-8")
        script.chmod(0o755)

    rows = [asdict(job) for job in jobs]
    manifest = output_dir / "transpolymer_external_reference_manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "transpolymer_external_reference_job_list.txt").write_text(
        "\n".join(job.script for job in jobs) + "\n", encoding="utf-8"
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument(
        "--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths_config)
    output_dir = (
        args.output_dir
        or paths.project_root / "slurm/generated/external_reference/transpolymer"
    ).resolve()
    jobs = write_matrix(
        paths,
        output_dir,
        baseline_config=args.baseline_config.resolve(),
    )
    print(f"transpolymer_external_reference={len(jobs)}; no jobs were submitted")


if __name__ == "__main__":
    main()
