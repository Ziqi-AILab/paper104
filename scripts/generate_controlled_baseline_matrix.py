#!/usr/bin/env python3
"""Generate the complete same-split controlled-baseline execution matrix.

Classical models are local CPU commands and never receive Slurm scripts.
Chemprop D-MPNN runs are one-A800 scripts.  This generator writes artifacts
only; it does not submit or execute any experiment.
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
    / "configs/downstream/controlled_baselines.yaml"
)
DETAILED_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
TWO_POINT_FRACTIONS = (0.10, 1.00)
SUBSAMPLES = (0, 1, 2, 3, 4)
TRANSPOLYMER_TASKS = ("EEA", "EGB", "EGC", "EI", "EPS", "NC", "OPV", "PE-II", "XC")
OPENPOLY_TASKS = (
    "OpenPoly:Atomization_Energy_eV",
    "OpenPoly:Bandgap_Bulk_eV",
    "OpenPoly:Bandgap_Chain_eV",
    "OpenPoly:Bandgap_eV",
    "OpenPoly:CH4_Permeability_Barrer",
    "OpenPoly:CO2_Permeability_Barrer",
    "OpenPoly:Compressive_Strength_MPa",
    "OpenPoly:Crystallization_Temperature_K",
    "OpenPoly:Crystallization_Tendency_percentage",
    "OpenPoly:Dielectric_Constant_Electronic",
    "OpenPoly:Dielectric_Constant_Ionic",
    "OpenPoly:Dielectric_Constant_Total",
    "OpenPoly:Electron_Affinity_eV",
    "OpenPoly:Elongation_at_Break_percentage",
    "OpenPoly:Flexural_Strength_MPa",
    "OpenPoly:H2_Permeability_Barrer",
    "OpenPoly:He_Permeability_Barrer",
    "OpenPoly:Ionization_Energy_eV",
    "OpenPoly:Lower_Critical_Solution_Temperature_K",
    "OpenPoly:N2_Permeability_Barrer",
    "OpenPoly:O2_Permeability_Barrer",
    "OpenPoly:Refractive_Index",
    "OpenPoly:Td_K",
    "OpenPoly:Tensile_Strength_MPa",
    "OpenPoly:Tg_K",
    "OpenPoly:Thermal_Conductivity_W_per_mK",
    "OpenPoly:Tm_K",
    "OpenPoly:Water_Contact_Angle",
    "OpenPoly:Water_Uptake_percentage",
    "OpenPoly:Youngs_Modulus_MPa",
)
OPENPOLY_HIGHLIGHTED = frozenset(
    {
        "OpenPoly:Bandgap_Chain_eV",
        "OpenPoly:CO2_Permeability_Barrer",
        "OpenPoly:Crystallization_Tendency_percentage",
        "OpenPoly:Refractive_Index",
        "OpenPoly:Tensile_Strength_MPa",
        "OpenPoly:Tg_K",
    }
)


@dataclass(frozen=True)
class BaselineSpec:
    dataset: Path
    split_manifest: Path
    task: str
    protocol: str
    protocol_role: str
    input_view: str
    fold: int
    label_column: str
    text_column: str
    task_type: str
    label_fraction: float
    subsample_id: int


@dataclass(frozen=True)
class ClassicalJob:
    row_index: int
    job_id: str
    method: str
    seed: int
    reporting_table: str
    task: str
    protocol: str
    protocol_role: str
    input_view: str
    fold: int
    label_fraction: float
    subsample_id: int
    dataset: str
    split_manifest: str
    label_column: str
    text_column: str
    task_type: str
    feature_cache: str
    output: str
    run_record: str
    command: str
    resource: str = "CPU"
    execution_path: str = "[CPU → local, no sbatch]"


@dataclass(frozen=True)
class ChempropJob:
    row_index: int
    job_id: str
    method: str
    seed: int
    reporting_table: str
    task: str
    protocol: str
    protocol_role: str
    input_view: str
    fold: int
    label_fraction: float
    subsample_id: int
    dataset: str
    split_manifest: str
    label_column: str
    text_column: str
    task_type: str
    epochs: int
    batch_size: int
    num_workers: int
    work_dir: str
    checkpoint_dir: str
    checkpoint_retention: str
    output: str
    run_record: str
    script: str
    command: str
    resource: str = "GPU-A800"
    execution_path: str = "[GPU-A800 → sbatch]"


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def _folds(path: Path) -> tuple[int, ...]:
    columns = set(pq.read_schema(path).names)
    if "evaluation_fold" in columns:
        column = "evaluation_fold"
    elif "fold" in columns:
        column = "fold"
    else:
        return (0,)
    values = pq.read_table(path, columns=[column]).column(0).to_pylist()
    return tuple(sorted({int(value) for value in values}))


def _subsamples(fraction: float) -> tuple[int, ...]:
    return (0,) if fraction >= 1.0 else SUBSAMPLES


def _expand_spec(
    *,
    dataset: Path,
    split_manifest: Path,
    task: str,
    protocol: str,
    protocol_role: str,
    input_view: str,
    label_column: str,
    text_column: str,
    fractions: tuple[float, ...],
    task_type: str = "regression",
) -> list[BaselineSpec]:
    return [
        BaselineSpec(
            dataset=dataset,
            split_manifest=split_manifest,
            task=task,
            protocol=protocol,
            protocol_role=protocol_role,
            input_view=input_view,
            fold=fold,
            label_column=label_column,
            text_column=text_column,
            task_type=task_type,
            label_fraction=fraction,
            subsample_id=subsample,
        )
        for fold in _folds(split_manifest)
        for fraction in fractions
        for subsample in _subsamples(fraction)
    ]


def build_specs(paths: ProjectPaths) -> list[BaselineSpec]:
    project = paths.project_root
    standardized = paths.shared_data_root / "standardized/v1"
    specs: list[BaselineSpec] = []

    aldeghi = standardized / "aldeghi_vipea/vipea.parquet"
    for task, label_column in (("EA", "ea_ev"), ("IP", "ip_ev")):
        for protocol, manifest_name in (
            ("pair_id", "aldeghi_pair_id.parquet"),
            ("family_ood", "aldeghi_family_ood.parquet"),
            ("butina_cluster", "aldeghi_butina_cluster.parquet"),
            ("random", "aldeghi_random.parquet"),
        ):
            specs.extend(
                _expand_spec(
                    dataset=aldeghi,
                    split_manifest=project / "data/splits" / manifest_name,
                    task=task,
                    protocol=protocol,
                    protocol_role="primary_polymer_evaluation",
                    input_view="structure_only",
                    label_column=label_column,
                    text_column="canonical_text",
                    fractions=DETAILED_FRACTIONS,
                )
            )

    for task in TRANSPOLYMER_TASKS:
        task_slug = task.casefold().replace("-", "_")
        specs.extend(
            _expand_spec(
                dataset=standardized / f"transpolymer/{task_slug}.parquet",
                split_manifest=project / f"data/splits/transpolymer_{task_slug}.parquet",
                task=task,
                protocol="group_ood",
                protocol_role="primary_polymer_evaluation",
                input_view="structure_only",
                label_column="label",
                text_column="structure_only_text",
                fractions=DETAILED_FRACTIONS if task in {"EGC", "XC"} else TWO_POINT_FRACTIONS,
            )
        )

    pe_i_dataset = project / "data/derived/transpolymer_pe_i_official.parquet"
    pe_i_manifest = project / "data/splits/transpolymer_pe_i_official_group_disjoint.parquet"
    for input_view, protocol_role, text_column in (
        ("structure_only", "primary_controlled_comparison", "structure_only_text"),
        (
            "structure_plus_metadata",
            "secondary_additional_information_reference",
            "structure_with_metadata_text",
        ),
    ):
        specs.extend(
            _expand_spec(
                dataset=pe_i_dataset,
                split_manifest=pe_i_manifest,
                task="PE-I",
                protocol="official_group_disjoint",
                protocol_role=protocol_role,
                input_view=input_view,
                label_column="label",
                text_column=text_column,
                fractions=TWO_POINT_FRACTIONS,
            )
        )

    if paths.formal_openpoly:
        openpoly = standardized / "openpoly/openpoly_live_2026_08_08.parquet"
        for task in OPENPOLY_TASKS:
            specs.extend(
                _expand_spec(
                    dataset=openpoly,
                    split_manifest=project / "data/splits/openpoly.parquet",
                    task=task,
                    protocol="external_group_split",
                    protocol_role="external_applicability",
                    input_view="structure_only",
                    label_column="label",
                    text_column="canonical_text",
                    fractions=(
                        TWO_POINT_FRACTIONS
                        if task in OPENPOLY_HIGHLIGHTED
                        else (1.0,)
                    ),
                )
            )

    for task in ("esol", "freesolv", "lipophilicity"):
        specs.extend(
            _expand_spec(
                dataset=standardized / f"molecular_retention/{task}.parquet",
                split_manifest=project / f"data/splits/{task}.parquet",
                task=task.upper(),
                protocol="molecular_scaffold",
                protocol_role="molecular_retention",
                input_view="structure_only",
                label_column="label",
                text_column="canonical_text",
                fractions=(1.0,),
            )
        )
    return specs


def _fraction_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _unit_parts(spec: BaselineSpec) -> tuple[str, ...]:
    return (
        _slug(spec.task),
        _slug(spec.protocol),
        _slug(spec.input_view),
        f"fold-{spec.fold}",
        f"fraction-{_fraction_tag(spec.label_fraction)}",
        f"subsample-{spec.subsample_id}",
    )


def _arguments(
    spec: BaselineSpec,
    *,
    method: str,
    seed: int,
    output: Path,
    run_record: Path,
    config_path: Path,
    work_dir: Path,
    checkpoint_dir: Path | None = None,
    checkpoint_retention: str = "trainable",
    epochs: int = 30,
    batch_size: int = 50,
    num_workers: int = 8,
    feature_cache: Path | None = None,
) -> list[str]:
    arguments = [
        "-m",
        "polyfid.cli",
        "downstream",
        "baseline",
        "--method",
        method,
        "--dataset",
        str(spec.dataset),
        "--split-manifest",
        str(spec.split_manifest),
        "--task",
        spec.task,
        "--protocol",
        spec.protocol,
        "--fold",
        str(spec.fold),
        "--label-column",
        spec.label_column,
        "--text-column",
        spec.text_column,
        "--task-type",
        spec.task_type,
        "--seeds",
        str(seed),
        "--label-fraction",
        f"{spec.label_fraction:g}",
        "--subsample-id",
        str(spec.subsample_id),
        "--input-view",
        spec.input_view,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--work-dir",
        str(work_dir),
        "--config-path",
        str(config_path),
        "--run-record",
        str(run_record),
        "--output",
        str(output),
    ]
    if checkpoint_dir is not None:
        arguments.extend(("--checkpoint-dir", str(checkpoint_dir)))
    arguments.extend(("--checkpoint-retention", checkpoint_retention))
    if feature_cache is not None:
        arguments.extend(("--feature-cache", str(feature_cache)))
    return arguments


def _load_config(path: Path) -> dict[str, object]:
    return dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def build_classical_jobs(
    paths: ProjectPaths,
    *,
    config_path: Path = DEFAULT_BASELINE_CONFIG,
) -> list[ClassicalJob]:
    config = _load_config(config_path)
    methods = dict(dict(config["classical"])["methods"])
    jobs: list[ClassicalJob] = []
    python = paths.project_root / ".conda/env/bin/python"
    for spec in build_specs(paths):
        parts = _unit_parts(spec)
        for method, method_config in methods.items():
            for seed in tuple(int(value) for value in dict(method_config)["seeds"]):
                job_id = "_".join(("classical", _slug(method), *parts, f"seed-{seed}"))
                output_root = (
                    paths.project_root
                    / "results/predictions/controlled_baselines"
                    / _slug(method)
                    / Path(*parts)
                    / f"seed-{seed}"
                )
                run_root = (
                    paths.project_root
                    / "runs/records/controlled_baselines"
                    / _slug(method)
                    / Path(*parts)
                    / f"seed-{seed}"
                )
                output = output_root / "predictions.parquet"
                run_record = run_root / "run.json"
                feature_family = "rdkit_2d" if method.startswith("rdkit") else "ecfp"
                feature_cache = (
                    paths.project_root
                    / "data/derived/controlled_baseline_features"
                    / feature_family
                    / f"{_slug(spec.task)}.npz"
                )
                arguments = _arguments(
                    spec,
                    method=method,
                    seed=seed,
                    output=output,
                    run_record=run_record,
                    config_path=config_path,
                    work_dir=paths.project_root / "runs/baselines/classical" / Path(*parts),
                    num_workers=8,
                    feature_cache=feature_cache,
                )
                jobs.append(
                    ClassicalJob(
                        row_index=len(jobs),
                        job_id=job_id,
                        method=method,
                        seed=seed,
                        reporting_table=str(config["reporting_table"]),
                        task=spec.task,
                        protocol=spec.protocol,
                        protocol_role=spec.protocol_role,
                        input_view=spec.input_view,
                        fold=spec.fold,
                        label_fraction=spec.label_fraction,
                        subsample_id=spec.subsample_id,
                        dataset=str(spec.dataset),
                        split_manifest=str(spec.split_manifest),
                        label_column=spec.label_column,
                        text_column=spec.text_column,
                        task_type=spec.task_type,
                        feature_cache=str(feature_cache),
                        output=str(output),
                        run_record=str(run_record),
                        command=shlex.join((str(python), *arguments)),
                    )
                )
    return jobs


def build_chemprop_jobs(
    paths: ProjectPaths,
    output_dir: Path,
    *,
    config_path: Path = DEFAULT_BASELINE_CONFIG,
) -> list[ChempropJob]:
    config = _load_config(config_path)
    chemprop = dict(config["chemprop"])
    jobs: list[ChempropJob] = []
    for spec in build_specs(paths):
        parts = _unit_parts(spec)
        for seed in tuple(int(value) for value in chemprop["seeds"]):
            job_id = "_".join(("chemprop", *parts, f"seed-{seed}"))
            output_root = (
                paths.project_root
                / "results/predictions/controlled_baselines/chemprop_dpnn"
                / Path(*parts)
                / f"seed-{seed}"
            )
            run_root = (
                paths.project_root
                / "runs/baselines/chemprop"
                / Path(*parts)
            )
            checkpoint_root = (
                paths.checkpoint_root
                / "downstream/controlled_baselines/chemprop_dpnn"
                / Path(*parts)
            )
            run_record = (
                paths.project_root
                / "runs/records/controlled_baselines/chemprop_dpnn"
                / Path(*parts)
                / f"seed-{seed}/run.json"
            )
            output = output_root / "predictions.parquet"
            script = output_dir / f"{job_id}.sbatch"
            arguments = _arguments(
                spec,
                method=str(chemprop["method"]),
                seed=seed,
                output=output,
                run_record=run_record,
                config_path=config_path,
                work_dir=run_root,
                checkpoint_dir=checkpoint_root,
                checkpoint_retention="metadata",
                epochs=int(chemprop["epochs"]),
                batch_size=int(chemprop["batch_size"]),
                num_workers=int(chemprop["num_workers"]),
            )
            jobs.append(
                ChempropJob(
                    row_index=len(jobs),
                    job_id=job_id,
                    method="chemprop_dpnn",
                    seed=seed,
                    reporting_table=str(config["reporting_table"]),
                    task=spec.task,
                    protocol=spec.protocol,
                    protocol_role=spec.protocol_role,
                    input_view=spec.input_view,
                    fold=spec.fold,
                    label_fraction=spec.label_fraction,
                    subsample_id=spec.subsample_id,
                    dataset=str(spec.dataset),
                    split_manifest=str(spec.split_manifest),
                    label_column=spec.label_column,
                    text_column=spec.text_column,
                    task_type=spec.task_type,
                    epochs=int(chemprop["epochs"]),
                    batch_size=int(chemprop["batch_size"]),
                    num_workers=int(chemprop["num_workers"]),
                    work_dir=str(run_root),
                    checkpoint_dir=str(checkpoint_root),
                    checkpoint_retention="metadata",
                    output=str(output),
                    run_record=str(run_record),
                    script=str(script),
                    command='"${PYTHON_BIN}" ' + shlex.join(arguments),
                )
            )
    return jobs


def _render_chemprop_script(job: ChempropJob) -> str:
    job_name = (
        f"p104-cp-{_slug(job.task)}-{_slug(job.protocol)}-"
        f"{_slug(job.input_view)}-f{job.fold}-p{_fraction_tag(job.label_fraction)}-"
        f"u{job.subsample_id}-s{job.seed}"
    )[:120]
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:a800:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/controlled_chemprop/%x-%j.out
#SBATCH --error=logs/slurm/controlled_chemprop/%x-%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-.}}"
PYTHON_BIN="${{PROJECT_DIR}}/.conda/env/bin/python"
cd "${{PROJECT_DIR}}"
mkdir -p "${{PROJECT_DIR}}/logs/slurm/controlled_chemprop"

{job.command}
'''


def _write_tsv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(
    paths: ProjectPaths,
    *,
    cpu_output_dir: Path,
    chemprop_output_dir: Path,
    config_path: Path = DEFAULT_BASELINE_CONFIG,
) -> tuple[list[ClassicalJob], list[ChempropJob]]:
    cpu_output_dir.mkdir(parents=True, exist_ok=True)
    chemprop_output_dir.mkdir(parents=True, exist_ok=True)
    classical_jobs = build_classical_jobs(paths, config_path=config_path)
    chemprop_jobs = build_chemprop_jobs(
        paths, chemprop_output_dir, config_path=config_path
    )

    _write_tsv(
        [asdict(job) for job in classical_jobs],
        cpu_output_dir / "classical_baseline_manifest.tsv",
    )
    (cpu_output_dir / "classical_baseline_command_list.txt").write_text(
        "\n".join(job.command for job in classical_jobs) + "\n", encoding="utf-8"
    )

    for job in chemprop_jobs:
        script = Path(job.script)
        script.write_text(_render_chemprop_script(job), encoding="utf-8")
        script.chmod(0o755)
    _write_tsv(
        [asdict(job) for job in chemprop_jobs],
        chemprop_output_dir / "chemprop_dpnn_manifest.tsv",
    )
    (chemprop_output_dir / "chemprop_dpnn_job_list.txt").write_text(
        "\n".join(job.script for job in chemprop_jobs) + "\n", encoding="utf-8"
    )
    return classical_jobs, chemprop_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--cpu-output-dir", type=Path)
    parser.add_argument("--chemprop-output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths_config)
    cpu_output = (
        args.cpu_output_dir
        or paths.project_root / "runs/manifests/controlled_baselines"
    ).resolve()
    chemprop_output = (
        args.chemprop_output_dir
        or paths.project_root / "slurm/generated/controlled_baselines/chemprop"
    ).resolve()
    classical_jobs, chemprop_jobs = write_matrix(
        paths,
        cpu_output_dir=cpu_output,
        chemprop_output_dir=chemprop_output,
        config_path=args.baseline_config.resolve(),
    )
    print(
        f"classical_local={len(classical_jobs)}; chemprop_a800={len(chemprop_jobs)}; "
        "no jobs were executed or submitted"
    )


if __name__ == "__main__":
    main()
