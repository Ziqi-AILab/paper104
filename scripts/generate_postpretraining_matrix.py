#!/usr/bin/env python3
"""Generate the complete Paper104 post-pretraining execution matrices.

The generator writes plans and executable scripts but never submits a Slurm
job.  Encoder forwards and supervised adaptation are one-A800 jobs.  Frozen
probes consume cached arrays locally on CPU and therefore receive a plain TSV
and command list, not sbatch files.
"""

from __future__ import annotations

import argparse
import csv
import shlex
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from polyfid.paths import DEFAULT_PATH_CONFIG, ProjectPaths, load_paths


FORMAL_ARMS = ("B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs")
FORMAL_SEEDS = (42, 123, 777, 2025, 2026)
MECHANISM_CONTROL_SEEDS = (42, 123, 777)
MECHANISM_CONTROL_STAGES = (
    ("matched_random_masking", "fidelity_boundary_mlm"),
    ("permuted_boundary_masking", "fidelity_boundary_mlm"),
    ("equivalence_only", "fidelity_relations"),
    ("topology_negative_only", "fidelity_relations"),
)
HEAD_SEEDS = (11, 22, 33)
SUBSAMPLES = (0, 1, 2, 3, 4)
DETAILED_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
TWO_POINT_FRACTIONS = (0.10, 1.00)

TRANSPOLYMER_TASKS = (
    "EEA",
    "EGB",
    "EGC",
    "EI",
    "EPS",
    "NC",
    "OPV",
    "PE-II",
    "XC",
)
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
class Representation:
    arm: str
    representation_seed: int
    checkpoint: Path
    base_checkpoint: Path
    replicate_role: str

    @property
    def label(self) -> str:
        if self.replicate_role == "fixed_reference":
            return "C_fixed"
        return f"{self.arm}_seed-{self.representation_seed}"


@dataclass(frozen=True)
class EvaluationJob:
    job_id: str
    stage: str
    arm: str
    representation_seed: int
    replicate_role: str
    checkpoint: str
    base_checkpoint: str
    input_path: str
    output_path: str
    script: str
    command: str
    resource: str = "GPU-A800"
    execution_path: str = "[GPU-A800 → sbatch]"


@dataclass(frozen=True)
class ProbeJob:
    row_index: int
    probe_id: str
    arm: str
    representation_seed: int
    replicate_role: str
    dataset: str
    split_manifest: str
    embeddings: str
    task: str
    protocol: str
    protocol_role: str
    input_view: str
    fold: int
    label_column: str
    task_type: str
    label_fraction: float
    subsample_id: int
    head_seeds: str
    learning_rate_recipe: str
    learning_rate: str
    output: str
    metrics_output: str
    probe_state_dir: str
    resource: str = "CPU"
    execution_path: str = "[CPU → local, no sbatch]"


@dataclass(frozen=True)
class AdaptationJob:
    job_id: str
    method: str
    arm: str
    representation_seed: int
    adaptation_seed: int
    checkpoint: str
    base_checkpoint: str
    dataset: str
    split_manifest: str
    task: str
    protocol: str
    fold: int
    label_column: str
    text_column: str
    task_type: str
    learning_rate: float
    checkpoint_retention: str
    output: str
    model_output: str
    run_record: str
    script: str
    command: str
    resource: str = "GPU-A800"
    execution_path: str = "[GPU-A800 → sbatch]"


@dataclass(frozen=True)
class TokenEfficiencyJob:
    job_id: str
    arm: str
    representation_seed: int
    checkpoint_fraction: float
    checkpoint_percent: int
    checkpoint: str
    base_checkpoint: str
    dataset: str
    split: str
    mask_seed: int
    output: str
    script: str
    command: str
    resource: str = "GPU-A800"
    execution_path: str = "[GPU-A800 → sbatch]"


@dataclass(frozen=True)
class ProbeSpec:
    dataset: Path
    split_manifest: Path
    task: str
    protocol: str
    protocol_role: str
    input_view: str
    folds: tuple[int, ...]
    label_column: str
    fractions: tuple[float, ...]
    embeddings_kind: str = "structure_only"
    task_type: str = "regression"


def build_representations(paths: ProjectPaths) -> list[Representation]:
    base = paths.shared_model_root / "chemberta_100m_mlm"
    representations = [
        Representation(
            arm="C",
            representation_seed=0,
            checkpoint=base,
            base_checkpoint=base,
            replicate_role="fixed_reference",
        )
    ]
    representations.extend(
        Representation(
            arm=arm,
            representation_seed=seed,
            checkpoint=(
                paths.checkpoint_root
                / "pretraining"
                / arm
                / f"seed-{seed}"
                / "token-100.pt"
            ),
            base_checkpoint=base,
            replicate_role="independent_pretraining",
        )
        for arm in FORMAL_ARMS
        for seed in FORMAL_SEEDS
    )
    return representations


def load_selected_warm_learning_rate(path: str | Path) -> float:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    recipe = values.get("selected_recipe", values)
    learning_rate = float(recipe["warm_learning_rate"])
    if learning_rate not in (1e-5, 3e-5, 1e-4):
        raise ValueError("selected warm learning rate is outside the declared candidates")
    return learning_rate


def build_property_development_representations(
    paths: ProjectPaths,
    *,
    warm_learning_rate: float,
) -> list[Representation]:
    base = paths.shared_model_root / "chemberta_100m_mlm"
    return [
        Representation(
            arm="A0",
            representation_seed=seed,
            checkpoint=(
                paths.checkpoint_root
                / "dev"
                / (
                    f"pilot_A0_seed-{seed}_lr-{warm_learning_rate:g}"
                    "_b-0.65_i-0.25"
                )
                / "final_model"
            ),
            base_checkpoint=base,
            replicate_role="development_pretraining",
        )
        for seed in (13, 37)
    ]


def _python_command(arguments: Sequence[str]) -> str:
    return '"${PYTHON_BIN}" ' + shlex.join(list(arguments))


def _script_name(prefix: str, representation: Representation) -> str:
    return f"{prefix}_{representation.label}.sbatch"


def _evaluation_commands(
    paths: ProjectPaths,
    representation: Representation,
) -> list[tuple[str, Path, Path, list[str], str]]:
    project = paths.project_root
    base_args = [
        "--checkpoint",
        str(representation.checkpoint),
        "--base-checkpoint",
        str(representation.base_checkpoint),
        "--method",
        representation.arm,
    ]
    structure_input = (
        project / "data/derived/evaluation_inputs/downstream_structure_only.parquet"
    )
    metadata_input = (
        project / "data/derived/evaluation_inputs/pe_i_structure_plus_metadata.parquet"
    )
    fidelity_input = (
        project / "data/derived/evaluation_inputs/fidelity_test_records.parquet"
    )
    relation_root = project / "data/relations/test"
    run_root = project / "runs"
    metric_root = project / "results/metrics"
    label = representation.label
    seed_args = ["--seed", str(representation.representation_seed)]

    return [
        (
            "embedding_structure_only",
            structure_input,
            run_root / f"embeddings/formal/{label}/structure_only.npz",
            [
                "scripts/cache_embeddings.py",
                "--dataset",
                str(structure_input),
                *base_args,
                *seed_args,
                "--text-column",
                "canonical_text",
                "--input-view",
                "valid",
                "--batch-size",
                "128",
                "--device",
                "cuda",
                "--output",
                str(run_root / f"embeddings/formal/{label}/structure_only.npz"),
            ],
            "2-00:00:00",
        ),
        (
            "embedding_pe_i_metadata",
            metadata_input,
            run_root / f"embeddings/formal/{label}/pe_i_structure_plus_metadata.npz",
            [
                "scripts/cache_embeddings.py",
                "--dataset",
                str(metadata_input),
                *base_args,
                *seed_args,
                "--text-column",
                "canonical_text",
                "--input-view",
                "structure_plus_metadata_secondary",
                "--batch-size",
                "128",
                "--device",
                "cuda",
                "--output",
                str(
                    run_root
                    / f"embeddings/formal/{label}/pe_i_structure_plus_metadata.npz"
                ),
            ],
            "1-00:00:00",
        ),
        (
            "fidelity_relations",
            relation_root / "pi1m_triplets.parquet",
            metric_root / f"fidelity/relations/{label}",
            [
                "scripts/evaluate_fidelity.py",
                "--relations",
                str(relation_root / "pi1m_triplets.parquet"),
                "--decoy-bank",
                str(relation_root / "retrieval_decoy_bank.parquet"),
                "--supplemental-controls",
                str(relation_root / "supplemental_topology_controls.parquet"),
                *base_args,
                *seed_args,
                "--batch-size",
                "128",
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--output",
                str(metric_root / f"fidelity/relations/{label}"),
            ],
            "2-00:00:00",
        ),
        (
            "fidelity_invariance",
            fidelity_input,
            metric_root / f"fidelity/invariance/{label}",
            [
                "scripts/evaluate_invariance.py",
                "--dataset",
                str(fidelity_input),
                *base_args,
                "--reference-checkpoint",
                str(representation.base_checkpoint),
                *seed_args,
                "--text-column",
                "canonical_psmiles",
                "--sample-id-column",
                "sample_id",
                "--layers",
                "0,3,6,9,12",
                "--randomized-views",
                "10",
                "--periodic-views",
                "10",
                "--atom-permutations",
                "10",
                "--batch-size",
                "128",
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--output",
                str(metric_root / f"fidelity/invariance/{label}"),
            ],
            "2-00:00:00",
        ),
        (
            "fidelity_boundary_mlm",
            fidelity_input,
            metric_root / f"fidelity/boundary_mlm/{label}",
            [
                "scripts/evaluate_boundary_mlm.py",
                "--dataset",
                str(fidelity_input),
                *base_args,
                *seed_args,
                "--scope",
                "repeat",
                "--text-column",
                "canonical_psmiles",
                "--sample-id-column",
                "sample_id",
                "--hidden-layers",
                "0,3,6,9,12",
                "--batch-size",
                "128",
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--output",
                str(metric_root / f"fidelity/boundary_mlm/{label}"),
            ],
            "2-00:00:00",
        ),
        (
            "retention_zinc50k_mlm",
            (
                paths.shared_data_root
                / "standardized/v1/zinc20/zinc20_1m.parquet"
            ),
            metric_root / f"retention/zinc50k_mlm/{label}",
            [
                "scripts/evaluate_zinc_retention.py",
                "--dataset",
                str(
                    paths.shared_data_root
                    / "standardized/v1/zinc20/zinc20_1m.parquet"
                ),
                *base_args,
                "--representation-seed",
                str(representation.representation_seed),
                "--split",
                "retention",
                "--mask-seed",
                "10407",
                "--mask-probability",
                "0.15",
                "--batch-size",
                "128",
                "--max-length",
                "256",
                "--device",
                "cuda",
                "--output",
                str(metric_root / f"retention/zinc50k_mlm/{label}"),
            ],
            "1-00:00:00",
        ),
    ]


def _render_a800_script(*, job_name: str, wall_time: str, command: str) -> str:
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpua800
#SBATCH --gres=gpu:a800:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time={wall_time}
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-.}}"
PYTHON_BIN="${{PROJECT_DIR}}/.conda/env/bin/python"
cd "${{PROJECT_DIR}}"
mkdir -p "${{PROJECT_DIR}}/logs/slurm"

{command}
'''


def build_evaluation_jobs(
    paths: ProjectPaths,
    output_dir: Path,
) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    for representation in build_representations(paths):
        for stage, input_path, output_path, arguments, wall_time in _evaluation_commands(
            paths, representation
        ):
            job_id = f"eval_{stage}_{representation.label}"
            script_path = output_dir / _script_name(job_id, representation)
            # job_id already includes the representation; avoid duplicating it in the file name.
            script_path = output_dir / f"{job_id}.sbatch"
            command = _python_command(arguments)
            jobs.append(
                EvaluationJob(
                    job_id=job_id,
                    stage=stage,
                    arm=representation.arm,
                    representation_seed=representation.representation_seed,
                    replicate_role=representation.replicate_role,
                    checkpoint=str(representation.checkpoint),
                    base_checkpoint=str(representation.base_checkpoint),
                    input_path=str(input_path),
                    output_path=str(output_path),
                    script=str(script_path),
                    command=command,
                )
            )
            script_path.write_text(
                _render_a800_script(
                    job_name=f"p104-{stage[:12]}-{representation.label}".replace(
                        "_", "-"
                    ),
                    wall_time=wall_time,
                    command=command,
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)
    return jobs


def build_mechanism_control_evaluation_jobs(
    paths: ProjectPaths,
    output_dir: Path,
) -> list[EvaluationJob]:
    """Build the prespecified targeted evaluations for the mechanism controls."""

    base_checkpoint = paths.shared_model_root / "chemberta_100m_mlm"
    jobs: list[EvaluationJob] = []
    for arm, selected_stage in MECHANISM_CONTROL_STAGES:
        for seed in MECHANISM_CONTROL_SEEDS:
            representation = Representation(
                arm=arm,
                representation_seed=seed,
                checkpoint=(
                    paths.checkpoint_root
                    / "pretraining"
                    / "controls"
                    / arm
                    / f"seed-{seed}"
                    / "token-100.pt"
                ),
                base_checkpoint=base_checkpoint,
                replicate_role="three_seed_mechanism_control",
            )
            selected = [
                item
                for item in _evaluation_commands(paths, representation)
                if item[0] == selected_stage
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"expected one {selected_stage} command for {representation.label}"
                )
            stage, input_path, output_path, arguments, _ = selected[0]
            job_id = f"eval_control_{stage}_{representation.label}"
            script_path = output_dir / f"{job_id}.sbatch"
            command = _python_command(arguments)
            jobs.append(
                EvaluationJob(
                    job_id=job_id,
                    stage=stage,
                    arm=arm,
                    representation_seed=seed,
                    replicate_role=representation.replicate_role,
                    checkpoint=str(representation.checkpoint),
                    base_checkpoint=str(base_checkpoint),
                    input_path=str(input_path),
                    output_path=str(output_path),
                    script=str(script_path),
                    command=command,
                )
            )
            script_path.write_text(
                _render_a800_script(
                    job_name=f"p104-ctrl-{selected_stage[:12]}-{representation.label}".replace(
                        "_", "-"
                    ),
                    wall_time="00:15:00",
                    command=command,
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)
    return jobs


def _subsamples_for_fraction(fraction: float) -> tuple[int, ...]:
    # At 100% every repeat contains exactly the same groups, so one execution
    # is sufficient.  Fractions below 100% retain five fixed sampling repeats.
    return (0,) if fraction >= 1.0 else SUBSAMPLES


def _probe_specs(paths: ProjectPaths) -> list[ProbeSpec]:
    project = paths.project_root
    standardized = paths.shared_data_root / "standardized/v1"
    specs: list[ProbeSpec] = []
    aldeghi = standardized / "aldeghi_vipea/vipea.parquet"
    aldeghi_protocols = (
        ("pair_id", "aldeghi_pair_id.parquet", (0,)),
        ("family_ood", "aldeghi_family_ood.parquet", tuple(range(9))),
        ("butina_cluster", "aldeghi_butina_cluster.parquet", tuple(range(5))),
        ("random", "aldeghi_random.parquet", tuple(range(10))),
    )
    for task, label_column in (("EA", "ea_ev"), ("IP", "ip_ev")):
        for protocol, manifest, folds in aldeghi_protocols:
            specs.append(
                ProbeSpec(
                    dataset=aldeghi,
                    split_manifest=project / "data/splits" / manifest,
                    task=task,
                    protocol=protocol,
                    protocol_role="primary_polymer_evaluation",
                    input_view="structure_only",
                    folds=folds,
                    label_column=label_column,
                    fractions=DETAILED_FRACTIONS,
                )
            )

    for task in TRANSPOLYMER_TASKS:
        slug = task.casefold().replace("-", "_")
        fractions = DETAILED_FRACTIONS if task in {"EGC", "XC"} else TWO_POINT_FRACTIONS
        specs.append(
            ProbeSpec(
                dataset=standardized / f"transpolymer/{slug}.parquet",
                split_manifest=project / f"data/splits/transpolymer_{slug}.parquet",
                task=task,
                protocol="group_ood",
                protocol_role="primary_polymer_evaluation",
                input_view="structure_only",
                folds=tuple(range(5)),
                label_column="label",
                fractions=fractions,
            )
        )

    pe_i_dataset = project / "data/derived/transpolymer_pe_i_official.parquet"
    pe_i_manifest = (
        project / "data/splits/transpolymer_pe_i_official_group_disjoint.parquet"
    )
    specs.extend(
        (
            ProbeSpec(
                dataset=pe_i_dataset,
                split_manifest=pe_i_manifest,
                task="PE-I",
                protocol="official_group_disjoint",
                protocol_role="primary_controlled_comparison",
                input_view="structure_only",
                folds=(0,),
                label_column="label",
                fractions=TWO_POINT_FRACTIONS,
            ),
            ProbeSpec(
                dataset=pe_i_dataset,
                split_manifest=pe_i_manifest,
                task="PE-I",
                protocol="official_group_disjoint",
                protocol_role="secondary_additional_information_reference",
                input_view="structure_plus_metadata",
                folds=(0,),
                label_column="label",
                fractions=TWO_POINT_FRACTIONS,
                embeddings_kind="pe_i_structure_plus_metadata",
            ),
        )
    )

    if paths.formal_openpoly:
        openpoly = standardized / "openpoly/openpoly_live_2026_08_08.parquet"
        for task in OPENPOLY_TASKS:
            specs.append(
                ProbeSpec(
                    dataset=openpoly,
                    split_manifest=project / "data/splits/openpoly.parquet",
                    task=task,
                    protocol="external_group_split",
                    protocol_role="external_applicability",
                    input_view="structure_only",
                    folds=(0,),
                    label_column="label",
                    fractions=(
                        TWO_POINT_FRACTIONS
                        if task in OPENPOLY_HIGHLIGHTED
                        else (1.0,)
                    ),
                )
            )

    for task in ("esol", "freesolv", "lipophilicity"):
        specs.append(
            ProbeSpec(
                dataset=standardized / f"molecular_retention/{task}.parquet",
                split_manifest=project / f"data/splits/{task}.parquet",
                task=task.upper(),
                protocol="molecular_scaffold",
                protocol_role="molecular_retention",
                input_view="structure_only",
                folds=(0,),
                label_column="label",
                fractions=(1.0,),
            )
        )
    return specs


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def build_probe_jobs(
    paths: ProjectPaths,
    *,
    learning_rate_recipe: Path,
) -> list[ProbeJob]:
    jobs: list[ProbeJob] = []
    row_index = 0
    for representation in build_representations(paths):
        embedding_root = paths.project_root / f"runs/embeddings/formal/{representation.label}"
        for spec in _probe_specs(paths):
            embeddings = embedding_root / (
                "pe_i_structure_plus_metadata.npz"
                if spec.embeddings_kind == "pe_i_structure_plus_metadata"
                else "structure_only.npz"
            )
            for fold in spec.folds:
                for fraction in spec.fractions:
                    for subsample in _subsamples_for_fraction(fraction):
                        fraction_tag = f"{fraction:g}".replace(".", "p")
                        task_slug = _slug(spec.task)
                        probe_id = (
                            f"probe_{representation.label}_{task_slug}_{spec.protocol}_"
                            f"{spec.input_view}_fold-{fold}_frac-{fraction_tag}_sub-{subsample}"
                        )
                        output_root = (
                            paths.project_root
                            / "results/predictions/frozen_probe"
                            / representation.label
                            / task_slug
                            / spec.protocol
                            / spec.input_view
                            / f"fold-{fold}"
                            / f"fraction-{fraction_tag}"
                            / f"subsample-{subsample}"
                        )
                        needs_probe_state = (
                            fraction == 1.0
                            and subsample == 0
                            and spec.protocol_role != "molecular_retention"
                        )
                        jobs.append(
                            ProbeJob(
                                row_index=row_index,
                                probe_id=probe_id,
                                arm=representation.arm,
                                representation_seed=representation.representation_seed,
                                replicate_role=representation.replicate_role,
                                dataset=str(spec.dataset),
                                split_manifest=str(spec.split_manifest),
                                embeddings=str(embeddings),
                                task=spec.task,
                                protocol=spec.protocol,
                                protocol_role=spec.protocol_role,
                                input_view=spec.input_view,
                                fold=fold,
                                label_column=spec.label_column,
                                task_type=spec.task_type,
                                label_fraction=fraction,
                                subsample_id=subsample,
                                head_seeds=",".join(map(str, HEAD_SEEDS)),
                                learning_rate_recipe=str(learning_rate_recipe),
                                learning_rate="",
                                output=str(output_root / "predictions.parquet"),
                                metrics_output=str(
                                    output_root / "validation_metrics.parquet"
                                ),
                                probe_state_dir=(
                                    str(output_root / "probe_states")
                                    if needs_probe_state
                                    else ""
                                ),
                            )
                        )
                        row_index += 1
    return jobs


def build_property_development_embedding_jobs(
    paths: ProjectPaths,
    output_dir: Path,
    *,
    warm_learning_rate: float,
) -> list[EvaluationJob]:
    jobs: list[EvaluationJob] = []
    inputs = (
        (
            "structure_only",
            paths.project_root
            / "data/derived/evaluation_inputs/downstream_structure_only.parquet",
            "canonical_text",
        ),
        (
            "pe_i_structure_plus_metadata",
            paths.project_root
            / "data/derived/evaluation_inputs/pe_i_structure_plus_metadata.parquet",
            "canonical_text",
        ),
    )
    for representation in build_property_development_representations(
        paths, warm_learning_rate=warm_learning_rate
    ):
        for view, dataset, text_column in inputs:
            output = (
                paths.project_root
                / "runs/embeddings/property_head_development"
                / representation.label
                / f"{view}.npz"
            )
            job_id = f"property_lr_embedding_{representation.label}_{view}"
            script_path = output_dir / f"{job_id}.sbatch"
            command = _python_command(
                [
                    "scripts/cache_embeddings.py",
                    "--dataset",
                    str(dataset),
                    "--checkpoint",
                    str(representation.checkpoint),
                    "--base-checkpoint",
                    str(representation.base_checkpoint),
                    "--method",
                    "A0_property_lr_development",
                    "--seed",
                    str(representation.representation_seed),
                    "--text-column",
                    text_column,
                    "--input-view",
                    "valid" if view == "structure_only" else view,
                    "--batch-size",
                    "128",
                    "--device",
                    "cuda",
                    "--output",
                    str(output),
                ]
            )
            job = EvaluationJob(
                job_id=job_id,
                stage="property_head_lr_embedding",
                arm="A0",
                representation_seed=representation.representation_seed,
                replicate_role="development_pretraining",
                checkpoint=str(representation.checkpoint),
                base_checkpoint=str(representation.base_checkpoint),
                input_path=str(dataset),
                output_path=str(output),
                script=str(script_path),
                command=command,
            )
            jobs.append(job)
            script_path.write_text(
                _render_a800_script(
                    job_name=f"p104-property-lr-{representation.representation_seed}-{_slug(view)}"[:120],
                    wall_time="2-00:00:00",
                    command=command,
                ),
                encoding="utf-8",
            )
            script_path.chmod(0o755)
    return jobs


def build_property_head_development_jobs(
    paths: ProjectPaths,
    *,
    warm_learning_rate: float,
) -> list[ProbeJob]:
    jobs: list[ProbeJob] = []
    row_index = 0
    representations = build_property_development_representations(
        paths, warm_learning_rate=warm_learning_rate
    )
    full_data_specs = [replace(spec, fractions=(1.0,)) for spec in _probe_specs(paths)]
    for representation in representations:
        embedding_root = (
            paths.project_root
            / "runs/embeddings/property_head_development"
            / representation.label
        )
        for learning_rate in (1e-4, 3e-4, 1e-3):
            lr_tag = f"{learning_rate:g}".replace(".", "p")
            for spec in full_data_specs:
                embeddings = embedding_root / (
                    "pe_i_structure_plus_metadata.npz"
                    if spec.embeddings_kind == "pe_i_structure_plus_metadata"
                    else "structure_only.npz"
                )
                for fold in spec.folds:
                    task_slug = _slug(spec.task)
                    probe_id = (
                        f"property_lr_dev_{representation.label}_{task_slug}_"
                        f"{spec.protocol}_{spec.input_view}_fold-{fold}_lr-{lr_tag}"
                    )
                    output_root = (
                        paths.project_root
                        / "runs/property_head_development"
                        / representation.label
                        / f"lr-{lr_tag}"
                        / task_slug
                        / spec.protocol
                        / spec.input_view
                        / f"fold-{fold}"
                    )
                    jobs.append(
                        ProbeJob(
                            row_index=row_index,
                            probe_id=probe_id,
                            arm="A0",
                            representation_seed=representation.representation_seed,
                            replicate_role="development_pretraining",
                            dataset=str(spec.dataset),
                            split_manifest=str(spec.split_manifest),
                            embeddings=str(embeddings),
                            task=spec.task,
                            protocol=spec.protocol,
                            protocol_role=f"development_validation:{spec.protocol_role}",
                            input_view=spec.input_view,
                            fold=fold,
                            label_column=spec.label_column,
                            task_type=spec.task_type,
                            label_fraction=1.0,
                            subsample_id=0,
                            head_seeds=",".join(map(str, HEAD_SEEDS)),
                            learning_rate_recipe="",
                            learning_rate=f"{learning_rate:g}",
                            output=str(output_root / "predictions.parquet"),
                            metrics_output=str(output_root / "validation_metrics.parquet"),
                            probe_state_dir=str(output_root / "probe_states"),
                        )
                    )
                    row_index += 1
    return jobs


def _adaptation_specs(paths: ProjectPaths) -> list[ProbeSpec]:
    project = paths.project_root
    standardized = paths.shared_data_root / "standardized/v1"
    return [
        ProbeSpec(
            dataset=standardized / "aldeghi_vipea/vipea.parquet",
            split_manifest=project / "data/splits/aldeghi_family_ood.parquet",
            task=task,
            protocol="family_ood",
            protocol_role="primary_adaptation",
            input_view="structure_only",
            folds=tuple(range(9)),
            label_column=label,
            fractions=(1.0,),
        )
        for task, label in (("EA", "ea_ev"), ("IP", "ip_ev"))
    ] + [
        ProbeSpec(
            dataset=standardized / f"transpolymer/{task.casefold()}.parquet",
            split_manifest=project / f"data/splits/transpolymer_{task.casefold()}.parquet",
            task=task,
            protocol="group_ood",
            protocol_role="primary_adaptation",
            input_view="structure_only",
            folds=tuple(range(5)),
            label_column="label",
            fractions=(1.0,),
        )
        for task in ("EGC", "XC")
    ]


def _render_adaptation_command(job: AdaptationJob) -> str:
    return _python_command(
        [
            "scripts/adapt_downstream.py",
            "--method",
            job.method,
            "--arm",
            job.arm,
            "--checkpoint",
            job.checkpoint,
            "--base-checkpoint",
            job.base_checkpoint,
            "--representation-seed",
            str(job.representation_seed),
            "--adaptation-seed",
            str(job.adaptation_seed),
            "--dataset",
            job.dataset,
            "--split-manifest",
            job.split_manifest,
            "--task",
            job.task,
            "--protocol",
            job.protocol,
            "--fold",
            str(job.fold),
            "--label-column",
            job.label_column,
            "--text-column",
            job.text_column,
            "--task-type",
            job.task_type,
            "--learning-rate",
            f"{job.learning_rate:g}",
            "--batch-size",
            "32",
            "--max-length",
            "256",
            "--max-epochs",
            "100",
            "--patience",
            "15",
            "--device",
            "cuda",
            "--output",
            job.output,
            "--model-output",
            job.model_output,
            "--checkpoint-retention",
            job.checkpoint_retention,
            "--run-record",
            job.run_record,
        ]
    )


def build_adaptation_jobs(
    paths: ProjectPaths,
    output_dir: Path,
) -> list[AdaptationJob]:
    jobs: list[AdaptationJob] = []
    representations = [
        representation
        for representation in build_representations(paths)
        if representation.arm != "C"
    ]
    for method in ("lora", "full"):
        for representation in representations:
            if method == "full" and representation.arm not in {"A0", "A_eqs"}:
                continue
            for spec in _adaptation_specs(paths):
                if method == "full" and spec.task not in {"EA", "EGC"}:
                    continue
                for fold in spec.folds:
                    for adaptation_seed in HEAD_SEEDS:
                        task_slug = _slug(spec.task)
                        job_id = (
                            f"adapt_{method}_{representation.label}_{task_slug}_"
                            f"{spec.protocol}_fold-{fold}_seed-{adaptation_seed}"
                        )
                        output_root = (
                            paths.project_root
                            / "results/predictions/adaptation"
                            / method
                            / representation.label
                            / task_slug
                            / spec.protocol
                            / f"fold-{fold}"
                            / f"adaptation-seed-{adaptation_seed}"
                        )
                        checkpoint_retention = (
                            "trainable" if method == "lora" else "metadata"
                        )
                        artifact_root = (
                            paths.checkpoint_root / "downstream"
                            if checkpoint_retention == "trainable"
                            else paths.project_root / "runs/adaptation/model_metadata"
                        )
                        model_output = (
                            artifact_root
                            / method
                            / representation.label
                            / task_slug
                            / spec.protocol
                            / f"fold-{fold}"
                            / (
                                f"adaptation-seed-{adaptation_seed}.pt"
                                if checkpoint_retention == "trainable"
                                else f"adaptation-seed-{adaptation_seed}.json"
                            )
                        )
                        script_path = output_dir / f"{job_id}.sbatch"
                        provisional = AdaptationJob(
                            job_id=job_id,
                            method=method,
                            arm=representation.arm,
                            representation_seed=representation.representation_seed,
                            adaptation_seed=adaptation_seed,
                            checkpoint=str(representation.checkpoint),
                            base_checkpoint=str(representation.base_checkpoint),
                            dataset=str(spec.dataset),
                            split_manifest=str(spec.split_manifest),
                            task=spec.task,
                            protocol=spec.protocol,
                            fold=fold,
                            label_column=spec.label_column,
                            text_column=(
                                "canonical_text"
                                if spec.task in {"EA", "IP"}
                                else "structure_only_text"
                            ),
                            task_type=spec.task_type,
                            learning_rate=1e-4,
                            checkpoint_retention=checkpoint_retention,
                            output=str(output_root / "predictions.parquet"),
                            model_output=str(model_output),
                            run_record=str(
                                paths.project_root
                                / "runs/adaptation"
                                / method
                                / representation.label
                                / task_slug
                                / spec.protocol
                                / f"fold-{fold}"
                                / f"adaptation-seed-{adaptation_seed}.json"
                            ),
                            script=str(script_path),
                            command="",
                        )
                        job = AdaptationJob(
                            **{
                                **asdict(provisional),
                                "command": _render_adaptation_command(provisional),
                            }
                        )
                        jobs.append(job)
                        script_path.write_text(
                            _render_a800_script(
                                job_name=f"p104-{job_id}".replace("_", "-")[:120],
                                wall_time=(
                                    "3-00:00:00" if method == "full" else "2-00:00:00"
                                ),
                                command=job.command,
                            ),
                            encoding="utf-8",
                        )
                        script_path.chmod(0o755)
    return jobs


def _write_tsv(rows: Iterable[object], path: Path) -> int:
    materialized = [asdict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        raise ValueError("cannot write an empty execution matrix")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(materialized[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def write_evaluation_matrix(paths: ProjectPaths, output_dir: Path) -> list[EvaluationJob]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_evaluation_jobs(paths, output_dir)
    _write_tsv(jobs, output_dir / "evaluation_manifest.tsv")
    (output_dir / "evaluation_job_list.txt").write_text(
        "\n".join(job.script for job in jobs) + "\n", encoding="utf-8"
    )
    return jobs


def write_mechanism_control_evaluation_matrix(
    paths: ProjectPaths,
    output_dir: Path,
) -> list[EvaluationJob]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_mechanism_control_evaluation_jobs(paths, output_dir)
    _write_tsv(jobs, output_dir / "mechanism_control_evaluation_manifest.tsv")
    (output_dir / "mechanism_control_evaluation_job_list.txt").write_text(
        "\n".join(job.script for job in jobs) + "\n", encoding="utf-8"
    )
    return jobs


def build_token_efficiency_jobs(
    paths: ProjectPaths,
    output_dir: Path,
) -> list[TokenEfficiencyJob]:
    """Build matched-token A0/B0 validation-MLM checkpoints for H1."""

    base_checkpoint = paths.shared_model_root / "chemberta_100m_mlm"
    dataset = (
        paths.project_root
        / "data/derived/pi1m_periodic_identity.parquet"
    )
    jobs: list[TokenEfficiencyJob] = []
    for arm in ("A0", "B0"):
        for seed in FORMAL_SEEDS:
            checkpoint_root = (
                paths.checkpoint_root / "pretraining" / arm / f"seed-{seed}"
            )
            for percent in (1, 5, 10, 25, 50, 100):
                checkpoint = checkpoint_root / f"token-{percent:03d}.pt"
                output = (
                    paths.project_root
                    / "results/metrics/token_efficiency"
                    / arm
                    / f"seed-{seed}"
                    / f"fraction-{percent:03d}"
                )
                job_id = f"token_efficiency_{arm}_seed-{seed}_fraction-{percent:03d}"
                script_path = output_dir / f"{job_id}.sbatch"
                command = _python_command(
                    [
                        "scripts/evaluate_uniform_mlm.py",
                        "--dataset",
                        str(dataset),
                        "--checkpoint",
                        str(checkpoint),
                        "--base-checkpoint",
                        str(base_checkpoint),
                        "--method",
                        arm,
                        "--representation-seed",
                        str(seed),
                        "--checkpoint-fraction",
                        str(percent / 100.0),
                        "--text-column",
                        "canonical_psmiles",
                        "--sample-id-column",
                        "sample_id",
                        "--split",
                        "validation",
                        "--split-column",
                        "split",
                        "--mask-seed",
                        "10407",
                        "--mask-probability",
                        "0.15",
                        "--batch-size",
                        "128",
                        "--max-length",
                        "256",
                        "--device",
                        "cuda",
                        "--output",
                        str(output),
                    ]
                )
                job = TokenEfficiencyJob(
                    job_id=job_id,
                    arm=arm,
                    representation_seed=seed,
                    checkpoint_fraction=percent / 100.0,
                    checkpoint_percent=percent,
                    checkpoint=str(checkpoint),
                    base_checkpoint=str(base_checkpoint),
                    dataset=str(dataset),
                    split="validation",
                    mask_seed=10407,
                    output=str(output),
                    script=str(script_path),
                    command=command,
                )
                jobs.append(job)
                script_path.write_text(
                    _render_a800_script(
                        job_name=f"p104-token-{arm}-{seed}-{percent:03d}",
                        wall_time="00:15:00",
                        command=command,
                    ),
                    encoding="utf-8",
                )
                script_path.chmod(0o755)
    return jobs


def write_token_efficiency_matrix(
    paths: ProjectPaths,
    output_dir: Path,
) -> list[TokenEfficiencyJob]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_token_efficiency_jobs(paths, output_dir)
    _write_tsv(jobs, output_dir / "token_efficiency_manifest.tsv")
    (output_dir / "token_efficiency_job_list.txt").write_text(
        "\n".join(job.script for job in jobs) + "\n", encoding="utf-8"
    )
    return jobs


def write_adaptation_matrix(paths: ProjectPaths, output_dir: Path) -> list[AdaptationJob]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_adaptation_jobs(paths, output_dir)
    _write_tsv(jobs, output_dir / "adaptation_manifest.tsv")
    (output_dir / "adaptation_job_list.txt").write_text(
        "\n".join(job.script for job in jobs) + "\n", encoding="utf-8"
    )
    for method in ("lora", "full"):
        method_jobs = [job for job in jobs if job.method == method]
        _write_tsv(method_jobs, output_dir / f"{method}_manifest.tsv")
        (output_dir / f"{method}_job_list.txt").write_text(
            "\n".join(job.script for job in method_jobs) + "\n",
            encoding="utf-8",
        )
    return jobs


def write_probe_matrix(
    paths: ProjectPaths,
    output_dir: Path,
    *,
    learning_rate_recipe: Path,
) -> list[ProbeJob]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_probe_jobs(paths, learning_rate_recipe=learning_rate_recipe)
    manifest = output_dir / "frozen_probe_manifest.tsv"
    _write_tsv(jobs, manifest)
    runner = paths.project_root / "scripts/run_local_probe_manifest.py"
    python = paths.project_root / ".conda/env/bin/python"
    commands = [
        shlex.join(
            [
                str(python),
                str(runner),
                "--manifest",
                str(manifest),
                "--row-index",
                str(job.row_index),
            ]
        )
        for job in jobs
    ]
    (output_dir / "frozen_probe_command_list.txt").write_text(
        "\n".join(commands) + "\n", encoding="utf-8"
    )
    return jobs


def write_property_head_development_matrix(
    paths: ProjectPaths,
    *,
    slurm_output_dir: Path,
    probe_output_dir: Path,
    warm_learning_rate: float,
) -> tuple[list[EvaluationJob], list[ProbeJob]]:
    slurm_output_dir.mkdir(parents=True, exist_ok=True)
    embedding_jobs = build_property_development_embedding_jobs(
        paths,
        slurm_output_dir,
        warm_learning_rate=warm_learning_rate,
    )
    _write_tsv(
        embedding_jobs,
        slurm_output_dir / "property_head_development_embedding_manifest.tsv",
    )
    (slurm_output_dir / "property_head_development_embedding_job_list.txt").write_text(
        "\n".join(job.script for job in embedding_jobs) + "\n",
        encoding="utf-8",
    )

    probe_output_dir.mkdir(parents=True, exist_ok=True)
    probe_jobs = build_property_head_development_jobs(
        paths, warm_learning_rate=warm_learning_rate
    )
    manifest = probe_output_dir / "property_head_development_manifest.tsv"
    _write_tsv(probe_jobs, manifest)
    runner = paths.project_root / "scripts/run_local_probe_manifest.py"
    python = paths.project_root / ".conda/env/bin/python"
    (probe_output_dir / "property_head_development_command_list.txt").write_text(
        "\n".join(
            shlex.join(
                [
                    str(python),
                    str(runner),
                    "--manifest",
                    str(manifest),
                    "--row-index",
                    str(job.row_index),
                ]
            )
            for job in probe_jobs
        )
        + "\n",
        encoding="utf-8",
    )
    return embedding_jobs, probe_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "matrix",
        choices=(
            "evaluation",
            "mechanism-control-evaluation",
            "token-efficiency",
            "adaptation",
            "probes",
            "property-head-development",
            "all",
        ),
    )
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--probe-output-dir", type=Path)
    parser.add_argument(
        "--property-head-recipe",
        type=Path,
        default=Path("configs/downstream/selected_property_head.yaml"),
    )
    parser.add_argument(
        "--pretraining-recipe",
        type=Path,
        default=Path("configs/pretraining/selected_recipe.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = load_paths(args.paths_config)
    output_root = (
        args.output_root
        or paths.project_root / "slurm/generated/postpretraining"
    ).resolve()
    counts: dict[str, int | str] = {}
    if args.matrix in {"evaluation", "all"}:
        counts["evaluation"] = len(
            write_evaluation_matrix(paths, output_root / "evaluation")
        )
    if args.matrix in {"mechanism-control-evaluation", "all"}:
        counts["mechanism_control_evaluation"] = len(
            write_mechanism_control_evaluation_matrix(
                paths, output_root / "mechanism_controls"
            )
        )
    if args.matrix in {"token-efficiency", "all"}:
        counts["token_efficiency"] = len(
            write_token_efficiency_matrix(
                paths, output_root / "token_efficiency"
            )
        )
    if args.matrix in {"adaptation", "all"}:
        counts["adaptation"] = len(
            write_adaptation_matrix(paths, output_root / "adaptation")
        )
    if args.matrix in {"probes", "all"}:
        probe_root = (
            args.probe_output_dir
            or paths.project_root / "runs/manifests/frozen_probes"
        ).resolve()
        recipe = (
            args.property_head_recipe
            if args.property_head_recipe.is_absolute()
            else paths.project_root / args.property_head_recipe
        )
        counts["probes"] = len(
            write_probe_matrix(paths, probe_root, learning_rate_recipe=recipe)
        )
    if args.matrix in {"property-head-development", "all"}:
        pretraining_recipe = (
            args.pretraining_recipe
            if args.pretraining_recipe.is_absolute()
            else paths.project_root / args.pretraining_recipe
        )
        if pretraining_recipe.is_file():
            warm_learning_rate = load_selected_warm_learning_rate(
                pretraining_recipe
            )
            embedding_jobs, probe_jobs = write_property_head_development_matrix(
                paths,
                slurm_output_dir=output_root / "property_head_development",
                probe_output_dir=(
                    paths.project_root
                    / "runs/manifests/property_head_development"
                ),
                warm_learning_rate=warm_learning_rate,
            )
            counts["property_head_development_embeddings"] = len(embedding_jobs)
            counts["property_head_development_probes"] = len(probe_jobs)
        elif args.matrix == "property-head-development":
            raise SystemExit(
                "property-head development requires the selected pretraining "
                f"recipe: {pretraining_recipe}"
            )
        else:
            counts["property_head_development"] = "pending_pretraining_recipe"
    print(
        ", ".join(f"{name}={count}" for name, count in counts.items())
        + "; no jobs were submitted"
    )


if __name__ == "__main__":
    main()
