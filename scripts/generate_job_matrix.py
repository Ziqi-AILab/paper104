#!/usr/bin/env python3
"""Generate one Paper104 continued-pretraining Slurm matrix at a time.

Development, formal-core, and control jobs are deliberately separate outputs.
Formal jobs can only be rendered from a recipe selected after development.
This module writes files only; it never invokes ``sbatch``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "slurm" / "generated"
SELECTED_RECIPE_PATH = PROJECT_ROOT / "configs" / "pretraining" / "selected_recipe.yaml"

DEVELOPMENT_SEEDS = (13, 37)
FORMAL_SEEDS = (42, 123, 777, 2025, 2026)
CONTROL_SEEDS = (42, 123, 777)

WARM_LEARNING_RATES = (1e-5, 3e-5, 1e-4)
SCRATCH_LEARNING_RATES = (3e-5, 1e-4, 3e-4)
MASK_RATIOS = (
    (0.50, 0.40),
    (0.65, 0.25),
    (0.80, 0.10),
)
FORMAL_ARMS = ("B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs")
SCRATCH_ARMS = frozenset(("B0", "B_eqs"))
CONTROLS = (
    "matched_random_masking",
    "permuted_boundary_masking",
    "equivalence_only",
    "topology_negative_only",
)


@dataclass(frozen=True)
class SelectedRecipe:
    warm_learning_rate: float
    scratch_learning_rate: float
    boundary_fraction: float
    interior_fraction: float


@dataclass(frozen=True)
class Job:
    matrix: str
    arm: str
    seed: int
    learning_rate: float
    boundary_fraction: float
    interior_fraction: float
    micro_batch_size: int
    gradient_accumulation: int
    script_name: str
    output_dir: str


def _format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def load_selected_recipe(path: Path) -> SelectedRecipe:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("selected recipe must be a YAML mapping")
    recipe_values: Mapping[str, Any] = values.get("selected_recipe", values)
    required = (
        "warm_learning_rate",
        "scratch_learning_rate",
        "boundary_fraction",
        "interior_fraction",
    )
    missing = [name for name in required if name not in recipe_values]
    if missing:
        raise ValueError(f"selected recipe is missing: {', '.join(missing)}")
    recipe = SelectedRecipe(**{name: float(recipe_values[name]) for name in required})
    if recipe.warm_learning_rate not in WARM_LEARNING_RATES:
        raise ValueError("warm_learning_rate was not a declared development candidate")
    if recipe.scratch_learning_rate not in SCRATCH_LEARNING_RATES:
        raise ValueError("scratch_learning_rate was not a declared development candidate")
    if (recipe.boundary_fraction, recipe.interior_fraction) not in MASK_RATIOS:
        raise ValueError("boundary/interior fractions were not a declared development candidate")
    return recipe


def build_development_jobs() -> list[Job]:
    jobs: list[Job] = []
    default_boundary, default_interior = MASK_RATIOS[1]
    for arm, learning_rates in (
        ("A0", WARM_LEARNING_RATES),
        ("B0", SCRATCH_LEARNING_RATES),
    ):
        for learning_rate in learning_rates:
            for seed in DEVELOPMENT_SEEDS:
                lr_tag = _format_float(learning_rate)
                jobs.append(
                    Job(
                        matrix="development",
                        arm=arm,
                        seed=seed,
                        learning_rate=learning_rate,
                        boundary_fraction=default_boundary,
                        interior_fraction=default_interior,
                        micro_batch_size=32,
                        gradient_accumulation=8,
                        script_name=f"dev_{arm}_lr{lr_tag}_seed{seed}.sbatch",
                        output_dir=f"runs/pretraining/development/{arm}/lr-{learning_rate:g}/seed-{seed}",
                    )
                )

    for learning_rate in WARM_LEARNING_RATES:
        for boundary_fraction, interior_fraction in MASK_RATIOS:
            for seed in DEVELOPMENT_SEEDS:
                lr_tag = _format_float(learning_rate)
                ratio_tag = (
                    f"b{_format_float(boundary_fraction)}"
                    f"_i{_format_float(interior_fraction)}"
                )
                jobs.append(
                    Job(
                        matrix="development",
                        arm="A_att",
                        seed=seed,
                        learning_rate=learning_rate,
                        boundary_fraction=boundary_fraction,
                        interior_fraction=interior_fraction,
                        micro_batch_size=32,
                        gradient_accumulation=8,
                        script_name=f"dev_A_att_lr{lr_tag}_{ratio_tag}_seed{seed}.sbatch",
                        output_dir=(
                            "runs/pretraining/development/A_att/"
                            f"lr-{learning_rate:g}_boundary-{boundary_fraction:g}"
                            f"_interior-{interior_fraction:g}/seed-{seed}"
                        ),
                    )
                )
    return jobs


def build_formal_jobs(recipe: SelectedRecipe) -> list[Job]:
    return [
        Job(
            matrix="formal",
            arm=arm,
            seed=seed,
            learning_rate=(
                recipe.scratch_learning_rate
                if arm in SCRATCH_ARMS
                else recipe.warm_learning_rate
            ),
            boundary_fraction=recipe.boundary_fraction,
            interior_fraction=recipe.interior_fraction,
            micro_batch_size=(8 if arm in {"A_gcl", "A_eqs", "B_eqs"} else 32),
            gradient_accumulation=(
                32 if arm in {"A_gcl", "A_eqs", "B_eqs"} else 8
            ),
            script_name=f"formal_{arm}_seed{seed}.sbatch",
            output_dir=f"runs/pretraining/formal/{arm}/seed-{seed}",
        )
        for arm in FORMAL_ARMS
        for seed in FORMAL_SEEDS
    ]


def build_control_jobs(recipe: SelectedRecipe) -> list[Job]:
    return [
        Job(
            matrix="controls",
            arm=control,
            seed=seed,
            learning_rate=recipe.warm_learning_rate,
            boundary_fraction=recipe.boundary_fraction,
            interior_fraction=recipe.interior_fraction,
            micro_batch_size=(
                8
                if control in {"equivalence_only", "topology_negative_only"}
                else 32
            ),
            gradient_accumulation=(
                32
                if control in {"equivalence_only", "topology_negative_only"}
                else 8
            ),
            script_name=f"control_{control}_seed{seed}.sbatch",
            output_dir=f"runs/pretraining/controls/{control}/seed-{seed}",
        )
        for control in CONTROLS
        for seed in CONTROL_SEEDS
    ]


def render_job(job: Job) -> str:
    stage = "pilot" if job.matrix == "development" else (
        "control" if job.matrix == "controls" else "formal"
    )
    job_name = f"p104-{Path(job.script_name).stem}".replace("_", "-")
    return f'''#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:4090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${{PROJECT_DIR:-.}}"
PYTHON_BIN="${{PROJECT_DIR}}/.conda/env/bin/python"
OUTPUT_DIR="${{PROJECT_DIR}}/{job.output_dir}"
cd "${{PROJECT_DIR}}"
mkdir -p "${{OUTPUT_DIR}}" "${{PROJECT_DIR}}/logs/slurm"

"${{PYTHON_BIN}}" scripts/run_pretraining.py \\
  --config configs/pretraining/core.yaml \\
  --arms-config configs/pretraining/arms.yaml \\
  --arm {job.arm} \\
  --seed {job.seed} \\
  --learning-rate {job.learning_rate:g} \\
  --boundary-fraction {job.boundary_fraction:g} \\
  --interior-fraction {job.interior_fraction:g} \\
  --micro-batch-size {job.micro_batch_size} \\
  --gradient-accumulation {job.gradient_accumulation} \\
  --resume-existing \\
  --stage {stage} \\
  --output-dir "${{OUTPUT_DIR}}"
'''


def write_matrix(matrix: str, output_dir: Path, recipe: SelectedRecipe | None = None) -> list[Job]:
    if matrix == "development":
        jobs = build_development_jobs()
    elif matrix == "formal":
        if recipe is None:
            raise ValueError("formal matrix requires a selected development recipe")
        jobs = build_formal_jobs(recipe)
    elif matrix == "controls":
        if recipe is None:
            raise ValueError("control matrix requires a selected development recipe")
        jobs = build_control_jobs(recipe)
    else:
        raise ValueError(f"unknown matrix: {matrix}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{matrix}_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "matrix",
                "arm",
                "seed",
                "learning_rate",
                "boundary_fraction",
                "interior_fraction",
                "micro_batch_size",
                "gradient_accumulation",
                "script",
                "output_dir",
                "resource",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for job in jobs:
            script_path = output_dir / job.script_name
            script_path.write_text(render_job(job), encoding="utf-8")
            script_path.chmod(0o755)
            try:
                manifest_script = script_path.relative_to(PROJECT_ROOT)
            except ValueError:
                manifest_script = script_path
            writer.writerow(
                {
                    "matrix": job.matrix,
                    "arm": job.arm,
                    "seed": job.seed,
                    "learning_rate": f"{job.learning_rate:g}",
                    "boundary_fraction": f"{job.boundary_fraction:g}",
                    "interior_fraction": f"{job.interior_fraction:g}",
                    "micro_batch_size": job.micro_batch_size,
                    "gradient_accumulation": job.gradient_accumulation,
                    "script": manifest_script,
                    "output_dir": job.output_dir,
                    "resource": "GPU-4090",
                }
            )

    (output_dir / f"{matrix}_job_list.txt").write_text(
        "\n".join(str(output_dir / job.script_name) for job in jobs) + "\n",
        encoding="utf-8",
    )
    return jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=("development", "formal", "controls"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = None
    if args.matrix != "development":
        if not SELECTED_RECIPE_PATH.is_file():
            raise SystemExit(
                "formal/control generation requires "
                "configs/pretraining/selected_recipe.yaml from the development selector"
            )
        recipe = load_selected_recipe(SELECTED_RECIPE_PATH)
    output_dir = (args.output_dir or (DEFAULT_OUTPUT_ROOT / args.matrix)).resolve()
    jobs = write_matrix(args.matrix, output_dir, recipe)
    print(f"Wrote {len(jobs)} {args.matrix} jobs to {output_dir}")


if __name__ == "__main__":
    main()
