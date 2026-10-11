#!/usr/bin/env python3
"""Clone 28 completed recipes into isolated, retained-weight recovery jobs.

Only destinations and retention change. Training hyperparameters, datasets,
splits, text handling, and validation selection are copied from the formal
commands. This generator neither trains models nor submits Slurm jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any

import yaml

from polyfid.paths import DEFAULT_PATH_CONFIG, ProjectPaths, load_paths


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs/downstream/cost_recovery.yaml"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def redirected_command(command: str, python: Path, replacements: dict[str, str]) -> str:
    """Replace only explicit execution-output arguments in an existing recipe."""

    arguments = shlex.split(command)
    arguments[0] = str(python)
    for option, value in replacements.items():
        position = arguments.index(option)
        arguments[position + 1] = str(value)
    return shlex.join(arguments)


def _option(command: str, name: str, default: str = "") -> str:
    arguments = shlex.split(command)
    return arguments[arguments.index(name) + 1] if name in arguments else default


def build_jobs(
    paths: ProjectPaths, config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = paths.project_root
    python = root / ".conda/env/bin/python"
    generated = root / "slurm/generated/cost_recovery"
    originals = {
        method: _rows(root / relative)
        for method, relative in config["original_manifests"].items()
    }
    preparation = _rows(root / config["original_periogt_prepare_manifest"])
    jobs: list[dict[str, Any]] = []
    prep_jobs: list[dict[str, Any]] = []
    for task in config["tasks"]:
        slug = str(task).lower()
        cache = root / "runs/cost_recovery/periogt_cache" / slug
        original_prep = next(row for row in preparation if row["task"].upper() == task.upper())
        prep_command = redirected_command(
            original_prep["command"], python, {"--cache-dir": str(cache)},
        )
        prep_jobs.append({
            "row_index": len(prep_jobs), "job_id": f"cost_recovery_periogt_prepare_{slug}",
            "task": task.upper(), "method": "periogt", "phase": "prepare",
            "dataset": original_prep["dataset"], "split_manifest": original_prep["split_manifest"],
            "cache_dir": str(cache), "output": str(cache), "command": prep_command,
            "resource": "CPU", "gpus": 0, "execution_path": "[CPU → local, no sbatch]",
        })
        for method in config["methods"]:
            matches = [
                row for row in originals[method]
                if row["task"].upper() == task.upper()
                and row["protocol"] == config["protocol"]
                and int(row["fold"]) == config["fold"]
                and int(row["seed"]) == config["seed"]
                and float(row.get("label_fraction", 1)) == config["label_fraction"]
                and int(row.get("subsample_id", 0)) == 0
                and row.get("input_view", "structure_only") == config["input_view"]
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one original recipe for {method}/{task}, found {len(matches)}")
            original = matches[0]
            cell = Path(method) / slug / "fold-0" / "seed-11"
            work = root / "runs/cost_recovery" / cell
            weights = paths.checkpoint_root / "cost_recovery" / cell
            output = root / "results/predictions/cost_recovery" / cell / "predictions.parquet"
            # The Chemprop runner appends seed-N to its --checkpoint-dir.
            cli_checkpoint_dir = weights.parent if method == "chemprop" else weights
            replacements = {
                "--work-dir": str(work), "--checkpoint-dir": str(cli_checkpoint_dir),
                "--checkpoint-retention": "trainable", "--output": str(output),
            }
            run_record = work / "run_record.json" if method == "chemprop" else work / "run/run.json"
            if method == "chemprop":
                replacements["--run-record"] = str(run_record)
            if method == "periogt":
                replacements["--cache-dir"] = str(cache)
            command = redirected_command(original["command"], python, replacements)
            artifacts = {
                "chemprop": ("fold_0/model_0/model.pt", "official_chemprop_checkpoint_with_scalers"),
                "transpolymer": ("transpolymer_best.pt", "payload_model_config_target_scaler"),
                "periogt": ("periogt_best.pth", "native_state_dict_with_separate_target_scaler"),
                "grin": ("grin.pt", "upstream_save_to_local_model_state_and_hyperparameters"),
            }
            source = {
                "chemprop": "random_initialization", "grin": "random_initialization",
                "transpolymer": str(paths.shared_model_root / "external_baselines/transpolymer/checkpoint"),
                "periogt": str(paths.shared_model_root / "external_baselines/periogt/checkpoint/pretrained_ckpt/pretrained/light/base.pth"),
            }[method]
            job_id = f"cost_recovery_{method}_{slug}_fold-0_seed-11"
            jobs.append({
                "row_index": len(jobs), "job_id": job_id,
                "method": method, "task": task.upper(), "protocol": config["protocol"],
                "fold": 0, "seed": 11, "label_fraction": 1.0, "subsample_id": 0,
                "input_view": "structure_only", "phase": "restore_selected_weights",
                "dataset": original["dataset"], "split_manifest": original["split_manifest"],
                "original_manifest": str(root / config["original_manifests"][method]),
                "original_job_id": original["job_id"], "original_output": original["output"],
                "original_command": original["command"], "work_dir": str(work),
                "checkpoint_dir": str(weights), "checkpoint_retention": "trainable",
                "selected_model_path": str(weights / artifacts[method][0]),
                "weights_format": artifacts[method][1], "backend_config_path": str(weights / "config.json"),
                "scaler_path": str(weights / "target_scaler.json"),
                "reload_check_path": str(weights / "reload_check.json"),
                "cache_dir": str(cache) if method == "periogt" else "",
                "shared_model_root": str(paths.shared_model_root), "model_source": source,
                "text_column": _option(command, "--text-column", "structure_only_text" if method == "transpolymer" else "canonical_text"),
                "max_length": _option(command, "--max-length", "not_applicable"),
                "dtype": "float64" if method == "transpolymer" else "float32",
                "batch_size": int(_option(command, "--batch-size")),
                "output": str(output), "prediction_path": str(output), "run_record": str(run_record),
                "script": str(generated / f"{job_id}.sbatch"), "command": command,
                "resource": "GPU-A800", "gpus": 1, "execution_path": "[GPU-A800 → sbatch]",
            })
    return jobs, prep_jobs


def render_script(job: dict[str, Any], paths: ProjectPaths, resources: dict[str, Any]) -> str:
    return f'''#!/usr/bin/env bash
#SBATCH --job-name=p104-cr-{job['method']}-{job['task'].lower()}
#SBATCH --partition={resources['partition']}
#SBATCH --gres={resources['gres']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['memory']}
#SBATCH --time={resources['wall_time']}
#SBATCH --output=logs/slurm/cost_recovery/%x-%j.out
#SBATCH --error=logs/slurm/cost_recovery/%x-%j.err

set -euo pipefail
export DGLBACKEND=pytorch
cd {shlex.quote(str(paths.project_root))}
mkdir -p logs/slurm/cost_recovery
{job['command']}
'''


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_matrix(paths: ProjectPaths, config: dict[str, Any]) -> dict[str, Any]:
    jobs, prep = build_jobs(paths, config)
    generated = paths.project_root / "slurm/generated/cost_recovery"
    manifests = paths.project_root / "runs/manifests/cost_recovery"
    generated.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        Path(job["script"]).write_text(render_script(job, paths, config["resource"]), encoding="utf-8")
    _write_tsv(manifests / "baseline_recovery_manifest.tsv", jobs)
    _write_tsv(manifests / "periogt_prepare_manifest.tsv", prep)
    for name, selected in (("recovery_all28.txt", jobs), ("recovery_egc_first4.txt", jobs[:4]), ("recovery_remaining24.txt", jobs[4:])):
        (generated / name).write_text("".join(str(job["script"]) + "\n" for job in selected), encoding="utf-8")
    (manifests / "periogt_prepare_command_list.txt").write_text("".join(row["command"] + "\n" for row in prep), encoding="utf-8")
    (paths.project_root / "logs/slurm/cost_recovery").mkdir(parents=True, exist_ok=True)
    return {"recovery_jobs": len(jobs), "local_cpu_preparations": len(prep), "manifest": str(manifests / "baseline_recovery_manifest.tsv"), "slurm_root": str(generated)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=DEFAULT_PATH_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(write_matrix(load_paths(args.paths), config), indent=2))


if __name__ == "__main__":
    main()
