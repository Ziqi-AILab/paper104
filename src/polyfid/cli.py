"""Single Paper104 command-line interface."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer

from polyfid.paths import DEFAULT_PATH_CONFIG, ProjectPaths


app = typer.Typer(name="polyfid", no_args_is_help=True)
assets_app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
relations_app = typer.Typer(no_args_is_help=True)
embeddings_app = typer.Typer(no_args_is_help=True)
downstream_app = typer.Typer(no_args_is_help=True)
fidelity_app = typer.Typer(no_args_is_help=True)
app.add_typer(assets_app, name="assets")
app.add_typer(data_app, name="data")
app.add_typer(relations_app, name="relations")
app.add_typer(embeddings_app, name="embeddings")
app.add_typer(downstream_app, name="downstream")
app.add_typer(fidelity_app, name="fidelity")


def _paths(path_config: Path) -> ProjectPaths:
    return ProjectPaths.from_yaml(path_config)


def _run_script(paths: ProjectPaths, name: str, arguments: list[str]) -> None:
    command = [sys.executable, str(paths.project_root / "scripts" / name), *arguments]
    result = subprocess.run(command, cwd=paths.project_root, check=False)
    if result.returncode:
        raise typer.Exit(result.returncode)


@app.command()
def status(
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Report actual local assets and observed outputs."""

    paths = _paths(path_config)
    generated_root = paths.project_root / "slurm/generated"
    pretraining_job_counts = {
        stage: len(list((generated_root / stage).glob("*.sbatch")))
        for stage in ("development", "formal", "controls")
    }
    values = {
        "project_root": str(paths.project_root),
        "environment_ready": (paths.project_root / ".conda/env/bin/python").is_file(),
        "pi1m_ready": (paths.shared_data_root / "standardized/v1/pi1m/pi1m_v2.parquet").is_file(),
        "chemberta_ready": (paths.shared_model_root / "chemberta_100m_mlm/model.safetensors").is_file(),
        "split_registry_ready": (paths.project_root / "data/splits/registry.json").is_file(),
        "relation_partitions": sorted(
            path.parent.name
            for path in (paths.project_root / "data/relations").glob("*/pi1m_triplets.parquet")
        ),
        "generated_pretraining_jobs": pretraining_job_counts,
        "generated_gpu_job_scripts_total": len(list(generated_root.glob("**/*.sbatch"))),
        "formal_pretraining_checkpoints": len(
            list((paths.checkpoint_root / "pretraining").glob("**/*.pt"))
        ),
        "development_pretraining_runs": len(
            list(
                (paths.project_root / "runs/pretraining/development").glob(
                    "**/run_result.json"
                )
            )
        ),
        "engineering_smoke_checkpoint_files": len(
            list((paths.checkpoint_root / "dev/smoke").glob("**/*.pt"))
        )
        + len(
            list((paths.checkpoint_root / "dev/external_smoke").glob("**/*.pt"))
        ),
        "prediction_files": len(list((paths.project_root / "results/predictions").glob("**/*.parquet"))),
    }
    if as_json:
        typer.echo(json.dumps(values, indent=2, ensure_ascii=False))
        return
    for key, value in values.items():
        typer.echo(f"{key}: {value}")


@assets_app.command("import-existing")
def assets_import(
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    from polyfid.workflows.assets import import_existing_assets

    typer.echo(json.dumps(import_existing_assets(_paths(path_config)), indent=2, ensure_ascii=False))


@data_app.command("prepare")
def data_prepare(
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    from polyfid.workflows.data import prepare_project_data

    typer.echo(json.dumps(prepare_project_data(_paths(path_config)), indent=2, ensure_ascii=False))


@data_app.command("rebuild-public")
def data_rebuild_public(
    stage: str = typer.Argument("all"),
    output_root: Path = typer.Option(
        ...,
        help="Explicit destination; the frozen shared standardized tables are not overwritten.",
    ),
    seed: int = typer.Option(42),
    pi1m_max_source_rows: int | None = typer.Option(None),
    zinc_smoke_count: int | None = typer.Option(None),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Rebuild public raw sources into a separate standardized-v1 directory."""

    paths = _paths(path_config)
    arguments = [
        stage,
        "--output-root",
        str(output_root),
        "--paths-config",
        str(path_config),
        "--seed",
        str(seed),
    ]
    if pi1m_max_source_rows is not None:
        arguments.extend(("--pi1m-max-source-rows", str(pi1m_max_source_rows)))
    if zinc_smoke_count is not None:
        arguments.extend(("--zinc-smoke-count", str(zinc_smoke_count)))
    _run_script(paths, "rebuild_public_data.py", arguments)


@relations_app.command("build")
def relations_build(
    partition: str = typer.Option("train"),
    target_triplets: int = typer.Option(10_000),
    max_index_rows: int | None = typer.Option(None),
    workers: int = typer.Option(12),
    seed: int = typer.Option(42),
    rebuild_index: bool = typer.Option(False),
    allow_near_formula_fallback: bool = typer.Option(False),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    from polyfid.workflows.relations import build_relation_set

    result = build_relation_set(
        _paths(path_config),
        partition=partition,
        target_triplets=target_triplets,
        max_index_rows=max_index_rows,
        workers=workers,
        seed=seed,
        rebuild_index=rebuild_index,
        allow_near_formula_fallback=allow_near_formula_fallback,
    )
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command("pretrain")
def pretrain(
    arm: str = typer.Option(...),
    seed: int = typer.Option(...),
    learning_rate: float | None = typer.Option(None),
    boundary_fraction: float = typer.Option(0.65),
    interior_fraction: float = typer.Option(0.25),
    max_rows: int | None = typer.Option(None),
    resume: Path | None = typer.Option(None),
    config: Path = typer.Option(Path("configs/pretraining/core.yaml")),
    arms_config: Path = typer.Option(Path("configs/pretraining/arms.yaml")),
    stage: str = typer.Option("formal"),
    output_dir: Path | None = typer.Option(None),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    paths = _paths(path_config)
    arguments = [
        "--arm", arm, "--seed", str(seed), "--config", str(config),
        "--arms-config", str(arms_config), "--stage", stage,
        "--paths-config", str(path_config),
        "--boundary-fraction", str(boundary_fraction),
        "--interior-fraction", str(interior_fraction),
    ]
    if learning_rate is not None:
        arguments.extend(("--learning-rate", str(learning_rate)))
    if max_rows is not None:
        arguments.extend(("--max-train-rows", str(max_rows), "--max-validation-rows", str(max_rows)))
    if resume is not None:
        arguments.extend(("--resume", str(resume)))
    if output_dir is not None:
        arguments.extend(("--output-dir", str(output_dir)))
    _run_script(paths, "run_pretraining.py", arguments)


@embeddings_app.command("cache")
def embeddings_cache(
    dataset: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    base_checkpoint: Path | None = typer.Option(None),
    method: str = typer.Option(...),
    seed: int = typer.Option(...),
    text_column: str = typer.Option("canonical_text"),
    output: Path = typer.Option(...),
    input_view: str = typer.Option("valid"),
    checkpoint_fraction: float | None = typer.Option(None),
    checkpoint_tokens: int | None = typer.Option(None),
    batch_size: int = typer.Option(128),
    max_rows: int | None = typer.Option(None),
    device: str = typer.Option("cuda"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    paths = _paths(path_config)
    arguments = [
        "--dataset", str(dataset), "--checkpoint", str(checkpoint),
        "--method", method, "--seed", str(seed), "--text-column", text_column,
        "--output", str(output), "--input-view", input_view,
        "--batch-size", str(batch_size), "--device", device,
    ]
    if max_rows is not None:
        arguments.extend(("--max-rows", str(max_rows)))
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    if checkpoint_fraction is not None:
        arguments.extend(("--checkpoint-fraction", str(checkpoint_fraction)))
    if checkpoint_tokens is not None:
        arguments.extend(("--checkpoint-tokens", str(checkpoint_tokens)))
    _run_script(
        paths,
        "cache_embeddings.py",
        arguments,
    )


@downstream_app.command("evaluate")
def downstream_evaluate(
    dataset: Path = typer.Option(...),
    split_manifest: Path = typer.Option(...),
    embeddings: Path = typer.Option(...),
    task: str = typer.Option(...),
    protocol: str = typer.Option("default"),
    label_column: str = typer.Option("label"),
    fold: int = typer.Option(0),
    label_fraction: float = typer.Option(1.0),
    subsample_id: int = typer.Option(0),
    learning_rate: float = typer.Option(3e-4),
    task_type: str = typer.Option("regression"),
    device: str = typer.Option(
        "cpu", help="Cached frozen-probe training is a local CPU stage."
    ),
    output: Path = typer.Option(...),
    metrics_output: Path | None = typer.Option(None),
    probe_state_dir: Path | None = typer.Option(None),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    arguments = [
        "--dataset", str(dataset), "--split-manifest", str(split_manifest),
        "--embeddings", str(embeddings), "--task", task,
        "--protocol", protocol,
        "--label-column", label_column, "--fold", str(fold),
        "--label-fraction", str(label_fraction), "--subsample-id", str(subsample_id),
        "--learning-rate", str(learning_rate), "--task-type", task_type,
        "--device", device, "--output", str(output),
    ]
    if probe_state_dir is not None:
        arguments.extend(("--probe-state-dir", str(probe_state_dir)))
    if metrics_output is not None:
        arguments.extend(("--metrics-output", str(metrics_output)))
    _run_script(_paths(path_config), "evaluate_downstream.py", arguments)


@downstream_app.command("select-lr")
def downstream_select_lr(
    validation_losses: Path = typer.Option(...),
    output: Path = typer.Option(...),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Select the shared property-head LR from validation losses only."""

    _run_script(
        _paths(path_config),
        "select_property_head_lr.py",
        [
            "--validation-losses",
            str(validation_losses),
            "--output",
            str(output),
        ],
    )


@downstream_app.command("adapt")
def downstream_adapt(
    method: str = typer.Option(..., help="LoRA or full fine-tuning."),
    arm: str = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    base_checkpoint: Path | None = typer.Option(None),
    representation_seed: int = typer.Option(...),
    checkpoint_fraction: float | None = typer.Option(None),
    checkpoint_tokens: int | None = typer.Option(None),
    adaptation_seed: int = typer.Option(...),
    dataset: Path = typer.Option(...),
    split_manifest: Path = typer.Option(...),
    task: str = typer.Option(...),
    protocol: str = typer.Option(...),
    fold: int = typer.Option(0),
    label_column: str | None = typer.Option(None),
    text_column: str | None = typer.Option(None),
    task_type: str = typer.Option("regression"),
    learning_rate: float = typer.Option(1e-4),
    weight_decay: float = typer.Option(0.01),
    batch_size: int = typer.Option(32),
    max_length: int = typer.Option(256),
    max_epochs: int = typer.Option(100),
    patience: int = typer.Option(15),
    lora_rank: int = typer.Option(8),
    lora_alpha: float = typer.Option(16.0),
    lora_dropout: float = typer.Option(0.05),
    device: str = typer.Option("cuda"),
    output: Path = typer.Option(...),
    model_output: Path = typer.Option(...),
    checkpoint_retention: str = typer.Option("trainable"),
    run_record: Path = typer.Option(...),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Run one predeclared same-split LoRA or full adaptation."""

    arguments = [
        "--method", method,
        "--arm", arm,
        "--checkpoint", str(checkpoint),
        "--representation-seed", str(representation_seed),
        "--adaptation-seed", str(adaptation_seed),
        "--dataset", str(dataset),
        "--split-manifest", str(split_manifest),
        "--task", task,
        "--protocol", protocol,
        "--fold", str(fold),
        "--task-type", task_type,
        "--learning-rate", str(learning_rate),
        "--weight-decay", str(weight_decay),
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--max-epochs", str(max_epochs),
        "--patience", str(patience),
        "--lora-rank", str(lora_rank),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--device", device,
        "--output", str(output),
        "--model-output", str(model_output),
        "--checkpoint-retention", checkpoint_retention,
        "--run-record", str(run_record),
    ]
    if checkpoint_fraction is not None:
        arguments.extend(("--checkpoint-fraction", str(checkpoint_fraction)))
    if checkpoint_tokens is not None:
        arguments.extend(("--checkpoint-tokens", str(checkpoint_tokens)))
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    if label_column is not None:
        arguments.extend(("--label-column", label_column))
    if text_column is not None:
        arguments.extend(("--text-column", text_column))
    _run_script(_paths(path_config), "adapt_downstream.py", arguments)


@downstream_app.command("baseline")
def downstream_baseline(
    method: str = typer.Option(...),
    dataset: Path = typer.Option(...),
    split_manifest: Path = typer.Option(...),
    task: str = typer.Option(...),
    protocol: str = typer.Option(...),
    fold: int = typer.Option(0),
    label_column: str | None = typer.Option(None),
    text_column: str | None = typer.Option(None),
    task_type: str = typer.Option("regression"),
    seeds: str = typer.Option("11,22,33"),
    label_fraction: float = typer.Option(1.0),
    subsample_id: int = typer.Option(0),
    input_view: str = typer.Option("structure_only"),
    feature_cache: Path | None = typer.Option(None),
    epochs: int = typer.Option(30),
    batch_size: int = typer.Option(50),
    num_workers: int = typer.Option(8),
    work_dir: Path = typer.Option(Path("runs/baselines/chemprop")),
    checkpoint_dir: Path | None = typer.Option(None),
    checkpoint_retention: str = typer.Option("trainable"),
    config_path: Path = typer.Option(
        Path("configs/downstream/controlled_baselines.yaml")
    ),
    run_record: Path | None = typer.Option(None),
    output: Path = typer.Option(...),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Run a same-split classical or Chemprop baseline."""

    arguments = [
        "run",
        "--method", method,
        "--dataset", str(dataset),
        "--split-manifest", str(split_manifest),
        "--task", task,
        "--protocol", protocol,
        "--fold", str(fold),
        "--task-type", task_type,
        "--seeds", seeds,
        "--label-fraction", str(label_fraction),
        "--subsample-id", str(subsample_id),
        "--input-view", input_view,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--work-dir", str(work_dir),
        "--checkpoint-retention", checkpoint_retention,
        "--config-path", str(config_path),
        "--output", str(output),
    ]
    if run_record is not None:
        arguments.extend(("--run-record", str(run_record)))
    if checkpoint_dir is not None:
        arguments.extend(("--checkpoint-dir", str(checkpoint_dir)))
    if feature_cache is not None:
        arguments.extend(("--feature-cache", str(feature_cache)))
    if label_column is not None:
        arguments.extend(("--label-column", label_column))
    if text_column is not None:
        arguments.extend(("--text-column", text_column))
    _run_script(_paths(path_config), "evaluate_classical_baselines.py", arguments)


@downstream_app.command("probe-external")
def downstream_probe_external(
    model_root: Path | None = typer.Option(None),
    output: Path | None = typer.Option(None),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Report which external baseline assets are available for integration."""

    paths = _paths(path_config)
    arguments = [
        "probe-external",
        "--model-root", str(model_root or paths.shared_model_root),
    ]
    if output is not None:
        arguments.extend(("--output", str(output)))
    _run_script(paths, "evaluate_classical_baselines.py", arguments)


@downstream_app.command("external-baseline")
def downstream_external_baseline(
    baseline: str = typer.Option("transpolymer"),
    dataset: Path = typer.Option(...),
    split_manifest: Path = typer.Option(...),
    task: str = typer.Option(...),
    protocol: str = typer.Option(...),
    fold: int = typer.Option(0),
    seed: int = typer.Option(...),
    label_column: str | None = typer.Option(None),
    text_column: str | None = typer.Option(None),
    epochs: int | None = typer.Option(None),
    patience: int | None = typer.Option(None),
    batch_size: int | None = typer.Option(None),
    learning_rate: float | None = typer.Option(None),
    max_length: int = typer.Option(512),
    num_workers: int | None = typer.Option(None),
    work_dir: Path = typer.Option(...),
    cache_dir: Path | None = typer.Option(None),
    checkpoint_dir: Path | None = typer.Option(None),
    checkpoint_retention: str = typer.Option("trainable"),
    output: Path = typer.Option(...),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Run a validated same-split external polymer adapter on A800."""

    paths = _paths(path_config)
    baseline_key = baseline.casefold().replace("_", "-")
    commands = {
        "transpolymer": "run-transpolymer",
        "periogt": "run-periogt",
        "grin": "run-grin",
    }
    if baseline_key not in commands:
        raise typer.BadParameter(f"unsupported runnable external baseline: {baseline}")
    defaults = {
        "transpolymer": (20, 5, 32, 0),
        "periogt": (50, 20, 32, 2),
        "grin": (500, 50, 128, 0),
    }
    default_epochs, default_patience, default_batch, default_workers = defaults[
        baseline_key
    ]
    arguments = [
        commands[baseline_key],
        "--dataset", str(dataset),
        "--split-manifest", str(split_manifest),
        "--task", task,
        "--protocol", protocol,
        "--fold", str(fold),
        "--model-root", str(paths.shared_model_root),
        "--seed", str(seed),
        "--epochs", str(epochs if epochs is not None else default_epochs),
        "--patience", str(patience if patience is not None else default_patience),
        "--batch-size", str(batch_size if batch_size is not None else default_batch),
        "--output-dir", str(work_dir),
        "--output", str(output),
        "--checkpoint-retention", checkpoint_retention,
    ]
    if baseline_key == "transpolymer":
        arguments.extend(
            (
                "--max-length", str(max_length),
                "--num-workers", str(
                    num_workers if num_workers is not None else default_workers
                ),
            )
        )
    elif baseline_key == "periogt":
        arguments.extend(
            (
                "--num-workers", str(
                    num_workers if num_workers is not None else default_workers
                ),
            )
        )
    if learning_rate is not None:
        arguments.extend(("--learning-rate", str(learning_rate)))
    if baseline_key == "periogt":
        if cache_dir is None:
            raise typer.BadParameter("PerioGT requires --cache-dir from external-prepare")
        arguments.extend(("--cache-dir", str(cache_dir)))
    if checkpoint_dir is not None:
        arguments.extend(("--checkpoint-dir", str(checkpoint_dir)))
    if label_column is not None:
        arguments.extend(("--label-column", label_column))
    if text_column is not None:
        arguments.extend(("--text-column", text_column))
    _run_script(paths, "evaluate_external_baseline.py", arguments)


@downstream_app.command("external-prepare")
def downstream_external_prepare(
    dataset: Path = typer.Option(...),
    split_manifest: Path = typer.Option(...),
    task: str = typer.Option(...),
    protocol: str = typer.Option(...),
    fold: int = typer.Option(0),
    cache_dir: Path = typer.Option(...),
    preprocess_workers: int = typer.Option(8),
    label_column: str | None = typer.Option(None),
    text_column: str | None = typer.Option(None),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Prepare the task-level PerioGT graph and periodic-prompt cache locally."""

    paths = _paths(path_config)
    arguments = [
        "prepare-periogt",
        "--dataset", str(dataset),
        "--split-manifest", str(split_manifest),
        "--task", task,
        "--protocol", protocol,
        "--fold", str(fold),
        "--model-root", str(paths.shared_model_root),
        "--preprocess-workers", str(preprocess_workers),
        "--cache-dir", str(cache_dir),
    ]
    if label_column is not None:
        arguments.extend(("--label-column", label_column))
    if text_column is not None:
        arguments.extend(("--text-column", text_column))
    _run_script(paths, "evaluate_external_baseline.py", arguments)


@fidelity_app.command("evaluate")
def fidelity_evaluate(
    relations: Path = typer.Option(...),
    decoy_bank: Path | None = typer.Option(None),
    supplemental_controls: Path | None = typer.Option(None),
    checkpoint: Path = typer.Option(...),
    base_checkpoint: Path | None = typer.Option(None),
    method: str = typer.Option(...),
    seed: int = typer.Option(...),
    output: Path = typer.Option(...),
    batch_size: int = typer.Option(128),
    max_length: int = typer.Option(256),
    device: str = typer.Option("cuda"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    arguments = [
        "--relations", str(relations), "--checkpoint", str(checkpoint),
        "--method", method, "--seed", str(seed), "--output", str(output),
        "--batch-size", str(batch_size), "--max-length", str(max_length),
        "--device", device,
    ]
    if decoy_bank is not None:
        arguments.extend(("--decoy-bank", str(decoy_bank)))
    if supplemental_controls is not None:
        arguments.extend(("--supplemental-controls", str(supplemental_controls)))
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    _run_script(_paths(path_config), "evaluate_fidelity.py", arguments)


@fidelity_app.command("invariance")
def fidelity_invariance(
    dataset: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    method: str = typer.Option(...),
    seed: int = typer.Option(...),
    output: Path = typer.Option(...),
    base_checkpoint: Path | None = typer.Option(None),
    reference_checkpoint: Path | None = typer.Option(None),
    text_column: str = typer.Option("canonical_psmiles"),
    sample_id_column: str = typer.Option("sample_id"),
    layers: str = typer.Option("0,3,6,9,12"),
    randomized_views: int = typer.Option(10),
    periodic_views: int = typer.Option(10),
    atom_permutations: int = typer.Option(10),
    batch_size: int = typer.Option(128),
    max_length: int = typer.Option(256),
    max_rows: int | None = typer.Option(None),
    device: str = typer.Option("cuda"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Evaluate legal-view consistency and separate applicability controls."""

    arguments = [
        "--dataset", str(dataset),
        "--checkpoint", str(checkpoint),
        "--method", method,
        "--seed", str(seed),
        "--output", str(output),
        "--text-column", text_column,
        "--sample-id-column", sample_id_column,
        "--layers", layers,
        "--randomized-views", str(randomized_views),
        "--periodic-views", str(periodic_views),
        "--atom-permutations", str(atom_permutations),
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--device", device,
    ]
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    if reference_checkpoint is not None:
        arguments.extend(("--reference-checkpoint", str(reference_checkpoint)))
    if max_rows is not None:
        arguments.extend(("--max-rows", str(max_rows)))
    _run_script(_paths(path_config), "evaluate_invariance.py", arguments)


@fidelity_app.command("boundary-mlm")
def fidelity_boundary_mlm(
    dataset: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    method: str = typer.Option(...),
    seed: int = typer.Option(...),
    output: Path = typer.Option(...),
    base_checkpoint: Path | None = typer.Option(None),
    scope: str = typer.Option("repeat"),
    text_column: str = typer.Option("canonical_psmiles"),
    sample_id_column: str = typer.Option("sample_id"),
    split: str | None = typer.Option(None),
    split_column: str = typer.Option("split"),
    hidden_layers: str = typer.Option("0,3,6,9,12"),
    batch_size: int = typer.Option(128),
    max_length: int = typer.Option(256),
    max_rows: int | None = typer.Option(None),
    device: str = typer.Option("cuda"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Evaluate atom-group MLM by attachment-distance stratum."""

    arguments = [
        "--dataset", str(dataset),
        "--checkpoint", str(checkpoint),
        "--method", method,
        "--seed", str(seed),
        "--output", str(output),
        "--scope", scope,
        "--text-column", text_column,
        "--sample-id-column", sample_id_column,
        "--split-column", split_column,
        "--hidden-layers", hidden_layers,
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--device", device,
    ]
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    if split is not None:
        arguments.extend(("--split", split))
    if max_rows is not None:
        arguments.extend(("--max-rows", str(max_rows)))
    _run_script(_paths(path_config), "evaluate_boundary_mlm.py", arguments)


@fidelity_app.command("retention")
def fidelity_retention(
    checkpoint: Path = typer.Option(...),
    method: str = typer.Option(...),
    representation_seed: int = typer.Option(...),
    output: Path = typer.Option(...),
    dataset: Path | None = typer.Option(None),
    base_checkpoint: Path | None = typer.Option(None),
    split: str = typer.Option("retention"),
    mask_seed: int = typer.Option(10_407),
    mask_probability: float = typer.Option(0.15),
    batch_size: int = typer.Option(128),
    max_length: int = typer.Option(256),
    max_rows: int | None = typer.Option(None),
    device: str = typer.Option("cuda"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Evaluate fixed-mask ZINC molecular MLM retention."""

    arguments = [
        "--checkpoint",
        str(checkpoint),
        "--method",
        method,
        "--representation-seed",
        str(representation_seed),
        "--output",
        str(output),
        "--split",
        split,
        "--mask-seed",
        str(mask_seed),
        "--mask-probability",
        str(mask_probability),
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(max_length),
        "--device",
        device,
    ]
    if dataset is not None:
        arguments.extend(("--dataset", str(dataset)))
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    if max_rows is not None:
        arguments.extend(("--max-rows", str(max_rows)))
    _run_script(_paths(path_config), "evaluate_zinc_retention.py", arguments)


@fidelity_app.command("uniform-mlm")
def fidelity_uniform_mlm(
    dataset: Path = typer.Option(...),
    checkpoint: Path = typer.Option(...),
    method: str = typer.Option(...),
    representation_seed: int = typer.Option(...),
    output: Path = typer.Option(...),
    base_checkpoint: Path | None = typer.Option(None),
    checkpoint_fraction: float | None = typer.Option(None),
    checkpoint_tokens: int | None = typer.Option(None),
    text_column: str = typer.Option("canonical_psmiles"),
    sample_id_column: str = typer.Option("sample_id"),
    split: str | None = typer.Option(None),
    split_column: str = typer.Option("split"),
    mask_seed: int = typer.Option(10_407),
    mask_probability: float = typer.Option(0.15),
    batch_size: int = typer.Option(128),
    max_length: int = typer.Option(256),
    max_rows: int | None = typer.Option(None),
    device: str = typer.Option("cuda"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Evaluate matched fixed masks for token-efficiency checkpoints."""

    arguments = [
        "--dataset", str(dataset),
        "--checkpoint", str(checkpoint),
        "--method", method,
        "--representation-seed", str(representation_seed),
        "--text-column", text_column,
        "--sample-id-column", sample_id_column,
        "--split-column", split_column,
        "--mask-seed", str(mask_seed),
        "--mask-probability", str(mask_probability),
        "--batch-size", str(batch_size),
        "--max-length", str(max_length),
        "--device", device,
        "--output", str(output),
    ]
    if base_checkpoint is not None:
        arguments.extend(("--base-checkpoint", str(base_checkpoint)))
    if checkpoint_fraction is not None:
        arguments.extend(("--checkpoint-fraction", str(checkpoint_fraction)))
    if checkpoint_tokens is not None:
        arguments.extend(("--checkpoint-tokens", str(checkpoint_tokens)))
    if split is not None:
        arguments.extend(("--split", split))
    if max_rows is not None:
        arguments.extend(("--max-rows", str(max_rows)))
    _run_script(_paths(path_config), "evaluate_uniform_mlm.py", arguments)


@fidelity_app.command("property-views")
def fidelity_property_views(
    dataset: Path = typer.Option(...),
    split_manifest: Path = typer.Option(...),
    probe_state: Path = typer.Option(...),
    reference_embeddings: Path = typer.Option(...),
    view_rows: Path = typer.Option(...),
    view_embeddings: Path = typer.Option(...),
    task: str = typer.Option(...),
    head_seed: int = typer.Option(...),
    output: Path = typer.Option(...),
    protocol: str = typer.Option("default"),
    fold: int = typer.Option(0),
    label_column: str = typer.Option("label"),
    text_column: str = typer.Option("canonical_text"),
    sample_id_column: str = typer.Option("sample_id"),
    group_id_column: str = typer.Option("group_id"),
    semantic_control_rows: Path | None = typer.Option(None),
    semantic_control_embeddings: Path | None = typer.Option(None),
    batch_size: int = typer.Option(4096),
    device: str = typer.Option("cpu"),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Apply one saved probe to equivalent views and separate scope controls."""

    arguments = [
        "--dataset", str(dataset),
        "--split-manifest", str(split_manifest),
        "--probe-state", str(probe_state),
        "--reference-embeddings", str(reference_embeddings),
        "--view-rows", str(view_rows),
        "--view-embeddings", str(view_embeddings),
        "--task", task,
        "--head-seed", str(head_seed),
        "--protocol", protocol,
        "--fold", str(fold),
        "--label-column", label_column,
        "--text-column", text_column,
        "--sample-id-column", sample_id_column,
        "--group-id-column", group_id_column,
        "--batch-size", str(batch_size),
        "--device", device,
        "--output", str(output),
    ]
    if semantic_control_rows is not None:
        arguments.extend(("--semantic-control-rows", str(semantic_control_rows)))
    if semantic_control_embeddings is not None:
        arguments.extend(
            ("--semantic-control-embeddings", str(semantic_control_embeddings))
        )
    _run_script(_paths(path_config), "evaluate_property_views.py", arguments)


@app.command("statistics")
def statistics(
    predictions: list[Path] = typer.Option([], "--predictions"),
    prediction_manifest: Path | None = typer.Option(None, "--prediction-manifest"),
    output: Path = typer.Option(...),
    bootstrap_resamples: int = typer.Option(10_000),
    bootstrap_seed: int = typer.Option(2026),
    progress_every: int = typer.Option(500),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    if bool(predictions) == bool(prediction_manifest):
        raise typer.BadParameter(
            "provide exactly one of --predictions or --prediction-manifest"
        )
    arguments: list[str] = []
    for path in predictions:
        arguments.extend(("--predictions", str(path)))
    if prediction_manifest is not None:
        arguments.extend(("--prediction-manifest", str(prediction_manifest)))
    arguments.extend(
        (
            "--output",
            str(output),
            "--bootstrap-resamples",
            str(bootstrap_resamples),
            "--bootstrap-seed",
            str(bootstrap_seed),
            "--progress-every",
            str(progress_every),
        )
    )
    _run_script(_paths(path_config), "run_statistics.py", arguments)


@app.command("token-efficiency-statistics")
def token_efficiency_statistics(
    manifest: Path = typer.Option(...),
    output: Path = typer.Option(...),
    bootstrap_resamples: int = typer.Option(10_000),
    bootstrap_seed: int = typer.Option(2026),
    bootstrap_chunk_size: int = typer.Option(16),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    """Analyse the formal fixed-mask A0/B0 learning curves on local CPU."""

    _run_script(
        _paths(path_config),
        "analyze_token_efficiency.py",
        [
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--bootstrap-resamples",
            str(bootstrap_resamples),
            "--bootstrap-seed",
            str(bootstrap_seed),
            "--bootstrap-chunk-size",
            str(bootstrap_chunk_size),
        ],
    )


@app.command("figures")
def figures(
    results_root: Path = typer.Option(Path("results")),
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    _run_script(_paths(path_config), "make_figures.py", ["--results-root", str(results_root)])


@app.command("paper")
def paper(
    path_config: Path = typer.Option(DEFAULT_PATH_CONFIG),
) -> None:
    _run_script(_paths(path_config), "build_paper.py", [])


if __name__ == "__main__":
    app()
