"""Executable, resumable Paper104 continued-pretraining runner."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from polyfid.data.mlm import (
    OnlineMLMCollator,
    RelationCycle,
    batched,
    encode_mlm_row,
    iter_encoded_mlm_examples,
    iter_pi1m_rows,
    read_relation_rows,
    validate_relation_training_rows,
)
from polyfid.models import arm_spec, build_chemberta
from polyfid.models.projection import RelationProjectionHead
from polyfid.paths import ProjectPaths
from polyfid.records import RunRecord

from .forward import PretrainingBatch, forward_pretraining_batch
from .optimization import build_optimizer_and_scheduler
from .progress import CheckpointSchedule
from .state import (
    TrainingState,
    checkpoint_write_lock,
    load_training_checkpoint,
    save_training_checkpoint,
)


@dataclass
class PretrainingRunConfig:
    arm: str
    seed: int
    learning_rate: float
    pi1m_path: str | Path | Iterable[Mapping[str, Any]]
    model_checkpoint: str | Path
    paths: ProjectPaths
    relation_path: str | Path | None = None
    protected_groups_path: str | Path | None = None
    resume_path: str | Path | None = None
    epochs: int = 3
    max_length: int = 256
    micro_batch_size: int = 32
    gradient_accumulation: int = 8
    mask_probability: float = 0.15
    boundary_fraction: float = 0.65
    interior_fraction: float = 0.25
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    optimizer_name: str = "adamw"
    scheduler_name: str = "linear"
    precision: str = "bf16"
    checkpoint_percentages: tuple[int, ...] = (1, 5, 10, 25, 50, 100)
    shuffle_buffer: int = 4096
    max_train_rows: int | None = None
    max_validation_rows: int | None = None
    stop_after_optimizer_steps: int | None = None
    run_id: str | None = None
    checkpoint_dir: str | Path | None = None
    command: tuple[str, ...] = field(default_factory=lambda: tuple(sys.argv))

    @property
    def effective_batch_size(self) -> int:
        return int(self.micro_batch_size) * int(self.gradient_accumulation)


@dataclass(frozen=True)
class PretrainingRunResult:
    run_id: str
    arm: str
    status: str
    checkpoint_dir: str
    metrics_path: str
    run_record_path: str
    optimizer_steps: int
    non_padding_tokens: int


def _protected_groups(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _rows(
    config: PretrainingRunConfig,
    *,
    split: str,
    epoch: int,
    protected: set[str],
) -> Iterable[dict[str, Any]]:
    return iter_pi1m_rows(
        config.pi1m_path,
        split=split,
        seed=config.seed,
        epoch=epoch,
        protected_repeat_groups=protected,
        shuffle_buffer=config.shuffle_buffer,
        max_rows=(
            config.max_train_rows if split == "train" else config.max_validation_rows
        ),
    )


def _count_tokens_and_rows(
    config: PretrainingRunConfig, tokenizer: Any, protected: set[str]
) -> tuple[int, int, int]:
    """Count exact clean anchor tokens with bounded tokenizer batches."""

    row_count = 0
    token_count = 0
    texts: list[str] = []
    overlength = 0
    for row in _rows(config, split="train", epoch=0, protected=protected):
        texts.append(str(row.get("canonical_psmiles") or row.get("canonical_text") or ""))
        row_count += 1
        if len(texts) < 512:
            continue
        encoded = tokenizer(texts, truncation=False)
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            if len(ids) > config.max_length:
                row_count -= 1
                overlength += 1
            else:
                token_count += sum(int(value) for value in mask)
        texts = []
    if texts:
        encoded = tokenizer(texts, truncation=False)
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            if len(ids) > config.max_length:
                row_count -= 1
                overlength += 1
            else:
                token_count += sum(int(value) for value in mask)
    if row_count == 0:
        raise ValueError("the PI1M training partition is empty")
    return row_count, token_count, overlength


def _to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _batch_to_device(batch: PretrainingBatch, device: Any) -> PretrainingBatch:
    relation = batch.relation_views
    if relation is not None:
        from .batching import RelationViewInputs

        relation = RelationViewInputs(
            anchors=_to_device(relation.anchors, device),
            positives=_to_device(relation.positives, device),
            hard_negatives=(
                None
                if relation.hard_negatives is None
                else _to_device(relation.hard_negatives, device)
            ),
        )
    return PretrainingBatch(
        mlm_inputs=_to_device(batch.mlm_inputs, device),
        labels=_to_device(batch.labels, device),
        relation_views=relation,
        masking_diagnostics=batch.masking_diagnostics,
    )


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if hasattr(value, "to"):
                state[name] = value.to(device)


def _parameter_counts(model: Any, projection: Any | None = None) -> tuple[int, int]:
    modules = (model,) if projection is None else (model, projection)
    total = sum(parameter.numel() for module in modules for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)


def _run_resources(
    *,
    torch: Any,
    device: Any,
    started: float,
    model: Any,
    projection: Any | None,
    cumulative_tokens: int,
    invocation_start_tokens: int,
    config: PretrainingRunConfig,
    **extra: float | int | str,
) -> dict[str, float | int | str]:
    wall_seconds = max(0.0, float(time.time() - started))
    processed_tokens = max(0, int(cumulative_tokens) - int(invocation_start_tokens))
    total_parameters, trainable_parameters = _parameter_counts(model, projection)
    cuda_run = device.type == "cuda" and torch.cuda.is_available()
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if cuda_run
        else 0.0
    )
    return {
        "device": str(device),
        "wall_seconds": wall_seconds,
        "gpu_hours": wall_seconds / 3600.0 if cuda_run else 0.0,
        "peak_cuda_memory_mb": peak_memory_mb,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "tokens_per_second": (
            float(processed_tokens) / wall_seconds if wall_seconds > 0 else 0.0
        ),
        "optimizer": config.optimizer_name,
        "scheduler": config.scheduler_name,
        "precision": config.precision,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "max_length": config.max_length,
        **extra,
    }


def _validation_mlm(
    model: Any,
    tokenizer: Any,
    config: PretrainingRunConfig,
    device: Any,
) -> dict[str, float]:
    import torch

    validation_arm = arm_spec("A0")
    collator = OnlineMLMCollator(
        tokenizer,
        validation_arm,
        seed=config.seed,
        epoch=0,
        max_length=config.max_length,
        mask_probability=config.mask_probability,
        boundary_fraction=config.boundary_fraction,
        interior_fraction=config.interior_fraction,
    )
    loss_sum = 0.0
    supervised_tokens = 0
    model.eval()
    with torch.no_grad():
        skipped: dict[str, int] = {}
        examples = iter_encoded_mlm_examples(
            _rows(config, split="validation", epoch=0, protected=set()),
            tokenizer,
            max_length=config.max_length,
            skipped=skipped,
        )
        for values in batched(examples, config.micro_batch_size):
            batch = _batch_to_device(collator(values), device)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and config.precision == "bf16",
            ):
                loss = forward_pretraining_batch(model, None, batch, validation_arm)
            count = int(batch.labels.ne(-100).sum().item())
            loss_sum += float(loss.mlm.detach().cpu()) * count
            supervised_tokens += count
    model.train()
    return {
        "validation_mlm_loss": loss_sum / max(1, supervised_tokens),
        "validation_supervised_tokens": float(supervised_tokens),
        "validation_overlength_rows": float(skipped.get("overlength", 0)),
    }


def _save_hf_model(
    model: Any,
    tokenizer: Any,
    projection: Any,
    destination: Path,
    *,
    write_lock_path: Path,
) -> None:
    staging_directory = os.environ.get("SLURM_TMPDIR")
    temporary = destination.with_name(f".{destination.name}.tmp")
    with tempfile.TemporaryDirectory(
        prefix=f"polyfid-{destination.name}-",
        dir=staging_directory,
    ) as staging_root:
        staging = Path(staging_root) / destination.name
        staging.mkdir()
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(staging)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(staging)
        if projection is not None:
            import torch

            torch.save(projection.state_dict(), staging / "relation_projection.pt")
        try:
            with checkpoint_write_lock(write_lock_path):
                shutil.copytree(staging, temporary)
                temporary.replace(destination)
        except OSError:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def _rewind_metrics_to_checkpoint(path: Path, state: TrainingState) -> None:
    """Discard metric rows written after the checkpoint being resumed."""

    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        optimizer_step = row.get("optimizer_step")
        if optimizer_step is not None and int(optimizer_step) > state.global_step:
            continue
        if (
            optimizer_step is None
            and "epoch" in row
            and int(row["epoch"]) >= state.epoch
        ):
            continue
        retained.append(line)
    temporary = path.with_name(f".{path.name}.resume.tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    temporary.replace(path)


def run_pretraining(
    config: PretrainingRunConfig,
    *,
    model_build: Any | None = None,
    relation_rows: Sequence[Any] | None = None,
) -> PretrainingRunResult:
    """Run one fixed reference or one fully resumable PI1M training arm."""

    import torch

    arm = arm_spec(config.arm)
    if config.precision != "bf16":
        raise ValueError("Paper104 pretraining currently supports precision=bf16")
    if config.effective_batch_size != 256:
        raise ValueError("Paper104 pretraining requires effective anchor batch 256")
    if (
        config.stop_after_optimizer_steps is not None
        and int(config.stop_after_optimizer_steps) <= 0
    ):
        raise ValueError("stop_after_optimizer_steps must be positive")
    active_relations = relation_rows
    if arm.needs_relation_views and active_relations is None:
        if config.relation_path is None:
            raise ValueError(f"{arm.name} requires --relation-path")
        active_relations = read_relation_rows(config.relation_path)
    if arm.needs_relation_views:
        validate_relation_training_rows(tuple(active_relations or ()))
    run_id = config.run_id or f"{arm.name}_seed-{int(config.seed)}"
    checkpoint_dir = (
        Path(config.checkpoint_dir)
        if config.checkpoint_dir is not None
        else config.paths.checkpoint_root / "pretraining" / arm.name / f"seed-{config.seed}"
    )
    metrics_path = config.paths.project_root / "results/metrics" / f"{run_id}.jsonl"
    run_record_path = config.paths.project_root / "runs" / f"{run_id}.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_lock_path = config.paths.checkpoint_root / ".checkpoint-write.lock"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    run_record_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    random.seed(int(config.seed))
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))

    build = model_build or build_chemberta(
        config.model_checkpoint,
        initialization=arm.initialization,
        seed=config.seed,
        max_length=config.max_length,
    )
    model, tokenizer = build.model, build.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    protected = _protected_groups(config.protected_groups_path)

    if not arm.train_on_pi1m:
        metrics = _validation_mlm(model, tokenizer, config, device)
        metrics_path.write_text(json.dumps({"stage": "reference", **metrics}) + "\n")
        record = RunRecord(
            run_id=run_id,
            stage="pretraining_reference",
            method=arm.name,
            seed=config.seed,
            config_path="configs/pretraining/core.yaml",
            command=config.command,
            input_paths=(str(config.pi1m_path), str(config.model_checkpoint)),
            output_path=str(config.model_checkpoint),
            status="fixed_reference",
            metrics=metrics,
            resources=_run_resources(
                torch=torch,
                device=device,
                started=started,
                model=model,
                projection=None,
                cumulative_tokens=0,
                invocation_start_tokens=0,
                config=config,
                optimizer_steps=0,
                non_padding_tokens=0,
                effective_batch_size=config.effective_batch_size,
            ),
        )
        run_record_path.write_text(
            json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return PretrainingRunResult(
            run_id, arm.name, "fixed_reference", str(checkpoint_dir),
            str(metrics_path), str(run_record_path), 0, 0
        )

    row_count, tokens_per_epoch, train_overlength = _count_tokens_and_rows(
        config, tokenizer, protected
    )
    micro_batches_per_epoch = math.ceil(row_count / config.micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / config.gradient_accumulation
    )
    total_optimizer_steps = optimizer_steps_per_epoch * int(config.epochs)
    total_non_padding_tokens = tokens_per_epoch * int(config.epochs)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    projection = (
        RelationProjectionHead(hidden_size, hidden_size, 256).to(device)
        if arm.needs_relation_views
        else None
    )
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        learning_rate=config.learning_rate,
        total_optimizer_steps=total_optimizer_steps,
        projection=projection,
        weight_decay=config.weight_decay,
        warmup_fraction=config.warmup_fraction,
        optimizer_name=config.optimizer_name,
        scheduler_name=config.scheduler_name,
    )
    optimized_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    state = TrainingState(arm.name, config.seed)
    schedule = CheckpointSchedule(
        total_non_padding_tokens,
        percentages=config.checkpoint_percentages,
    )
    if config.resume_path is not None:
        state, schedule_values = load_training_checkpoint(
            config.resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            projection=projection,
            map_location="cpu",
        )
        if state.arm != arm.name or state.seed != config.seed:
            raise ValueError("resume checkpoint belongs to a different arm or seed")
        if schedule_values is not None:
            schedule = CheckpointSchedule.from_dict(schedule_values)
        model.to(device)
        if projection is not None:
            projection.to(device)
        _optimizer_to_device(optimizer, device)
        _rewind_metrics_to_checkpoint(metrics_path, state)
    invocation_start_tokens = state.non_padding_tokens

    relation_cycle = (
        RelationCycle(
            tuple(active_relations or ()),
            objective=arm.relation_objective,
            seed=config.seed,
        )
        if arm.needs_relation_views
        else None
    )
    if relation_cycle is not None:
        prior_rows = min(state.epoch, config.epochs) * row_count
        current_steps = max(0, state.global_step - state.epoch * optimizer_steps_per_epoch)
        current_batches = min(
            micro_batches_per_epoch, current_steps * config.gradient_accumulation
        )
        relation_cycle.position = prior_rows + min(
            row_count, current_batches * config.micro_batch_size
        )

    metrics_file = metrics_path.open("a" if config.resume_path else "w", encoding="utf-8")
    stopped_early = False
    if config.resume_path is None:
        metrics_file.write(
            json.dumps(
                {
                    "stage": "data_census",
                    "train_rows": row_count,
                    "train_overlength_rows": train_overlength,
                    "tokens_per_epoch": tokens_per_epoch,
                }
            )
            + "\n"
        )
        metrics_file.flush()
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(state.epoch, int(config.epochs)):
            completed_steps = max(
                0, state.global_step - epoch * optimizer_steps_per_epoch
            ) if epoch == state.epoch else 0
            skip_batches = min(
                micro_batches_per_epoch,
                completed_steps * config.gradient_accumulation,
            )
            collator = OnlineMLMCollator(
                tokenizer,
                arm,
                seed=config.seed,
                epoch=epoch,
                max_length=config.max_length,
                mask_probability=config.mask_probability,
                boundary_fraction=config.boundary_fraction,
                interior_fraction=config.interior_fraction,
                relation_cycle=relation_cycle,
            )
            raw_rows = _rows(config, split="train", epoch=epoch, protected=protected)
            skipped_rows: dict[str, int] = {}
            accumulated_masking: Counter[str] = Counter()
            examples = iter_encoded_mlm_examples(
                raw_rows,
                tokenizer,
                max_length=config.max_length,
                skipped=skipped_rows,
            )
            for batch_index, values in enumerate(batched(examples, config.micro_batch_size)):
                if batch_index < skip_batches:
                    continue
                batch = _batch_to_device(collator(values), device)
                accumulated_masking.update(batch.masking_diagnostics or {})
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and config.precision == "bf16",
                ):
                    loss = forward_pretraining_batch(model, projection, batch, arm)
                    group_start = (
                        batch_index // config.gradient_accumulation
                    ) * config.gradient_accumulation
                    accumulation_width = min(
                        config.gradient_accumulation,
                        micro_batches_per_epoch - group_start,
                    )
                    scaled_loss = loss.total / accumulation_width
                scaled_loss.backward()
                state.non_padding_tokens += int(batch.mlm_inputs["attention_mask"].sum().item())
                final_batch = batch_index + 1 == micro_batches_per_epoch
                update = (
                    (batch_index + 1) % config.gradient_accumulation == 0
                    or final_batch
                )
                if not update:
                    continue
                # Relation arms optimize the projection together with the
                # backbone, so both must participate in the same clipping
                # step.
                torch.nn.utils.clip_grad_norm_(optimized_parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1
                metric_row = {
                    "epoch": epoch,
                    "optimizer_step": state.global_step,
                    "non_padding_tokens": state.non_padding_tokens,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    **loss.detached_metrics(),
                    **dict(accumulated_masking),
                }
                metrics_file.write(json.dumps(metric_row) + "\n")
                metrics_file.flush()
                accumulated_masking.clear()
                for percentage in schedule.newly_reached(state.non_padding_tokens):
                    save_training_checkpoint(
                        checkpoint_dir / f"token-{percentage:03d}.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        projection=projection,
                        state=state,
                        checkpoint_schedule=schedule,
                        write_lock_path=write_lock_path,
                    )
                if (
                    config.stop_after_optimizer_steps is not None
                    and state.global_step >= int(config.stop_after_optimizer_steps)
                    and state.global_step < total_optimizer_steps
                ):
                    save_training_checkpoint(
                        checkpoint_dir / "last.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        projection=projection,
                        state=state,
                        checkpoint_schedule=schedule,
                        write_lock_path=write_lock_path,
                    )
                    stopped_early = True
                    break
            if stopped_early:
                break
            state.epoch = epoch + 1
            if skipped_rows.get("overlength", 0):
                metrics_file.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_overlength_rows": skipped_rows["overlength"],
                        }
                    )
                    + "\n"
                )
            validation = _validation_mlm(model, tokenizer, config, device)
            metrics_file.write(json.dumps({"epoch": epoch, **validation}) + "\n")
            metrics_file.flush()
            save_training_checkpoint(
                checkpoint_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                projection=projection,
                state=state,
                checkpoint_schedule=schedule,
                write_lock_path=write_lock_path,
            )
    finally:
        metrics_file.close()

    if stopped_early:
        partial_metrics = {
            "optimizer_steps": float(state.global_step),
            "non_padding_tokens": float(state.non_padding_tokens),
        }
        record = RunRecord(
            run_id=run_id,
            stage="continued_pretraining",
            method=arm.name,
            seed=config.seed,
            config_path="configs/pretraining/core.yaml",
            command=config.command,
            input_paths=tuple(
                str(value)
                for value in (
                    config.pi1m_path,
                    config.model_checkpoint,
                    config.relation_path,
                )
                if value is not None
            ),
            output_path=str(checkpoint_dir / "last.pt"),
            status="partial",
            metrics=partial_metrics,
            resources=_run_resources(
                torch=torch,
                device=device,
                started=started,
                model=model,
                projection=projection,
                cumulative_tokens=state.non_padding_tokens,
                invocation_start_tokens=invocation_start_tokens,
                config=config,
                optimizer_steps=state.global_step,
                non_padding_tokens=state.non_padding_tokens,
                effective_batch_size=config.effective_batch_size,
            ),
        )
        run_record_path.write_text(
            json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return PretrainingRunResult(
            run_id=run_id,
            arm=arm.name,
            status="partial",
            checkpoint_dir=str(checkpoint_dir),
            metrics_path=str(metrics_path),
            run_record_path=str(run_record_path),
            optimizer_steps=state.global_step,
            non_padding_tokens=state.non_padding_tokens,
        )

    final_model = checkpoint_dir / "final_model"
    _save_hf_model(
        model,
        tokenizer,
        projection,
        final_model,
        write_lock_path=write_lock_path,
    )
    final_metrics = _validation_mlm(model, tokenizer, config, device)
    record = RunRecord(
        run_id=run_id,
        stage="continued_pretraining",
        method=arm.name,
        seed=config.seed,
        config_path="configs/pretraining/core.yaml",
        command=config.command,
        input_paths=tuple(
            str(value)
            for value in (
                config.pi1m_path,
                config.model_checkpoint,
                config.relation_path,
            )
            if value is not None
        ),
        output_path=str(final_model),
        status="complete",
        metrics=final_metrics,
        resources=_run_resources(
            torch=torch,
            device=device,
            started=started,
            model=model,
            projection=projection,
            cumulative_tokens=state.non_padding_tokens,
            invocation_start_tokens=invocation_start_tokens,
            config=config,
            optimizer_steps=state.global_step,
            non_padding_tokens=state.non_padding_tokens,
            effective_batch_size=config.effective_batch_size,
            boundary_fraction=config.boundary_fraction,
            interior_fraction=config.interior_fraction,
            train_overlength_rows_per_epoch=train_overlength,
        ),
    )
    run_record_path.write_text(
        json.dumps(record.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return PretrainingRunResult(
        run_id=run_id,
        arm=arm.name,
        status="complete",
        checkpoint_dir=str(checkpoint_dir),
        metrics_path=str(metrics_path),
        run_record_path=str(run_record_path),
        optimizer_steps=state.global_step,
        non_padding_tokens=state.non_padding_tokens,
    )


__all__ = ["PretrainingRunConfig", "PretrainingRunResult", "run_pretraining"]
