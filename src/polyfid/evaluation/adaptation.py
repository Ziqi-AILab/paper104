"""Same-split supervised adaptation for the selected Paper104 tasks.

This module keeps split handling shared between LoRA, full fine-tuning, and
the external baselines.  The training functions accept already constructed
models and tokenizers so the CPU tests can exercise the complete optimization
path without downloading or instantiating ChemBERTa.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from polyfid.records import PredictionRecord


AdaptationMethod = Literal["lora", "full"]
TaskType = Literal["regression", "binary"]

LORA_ARMS = frozenset({"B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs"})
FULL_ARMS = frozenset({"A0", "A_eqs"})
FULL_TASKS = frozenset({"EA", "EGC"})


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError("downstream tables must be parquet or CSV")


def _task_rows(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    if "task" not in frame:
        return frame
    return frame.loc[
        frame["task"].astype(str).str.casefold().eq(str(task).casefold())
    ].copy()


def _role_rows(roles: pd.DataFrame, *, task: str, protocol: str, fold: int) -> pd.DataFrame:
    roles = _task_rows(roles, task)
    if "protocol" in roles:
        roles = roles.loc[roles["protocol"].astype(str).eq(str(protocol))]
    if "evaluation_fold" in roles:
        roles = roles.loc[roles["evaluation_fold"].astype(int).eq(int(fold))]
    elif "fold" in roles:
        roles = roles.loc[roles["fold"].astype(int).eq(int(fold))]
    return roles.copy()


def _preferred_series(frame: pd.DataFrame, names: Sequence[str], default: Any) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(default, index=frame.index)


def prepare_downstream_frame(
    dataset: str | Path | pd.DataFrame,
    split_manifest: str | Path | pd.DataFrame,
    *,
    task: str,
    protocol: str,
    fold: int,
    label_column: str,
    text_column: str,
) -> pd.DataFrame:
    """Join a standardized endpoint table to one declared evaluation fold.

    Roles from the manifest take precedence over any source-table split.  A
    normalized ``group_id`` is returned and checked for disjoint train,
    validation, and test membership.
    """

    records = dataset.copy() if isinstance(dataset, pd.DataFrame) else _read_table(dataset)
    roles = (
        split_manifest.copy()
        if isinstance(split_manifest, pd.DataFrame)
        else _read_table(split_manifest)
    )
    records = _task_rows(records, task)
    roles = _role_rows(roles, task=task, protocol=protocol, fold=fold)
    if records.empty or roles.empty:
        raise ValueError(f"no rows for task={task}, protocol={protocol}, fold={fold}")

    join_keys = ["sample_id"]
    if "task" in records and "task" in roles:
        join_keys.append("task")
    if roles.duplicated(join_keys).any():
        raise ValueError("the selected manifest has duplicate sample roles")

    role_values = roles.copy()
    role_values = role_values.rename(
        columns={
            column: f"role_{column}"
            for column in role_values.columns
            if column not in join_keys
        }
    )
    frame = records.merge(role_values, on=join_keys, how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("the endpoint table and split manifest do not share samples")
    if label_column not in frame:
        raise ValueError(f"label column {label_column!r} is unavailable")
    if text_column not in frame:
        raise ValueError(f"text column {text_column!r} is unavailable")

    frame["split"] = _preferred_series(frame, ("role_split", "split"), "")
    frame["group_id"] = _preferred_series(
        frame,
        (
            "role_monomer_pair_group_id",
            "role_group_id",
            "monomer_pair_group_id",
            "group_id",
        ),
        frame["sample_id"].astype(str),
    ).astype(str)
    frame["family_id"] = _preferred_series(
        frame, ("role_family_id", "family_id"), None
    )
    frame["cluster_id"] = _preferred_series(
        frame, ("role_cluster_id", "cluster_id"), None
    )
    frame["structural_fidelity_eligible"] = _preferred_series(
        frame,
        (
            "role_structural_fidelity_eligible",
            "structural_fidelity_eligible",
            "role_repeat_eligible",
            "repeat_eligible",
        ),
        None,
    )
    frame["scope_class"] = _preferred_series(
        frame, ("role_scope_class", "scope_class"), None
    )
    frame["evaluation_fold"] = int(fold)
    frame["label"] = pd.to_numeric(frame[label_column], errors="coerce")
    frame["model_text"] = frame[text_column].astype(str)
    frame = frame.loc[frame["label"].notna() & frame["model_text"].ne("")].copy()

    required_roles = {"train", "validation", "test"}
    role_names = set(frame["split"].astype(str))
    if not required_roles.issubset(role_names):
        counts = frame["split"].value_counts().to_dict()
        raise ValueError(f"evaluation fold lacks a required split: {counts}")
    groups = {
        role: set(frame.loc[frame["split"].eq(role), "group_id"].astype(str))
        for role in sorted(required_roles)
    }
    if any(
        groups[left].intersection(groups[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("structure groups cross train, validation, or test roles")
    return frame.reset_index(drop=True)


class LoRALinear:
    """Factory namespace used to avoid importing torch at module import time."""

    @staticmethod
    def wrap(linear: Any, *, rank: int, alpha: float, dropout: float) -> Any:
        import torch
        from torch import nn

        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        class _LoRALinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = linear
                for parameter in self.base.parameters():
                    parameter.requires_grad_(False)
                self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
                self.lora_b = nn.Linear(rank, linear.out_features, bias=False)
                self.dropout = nn.Dropout(dropout)
                self.scale = float(alpha) / float(rank)
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_b.weight)

            def forward(self, inputs: Any) -> Any:
                return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scale

        return _LoRALinear()


def _set_child(module: Any, dotted_name: str, child: Any) -> None:
    parent = module
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


def inject_attention_lora(
    model: Any,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_names: Sequence[str] = ("query", "value"),
) -> tuple[str, ...]:
    """Replace the query/value linear layers inside attention blocks."""

    from torch import nn

    replacements: list[tuple[str, Any]] = []
    targets = set(target_names)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.rsplit(".", 1)[-1] not in targets or "attention" not in name:
            continue
        replacements.append(
            (name, LoRALinear.wrap(module, rank=rank, alpha=alpha, dropout=dropout))
        )
    for name, replacement in replacements:
        _set_child(model, name, replacement)
    if not replacements:
        raise ValueError("no attention query/value linear layers were found for LoRA")
    return tuple(name for name, _ in replacements)


def _encoder_from_mlm(model: Any) -> Any:
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def build_adaptation_model(
    checkpoint: str | Path,
    *,
    base_checkpoint: str | Path | None = None,
    method: AdaptationMethod,
    task_type: TaskType,
    rank: int = 8,
    alpha: float = 16.0,
    lora_dropout: float = 0.05,
    head_dropout: float = 0.1,
    local_files_only: bool = True,
) -> tuple[Any, Any, tuple[str, ...]]:
    """Load a Paper104 checkpoint and attach the common two-layer head."""

    import torch
    from torch import nn
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    checkpoint_path = Path(checkpoint)
    model_source = (
        checkpoint_path
        if checkpoint_path.is_dir() and (checkpoint_path / "config.json").is_file()
        else Path(base_checkpoint or checkpoint)
    )
    tokenizer_source = (
        checkpoint_path
        if checkpoint_path.is_dir() and (checkpoint_path / "tokenizer.json").is_file()
        else Path(base_checkpoint or checkpoint)
    )
    if checkpoint_path.is_file() and base_checkpoint is None:
        raise ValueError("a file-style training checkpoint requires base_checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, local_files_only=local_files_only, use_fast=True
    )
    mlm = AutoModelForMaskedLM.from_pretrained(
        model_source, local_files_only=local_files_only
    )
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        mlm.load_state_dict(payload["model"])
    encoder = _encoder_from_mlm(mlm)
    hidden_size = int(getattr(mlm.config, "hidden_size", 768))

    class _PropertyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(256, 1),
            )

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            special_tokens_mask: Any | None = None,
        ) -> Any:
            result = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = result.last_hidden_state
            valid = attention_mask.bool()
            if special_tokens_mask is not None:
                valid = valid & ~special_tokens_mask.bool()
            weights = valid.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            return self.head(pooled).squeeze(-1)

    property_model = _PropertyModel()
    replaced: tuple[str, ...] = ()
    if method == "lora":
        for parameter in property_model.encoder.parameters():
            parameter.requires_grad_(False)
        replaced = inject_attention_lora(
            property_model.encoder,
            rank=rank,
            alpha=alpha,
            dropout=lora_dropout,
        )
    elif method == "full":
        for parameter in property_model.encoder.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError("adaptation method must be lora or full")
    for parameter in property_model.head.parameters():
        parameter.requires_grad_(True)
    return property_model, tokenizer, replaced


@dataclass(frozen=True)
class AdaptationConfig:
    method: AdaptationMethod
    task_type: TaskType = "regression"
    seed: int = 11
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_length: int = 256
    max_epochs: int = 100
    patience: int = 15
    device: str = "cuda"


@dataclass(frozen=True)
class AdaptationResult:
    predictions: np.ndarray
    best_validation_loss: float
    completed_epochs: int
    trainable_parameters: int
    wall_seconds: float
    peak_cuda_memory_bytes: int | None


class _TextDataset:
    def __init__(self, texts: Sequence[str], targets: Sequence[float]) -> None:
        self.texts = list(texts)
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, float]:
        return self.texts[index], float(self.targets[index])


def _collator(tokenizer: Any, *, max_length: int):
    import torch

    def collate(rows: Sequence[tuple[str, float]]) -> dict[str, Any]:
        texts, targets = zip(*rows, strict=True)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        encoded["targets"] = torch.as_tensor(targets, dtype=torch.float32)
        return encoded

    return collate


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in batch.items()}


def _predict(model: Any, loader: Any, device: Any, *, task_type: TaskType, center: float, scale: float) -> np.ndarray:
    import torch

    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for raw in loader:
            batch = _move(raw, device)
            batch.pop("targets")
            output = model(**batch)
            if task_type == "binary":
                output = torch.sigmoid(output)
            else:
                output = output * scale + center
            values.append(output.float().cpu().numpy())
    return np.concatenate(values) if values else np.empty(0, dtype=float)


def train_adaptation(
    frame: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    config: AdaptationConfig,
) -> AdaptationResult:
    """Optimize one adapter/head and return predictions in test row order."""

    import torch
    from torch import nn

    started = time.time()
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    device = torch.device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    train = frame.loc[frame["split"].eq("train")]
    validation = frame.loc[frame["split"].eq("validation")]
    test = frame.loc[frame["split"].eq("test")]
    if config.task_type == "regression":
        center = float(train["label"].mean())
        scale = float(max(train["label"].std(ddof=0), 1e-8))
        target = lambda values: (np.asarray(values, dtype=float) - center) / scale
        loss_function: Any = nn.MSELoss()
    else:
        center, scale = 0.0, 1.0
        target = lambda values: np.asarray(values, dtype=float)
        loss_function = nn.BCEWithLogitsLoss()

    generator = torch.Generator().manual_seed(int(config.seed))
    make_loader = lambda part, shuffle: torch.utils.data.DataLoader(
        _TextDataset(part["model_text"].tolist(), target(part["label"])),
        batch_size=int(config.batch_size),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=_collator(tokenizer, max_length=config.max_length),
    )
    train_loader = make_loader(train, True)
    validation_loader = make_loader(validation, False)
    test_loader = make_loader(test, False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    completed = 0
    for epoch in range(int(config.max_epochs)):
        model.train()
        for raw in train_loader:
            batch = _move(raw, device)
            targets = batch.pop("targets")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss = loss_function(model(**batch), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        model.eval()
        validation_sum = 0.0
        validation_count = 0
        with torch.inference_mode():
            for raw in validation_loader:
                batch = _move(raw, device)
                targets = batch.pop("targets")
                loss = loss_function(model(**batch), targets)
                validation_sum += float(loss.detach().cpu()) * len(targets)
                validation_count += len(targets)
        validation_loss = validation_sum / max(1, validation_count)
        completed = epoch + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    if best_state is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in best_state:
                    parameter.copy_(best_state[name].to(parameter.device))
    predictions = _predict(
        model,
        test_loader,
        device,
        task_type=config.task_type,
        center=center,
        scale=scale,
    )
    return AdaptationResult(
        predictions=predictions,
        best_validation_loss=best_loss,
        completed_epochs=completed,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        wall_seconds=time.time() - started,
        peak_cuda_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    )


def validate_adaptation_scope(*, arm: str, task: str, protocol: str, method: AdaptationMethod) -> None:
    normalized_task = str(task).upper()
    if normalized_task in {"EA", "IP"}:
        if protocol != "family_ood":
            raise ValueError("EA/IP adaptation is predeclared only for family_ood")
    elif normalized_task in {"EGC", "XC"}:
        if protocol != "group_ood":
            raise ValueError("Egc/Xc adaptation is predeclared only for group_ood")
    else:
        raise ValueError("adaptation is limited to EA, IP, Egc, and Xc")
    if method == "lora" and arm not in LORA_ARMS:
        raise ValueError(f"LoRA arm {arm!r} is outside the predeclared matrix")
    if method == "full" and (arm not in FULL_ARMS or normalized_task not in FULL_TASKS):
        raise ValueError("full fine-tuning is limited to A0/A_eqs on EA/Egc")


def prediction_records(
    test: pd.DataFrame,
    predictions: Sequence[float],
    *,
    task: str,
    method: str,
    arm: str,
    protocol: str,
    fold: int,
    representation_seed: int,
    adaptation_seed: int,
    representation_path: str,
    label_fraction: float = 1.0,
    subsample_id: int = 0,
    input_view: str = "valid",
    checkpoint_fraction: float | None = None,
    checkpoint_tokens: int | None = None,
) -> list[PredictionRecord]:
    if len(test) != len(predictions):
        raise ValueError("prediction count does not match held-out rows")
    records: list[PredictionRecord] = []
    for row, prediction in zip(test.to_dict(orient="records"), predictions, strict=True):
        eligibility = row.get(
            "structural_fidelity_eligible", row.get("repeat_eligible")
        )
        if eligibility is not None and not pd.isna(eligibility):
            eligibility = bool(eligibility)
        else:
            eligibility = None
        scope_class = row.get("scope_class")
        if scope_class is not None and pd.isna(scope_class):
            scope_class = None
        row_checkpoint_fraction = row.get("checkpoint_fraction", checkpoint_fraction)
        if row_checkpoint_fraction is not None and pd.isna(row_checkpoint_fraction):
            row_checkpoint_fraction = checkpoint_fraction
        row_checkpoint_tokens = row.get("checkpoint_tokens", checkpoint_tokens)
        if row_checkpoint_tokens is not None and pd.isna(row_checkpoint_tokens):
            row_checkpoint_tokens = checkpoint_tokens
        records.append(
            PredictionRecord(
                sample_id=str(row["sample_id"]),
                task=str(task),
                split="test",
                group_id=str(row["group_id"]),
                y_true=float(row["label"]),
                y_pred=float(prediction),
                method=f"{arm}_{method}",
                representation_seed=int(representation_seed),
                head_seed=int(adaptation_seed),
                protocol=str(protocol),
                family_id=None if pd.isna(row.get("family_id")) else str(row.get("family_id")),
                cluster_id=None if pd.isna(row.get("cluster_id")) else str(row.get("cluster_id")),
                fold=str(fold),
                label_fraction=float(label_fraction),
                subsample_id=int(subsample_id),
                input_view=str(input_view),
                representation_path=str(representation_path),
                structural_fidelity_eligible=eligibility,
                scope_class=None if scope_class is None else str(scope_class),
                checkpoint_fraction=(
                    None
                    if row_checkpoint_fraction is None
                    else float(row_checkpoint_fraction)
                ),
                checkpoint_tokens=(
                    None if row_checkpoint_tokens is None else int(row_checkpoint_tokens)
                ),
            )
        )
    return records


def trainable_state(model: Any) -> dict[str, Any]:
    """Return the learned adaptation tensors for a compact restorable output."""

    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


__all__ = [
    "AdaptationConfig",
    "AdaptationResult",
    "FULL_ARMS",
    "FULL_TASKS",
    "LORA_ARMS",
    "build_adaptation_model",
    "inject_attention_lora",
    "prediction_records",
    "prepare_downstream_frame",
    "train_adaptation",
    "trainable_state",
    "validate_adaptation_scope",
]
