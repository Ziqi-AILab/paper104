"""Thin same-split runner around the official TransPolymer release."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polyfid.baselines.restoration import (
    verify_prediction_roundtrip,
    write_retained_description,
)


@dataclass(frozen=True)
class TransPolymerRunConfig:
    epochs: int = 20
    patience: int = 5
    batch_size: int = 32
    max_length: int = 512
    encoder_learning_rate: float = 5e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    dropout: float = 0.1
    num_workers: int = 0
    use_float64: bool = True


def _patch_upstream_tokenizer(tokenizer_class: type) -> None:
    """Bridge the official 4.20-era tokenizer to the pinned Transformers API.

    The upstream tokenizer initializes its vocabulary after the parent class,
    while newer Transformers queries it during parent initialization.  Giving
    that early query an empty vocabulary and restoring the true vocabulary
    length reproduces the official token IDs without changing tokenization.
    """

    tokenizer_class.encoder = {}

    def vocabulary_length(instance: Any) -> int:
        return instance.vocab_size + sum(
            int(index) >= instance.vocab_size
            for index in instance.added_tokens_encoder.values()
        )

    tokenizer_class.__len__ = vocabulary_length


def load_transpolymer_tokenizer(
    model_root: str | Path, *, max_length: int = 512
) -> Any:
    from transformers import logging as transformers_logging

    root = Path(model_root) / "external_baselines/transpolymer"
    source = root / "source"
    tokenizer_path = root / "tokenizer/roberta-base"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from PolymerSmilesTokenization import PolymerSmilesTokenizer

    _patch_upstream_tokenizer(PolymerSmilesTokenizer)
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        tokenizer = PolymerSmilesTokenizer.from_pretrained(
            str(tokenizer_path), max_len=int(max_length), local_files_only=True
        )
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    if len(tokenizer) != tokenizer.vocab_size:
        raise RuntimeError("unexpected TransPolymer tokenizer vocabulary size")
    return tokenizer


def load_transpolymer_encoder(model_root: str | Path) -> Any:
    from transformers import RobertaModel

    checkpoint = Path(model_root) / "external_baselines/transpolymer/checkpoint"
    return RobertaModel.from_pretrained(
        str(checkpoint), local_files_only=True, add_pooling_layer=False
    )


def probe_transpolymer_load(model_root: str | Path) -> dict[str, Any]:
    """Load the real local tokenizer/checkpoint and report architecture facts."""

    tokenizer = load_transpolymer_tokenizer(model_root, max_length=32)
    encoder = load_transpolymer_encoder(model_root)
    token_ids = tokenizer("*CC*", add_special_tokens=True)["input_ids"]
    return {
        "tokenizer_vocabulary_size": int(len(tokenizer)),
        "hidden_size": int(encoder.config.hidden_size),
        "encoder_layers": int(encoder.config.num_hidden_layers),
        "representative_token_count": int(len(token_ids)),
        "checkpoint_loaded": True,
    }


class _TextDataset:
    def __new__(cls, frame: pd.DataFrame, tokenizer: Any, maximum_length: int) -> Any:
        import torch
        from torch.utils.data import Dataset

        class DatasetImpl(Dataset):
            def __init__(self) -> None:
                self.frame = frame.reset_index(drop=True)

            def __len__(self) -> int:
                return len(self.frame)

            def __getitem__(self, index: int) -> dict[str, Any]:
                row = self.frame.iloc[index]
                encoded = tokenizer(
                    str(row["smiles"]),
                    add_special_tokens=True,
                    max_length=int(maximum_length),
                    return_token_type_ids=False,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoded["input_ids"].flatten(),
                    "attention_mask": encoded["attention_mask"].flatten(),
                    "value": torch.tensor(float(row["scaled_value"])),
                }

        return DatasetImpl()


def _regressor(encoder: Any, dropout: float) -> Any:
    from torch import nn

    class OfficialRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.PretrainedModel = copy.deepcopy(encoder)
            hidden = int(self.PretrainedModel.config.hidden_size)
            self.Regressor = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            output = self.PretrainedModel(
                input_ids=input_ids, attention_mask=attention_mask
            )
            return self.Regressor(output.last_hidden_state[:, 0, :])

    return OfficialRegressor()


def _predict(model: Any, loader: Any, device: Any, scaler: Any) -> np.ndarray:
    import torch

    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            )
            values.append(output.detach().cpu().numpy().reshape(-1, 1))
    return scaler.inverse_transform(np.concatenate(values, axis=0)).reshape(-1)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_transpolymer_fold(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    model_root: str | Path,
    seed: int,
    config: TransPolymerRunConfig = TransPolymerRunConfig(),
    checkpoint_dir: str | Path | None = None,
    checkpoint_retention: str = "trainable",
    require_a800: bool = True,
) -> Path:
    """Fine-tune the official checkpoint on one Paper104 fold."""

    import torch
    from sklearn.preprocessing import StandardScaler
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError("formal TransPolymer fine-tuning requires an A800 GPU job")
    device = torch.device("cuda")
    if require_a800 and "A800" not in torch.cuda.get_device_name(device).upper():
        raise RuntimeError("formal TransPolymer fine-tuning must run on an A800")

    _seed_everything(int(seed))
    source = Path(dataset_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_destination = (
        destination if checkpoint_dir is None else Path(checkpoint_dir)
    )
    checkpoint_destination.mkdir(parents=True, exist_ok=True)
    tables = {
        split: pd.read_csv(source / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    scaler = StandardScaler().fit(tables["train"][["value"]])
    for frame in tables.values():
        frame["scaled_value"] = scaler.transform(frame[["value"]]).reshape(-1)

    tokenizer = load_transpolymer_tokenizer(model_root, max_length=config.max_length)
    encoder = load_transpolymer_encoder(model_root)
    model = _regressor(encoder, config.dropout).to(device)
    if config.use_float64:
        model = model.double()
    torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator().manual_seed(int(seed))
    loaders = {
        split: DataLoader(
            _TextDataset(frame, tokenizer, config.max_length),
            batch_size=int(config.batch_size),
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=int(config.num_workers),
        )
        for split, frame in tables.items()
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.PretrainedModel.parameters(),
                "lr": float(config.encoder_learning_rate),
                "weight_decay": 0.0,
            },
            {
                "params": model.Regressor.parameters(),
                "lr": float(config.head_learning_rate),
                "weight_decay": float(config.weight_decay),
            },
        ]
    )
    total_steps = max(1, int(config.epochs) * len(loaders["train"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config.warmup_ratio)),
        num_training_steps=total_steps,
    )
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    training_started = time.perf_counter()
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device)
            ).reshape(-1)
            target = batch["value"].to(device=device, dtype=prediction.dtype)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
        validation = _predict(model, loaders["validation"], device, scaler)
        validation_loss = float(
            np.mean(
                np.square(
                    validation
                    - tables["validation"]["value"].to_numpy(dtype=float)
                )
            )
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        if epoch - best_epoch >= int(config.patience):
            break
    training_seconds = time.perf_counter() - training_started
    if best_state is None:
        raise RuntimeError("TransPolymer did not produce a validation-selected state")
    model.load_state_dict(best_state)

    inference_started = time.perf_counter()
    outputs = {
        split: _predict(model, loaders[split], device, scaler)
        for split in ("validation", "test")
    }
    inference_seconds = time.perf_counter() - inference_started
    prediction_path = destination / "backend_predictions.csv"
    pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": tables[split]["sample_id"].astype(str),
                    "split": split,
                    "prediction": outputs[split],
                }
            )
            for split in ("validation", "test")
        ],
        ignore_index=True,
    ).to_csv(prediction_path, index=False)
    if checkpoint_retention not in {"trainable", "metadata"}:
        raise ValueError("checkpoint_retention must be trainable or metadata")
    selected_tensor_bytes = int(
        sum(value.numel() * value.element_size() for value in best_state.values())
    )
    if checkpoint_retention == "trainable":
        model_artifact = checkpoint_destination / "transpolymer_best.pt"
        target_scaler = {
            "kind": "standard",
            "mean": float(scaler.mean_[0]),
            "std": float(scaler.scale_[0]),
        }
        torch.save(
            {
                "model": best_state,
                "best_epoch": int(best_epoch),
                "best_validation_mse": float(best_loss),
                "seed": int(seed),
                "config": asdict(config),
                "target_scaler": target_scaler,
            },
            model_artifact,
        )
        write_retained_description(
            checkpoint_destination, config=asdict(config), target_scaler=target_scaler,
        )
        restored = torch.load(model_artifact, map_location="cpu", weights_only=False)
        model.load_state_dict(restored["model"])
        scaler.mean_ = np.asarray([restored["target_scaler"]["mean"]])
        scaler.scale_ = np.asarray([restored["target_scaler"]["std"]])
        verify_prediction_roundtrip(
            outputs["test"], _predict(model, loaders["test"], device, scaler),
            checkpoint_destination / "reload_check.json",
        )
        del restored
    else:
        model_artifact = checkpoint_destination / "model_metadata.json"
        model_artifact.write_text(
            json.dumps(
                {
                    "artifact_type": "transpolymer_model_metadata",
                    "checkpoint_retention": "metadata",
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_validation_mse": float(best_loss),
                    "validation_selected_tensor_bytes": selected_tensor_bytes,
                    "source_checkpoint": str(
                        Path(model_root) / "external_baselines/transpolymer/checkpoint"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    model_output_bytes = int(model_artifact.stat().st_size)
    report = {
        "backend": "official TransPolymer tokenizer, pretrained encoder, and regressor",
        "protocol": "Paper104 fixed split; checkpoint selected on validation only",
        "seed": int(seed),
        "config": asdict(config),
        "best_epoch": int(best_epoch),
        "best_validation_mse": float(best_loss),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameters": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "checkpoint_retention": checkpoint_retention,
        "validation_selected_tensor_bytes": selected_tensor_bytes,
        "model_output_bytes": model_output_bytes,
        "model_artifact": str(model_artifact),
        "prediction_path": str(prediction_path),
    }
    (destination / "run.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return prediction_path


__all__ = [
    "TransPolymerRunConfig",
    "load_transpolymer_encoder",
    "load_transpolymer_tokenizer",
    "probe_transpolymer_load",
    "run_transpolymer_fold",
]
