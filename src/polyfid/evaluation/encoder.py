"""ChemBERTa embedding extraction for formal A800 evaluation jobs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EmbeddingTable


def load_encoder_checkpoint(
    checkpoint: str | Path,
    *,
    base_checkpoint: str | Path | None = None,
    device: str = "cuda",
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    path = Path(checkpoint)
    tokenizer_source = path if (path / "tokenizer.json").is_file() else Path(base_checkpoint or checkpoint)
    model_source = path if (path / "config.json").is_file() else Path(base_checkpoint or checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(model_source, local_files_only=True)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, tokenizer


def extract_embeddings(
    model: Any,
    tokenizer: Any,
    sample_ids: Sequence[str],
    texts: Sequence[str],
    *,
    method: str,
    representation_seed: int,
    input_view: str = "valid",
    max_length: int = 256,
    batch_size: int = 128,
    device: str = "cuda",
) -> EmbeddingTable:
    table = extract_hidden_layer_embeddings(
        model,
        tokenizer,
        sample_ids,
        texts,
        layers=(-1,),
        method=method,
        representation_seed=representation_seed,
        input_view=input_view,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )[-1]
    # A single final representation cache records the scientific input view,
    # while multi-layer diagnostic caches retain their explicit layer suffix.
    return replace(table, input_view=str(input_view))


def extract_hidden_layer_embeddings(
    model: Any,
    tokenizer: Any,
    sample_ids: Sequence[str],
    texts: Sequence[str],
    *,
    layers: Sequence[int],
    method: str,
    representation_seed: int,
    input_view: str = "valid",
    max_length: int = 256,
    batch_size: int = 128,
    device: str = "cuda",
) -> dict[int, EmbeddingTable]:
    """Mean-pool selected hidden layers in one encoder pass per batch.

    Layer ``0`` is the embedding output exposed by Transformers and ``-1`` is
    the final encoder layer.  The returned mapping keeps the requested layer
    labels, which makes caches from different checkpoints directly alignable
    for layer-drift and retention analyses.
    """

    import torch

    requested = tuple(dict.fromkeys(int(layer) for layer in layers))
    if not requested:
        raise ValueError("at least one hidden layer must be requested")
    if len(sample_ids) != len(texts):
        raise ValueError("sample IDs and texts must align")

    outputs: dict[int, list[np.ndarray]] = {layer: [] for layer in requested}
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            special = encoded.pop("special_tokens_mask").to(device)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            result = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden_states = result.hidden_states
            valid = encoded["attention_mask"].bool() & ~special.bool()
            for layer in requested:
                resolved = layer if layer >= 0 else len(hidden_states) + layer
                if resolved < 0 or resolved >= len(hidden_states):
                    raise ValueError(
                        f"hidden layer {layer} is unavailable; model returned "
                        f"{len(hidden_states)} states"
                    )
                hidden = hidden_states[resolved]
                weights = valid.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
                outputs[layer].append(pooled.float().cpu().numpy())

    if not texts:
        raise ValueError("cannot extract an embedding table from zero texts")
    identifiers = np.asarray(sample_ids, dtype=str)
    return {
        layer: EmbeddingTable(
            sample_ids=identifiers,
            embeddings=np.concatenate(blocks, axis=0),
            method=method,
            representation_seed=representation_seed,
            input_view=f"{input_view}:layer={layer}",
        )
        for layer, blocks in outputs.items()
    }


__all__ = [
    "extract_embeddings",
    "extract_hidden_layer_embeddings",
    "load_encoder_checkpoint",
]
