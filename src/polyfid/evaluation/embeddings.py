"""Cached pre-projection encoder representations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from polyfid.io import atomic_save_npz


@dataclass(frozen=True)
class EmbeddingTable:
    sample_ids: np.ndarray
    embeddings: np.ndarray
    method: str
    representation_seed: int
    input_view: str = "valid"
    checkpoint_fraction: float | None = None
    checkpoint_tokens: int | None = None

    def __post_init__(self) -> None:
        if len(self.sample_ids) != len(self.embeddings):
            raise ValueError("sample IDs and embedding rows must align")


def save_embedding_table(table: EmbeddingTable, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample_ids": np.asarray(table.sample_ids, dtype=str),
        "embeddings": np.asarray(table.embeddings, dtype=np.float32),
        "method": np.asarray(table.method),
        "representation_seed": np.asarray(table.representation_seed, dtype=np.int64),
        "input_view": np.asarray(table.input_view),
    }
    if table.checkpoint_fraction is not None:
        payload["checkpoint_fraction"] = np.asarray(
            table.checkpoint_fraction, dtype=np.float64
        )
    if table.checkpoint_tokens is not None:
        payload["checkpoint_tokens"] = np.asarray(table.checkpoint_tokens, dtype=np.int64)
    atomic_save_npz(destination, payload)
    return destination


def load_embedding_table(path: str | Path) -> EmbeddingTable:
    with np.load(path, allow_pickle=False) as values:
        return EmbeddingTable(
            sample_ids=values["sample_ids"].astype(str),
            embeddings=values["embeddings"].astype(np.float32),
            method=str(values["method"].item()),
            representation_seed=int(values["representation_seed"].item()),
            input_view=str(values["input_view"].item()),
            checkpoint_fraction=(
                float(values["checkpoint_fraction"].item())
                if "checkpoint_fraction" in values.files
                else None
            ),
            checkpoint_tokens=(
                int(values["checkpoint_tokens"].item())
                if "checkpoint_tokens" in values.files
                else None
            ),
        )


def align_embeddings(
    table: EmbeddingTable,
    sample_ids: np.ndarray,
) -> np.ndarray:
    index = {str(sample_id): row for row, sample_id in enumerate(table.sample_ids)}
    missing = [str(sample_id) for sample_id in sample_ids if str(sample_id) not in index]
    if missing:
        raise ValueError(f"embedding cache lacks {len(missing)} requested samples")
    return table.embeddings[[index[str(sample_id)] for sample_id in sample_ids]]


__all__ = [
    "EmbeddingTable",
    "align_embeddings",
    "load_embedding_table",
    "save_embedding_table",
]
