from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polyfid.evaluation.embeddings import (
    EmbeddingTable,
    load_embedding_table,
    save_embedding_table,
)
from polyfid.evaluation.predictions import load_predictions, prediction_frame, write_predictions
from polyfid.records import PredictionRecord


def test_prediction_record_context_round_trip_and_legacy_compatibility(
    tmp_path: Path,
) -> None:
    record = PredictionRecord(
        sample_id="p1",
        task="Egc",
        split="test",
        group_id="g1",
        y_true=1.0,
        y_pred=1.2,
        method="A_eqs",
        representation_seed=42,
        head_seed=11,
        structural_fidelity_eligible=True,
        scope_class="linear_dual_attachment_repeat",
        checkpoint_fraction=0.25,
        checkpoint_tokens=123_456,
    )
    path = write_predictions([record], tmp_path / "predictions.parquet")
    loaded = load_predictions(path).iloc[0]
    assert bool(loaded["structural_fidelity_eligible"])
    assert loaded["scope_class"] == "linear_dual_attachment_repeat"
    assert loaded["checkpoint_fraction"] == 0.25
    assert loaded["checkpoint_tokens"] == 123_456

    legacy = prediction_frame(
        pd.DataFrame(
            [
                {
                    "sample_id": "legacy",
                    "task": "EA",
                    "split": "test",
                    "group_id": "g2",
                    "y_true": 0.0,
                    "y_pred": 0.1,
                    "method": "C",
                    "representation_seed": 42,
                    "head_seed": 11,
                }
            ]
        )
    )
    assert pd.isna(legacy.loc[0, "structural_fidelity_eligible"])
    assert pd.isna(legacy.loc[0, "scope_class"])
    assert pd.isna(legacy.loc[0, "checkpoint_fraction"])
    assert pd.isna(legacy.loc[0, "checkpoint_tokens"])


def test_embedding_checkpoint_progress_is_optional_and_backward_compatible(
    tmp_path: Path,
) -> None:
    table = EmbeddingTable(
        sample_ids=np.asarray(["s1"]),
        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
        method="A_att",
        representation_seed=123,
        checkpoint_fraction=0.5,
        checkpoint_tokens=9_000,
    )
    loaded = load_embedding_table(save_embedding_table(table, tmp_path / "new.npz"))
    assert loaded.checkpoint_fraction == 0.5
    assert loaded.checkpoint_tokens == 9_000

    np.savez_compressed(
        tmp_path / "legacy.npz",
        sample_ids=np.asarray(["s1"]),
        embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
        method=np.asarray("C"),
        representation_seed=np.asarray(42, dtype=np.int64),
        input_view=np.asarray("valid"),
    )
    legacy = load_embedding_table(tmp_path / "legacy.npz")
    assert legacy.checkpoint_fraction is None
    assert legacy.checkpoint_tokens is None
