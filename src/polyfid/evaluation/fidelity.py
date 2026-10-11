"""Structural-fidelity metrics kept separate by scientific question."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _normalise(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def row_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity for aligned rows."""

    left_array = _normalise(left)
    right_array = _normalise(right)
    if left_array.shape != right_array.shape:
        raise ValueError("aligned cosine inputs must have the same shape")
    return np.sum(left_array * right_array, axis=-1)


def equivalence_metrics(
    anchors: np.ndarray,
    positives: np.ndarray,
    decoys: np.ndarray,
) -> dict[str, float]:
    """Cosine and retrieval metrics for one positive plus matched decoys."""

    anchor = _normalise(anchors)
    positive = _normalise(positives)
    negative = _normalise(decoys)
    positive_score = np.sum(anchor * positive, axis=-1)
    decoy_score = np.einsum("bd,bkd->bk", anchor, negative)
    ranks = 1 + np.sum(decoy_score >= positive_score[:, None], axis=1)
    return {
        "equivalence_cosine": float(np.mean(positive_score)),
        "recall_at_1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


def topology_metrics(
    anchors: np.ndarray,
    positives: np.ndarray,
    topology_negatives: np.ndarray,
) -> dict[str, float]:
    """Topology discrimination against an aligned near-negative."""

    positive_score = row_cosine_similarity(anchors, positives)
    negative_score = row_cosine_similarity(anchors, topology_negatives)
    margin = positive_score - negative_score
    return {
        "topology_margin": float(np.mean(margin)),
        "triplet_accuracy": float(np.mean(margin > 0.0)),
    }


def _decoy_index_matrix(relations: pd.DataFrame, *, bank_size: int) -> np.ndarray:
    """Read the reusable, relation-builder-selected retrieval candidates."""

    column = "retrieval_decoy_bank_indices"
    if column not in relations.columns:
        raise ValueError(
            "relation table lacks retrieval_decoy_bank_indices; rebuild the relation "
            "set and use its fixed retrieval_decoy_bank.parquet"
        )
    rows = [np.asarray(values, dtype=int) for values in relations[column]]
    if not rows or not rows[0].size:
        raise ValueError("relation table does not contain retrieval decoys")
    widths = {int(values.size) for values in rows}
    if len(widths) != 1:
        raise ValueError("retrieval decoy lists must have one common length")
    indices = np.stack(rows)
    if indices.min() < 0 or indices.max() >= int(bank_size):
        raise ValueError("retrieval decoy index is outside the fixed decoy bank")
    return indices


def evaluate_relation_embeddings(
    relations: pd.DataFrame,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    retrieval_decoy_bank: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return one metric row per verified-relation triplet and aggregate summaries.

    Retrieval candidates are *only* those fixed by the relation-construction
    step.  This makes the evaluated candidate set identical across checkpoints
    and prevents an evaluation script from silently inventing easier decoys.
    """

    count = len(relations)
    if any(len(values) != count for values in (anchors, positives, negatives)):
        raise ValueError("relation rows and the three embedding blocks must align")
    bank_array = _normalise(retrieval_decoy_bank)
    decoy_indices = _decoy_index_matrix(relations, bank_size=len(bank_array))
    anchor_array = _normalise(anchors)
    positive_array = _normalise(positives)
    negative_array = _normalise(negatives)
    positive_score = np.sum(anchor_array * positive_array, axis=1)
    negative_score = np.sum(anchor_array * negative_array, axis=1)
    if bank_array.ndim != 2 or bank_array.shape[1] != anchor_array.shape[1]:
        raise ValueError("retrieval decoy bank and anchor embeddings must share width")
    decoy_scores = np.einsum("bd,bkd->bk", anchor_array, bank_array[decoy_indices])
    ranks = 1 + np.sum(decoy_scores >= positive_score[:, None], axis=1)
    margins = positive_score - negative_score

    details = relations.copy()
    details["anchor_positive_cosine"] = positive_score
    details["anchor_negative_cosine"] = negative_score
    details["topology_margin"] = margins
    details["triplet_correct"] = margins > 0.0
    details["retrieval_rank"] = ranks
    details["reciprocal_rank"] = 1.0 / ranks
    details["retrieval_correct_at_1"] = ranks == 1
    details["retrieval_max_decoy_cosine"] = np.max(decoy_scores, axis=1)
    details["retrieval_mean_decoy_cosine"] = np.mean(decoy_scores, axis=1)
    metrics: dict[str, float | int] = {
        "equivalence_cosine": float(np.mean(positive_score)),
        "recall_at_1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "topology_margin": float(np.mean(margins)),
        "triplet_accuracy": float(np.mean(margins > 0.0)),
        "triplets": int(count),
        "decoys_per_anchor": int(decoy_indices.shape[1]),
        "decoy_bank_size": int(len(bank_array)),
    }
    return details, metrics


def evaluate_topology_controls(
    controls: pd.DataFrame,
    anchors: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Score supplemental topology controls without pooling their families."""

    count = len(controls)
    if any(len(values) != count for values in (anchors, positives, negatives)):
        raise ValueError("control rows and embedding blocks must align")
    details = controls.copy()
    details["anchor_positive_cosine"] = row_cosine_similarity(anchors, positives)
    details["anchor_negative_cosine"] = row_cosine_similarity(anchors, negatives)
    details["topology_margin"] = (
        details["anchor_positive_cosine"] - details["anchor_negative_cosine"]
    )
    details["triplet_correct"] = details["topology_margin"] > 0.0
    summaries: dict[str, dict[str, float | int]] = {}
    for family, cell in details.groupby("negative_relation", sort=True):
        summaries[str(family)] = {
            "n": int(len(cell)),
            "topology_margin": float(cell["topology_margin"].mean()),
            "triplet_accuracy": float(cell["triplet_correct"].mean()),
            "anchor_positive_cosine": float(cell["anchor_positive_cosine"].mean()),
            "anchor_negative_cosine": float(cell["anchor_negative_cosine"].mean()),
        }
    return details, summaries


def equivalent_prediction_variation(predictions: np.ndarray) -> float:
    """Mean within-record standard deviation across equivalent input views."""

    values = np.asarray(predictions, dtype=float)
    return float(np.mean(np.std(values, axis=1, ddof=0)))


def boundary_mlm_summary(
    token_losses: np.ndarray,
    boundary_classes: np.ndarray,
    correct: np.ndarray | None = None,
    group_ids: Sequence[str] | None = None,
    scope: str = "repeat",
) -> dict[str, Mapping[str, float | int | str]]:
    """Summarise local MLM; group estimates weight each atom group equally."""

    losses = np.asarray(token_losses, dtype=float)
    classes = np.asarray(boundary_classes).astype(str)
    correctness = None if correct is None else np.asarray(correct, dtype=bool)
    groups = None if group_ids is None else np.asarray(group_ids).astype(str)
    output: dict[str, Mapping[str, float | int | str]] = {}
    for name in ("distance_0", "distance_1", "distance_2", "distance_3_plus", "interior"):
        selected = classes == name
        row: dict[str, float | int | str] = {
            "n_tokens": int(np.sum(selected)),
            "mean_nll": float(np.mean(losses[selected])) if np.any(selected) else float("nan"),
            "alias_of": "interior" if scope == "repeat" and name == "distance_3_plus" else "",
            "independent_evidence": int(not (scope == "repeat" and name == "distance_3_plus")),
        }
        if correctness is not None:
            row["accuracy"] = (
                float(np.mean(correctness[selected])) if np.any(selected) else float("nan")
            )
        if groups is not None:
            row["n_groups"] = int(np.unique(groups[selected]).size)
            selected_groups = groups[selected]
            group_nll = [
                float(np.mean(losses[selected][selected_groups == group]))
                for group in np.unique(selected_groups)
            ]
            row["atom_group_mean_nll"] = (
                float(np.mean(group_nll)) if group_nll else float("nan")
            )
            if correctness is not None:
                group_accuracy = [
                    float(np.mean(correctness[selected][selected_groups == group]))
                    for group in np.unique(selected_groups)
                ]
                row["atom_group_accuracy"] = (
                    float(np.mean(group_accuracy))
                    if group_accuracy
                    else float("nan")
                )
        output[name] = row
    return output


__all__ = [
    "boundary_mlm_summary",
    "equivalence_metrics",
    "equivalent_prediction_variation",
    "evaluate_relation_embeddings",
    "evaluate_topology_controls",
    "row_cosine_similarity",
    "topology_metrics",
]
