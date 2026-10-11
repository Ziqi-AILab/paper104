"""Paired statistics for the targeted Paper104 mechanism controls.

This module keeps two scientific questions separate:

* boundary masking controls are compared with A0 at matched representation
  seeds, while the five-seed A_att/A0 comparison remains a distinct main
  boundary result;
* relation-loss component controls are compared with the A_eqs full objective
  on their three shared seeds, overall and by verified positive-relation
  family.

The controls are supplementary analyses.  They are not added to the seven-arm
Figure 4 comparison and are never collapsed into a structural-fidelity score.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from polyfid.io import atomic_write_csv, atomic_write_text


FORMAL_REPRESENTATION_SEEDS = (42, 123, 777, 2025, 2026)
MECHANISM_CONTROL_SEEDS = (42, 123, 777)
BOUNDARY_STRATA = ("distance_0", "distance_1", "distance_2", "interior")


@dataclass(frozen=True)
class _Comparison:
    comparison: str
    candidate: str
    reference: str
    seeds: tuple[int, ...]
    analysis_role: str


BOUNDARY_COMPARISONS = (
    _Comparison(
        "boundary_supervision",
        "A_att",
        "A0",
        FORMAL_REPRESENTATION_SEEDS,
        "main_five_seed_boundary_comparison",
    ),
    _Comparison(
        "matched_random_masking_control",
        "matched_random_masking",
        "A0",
        MECHANISM_CONTROL_SEEDS,
        "supplementary_three_seed_mechanism_control",
    ),
    _Comparison(
        "permuted_boundary_masking_control",
        "permuted_boundary_masking",
        "A0",
        MECHANISM_CONTROL_SEEDS,
        "supplementary_three_seed_mechanism_control",
    ),
)

RELATION_COMPARISONS = (
    _Comparison(
        "equivalence_only_vs_full_objective",
        "equivalence_only",
        "A_eqs",
        MECHANISM_CONTROL_SEEDS,
        "supplementary_three_seed_mechanism_control",
    ),
    _Comparison(
        "topology_negative_only_vs_full_objective",
        "topology_negative_only",
        "A_eqs",
        MECHANISM_CONTROL_SEEDS,
        "supplementary_three_seed_mechanism_control",
    ),
)

BOUNDARY_PAIR_KEYS = (
    "sample_id",
    "group_id",
    "token_position",
    "target_token_id",
)
RELATION_PAIR_KEYS = (
    "triplet_id",
    "anchor_sample_id",
    "positive_sample_id",
    "negative_sample_id",
    "positive_relation",
    "negative_relation",
)
RELATION_METRICS: Mapping[str, str] = {
    "equivalence_cosine": "anchor_positive_cosine",
    "recall_at_1": "retrieval_correct_at_1",
    "mrr": "reciprocal_rank",
    "topology_margin": "topology_margin",
    "triplet_accuracy": "triplet_correct",
}


def _result_path(root: str | Path, method: str, seed: int, filename: str) -> Path:
    return Path(root) / f"{method}_seed-{seed}" / filename


def _read_boundary_cell(root: str | Path, method: str, seed: int) -> pd.DataFrame:
    path = _result_path(root, method, seed, "token_metrics.parquet")
    columns = [
        *BOUNDARY_PAIR_KEYS,
        "boundary_stratum",
        "interior_region",
        "nll",
        "correct",
        "method",
        "representation_seed",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty:
        raise ValueError(f"boundary MLM result is empty: {path}")
    if not frame["method"].astype(str).eq(method).all():
        raise ValueError(f"boundary MLM method does not match {method}: {path}")
    if not pd.to_numeric(frame["representation_seed"], errors="raise").astype(int).eq(
        seed
    ).all():
        raise ValueError(f"boundary MLM seed does not match {seed}: {path}")
    frame = frame.copy()
    frame["analysis_stratum"] = frame["boundary_stratum"].astype(str)
    frame.loc[frame["interior_region"].astype(bool), "analysis_stratum"] = "interior"
    frame = frame.loc[frame["analysis_stratum"].isin(BOUNDARY_STRATA)].copy()
    frame["nll"] = pd.to_numeric(frame["nll"], errors="raise").astype(float)
    frame["correct"] = frame["correct"].astype(bool).astype(float)
    return frame


def _read_relation_cell(root: str | Path, method: str, seed: int) -> pd.DataFrame:
    path = _result_path(root, method, seed, "triplet_metrics.parquet")
    columns = [*RELATION_PAIR_KEYS, *RELATION_METRICS.values(), "method", "representation_seed"]
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    if frame.empty:
        raise ValueError(f"relation-fidelity result is empty: {path}")
    if not frame["method"].astype(str).eq(method).all():
        raise ValueError(f"relation-fidelity method does not match {method}: {path}")
    if not pd.to_numeric(frame["representation_seed"], errors="raise").astype(int).eq(
        seed
    ).all():
        raise ValueError(f"relation-fidelity seed does not match {seed}: {path}")
    frame = frame.copy()
    for column in RELATION_METRICS.values():
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    return frame


def _sorted_pair(frame: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    return frame.sort_values(list(keys), kind="mergesort").reset_index(drop=True)


def pair_boundary_cells(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    comparison: _Comparison,
    seed: int,
) -> pd.DataFrame:
    """Pair fixed atom-token targets, then give each atom group one row."""

    if "analysis_stratum" not in candidate.columns:
        candidate = candidate.copy()
        candidate["analysis_stratum"] = candidate["boundary_stratum"].astype(str)
        candidate.loc[candidate["interior_region"].astype(bool), "analysis_stratum"] = "interior"
    if "analysis_stratum" not in reference.columns:
        reference = reference.copy()
        reference["analysis_stratum"] = reference["boundary_stratum"].astype(str)
        reference.loc[reference["interior_region"].astype(bool), "analysis_stratum"] = "interior"
    keys = (*BOUNDARY_PAIR_KEYS, "analysis_stratum")
    left = _sorted_pair(candidate, keys)
    right = _sorted_pair(reference, keys)
    left_index = pd.MultiIndex.from_frame(left.loc[:, keys])
    right_index = pd.MultiIndex.from_frame(right.loc[:, keys])
    if not left_index.equals(right_index):
        raise ValueError(
            f"boundary comparison {comparison.comparison}/seed-{seed} does not share "
            "the same fixed atom-token targets"
        )
    paired_tokens = pd.DataFrame(
        {
            "sample_id": left["sample_id"].astype(str).to_numpy(),
            "group_id": left["group_id"].astype(str).to_numpy(),
            "stratum": left["analysis_stratum"].astype(str).to_numpy(),
            "candidate_nll": left["nll"].to_numpy(dtype=float),
            "reference_nll": right["nll"].to_numpy(dtype=float),
            "candidate_accuracy": left["correct"].to_numpy(dtype=float),
            "reference_accuracy": right["correct"].to_numpy(dtype=float),
        }
    )
    groups = (
        paired_tokens.groupby(["sample_id", "group_id", "stratum"], sort=True, as_index=False)
        .agg(
            candidate_nll=("candidate_nll", "mean"),
            reference_nll=("reference_nll", "mean"),
            candidate_accuracy=("candidate_accuracy", "mean"),
            reference_accuracy=("reference_accuracy", "mean"),
            token_count=("candidate_nll", "size"),
        )
        .assign(
            comparison=comparison.comparison,
            candidate=comparison.candidate,
            reference=comparison.reference,
            analysis_role=comparison.analysis_role,
            representation_seed=int(seed),
        )
    )
    return groups


def pair_relation_cells(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    comparison: _Comparison,
    seed: int,
) -> pd.DataFrame:
    """Pair one fixed verified-relation row across two checkpoints."""

    left = _sorted_pair(candidate, RELATION_PAIR_KEYS)
    right = _sorted_pair(reference, RELATION_PAIR_KEYS)
    left_index = pd.MultiIndex.from_frame(left.loc[:, RELATION_PAIR_KEYS])
    right_index = pd.MultiIndex.from_frame(right.loc[:, RELATION_PAIR_KEYS])
    if not left_index.equals(right_index):
        raise ValueError(
            f"relation comparison {comparison.comparison}/seed-{seed} does not share "
            "the fixed relation rows"
        )
    paired = pd.DataFrame(
        {
            "triplet_id": left["triplet_id"].astype(str).to_numpy(),
            "sample_id": left["anchor_sample_id"].astype(str).to_numpy(),
            "relation_family": left["positive_relation"].astype(str).to_numpy(),
            "comparison": comparison.comparison,
            "candidate": comparison.candidate,
            "reference": comparison.reference,
            "analysis_role": comparison.analysis_role,
            "representation_seed": int(seed),
        }
    )
    for metric, column in RELATION_METRICS.items():
        paired[f"candidate_{metric}"] = left[column].to_numpy(dtype=float)
        paired[f"reference_{metric}"] = right[column].to_numpy(dtype=float)
    return paired


def _boundary_method_summary(cells: Mapping[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, seed), frame in sorted(cells.items()):
        for stratum, cell in frame.groupby("analysis_stratum", sort=True):
            group_values = (
                cell.groupby(["sample_id", "group_id"], sort=False, as_index=False)
                .agg(group_nll=("nll", "mean"), group_accuracy=("correct", "mean"))
            )
            rows.append(
                {
                    "method": method,
                    "representation_seed": int(seed),
                    "stratum": str(stratum),
                    "n_tokens": int(len(cell)),
                    "n_atom_groups": int(len(group_values)),
                    "n_samples": int(cell["sample_id"].astype(str).nunique()),
                    "token_mean_nll": float(cell["nll"].mean()),
                    "token_accuracy": float(cell["correct"].mean()),
                    "atom_group_mean_nll": float(group_values["group_nll"].mean()),
                    "atom_group_accuracy": float(group_values["group_accuracy"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _boundary_seed_effects(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "comparison",
        "candidate",
        "reference",
        "analysis_role",
        "representation_seed",
        "stratum",
    ]
    for values, cell in paired.groupby(keys, sort=True):
        candidate_nll = float(cell["candidate_nll"].mean())
        reference_nll = float(cell["reference_nll"].mean())
        candidate_accuracy = float(cell["candidate_accuracy"].mean())
        reference_accuracy = float(cell["reference_accuracy"].mean())
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_samples": int(cell["sample_id"].nunique()),
                "n_atom_groups": int(len(cell)),
                "candidate_atom_group_mean_nll": candidate_nll,
                "reference_atom_group_mean_nll": reference_nll,
                "nll_difference": candidate_nll - reference_nll,
                "log_nll_ratio": float(np.log(candidate_nll / reference_nll)),
                "candidate_atom_group_accuracy": candidate_accuracy,
                "reference_atom_group_accuracy": reference_accuracy,
                "accuracy_difference_pp": 100.0 * (candidate_accuracy - reference_accuracy),
            }
        )
    return pd.DataFrame(rows)


def _relation_scopes(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    yield "overall", "all", frame
    for family, cell in frame.groupby("relation_family", sort=True):
        yield "relation_family", str(family), cell


def _relation_method_summary(cells: Mapping[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (method, seed), frame in sorted(cells.items()):
        renamed = frame.rename(columns={value: key for key, value in RELATION_METRICS.items()})
        renamed["relation_family"] = renamed["positive_relation"].astype(str)
        for scope, family, cell in _relation_scopes(renamed):
            row: dict[str, Any] = {
                "method": method,
                "representation_seed": int(seed),
                "analysis_scope": scope,
                "relation_family": family,
                "n_triplets": int(len(cell)),
                "n_anchors": int(cell["anchor_sample_id"].astype(str).nunique()),
            }
            for metric in RELATION_METRICS:
                row[metric] = float(cell[metric].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _relation_seed_effects(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = [
        "comparison",
        "candidate",
        "reference",
        "analysis_role",
        "representation_seed",
    ]
    for values, seed_cell in paired.groupby(group_keys, sort=True):
        for scope, family, cell in _relation_scopes(seed_cell):
            common = {
                **dict(zip(group_keys, values, strict=True)),
                "analysis_scope": scope,
                "relation_family": family,
                "n_triplets": int(len(cell)),
                "n_anchors": int(cell["sample_id"].nunique()),
            }
            for metric in RELATION_METRICS:
                candidate = float(cell[f"candidate_{metric}"].mean())
                reference = float(cell[f"reference_{metric}"].mean())
                rows.append(
                    {
                        **common,
                        "metric": metric,
                        "candidate_mean": candidate,
                        "reference_mean": reference,
                        "candidate_minus_full_objective": candidate - reference,
                    }
                )
    return pd.DataFrame(rows)


def _confidence_interval(draws: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper)


def _two_sided_zero_p_value(draws: np.ndarray) -> float:
    left = (np.count_nonzero(draws <= 0.0) + 1.0) / (len(draws) + 1.0)
    right = (np.count_nonzero(draws >= 0.0) + 1.0) / (len(draws) + 1.0)
    return float(min(1.0, 2.0 * min(left, right)))


def _cluster_mean_draws(
    frame: pd.DataFrame,
    metric_pairs: Mapping[str, tuple[str, str]],
    *,
    n_resamples: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resample representation seeds, samples, and all leaves within samples."""

    seeds = np.sort(frame["representation_seed"].astype(int).unique())
    metric_names = tuple(metric_pairs)
    candidate_inner = np.empty((n_resamples, len(seeds), len(metric_names)), dtype=float)
    reference_inner = np.empty_like(candidate_inner)
    point_candidate = np.empty((len(seeds), len(metric_names)), dtype=float)
    point_reference = np.empty_like(point_candidate)

    for seed_index, seed in enumerate(seeds):
        seed_cell = frame.loc[frame["representation_seed"].astype(int).eq(int(seed))]
        aggregations: dict[str, tuple[str, str]] = {"leaf_count": ("sample_id", "size")}
        for metric, (candidate_column, reference_column) in metric_pairs.items():
            aggregations[f"candidate_{metric}_sum"] = (candidate_column, "sum")
            aggregations[f"reference_{metric}_sum"] = (reference_column, "sum")
        samples = seed_cell.groupby("sample_id", sort=True, as_index=False).agg(**aggregations)
        counts = samples["leaf_count"].to_numpy(dtype=float)
        candidate_sums = np.column_stack(
            [samples[f"candidate_{metric}_sum"].to_numpy(dtype=float) for metric in metric_names]
        )
        reference_sums = np.column_stack(
            [samples[f"reference_{metric}_sum"].to_numpy(dtype=float) for metric in metric_names]
        )
        point_candidate[seed_index] = candidate_sums.sum(axis=0) / counts.sum()
        point_reference[seed_index] = reference_sums.sum(axis=0) / counts.sum()
        sample_count = len(samples)
        for start in range(0, n_resamples, chunk_size):
            stop = min(n_resamples, start + chunk_size)
            indices = rng.integers(
                0,
                sample_count,
                size=(stop - start, sample_count),
                dtype=np.int32,
            )
            denominators = counts[indices].sum(axis=1)[:, None]
            candidate_inner[start:stop, seed_index] = candidate_sums[indices].sum(axis=1) / denominators
            reference_inner[start:stop, seed_index] = reference_sums[indices].sum(axis=1) / denominators

    outer = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)), dtype=np.int32)
    selected_candidate = np.take_along_axis(candidate_inner, outer[:, :, None], axis=1)
    selected_reference = np.take_along_axis(reference_inner, outer[:, :, None], axis=1)
    return (
        {
            metric: point_candidate[:, index]
            for index, metric in enumerate(metric_names)
        },
        {
            metric: point_reference[:, index]
            for index, metric in enumerate(metric_names)
        },
        {
            f"candidate_{metric}": selected_candidate[:, :, index]
            for index, metric in enumerate(metric_names)
        }
        | {
            f"reference_{metric}": selected_reference[:, :, index]
            for index, metric in enumerate(metric_names)
        },
    )


def bootstrap_boundary_comparisons(
    paired: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 2026,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """Paired boundary effects with seeds outermost and samples clustered."""

    if n_resamples < 1 or chunk_size < 1:
        raise ValueError("bootstrap resamples and chunk size must be positive")
    rows: list[dict[str, Any]] = []
    keys = ["comparison", "candidate", "reference", "analysis_role", "stratum"]
    for index, (values, cell) in enumerate(paired.groupby(keys, sort=True)):
        rng = np.random.default_rng(seed + index)
        point_candidate, point_reference, draws = _cluster_mean_draws(
            cell,
            {
                "nll": ("candidate_nll", "reference_nll"),
                "accuracy": ("candidate_accuracy", "reference_accuracy"),
            },
            n_resamples=n_resamples,
            rng=rng,
            chunk_size=chunk_size,
        )
        nll_difference_draws = (
            draws["candidate_nll"] - draws["reference_nll"]
        ).mean(axis=1)
        log_ratio_draws = np.log(
            draws["candidate_nll"] / draws["reference_nll"]
        ).mean(axis=1)
        accuracy_draws = 100.0 * (
            draws["candidate_accuracy"] - draws["reference_accuracy"]
        ).mean(axis=1)
        nll_lower, nll_upper = _confidence_interval(nll_difference_draws)
        ratio_lower, ratio_upper = _confidence_interval(log_ratio_draws)
        accuracy_lower, accuracy_upper = _confidence_interval(accuracy_draws)
        point_log_ratio = float(
            np.log(point_candidate["nll"] / point_reference["nll"]).mean()
        )
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "n_representation_seeds": int(cell["representation_seed"].nunique()),
                "n_samples_per_seed_min": int(cell.groupby("representation_seed")["sample_id"].nunique().min()),
                "n_atom_groups_per_seed_min": int(cell.groupby("representation_seed").size().min()),
                "candidate_atom_group_mean_nll": float(point_candidate["nll"].mean()),
                "reference_atom_group_mean_nll": float(point_reference["nll"].mean()),
                "nll_difference": float((point_candidate["nll"] - point_reference["nll"]).mean()),
                "nll_difference_ci_lower": nll_lower,
                "nll_difference_ci_upper": nll_upper,
                "log_nll_ratio": point_log_ratio,
                "log_nll_ratio_ci_lower": ratio_lower,
                "log_nll_ratio_ci_upper": ratio_upper,
                "relative_nll_percent": float(100.0 * np.expm1(point_log_ratio)),
                "relative_nll_percent_ci_lower": float(100.0 * np.expm1(ratio_lower)),
                "relative_nll_percent_ci_upper": float(100.0 * np.expm1(ratio_upper)),
                "probability_candidate_lower_nll": float(np.mean(log_ratio_draws < 0.0)),
                "nll_p_value_two_sided": _two_sided_zero_p_value(log_ratio_draws),
                "candidate_atom_group_accuracy": float(point_candidate["accuracy"].mean()),
                "reference_atom_group_accuracy": float(point_reference["accuracy"].mean()),
                "accuracy_difference_pp": float(100.0 * (point_candidate["accuracy"] - point_reference["accuracy"]).mean()),
                "accuracy_difference_pp_ci_lower": accuracy_lower,
                "accuracy_difference_pp_ci_upper": accuracy_upper,
                "probability_candidate_higher_accuracy": float(np.mean(accuracy_draws > 0.0)),
                "accuracy_p_value_two_sided": _two_sided_zero_p_value(accuracy_draws),
                "bootstrap_resamples": int(n_resamples),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_relation_comparisons(
    paired: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 2026,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """Compare each loss-component control with A_eqs on three shared seeds."""

    if n_resamples < 1 or chunk_size < 1:
        raise ValueError("bootstrap resamples and chunk size must be positive")
    rows: list[dict[str, Any]] = []
    base_keys = ["comparison", "candidate", "reference", "analysis_role"]
    offset = 0
    for values, comparison_cell in paired.groupby(base_keys, sort=True):
        for scope, family, cell in _relation_scopes(comparison_cell):
            for metric in RELATION_METRICS:
                rng = np.random.default_rng(seed + offset)
                offset += 1
                point_candidate, point_reference, draws = _cluster_mean_draws(
                    cell,
                    {metric: (f"candidate_{metric}", f"reference_{metric}")},
                    n_resamples=n_resamples,
                    rng=rng,
                    chunk_size=chunk_size,
                )
                effect_draws = (
                    draws[f"candidate_{metric}"] - draws[f"reference_{metric}"]
                ).mean(axis=1)
                lower, upper = _confidence_interval(effect_draws)
                point_effect = float(
                    (point_candidate[metric] - point_reference[metric]).mean()
                )
                rows.append(
                    {
                        **dict(zip(base_keys, values, strict=True)),
                        "analysis_scope": scope,
                        "relation_family": family,
                        "metric": metric,
                        "candidate_mean": float(point_candidate[metric].mean()),
                        "full_objective_mean": float(point_reference[metric].mean()),
                        "candidate_minus_full_objective": point_effect,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "probability_candidate_higher": float(np.mean(effect_draws > 0.0)),
                        "p_value_two_sided": _two_sided_zero_p_value(effect_draws),
                        "n_representation_seeds": int(cell["representation_seed"].nunique()),
                        "n_triplets_per_seed_min": int(cell.groupby("representation_seed").size().min()),
                        "n_anchors_per_seed_min": int(cell.groupby("representation_seed")["sample_id"].nunique().min()),
                        "bootstrap_resamples": int(n_resamples),
                    }
                )
    return pd.DataFrame(rows)


def load_boundary_inputs(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read all boundary cells needed by the main and control comparisons."""

    required = {
        (method, seed)
        for comparison in BOUNDARY_COMPARISONS
        for method in (comparison.candidate, comparison.reference)
        for seed in comparison.seeds
    }
    cells = {
        (method, seed): _read_boundary_cell(root, method, seed)
        for method, seed in sorted(required)
    }
    inventory = pd.DataFrame(
        [
            {
                "analysis_domain": "boundary_mlm",
                "method": method,
                "representation_seed": seed,
                "result_path": str(_result_path(root, method, seed, "token_metrics.parquet")),
                "rows": int(len(frame)),
                "samples": int(frame["sample_id"].nunique()),
            }
            for (method, seed), frame in sorted(cells.items())
        ]
    )
    paired = pd.concat(
        [
            pair_boundary_cells(
                cells[(comparison.candidate, seed)],
                cells[(comparison.reference, seed)],
                comparison=comparison,
                seed=seed,
            )
            for comparison in BOUNDARY_COMPARISONS
            for seed in comparison.seeds
        ],
        ignore_index=True,
    )
    return _boundary_method_summary(cells), paired, inventory


def load_relation_inputs(
    root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read relation controls and their shared-seed A_eqs reference cells."""

    required = {
        (method, seed)
        for comparison in RELATION_COMPARISONS
        for method in (comparison.candidate, comparison.reference)
        for seed in comparison.seeds
    }
    cells = {
        (method, seed): _read_relation_cell(root, method, seed)
        for method, seed in sorted(required)
    }
    inventory = pd.DataFrame(
        [
            {
                "analysis_domain": "relation_fidelity",
                "method": method,
                "representation_seed": seed,
                "result_path": str(_result_path(root, method, seed, "triplet_metrics.parquet")),
                "rows": int(len(frame)),
                "samples": int(frame["anchor_sample_id"].nunique()),
            }
            for (method, seed), frame in sorted(cells.items())
        ]
    )
    paired = pd.concat(
        [
            pair_relation_cells(
                cells[(comparison.candidate, seed)],
                cells[(comparison.reference, seed)],
                comparison=comparison,
                seed=seed,
            )
            for comparison in RELATION_COMPARISONS
            for seed in comparison.seeds
        ],
        ignore_index=True,
    )
    return _relation_method_summary(cells), paired, inventory


def _supplementary_readout(
    boundary_main: pd.DataFrame,
    boundary_controls: pd.DataFrame,
    relation_controls: pd.DataFrame,
) -> str:
    lines = [
        "# Mechanism-control supplementary readout",
        "",
        "These tables remain separate from the seven-arm Figure 4 analysis. "
        "The two loss-component experiments have three representation seeds and "
        "therefore support interpretation rather than a broad standalone claim.",
        "",
        "## Boundary supervision: five-seed main comparison",
        "",
        "| Stratum | NLL change (%) | 95% CI | Accuracy change (pp) | 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in boundary_main.sort_values("stratum").itertuples(index=False):
        lines.append(
            f"| {row.stratum} | {row.relative_nll_percent:.2f} | "
            f"[{row.relative_nll_percent_ci_lower:.2f}, {row.relative_nll_percent_ci_upper:.2f}] | "
            f"{row.accuracy_difference_pp:.2f} | "
            f"[{row.accuracy_difference_pp_ci_lower:.2f}, {row.accuracy_difference_pp_ci_upper:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Boundary masking controls: three shared seeds",
            "",
            "| Control vs A0 | Stratum | NLL change (%) | 95% CI | Accuracy change (pp) | 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in boundary_controls.sort_values(["candidate", "stratum"]).itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.stratum} | {row.relative_nll_percent:.2f} | "
            f"[{row.relative_nll_percent_ci_lower:.2f}, {row.relative_nll_percent_ci_upper:.2f}] | "
            f"{row.accuracy_difference_pp:.2f} | "
            f"[{row.accuracy_difference_pp_ci_lower:.2f}, {row.accuracy_difference_pp_ci_upper:.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Relation-loss component controls: three shared seeds",
            "",
            "Effects are control minus A_eqs full objective; higher is better for all five metrics.",
            "",
            "| Control | Scope | Family | Metric | Effect | 95% CI |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in relation_controls.sort_values(
        ["candidate", "analysis_scope", "relation_family", "metric"]
    ).itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.analysis_scope} | {row.relation_family} | "
            f"{row.metric} | {row.candidate_minus_full_objective:.4f} | "
            f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}] |"
        )
    return "\n".join(lines) + "\n"


def write_mechanism_control_analysis_outputs(
    boundary_root: str | Path,
    relation_root: str | Path,
    output: str | Path,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    bootstrap_chunk_size: int = 64,
) -> dict[str, Any]:
    """Run the local analysis and write compact supplementary tables."""

    boundary_summary, boundary_paired, boundary_inventory = load_boundary_inputs(
        boundary_root
    )
    relation_summary, relation_paired, relation_inventory = load_relation_inputs(
        relation_root
    )
    boundary_seed_effects = _boundary_seed_effects(boundary_paired)
    boundary_aggregate = bootstrap_boundary_comparisons(
        boundary_paired,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
        chunk_size=bootstrap_chunk_size,
    )
    boundary_main = boundary_aggregate.loc[
        boundary_aggregate["analysis_role"].eq("main_five_seed_boundary_comparison")
    ].reset_index(drop=True)
    boundary_controls = boundary_aggregate.loc[
        boundary_aggregate["analysis_role"].eq(
            "supplementary_three_seed_mechanism_control"
        )
    ].reset_index(drop=True)
    relation_seed_effects = _relation_seed_effects(relation_paired)
    relation_controls = bootstrap_relation_comparisons(
        relation_paired,
        n_resamples=n_resamples,
        seed=bootstrap_seed + 1000,
        chunk_size=bootstrap_chunk_size,
    )

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    inventory = pd.concat([boundary_inventory, relation_inventory], ignore_index=True)
    atomic_write_csv(inventory, destination / "result_inventory.csv")
    atomic_write_csv(boundary_summary, destination / "boundary_method_summary.csv")
    atomic_write_csv(boundary_seed_effects, destination / "boundary_seed_effects.csv")
    atomic_write_csv(boundary_main, destination / "boundary_main_comparison.csv")
    atomic_write_csv(
        boundary_controls, destination / "boundary_control_comparisons.csv"
    )
    atomic_write_csv(relation_summary, destination / "relation_method_summary.csv")
    atomic_write_csv(relation_seed_effects, destination / "relation_seed_effects.csv")
    atomic_write_csv(
        relation_controls, destination / "relation_control_comparisons.csv"
    )
    atomic_write_text(
        _supplementary_readout(boundary_main, boundary_controls, relation_controls),
        destination / "supplementary_readout.md",
    )
    summary = {
        "status": "complete",
        "analysis": "Paper104 targeted mechanism controls",
        "boundary_root": str(Path(boundary_root).resolve()),
        "relation_root": str(Path(relation_root).resolve()),
        "result_cells": int(len(inventory)),
        "boundary_main_comparison": "A_att vs A0 on five representation seeds",
        "boundary_controls": [
            "matched_random_masking vs A0 on seeds 42, 123, 777",
            "permuted_boundary_masking vs A0 on seeds 42, 123, 777",
        ],
        "relation_controls": [
            "equivalence_only vs A_eqs full objective on seeds 42, 123, 777",
            "topology_negative_only vs A_eqs full objective on seeds 42, 123, 777",
        ],
        "boundary_strata": list(BOUNDARY_STRATA),
        "relation_family_definition": "positive_relation",
        "relation_metrics": list(RELATION_METRICS),
        "boundary_estimand": "representation-seed-equal effect on atom-group-equal NLL and accuracy",
        "relation_estimand": "representation-seed-equal candidate minus full-objective effect",
        "bootstrap_hierarchy": "representation seed -> held-out sample/anchor; atom groups or triplets remain nested",
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "multiplicity_adjustment": "none; these are separated mechanism readouts, not an added primary comparison family",
        "claim_scope": "three-seed controls are supplementary evidence and are not included in the main seven-arm Figure 4",
        "combined_score": "not computed",
    }
    atomic_write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        destination / "summary.json",
    )
    return summary


__all__ = [
    "BOUNDARY_COMPARISONS",
    "BOUNDARY_STRATA",
    "FORMAL_REPRESENTATION_SEEDS",
    "MECHANISM_CONTROL_SEEDS",
    "RELATION_COMPARISONS",
    "RELATION_METRICS",
    "bootstrap_boundary_comparisons",
    "bootstrap_relation_comparisons",
    "load_boundary_inputs",
    "load_relation_inputs",
    "pair_boundary_cells",
    "pair_relation_cells",
    "write_mechanism_control_analysis_outputs",
]
