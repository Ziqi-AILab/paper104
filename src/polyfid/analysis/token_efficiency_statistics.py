"""Paired fixed-mask MLM statistics for the A0/B0 token-efficiency curves.

The evaluator writes one row per supervised token.  Inference here stores the
NLL sum and token count for each validation sample, then resamples complete
samples inside independently trained representation seeds.  This preserves the
reported token-weighted NLL without treating token positions as independent
training runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from polyfid.io import atomic_write_csv, atomic_write_text


FORMAL_REPRESENTATION_SEEDS = (42, 123, 777, 2025, 2026)
TOKEN_EFFICIENCY_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
MASK_ROW_COLUMNS = (
    "sample_id",
    "atom_group_id",
    "token_position",
    "target_token_id",
)


@dataclass(frozen=True)
class _PreparedSeedCurve:
    seed: int
    fractions: np.ndarray
    candidate_checkpoint_tokens: np.ndarray
    baseline_checkpoint_tokens: np.ndarray
    candidate_nll_sum: np.ndarray
    baseline_nll_sum: np.ndarray
    supervised_tokens: np.ndarray


def load_token_efficiency_manifest(path: str | Path) -> pd.DataFrame:
    """Load the generated A0/B0 evaluation matrix."""

    source = Path(path)
    separator = "\t" if source.suffix.casefold() in {".tsv", ".txt"} else ","
    manifest = pd.read_csv(source, sep=separator)
    required = {
        "arm",
        "representation_seed",
        "checkpoint_fraction",
        "checkpoint_percent",
        "mask_seed",
        "output",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"token-efficiency manifest is missing columns: {missing}")
    return manifest


def select_formal_token_efficiency_grid(
    manifest: pd.DataFrame,
    *,
    seeds: Iterable[int] = FORMAL_REPRESENTATION_SEEDS,
    fractions: Iterable[float] = TOKEN_EFFICIENCY_FRACTIONS,
) -> pd.DataFrame:
    """Select the declared 2 x seed x checkpoint grid and require one row/cell."""

    expected_seeds = tuple(int(value) for value in seeds)
    expected_fractions = tuple(float(value) for value in fractions)
    selected = manifest.loc[
        manifest["arm"].astype(str).isin({"A0", "B0"})
        & pd.to_numeric(manifest["representation_seed"], errors="raise")
        .astype(int)
        .isin(expected_seeds)
        & pd.to_numeric(manifest["checkpoint_fraction"], errors="raise")
        .astype(float)
        .isin(expected_fractions)
    ].copy()
    cell_columns = ["arm", "representation_seed", "checkpoint_fraction"]
    if selected.duplicated(cell_columns).any():
        raise ValueError("token-efficiency manifest has duplicate method/seed/fraction rows")
    expected = {
        (method, seed, fraction)
        for method in ("A0", "B0")
        for seed in expected_seeds
        for fraction in expected_fractions
    }
    observed = {
        (str(row.arm), int(row.representation_seed), float(row.checkpoint_fraction))
        for row in selected.itertuples(index=False)
    }
    missing = sorted(expected.difference(observed))
    if missing:
        raise ValueError(
            "token-efficiency grid is incomplete; missing "
            + ", ".join(f"{arm}/seed-{seed}/{fraction:g}" for arm, seed, fraction in missing)
        )
    return selected.sort_values(cell_columns).reset_index(drop=True)


def _single_integer(frame: pd.DataFrame, column: str, path: Path) -> int:
    values = pd.to_numeric(frame[column], errors="raise").dropna().astype(np.int64).unique()
    if len(values) != 1:
        raise ValueError(f"{path} does not contain one {column} value")
    return int(values[0])


def _read_cell(row: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_root = Path(row.output)
    token_path = output_root / "token_metrics.parquet"
    report_path = output_root / "metrics.json"
    if not token_path.is_file():
        raise FileNotFoundError(token_path)
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    columns = [
        *MASK_ROW_COLUMNS,
        "nll",
        "method",
        "representation_seed",
        "checkpoint_fraction",
        "checkpoint_tokens",
    ]
    frame = pd.read_parquet(token_path, columns=columns)
    if frame.empty:
        raise ValueError(f"token-efficiency result is empty: {token_path}")
    expected_method = str(row.arm)
    expected_seed = int(row.representation_seed)
    expected_fraction = float(row.checkpoint_fraction)
    if not frame["method"].astype(str).eq(expected_method).all():
        raise ValueError(f"{token_path} does not match method {expected_method}")
    if not pd.to_numeric(frame["representation_seed"], errors="raise").astype(int).eq(
        expected_seed
    ).all():
        raise ValueError(f"{token_path} does not match representation seed {expected_seed}")
    if not np.allclose(
        pd.to_numeric(frame["checkpoint_fraction"], errors="raise"),
        expected_fraction,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"{token_path} does not match checkpoint fraction {expected_fraction}")
    nll = pd.to_numeric(frame["nll"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(nll).all():
        raise ValueError(f"{token_path} contains a non-finite NLL")
    if frame.duplicated(list(MASK_ROW_COLUMNS)).any():
        raise ValueError(f"{token_path} contains duplicate fixed-mask target rows")
    checkpoint_tokens = _single_integer(frame, "checkpoint_tokens", token_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if int(report["checkpoint_tokens"]) != checkpoint_tokens:
        raise ValueError(f"{report_path} and {token_path} disagree on checkpoint tokens")
    return frame, {
        "method": expected_method,
        "representation_seed": expected_seed,
        "checkpoint_fraction": expected_fraction,
        "checkpoint_percent": int(row.checkpoint_percent),
        "checkpoint_tokens": checkpoint_tokens,
        "mask_seed": int(row.mask_seed),
        "supervised_tokens": int(len(frame)),
        "samples": int(frame["sample_id"].astype(str).nunique()),
        "token_metrics_path": str(token_path),
        "metrics_path": str(report_path),
    }


def _sort_mask_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(MASK_ROW_COLUMNS), kind="mergesort").reset_index(
        drop=True
    )


def _pair_cell(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    seed: int,
    fraction: float,
    candidate_checkpoint_tokens: int,
    baseline_checkpoint_tokens: int,
) -> pd.DataFrame:
    left = _sort_mask_rows(candidate)
    right = _sort_mask_rows(baseline)
    left_keys = pd.MultiIndex.from_frame(left.loc[:, MASK_ROW_COLUMNS])
    right_keys = pd.MultiIndex.from_frame(right.loc[:, MASK_ROW_COLUMNS])
    if not left_keys.equals(right_keys):
        raise ValueError(
            f"A0 and B0 do not share fixed-mask targets for seed {seed}, "
            f"fraction {fraction:g}"
        )
    token_rows = pd.DataFrame(
        {
            "sample_id": left["sample_id"].astype(str).to_numpy(),
            "nll_candidate": pd.to_numeric(left["nll"], errors="raise").to_numpy(
                dtype=float
            ),
            "nll_baseline": pd.to_numeric(right["nll"], errors="raise").to_numpy(
                dtype=float
            ),
        }
    )
    samples = (
        token_rows.groupby("sample_id", sort=True, as_index=False)
        .agg(
            candidate_nll_sum=("nll_candidate", "sum"),
            baseline_nll_sum=("nll_baseline", "sum"),
            supervised_tokens=("nll_candidate", "size"),
        )
        .assign(
            representation_seed=int(seed),
            checkpoint_fraction=float(fraction),
            checkpoint_percent=int(round(100.0 * fraction)),
            candidate_checkpoint_tokens=int(candidate_checkpoint_tokens),
            baseline_checkpoint_tokens=int(baseline_checkpoint_tokens),
        )
    )
    return samples


def load_paired_token_efficiency_samples(
    manifest: pd.DataFrame,
    *,
    seeds: Iterable[int] = FORMAL_REPRESENTATION_SEEDS,
    fractions: Iterable[float] = TOKEN_EFFICIENCY_FRACTIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read each A0/B0 cell and retain sample-level token sufficient statistics."""

    selected = select_formal_token_efficiency_grid(
        manifest, seeds=seeds, fractions=fractions
    )
    paired_blocks: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, Any]] = []
    reference_mask_rows: pd.MultiIndex | None = None
    for (seed, fraction), cell in selected.groupby(
        ["representation_seed", "checkpoint_fraction"], sort=True
    ):
        rows = {str(row.arm): row for row in cell.itertuples(index=False)}
        candidate, candidate_inventory = _read_cell(rows["A0"])
        baseline, baseline_inventory = _read_cell(rows["B0"])
        candidate_mask_rows = pd.MultiIndex.from_frame(
            _sort_mask_rows(candidate).loc[:, MASK_ROW_COLUMNS]
        )
        if reference_mask_rows is None:
            reference_mask_rows = candidate_mask_rows
        elif not reference_mask_rows.equals(candidate_mask_rows):
            raise ValueError(
                "token-efficiency checkpoints do not share the same fixed-mask targets"
            )
        if candidate_inventory["mask_seed"] != baseline_inventory["mask_seed"]:
            raise ValueError(
                f"A0 and B0 mask seeds differ for seed {int(seed)}, "
                f"fraction {float(fraction):g}"
            )
        paired_blocks.append(
            _pair_cell(
                candidate,
                baseline,
                seed=int(seed),
                fraction=float(fraction),
                candidate_checkpoint_tokens=int(
                    candidate_inventory["checkpoint_tokens"]
                ),
                baseline_checkpoint_tokens=int(
                    baseline_inventory["checkpoint_tokens"]
                ),
            )
        )
        inventory_rows.extend((candidate_inventory, baseline_inventory))
    paired = pd.concat(paired_blocks, ignore_index=True)
    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["method", "representation_seed", "checkpoint_fraction"]
    )
    return paired, inventory.reset_index(drop=True)


def _prepare_seed_curves(paired: pd.DataFrame) -> tuple[_PreparedSeedCurve, ...]:
    required = {
        "sample_id",
        "representation_seed",
        "checkpoint_fraction",
        "candidate_checkpoint_tokens",
        "baseline_checkpoint_tokens",
        "candidate_nll_sum",
        "baseline_nll_sum",
        "supervised_tokens",
    }
    missing = sorted(required.difference(paired.columns))
    if missing:
        raise ValueError(f"paired token-efficiency samples are missing columns: {missing}")
    prepared: list[_PreparedSeedCurve] = []
    shared_fractions: np.ndarray | None = None
    for seed, cell in paired.groupby("representation_seed", sort=True):
        fractions = np.sort(cell["checkpoint_fraction"].astype(float).unique())
        if shared_fractions is None:
            shared_fractions = fractions
        elif not np.array_equal(fractions, shared_fractions):
            raise ValueError("representation seeds do not share the same checkpoint fractions")
        candidate = cell.pivot(
            index="sample_id",
            columns="checkpoint_fraction",
            values="candidate_nll_sum",
        ).reindex(columns=fractions)
        baseline = cell.pivot(
            index="sample_id",
            columns="checkpoint_fraction",
            values="baseline_nll_sum",
        ).reindex(index=candidate.index, columns=fractions)
        token_counts = cell.pivot(
            index="sample_id",
            columns="checkpoint_fraction",
            values="supervised_tokens",
        ).reindex(index=candidate.index, columns=fractions)
        if (
            candidate.isna().any().any()
            or baseline.isna().any().any()
            or token_counts.isna().any().any()
        ):
            raise ValueError(
                f"seed {int(seed)} does not contain the same samples at every fraction"
            )
        token_table = cell[
            [
                "checkpoint_fraction",
                "candidate_checkpoint_tokens",
                "baseline_checkpoint_tokens",
            ]
        ].drop_duplicates()
        if token_table["checkpoint_fraction"].duplicated().any():
            raise ValueError(f"seed {int(seed)} has multiple token budgets at one fraction")
        token_table = token_table.sort_values("checkpoint_fraction")
        candidate_checkpoint_tokens = token_table[
            "candidate_checkpoint_tokens"
        ].to_numpy(dtype=np.int64)
        baseline_checkpoint_tokens = token_table[
            "baseline_checkpoint_tokens"
        ].to_numpy(dtype=np.int64)
        for method, checkpoint_tokens in (
            ("A0", candidate_checkpoint_tokens),
            ("B0", baseline_checkpoint_tokens),
        ):
            if np.any(checkpoint_tokens <= 0) or np.any(
                np.diff(checkpoint_tokens) <= 0
            ):
                raise ValueError(
                    f"seed {int(seed)} {method} checkpoint tokens are not strictly increasing"
                )
        prepared.append(
            _PreparedSeedCurve(
                seed=int(seed),
                fractions=fractions,
                candidate_checkpoint_tokens=candidate_checkpoint_tokens,
                baseline_checkpoint_tokens=baseline_checkpoint_tokens,
                candidate_nll_sum=candidate.to_numpy(dtype=float),
                baseline_nll_sum=baseline.to_numpy(dtype=float),
                supervised_tokens=token_counts.to_numpy(dtype=float),
            )
        )
    if not prepared:
        raise ValueError("paired token-efficiency samples are empty")
    return tuple(prepared)


def compute_seed_curves(paired: pd.DataFrame) -> pd.DataFrame:
    """Return token-weighted NLL summaries for each independent seed and budget."""

    rows: list[dict[str, Any]] = []
    for representation in _prepare_seed_curves(paired):
        token_totals = representation.supervised_tokens.sum(axis=0)
        candidate = representation.candidate_nll_sum.sum(axis=0) / token_totals
        baseline = representation.baseline_nll_sum.sum(axis=0) / token_totals
        for index, fraction in enumerate(representation.fractions):
            log_ratio = float(np.log(candidate[index] / baseline[index]))
            rows.append(
                {
                    "comparison": "molecular_initialization",
                    "candidate": "A0",
                    "baseline": "B0",
                    "representation_seed": representation.seed,
                    "checkpoint_fraction": float(fraction),
                    "checkpoint_percent": int(round(100.0 * fraction)),
                    "candidate_checkpoint_tokens": int(
                        representation.candidate_checkpoint_tokens[index]
                    ),
                    "baseline_checkpoint_tokens": int(
                        representation.baseline_checkpoint_tokens[index]
                    ),
                    "n_samples": int(representation.candidate_nll_sum.shape[0]),
                    "supervised_tokens": int(token_totals[index]),
                    "candidate_token_mean_nll": float(candidate[index]),
                    "baseline_token_mean_nll": float(baseline[index]),
                    "delta_nll": float(candidate[index] - baseline[index]),
                    "log_nll_ratio": log_ratio,
                    "relative_percent": float(100.0 * np.expm1(log_ratio)),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_token_mean_nll(
    representation: _PreparedSeedCurve,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(
        (
            representation.candidate_nll_sum,
            representation.baseline_nll_sum,
            representation.supervised_tokens,
        ),
        axis=1,
    )
    sample_count = values.shape[0]
    fraction_count = representation.candidate_nll_sum.shape[1]
    candidate = np.empty((n_resamples, fraction_count), dtype=float)
    baseline = np.empty_like(candidate)
    for start in range(0, n_resamples, chunk_size):
        stop = min(n_resamples, start + chunk_size)
        indices = rng.integers(
            0,
            sample_count,
            size=(stop - start, sample_count),
            dtype=np.int32,
        )
        totals = values[indices].sum(axis=1, dtype=np.float64)
        denominator = totals[:, 2 * fraction_count :]
        candidate[start:stop] = totals[:, :fraction_count] / denominator
        baseline[start:stop] = (
            totals[:, fraction_count : 2 * fraction_count] / denominator
        )
    return candidate, baseline


def _confidence_interval(draws: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return float(lower), float(upper)


def _common_log_token_interval(
    prepared: tuple[_PreparedSeedCurve, ...]
) -> tuple[float, float]:
    lower = max(
        float(np.log(tokens[0]))
        for representation in prepared
        for tokens in (
            representation.candidate_checkpoint_tokens,
            representation.baseline_checkpoint_tokens,
        )
    )
    upper = min(
        float(np.log(tokens[-1]))
        for representation in prepared
        for tokens in (
            representation.candidate_checkpoint_tokens,
            representation.baseline_checkpoint_tokens,
        )
    )
    if upper <= lower:
        raise ValueError("token-efficiency curves have no common log-token interval")
    return lower, upper


def _linear_curve_average_weights(
    x: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    """Weights for the normalized integral of a piecewise-linear curve."""

    if np.any(np.diff(x) <= 0.0) or lower < x[0] or upper > x[-1] or upper <= lower:
        raise ValueError("invalid log-token coordinates for curve integration")
    knots = np.unique(
        np.concatenate(
            (
                np.asarray([lower]),
                x[(x > lower) & (x < upper)],
                np.asarray([upper]),
            )
        )
    )
    basis = np.eye(len(x), dtype=float)
    interpolated = np.column_stack(
        [np.interp(knots, x, basis[:, index]) for index in range(len(x))]
    )
    return np.trapz(interpolated, x=knots, axis=0) / (upper - lower)


def _two_sided_zero_p_value(draws: np.ndarray) -> float:
    left = (np.count_nonzero(draws <= 0.0) + 1.0) / (len(draws) + 1.0)
    right = (np.count_nonzero(draws >= 0.0) + 1.0) / (len(draws) + 1.0)
    return float(min(1.0, 2.0 * min(left, right)))


def paired_token_efficiency_bootstrap(
    paired: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    seed: int = 2026,
    chunk_size: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap fraction effects and a normalized log-token curve summary.

    For each bootstrap replicate, samples are resampled once within each
    observed representation seed and reused across all six checkpoints.  The
    resulting seed curves are then resampled at the representation-seed level.
    """

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    prepared = _prepare_seed_curves(paired)
    fractions = prepared[0].fractions
    rng = np.random.default_rng(seed)
    outer = rng.integers(
        0,
        len(prepared),
        size=(n_resamples, len(prepared)),
        dtype=np.int32,
    )
    candidate_selected = np.empty(
        (n_resamples, len(prepared), len(fractions)), dtype=float
    )
    baseline_selected = np.empty_like(candidate_selected)
    for outer_slot in range(len(prepared)):
        for representation_index, representation in enumerate(prepared):
            draw_rows = np.flatnonzero(outer[:, outer_slot] == representation_index)
            if not len(draw_rows):
                continue
            candidate_means, baseline_means = _bootstrap_token_mean_nll(
                representation,
                n_resamples=len(draw_rows),
                rng=rng,
                chunk_size=chunk_size,
            )
            candidate_selected[draw_rows, outer_slot] = candidate_means
            baseline_selected[draw_rows, outer_slot] = baseline_means
    delta_draws = (candidate_selected - baseline_selected).mean(axis=1)
    log_ratio_draws = np.log(candidate_selected / baseline_selected).mean(axis=1)

    seed_curves = compute_seed_curves(paired)
    fraction_rows: list[dict[str, Any]] = []
    for fraction_index, fraction in enumerate(fractions):
        cell = seed_curves.loc[
            np.isclose(seed_curves["checkpoint_fraction"], fraction)
        ]
        delta_lower, delta_upper = _confidence_interval(
            delta_draws[:, fraction_index]
        )
        ratio_lower, ratio_upper = _confidence_interval(
            log_ratio_draws[:, fraction_index]
        )
        log_ratio = float(cell["log_nll_ratio"].mean())
        fraction_rows.append(
            {
                "comparison": "molecular_initialization",
                "candidate": "A0",
                "baseline": "B0",
                "checkpoint_fraction": float(fraction),
                "checkpoint_percent": int(round(100.0 * fraction)),
                "candidate_checkpoint_tokens_mean": float(
                    cell["candidate_checkpoint_tokens"].mean()
                ),
                "baseline_checkpoint_tokens_mean": float(
                    cell["baseline_checkpoint_tokens"].mean()
                ),
                "candidate_token_mean_nll": float(
                    cell["candidate_token_mean_nll"].mean()
                ),
                "baseline_token_mean_nll": float(
                    cell["baseline_token_mean_nll"].mean()
                ),
                "delta_nll": float(cell["delta_nll"].mean()),
                "delta_nll_ci_lower": delta_lower,
                "delta_nll_ci_upper": delta_upper,
                "log_nll_ratio": log_ratio,
                "log_nll_ratio_ci_lower": ratio_lower,
                "log_nll_ratio_ci_upper": ratio_upper,
                "relative_percent": float(100.0 * np.expm1(log_ratio)),
                "relative_percent_ci_lower": float(100.0 * np.expm1(ratio_lower)),
                "relative_percent_ci_upper": float(100.0 * np.expm1(ratio_upper)),
                "probability_a0_lower_nll": float(
                    np.mean(log_ratio_draws[:, fraction_index] < 0.0)
                ),
                "n_representation_seeds": int(len(prepared)),
                "n_samples_per_seed_min": int(cell["n_samples"].min()),
                "n_samples_per_seed_max": int(cell["n_samples"].max()),
                "bootstrap_resamples": int(n_resamples),
            }
        )

    common_lower, common_upper = _common_log_token_interval(prepared)
    seed_auc_rows: list[dict[str, Any]] = []
    selected_candidate_auc = np.empty((n_resamples, len(prepared)), dtype=float)
    selected_baseline_auc = np.empty_like(selected_candidate_auc)
    for representation_index, representation in enumerate(prepared):
        candidate_x = np.log(
            representation.candidate_checkpoint_tokens.astype(float)
        )
        baseline_x = np.log(
            representation.baseline_checkpoint_tokens.astype(float)
        )
        candidate_weights = _linear_curve_average_weights(
            candidate_x, common_lower, common_upper
        )
        baseline_weights = _linear_curve_average_weights(
            baseline_x, common_lower, common_upper
        )
        for outer_slot in range(len(prepared)):
            draw_rows = np.flatnonzero(outer[:, outer_slot] == representation_index)
            selected_candidate_auc[draw_rows, outer_slot] = (
                candidate_selected[draw_rows, outer_slot] @ candidate_weights
            )
            selected_baseline_auc[draw_rows, outer_slot] = (
                baseline_selected[draw_rows, outer_slot] @ baseline_weights
            )
        token_totals = representation.supervised_tokens.sum(axis=0)
        candidate_point = representation.candidate_nll_sum.sum(axis=0) / token_totals
        baseline_point = representation.baseline_nll_sum.sum(axis=0) / token_totals
        candidate_auc = float(candidate_point @ candidate_weights)
        baseline_auc = float(baseline_point @ baseline_weights)
        seed_log_ratio = float(np.log(candidate_auc / baseline_auc))
        seed_auc_rows.append(
            {
                "comparison": "molecular_initialization",
                "candidate": "A0",
                "baseline": "B0",
                "representation_seed": representation.seed,
                "common_token_lower": float(np.exp(common_lower)),
                "common_token_upper": float(np.exp(common_upper)),
                "candidate_log_token_average_nll": candidate_auc,
                "baseline_log_token_average_nll": baseline_auc,
                "log_token_average_delta_nll": candidate_auc - baseline_auc,
                "log_ratio_of_log_token_average_nll": seed_log_ratio,
                "relative_percent": float(100.0 * np.expm1(seed_log_ratio)),
            }
        )
    delta_auc_draws = (selected_candidate_auc - selected_baseline_auc).mean(axis=1)
    log_ratio_auc_draws = np.log(
        selected_candidate_auc / selected_baseline_auc
    ).mean(axis=1)
    delta_auc_lower, delta_auc_upper = _confidence_interval(delta_auc_draws)
    ratio_auc_lower, ratio_auc_upper = _confidence_interval(log_ratio_auc_draws)
    seed_auc = pd.DataFrame(seed_auc_rows)
    log_ratio_auc = float(seed_auc["log_ratio_of_log_token_average_nll"].mean())
    curve = pd.DataFrame(
        [
            {
                "comparison": "molecular_initialization",
                "candidate": "A0",
                "baseline": "B0",
                "curve_axis": "natural_log_checkpoint_tokens",
                "curve_range": "common_observed_1_to_100_percent_token_interval",
                "common_token_lower": float(np.exp(common_lower)),
                "common_token_upper": float(np.exp(common_upper)),
                "candidate_log_token_average_nll": float(
                    seed_auc["candidate_log_token_average_nll"].mean()
                ),
                "baseline_log_token_average_nll": float(
                    seed_auc["baseline_log_token_average_nll"].mean()
                ),
                "log_token_average_delta_nll": float(
                    seed_auc["log_token_average_delta_nll"].mean()
                ),
                "log_token_average_delta_nll_ci_lower": delta_auc_lower,
                "log_token_average_delta_nll_ci_upper": delta_auc_upper,
                "log_ratio_of_log_token_average_nll": log_ratio_auc,
                "log_ratio_ci_lower": ratio_auc_lower,
                "log_ratio_ci_upper": ratio_auc_upper,
                "relative_percent": float(
                    100.0 * np.expm1(log_ratio_auc)
                ),
                "relative_percent_ci_lower": float(
                    100.0 * np.expm1(ratio_auc_lower)
                ),
                "relative_percent_ci_upper": float(
                    100.0 * np.expm1(ratio_auc_upper)
                ),
                "probability_a0_lower_log_token_average_nll": float(
                    np.mean(delta_auc_draws < 0.0)
                ),
                "p_value_two_sided": _two_sided_zero_p_value(delta_auc_draws),
                "n_representation_seeds": int(len(prepared)),
                "bootstrap_resamples": int(n_resamples),
            }
        ]
    )
    return pd.DataFrame(fraction_rows), seed_auc, curve


def write_token_efficiency_analysis_outputs(
    manifest_path: str | Path,
    output: str | Path,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    bootstrap_chunk_size: int = 16,
    seeds: Iterable[int] = FORMAL_REPRESENTATION_SEEDS,
    fractions: Iterable[float] = TOKEN_EFFICIENCY_FRACTIONS,
) -> dict[str, Any]:
    """Run the local paired analysis and write compact reviewer-facing tables."""

    manifest_source = Path(manifest_path).resolve()
    manifest = load_token_efficiency_manifest(manifest_source)
    selected_seeds = tuple(int(value) for value in seeds)
    selected_fractions = tuple(float(value) for value in fractions)
    paired, inventory = load_paired_token_efficiency_samples(
        manifest, seeds=selected_seeds, fractions=selected_fractions
    )
    seed_curves = compute_seed_curves(paired)
    fraction_effects, seed_auc, curve_summary = paired_token_efficiency_bootstrap(
        paired,
        n_resamples=n_resamples,
        seed=bootstrap_seed,
        chunk_size=bootstrap_chunk_size,
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(inventory, destination / "result_inventory.csv")
    atomic_write_csv(seed_curves, destination / "seed_curves.csv")
    atomic_write_csv(fraction_effects, destination / "fraction_effects.csv")
    atomic_write_csv(seed_auc, destination / "seed_auc.csv")
    atomic_write_csv(curve_summary, destination / "curve_summary.csv")
    summary = {
        "source": "fixed-mask PI1M validation token metrics",
        "manifest": str(manifest_source),
        "comparison": "A0 vs B0 molecular initialization",
        "efficiency_scope": (
            "PI1M continued-pretraining token budget only; upstream molecular "
            "pretraining cost is outside this curve"
        ),
        "methods": ["A0", "B0"],
        "representation_seeds": list(selected_seeds),
        "checkpoint_fractions": list(selected_fractions),
        "result_cells": int(len(inventory)),
        "paired_sample_rows": int(len(paired)),
        "sample_estimand": (
            "sample-level NLL sums and token counts are resampled as a cluster; "
            "their ratio retains token-weighted mean NLL"
        ),
        "bootstrap_hierarchy": "representation seed -> validation sample",
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "curve_summary": (
            "normalized piecewise-linear mean over the common observed "
            "natural-log checkpoint-token interval"
        ),
        "effect_direction": "negative delta or relative percent means lower A0 NLL",
        "inference_scope": (
            "per-fraction intervals describe the learning curve; the log-token "
            "average is the single cross-budget summary"
        ),
        "multiplicity_adjustment": "none within this prespecified H1 learning-curve analysis",
    }
    atomic_write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        destination / "summary.json",
    )
    return summary


__all__ = [
    "FORMAL_REPRESENTATION_SEEDS",
    "TOKEN_EFFICIENCY_FRACTIONS",
    "compute_seed_curves",
    "load_paired_token_efficiency_samples",
    "load_token_efficiency_manifest",
    "paired_token_efficiency_bootstrap",
    "select_formal_token_efficiency_grid",
    "write_token_efficiency_analysis_outputs",
]
