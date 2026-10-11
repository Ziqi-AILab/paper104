"""Paired hierarchical bootstrap for the three declared comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd


PRIMARY_COMPARISONS = {
    "molecular_initialization": ("A0", "B0"),
    "boundary_supervision": ("A_att", "A0"),
    "verified_relations": ("A_eqs", "A_gcl"),
}


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    probability_candidate_better: float
    p_value_two_sided: float
    resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_hierarchical_bootstrap(
    paired_rows: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    lower_is_better: bool = True,
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> BootstrapResult:
    """Resample representation seeds, heads, structure groups, then rows."""

    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    representation_seeds = np.sort(paired_rows["representation_seed"].unique())
    for draw_index in range(n_resamples):
        pieces: list[pd.DataFrame] = []
        for representation_seed in rng.choice(representation_seeds, len(representation_seeds), replace=True):
            seed_cell = paired_rows.loc[paired_rows["representation_seed"].eq(representation_seed)]
            heads = np.sort(seed_cell["head_seed"].unique())
            for head in rng.choice(heads, len(heads), replace=True):
                head_cell = seed_cell.loc[seed_cell["head_seed"].eq(head)]
                for _, task_cell in head_cell.groupby("task", sort=True):
                    if "family_id" in task_cell and task_cell["family_id"].notna().any():
                        families = task_cell["family_id"].dropna().astype(str).unique()
                        for family in rng.choice(families, len(families), replace=True):
                            family_cell = task_cell.loc[task_cell["family_id"].astype(str).eq(str(family))]
                            groups = family_cell["group_id"].astype(str).unique()
                            for group in rng.choice(groups, len(groups), replace=True):
                                group_cell = family_cell.loc[family_cell["group_id"].astype(str).eq(str(group))]
                                pieces.append(group_cell.iloc[rng.integers(0, len(group_cell), len(group_cell))])
                    else:
                        groups = task_cell["group_id"].astype(str).unique()
                        for group in rng.choice(groups, len(groups), replace=True):
                            group_cell = task_cell.loc[task_cell["group_id"].astype(str).eq(str(group))]
                            pieces.append(group_cell.iloc[rng.integers(0, len(group_cell), len(group_cell))])
        draws[draw_index] = statistic(pd.concat(pieces, ignore_index=True))
    estimate = float(statistic(paired_rows))
    lower, upper = np.quantile(draws, (0.025, 0.975))
    probability = np.mean(draws < 0.0) if lower_is_better else np.mean(draws > 0.0)
    left = (np.count_nonzero(draws <= 0.0) + 1.0) / (len(draws) + 1.0)
    right = (np.count_nonzero(draws >= 0.0) + 1.0) / (len(draws) + 1.0)
    return BootstrapResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        probability_candidate_better=float(probability),
        p_value_two_sided=float(min(1.0, 2.0 * min(left, right))),
        resamples=n_resamples,
    )
