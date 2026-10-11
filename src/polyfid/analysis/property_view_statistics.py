"""Record-level statistics for predictions across verified equivalent views.

The property-view cache is deliberately analysed from the compact
``equivalent_view_within_record.parquet`` files.  The much larger per-view
prediction tables are not duplicated or loaded for this stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from polyfid.evaluation.bootstrap import PRIMARY_COMPARISONS
from polyfid.io import atomic_write_csv, atomic_write_text


VIEW_METRICS = {
    "within_record_prediction_std": "mean_property_prediction_std",
    "mean_absolute_view_delta": "mean_absolute_view_delta",
}

MANIFEST_PAIR_COLUMNS = (
    "representation_seed",
    "task",
    "protocol",
    "fold",
    "input_view",
    "partition_id",
    "label_fraction",
    "subsample_id",
)


@dataclass(frozen=True)
class PropertyViewBootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    probability_candidate_better: float
    p_value_two_sided: float
    resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class _PreparedTask:
    name: str
    candidate: np.ndarray
    baseline: np.ndarray
    group_starts: np.ndarray
    group_sizes: np.ndarray
    flat_group_members: np.ndarray


@dataclass(frozen=True)
class _PreparedRepresentation:
    seed: int
    heads: np.ndarray
    tasks: tuple[_PreparedTask, ...]


def load_property_view_manifest(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    separator = "\t" if source.suffix.casefold() in {".tsv", ".txt"} else ","
    manifest = pd.read_csv(source, sep=separator)
    required = {
        "arm",
        "representation_label",
        "representation_seed",
        "task",
        "protocol",
        "fold",
        "input_view",
        "partition_id",
        "head_seeds",
        "label_fraction",
        "subsample_id",
        "output_root",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"property-view manifest is missing columns: {missing}")
    return manifest


def _head_seeds(value: Any) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),)
    return tuple(int(item.strip()) for item in str(value).split(",") if item.strip())


def _record_path(output_root: str | Path, head_seed: int) -> Path:
    return (
        Path(output_root)
        / f"head-seed-{int(head_seed)}"
        / "equivalent_view_within_record.parquet"
    )


def _validate_nonempty_table(
    frame: pd.DataFrame,
    row: Any,
    head_seed: int,
    *,
    method: str | None = None,
) -> None:
    expected = {
        "method": str(method if method is not None else row.arm),
        "representation_seed": int(row.representation_seed),
        "head_seed": int(head_seed),
        "task": str(row.task),
        "protocol": str(row.protocol),
    }
    for column, value in expected.items():
        if column not in frame:
            raise ValueError(f"property-view table is missing {column}")
        observed = frame[column]
        if column in {"representation_seed", "head_seed"}:
            matches = pd.to_numeric(observed, errors="raise").astype(int).eq(int(value))
        else:
            matches = observed.astype(str).eq(str(value))
        if not matches.all():
            raise ValueError(f"property-view table does not match {column}={value}")


def stream_property_view_task_metrics(
    manifest: pd.DataFrame,
    *,
    progress_every: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise each formal record-level result without loading the corpus."""

    metric_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    total = sum(len(_head_seeds(value)) for value in manifest["head_seeds"])
    processed = 0
    for row in manifest.itertuples(index=False):
        for head_seed in _head_seeds(row.head_seeds):
            path = _record_path(row.output_root, head_seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_parquet(path)
            report_path = path.parent / "metrics.json"
            report = (
                json.loads(report_path.read_text(encoding="utf-8"))
                if report_path.is_file()
                else {}
            )
            processed += 1
            inventory_rows.append(
                {
                    "representation_label": str(row.representation_label),
                    "method": str(row.arm),
                    "representation_seed": int(row.representation_seed),
                    "head_seed": int(head_seed),
                    "task": str(row.task),
                    "protocol": str(row.protocol),
                    "fold": int(row.fold),
                    "input_view": str(row.input_view),
                    "records": int(len(frame)),
                    "test_records": int(report.get("test_records", 0)),
                    "eligible_test_records": int(
                        report.get("eligible_test_records", 0)
                    ),
                    "view_families": int(frame["view_family"].nunique())
                    if not frame.empty
                    else 0,
                    "status": "nonempty" if not frame.empty else "no_eligible_views",
                    "record_path": str(path),
                }
            )
            if not frame.empty:
                _validate_nonempty_table(frame, row, head_seed)
                for view_family, cell in frame.groupby("view_family", sort=True):
                    metric_rows.append(
                        {
                            "representation_label": str(row.representation_label),
                            "method": str(row.arm),
                            "representation_seed": int(row.representation_seed),
                            "head_seed": int(head_seed),
                            "task": str(row.task),
                            "protocol": str(row.protocol),
                            "fold": int(row.fold),
                            "input_view": str(row.input_view),
                            "label_fraction": float(row.label_fraction),
                            "subsample_id": int(row.subsample_id),
                            "view_family": str(view_family),
                            "task_type": str(cell["task_type"].iloc[0]),
                            "scope_class": str(cell["scope_class"].iloc[0]),
                            "n_records": int(len(cell)),
                            "n_groups": int(cell["group_id"].astype(str).nunique()),
                            "n_equivalent_views": int(
                                cell["n_equivalent_views"].sum()
                            ),
                            "mean_equivalent_views_per_record": float(
                                cell["n_equivalent_views"].mean()
                            ),
                            "mean_property_prediction_std": float(
                                cell["property_prediction_std"].mean()
                            ),
                            "median_property_prediction_std": float(
                                cell["property_prediction_std"].median()
                            ),
                            "mean_absolute_view_delta": float(
                                cell["mean_absolute_view_delta"].mean()
                            ),
                            "median_absolute_view_delta": float(
                                cell["mean_absolute_view_delta"].median()
                            ),
                            "mean_max_absolute_view_delta": float(
                                cell["max_absolute_view_delta"].mean()
                            ),
                            "reference_rmse": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(
                                            cell["reference_y_pred"].to_numpy(dtype=float)
                                            - cell["y_true"].to_numpy(dtype=float)
                                        )
                                    )
                                )
                            ),
                            "reference_prediction_std": float(
                                cell["reference_y_pred"].std(ddof=0)
                            ),
                            "zero_property_prediction_std_records": int(
                                cell["property_prediction_std"].eq(0.0).sum()
                            ),
                        }
                    )
            if progress_every > 0 and (
                processed % progress_every == 0 or processed == total
            ):
                print(f"streamed_property_view_files={processed}/{total}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(inventory_rows)


def compute_property_view_task_effects(task_metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair methods and pool the fixed OOD folds into one task/head effect."""

    merge_keys = [
        "representation_seed",
        "head_seed",
        "task",
        "protocol",
        "fold",
        "input_view",
        "label_fraction",
        "subsample_id",
        "view_family",
    ]
    task_keys = [column for column in merge_keys if column != "fold"]
    rows: list[dict[str, Any]] = []
    for comparison, (candidate, baseline) in PRIMARY_COMPARISONS.items():
        left = task_metrics.loc[task_metrics["method"].eq(candidate)].copy()
        right = task_metrics.loc[task_metrics["method"].eq(baseline)].copy()
        paired = left.merge(
            right,
            on=merge_keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_candidate", "_baseline"),
        )
        if not paired["n_records_candidate"].astype(int).eq(
            paired["n_records_baseline"].astype(int)
        ).all():
            raise ValueError(
                f"{candidate}/{baseline} have different property-view record counts"
            )
        for keys, cell in paired.groupby(task_keys, dropna=False, sort=True):
            values_by_key = dict(zip(task_keys, keys, strict=True))
            if not cell["n_equivalent_views_candidate"].astype(int).eq(
                cell["n_equivalent_views_baseline"].astype(int)
            ).all():
                raise ValueError(
                    f"{candidate}/{baseline} use different equivalent-view counts"
                )
            values: dict[str, Any] = {
                "comparison": comparison,
                "candidate": candidate,
                "baseline": baseline,
                **values_by_key,
                "n_folds": int(cell["fold"].nunique()),
                "n_paired_records": int(cell["n_records_candidate"].sum()),
                "n_groups": int(cell["n_groups_candidate"].sum()),
                "n_equivalent_views": int(
                    cell["n_equivalent_views_candidate"].sum()
                ),
            }
            for metric, source in VIEW_METRICS.items():
                weights = cell["n_records_candidate"].to_numpy(dtype=float)
                candidate_value = float(
                    np.average(
                        cell[f"{source}_candidate"].to_numpy(dtype=float),
                        weights=weights,
                    )
                )
                baseline_value = float(
                    np.average(
                        cell[f"{source}_baseline"].to_numpy(dtype=float),
                        weights=weights,
                    )
                )
                if candidate_value <= 0.0 or baseline_value <= 0.0:
                    raise ValueError("property-view task means must be positive")
                values[f"{metric}_candidate"] = candidate_value
                values[f"{metric}_baseline"] = baseline_value
                values[f"{metric}_paired_log_ratio"] = float(
                    np.log(candidate_value / baseline_value)
                )
            candidate_rmse = float(
                np.sqrt(
                    np.average(
                        np.square(cell["reference_rmse_candidate"].to_numpy(dtype=float)),
                        weights=weights,
                    )
                )
            )
            baseline_rmse = float(
                np.sqrt(
                    np.average(
                        np.square(cell["reference_rmse_baseline"].to_numpy(dtype=float)),
                        weights=weights,
                    )
                )
            )
            values["reference_rmse_candidate"] = candidate_rmse
            values["reference_rmse_baseline"] = baseline_rmse
            values["reference_rmse_paired_log_ratio"] = float(
                np.log(candidate_rmse / baseline_rmse)
            )
            candidate_dispersion = cell[
                "reference_prediction_std_candidate"
            ].to_numpy(dtype=float)
            baseline_dispersion = cell[
                "reference_prediction_std_baseline"
            ].to_numpy(dtype=float)
            values["reference_prediction_std_candidate"] = float(
                np.average(candidate_dispersion, weights=weights)
            )
            values["reference_prediction_std_baseline"] = float(
                np.average(baseline_dispersion, weights=weights)
            )
            if np.all(candidate_dispersion > 0.0) and np.all(
                baseline_dispersion > 0.0
            ):
                candidate_normalized = float(
                    np.average(
                        cell["mean_property_prediction_std_candidate"].to_numpy(
                            dtype=float
                        )
                        / candidate_dispersion,
                        weights=weights,
                    )
                )
                baseline_normalized = float(
                    np.average(
                        cell["mean_property_prediction_std_baseline"].to_numpy(
                            dtype=float
                        )
                        / baseline_dispersion,
                        weights=weights,
                    )
                )
                values["dispersion_normalized_stability_candidate"] = (
                    candidate_normalized
                )
                values["dispersion_normalized_stability_baseline"] = baseline_normalized
                values["dispersion_normalized_stability_paired_log_ratio"] = float(
                    np.log(candidate_normalized / baseline_normalized)
                )
            else:
                values["dispersion_normalized_stability_candidate"] = np.nan
                values["dispersion_normalized_stability_baseline"] = np.nan
                values["dispersion_normalized_stability_paired_log_ratio"] = np.nan
            rows.append(values)
    return pd.DataFrame(rows)


def _primary_protocol(task: str, protocols: Iterable[str]) -> str | None:
    available = sorted({str(value) for value in protocols})
    name = str(task).casefold().rsplit(":", maxsplit=1)[-1].replace("-", "_")
    if name in {"ea", "ip"} and "family_ood" in available:
        return "family_ood"
    if "group_ood" in available:
        return "group_ood"
    if "official_group_disjoint" in available:
        return "official_group_disjoint"
    if len(available) == 1:
        return available[0]
    return None


def select_primary_property_view_manifest(
    manifest: pd.DataFrame, candidate: str, baseline: str
) -> pd.DataFrame:
    """Choose the full-data structure-only protocol once per polymer task."""

    selected = manifest.loc[
        manifest["arm"].astype(str).isin([candidate, baseline])
        & manifest["input_view"].astype(str).eq("structure_only")
        & np.isclose(pd.to_numeric(manifest["label_fraction"]), 1.0)
        & pd.to_numeric(manifest["subsample_id"], errors="raise").astype(int).eq(0)
    ].copy()
    task_names = selected["task"].astype(str).str.casefold()
    selected = selected.loc[
        ~task_names.str.startswith("openpoly:")
        & ~task_names.isin({"esol", "freesolv", "lipophilicity", "zinc", "zinc_mlm"})
    ]
    blocks: list[pd.DataFrame] = []
    for task, cell in selected.groupby("task", sort=True):
        protocol = _primary_protocol(str(task), cell["protocol"])
        if protocol is not None:
            blocks.append(cell.loc[cell["protocol"].astype(str).eq(protocol)])
    return pd.concat(blocks, ignore_index=True) if blocks else selected.iloc[0:0]


def _paired_manifest_rows(
    manifest: pd.DataFrame, candidate: str, baseline: str
) -> pd.DataFrame:
    selected = select_primary_property_view_manifest(manifest, candidate, baseline)
    left = selected.loc[selected["arm"].astype(str).eq(candidate)].copy()
    right = selected.loc[selected["arm"].astype(str).eq(baseline)].copy()
    return left.merge(
        right,
        on=list(MANIFEST_PAIR_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )


def load_primary_paired_property_views(
    manifest: pd.DataFrame,
    candidate: str,
    baseline: str,
) -> pd.DataFrame:
    """Load only the record-level rows needed by one formal comparison."""

    blocks: list[pd.DataFrame] = []
    for row in _paired_manifest_rows(manifest, candidate, baseline).itertuples(index=False):
        candidate_heads = _head_seeds(row.head_seeds_candidate)
        baseline_heads = _head_seeds(row.head_seeds_baseline)
        if candidate_heads != baseline_heads:
            raise ValueError(f"{candidate}/{baseline} use different property-head seeds")
        for head_seed in candidate_heads:
            candidate_frame = pd.read_parquet(
                _record_path(row.output_root_candidate, head_seed)
            )
            baseline_frame = pd.read_parquet(
                _record_path(row.output_root_baseline, head_seed)
            )
            if candidate_frame.empty and baseline_frame.empty:
                continue
            if candidate_frame.empty != baseline_frame.empty:
                raise ValueError(f"{candidate}/{baseline} differ in eligible view coverage")
            _validate_nonempty_table(
                candidate_frame, row, head_seed, method=candidate
            )
            _validate_nonempty_table(
                baseline_frame, row, head_seed, method=baseline
            )
            if not candidate_frame["method"].astype(str).eq(candidate).all():
                raise ValueError("candidate property-view method does not match manifest")
            if not baseline_frame["method"].astype(str).eq(baseline).all():
                raise ValueError("baseline property-view method does not match manifest")
            keys = ["sample_id", "view_family"]
            left = candidate_frame.loc[
                :,
                keys
                + [
                    "group_id",
                    "n_equivalent_views",
                    "property_prediction_std",
                    "mean_absolute_view_delta",
                ],
            ].rename(
                columns={
                    "property_prediction_std": "property_prediction_std_candidate",
                    "mean_absolute_view_delta": "mean_absolute_view_delta_candidate",
                }
            )
            right = baseline_frame.loc[
                :,
                keys
                + [
                    "group_id",
                    "n_equivalent_views",
                    "property_prediction_std",
                    "mean_absolute_view_delta",
                ],
            ].rename(
                columns={
                    "group_id": "group_id_baseline",
                    "n_equivalent_views": "n_equivalent_views_baseline",
                    "property_prediction_std": "property_prediction_std_baseline",
                    "mean_absolute_view_delta": "mean_absolute_view_delta_baseline",
                }
            )
            paired = left.merge(
                right,
                on=keys,
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            if not paired["_merge"].eq("both").all():
                raise ValueError(f"{candidate}/{baseline} do not cover the same records")
            paired = paired.drop(columns="_merge")
            if not paired["group_id"].astype(str).eq(
                paired["group_id_baseline"].astype(str)
            ).all():
                raise ValueError(f"{candidate}/{baseline} disagree on structure groups")
            if not paired["n_equivalent_views"].astype(int).eq(
                paired["n_equivalent_views_baseline"].astype(int)
            ).all():
                raise ValueError(f"{candidate}/{baseline} use different view counts")
            paired = paired.drop(
                columns=["group_id_baseline", "n_equivalent_views_baseline"]
            )
            paired["representation_seed"] = int(row.representation_seed)
            paired["head_seed"] = int(head_seed)
            paired["task"] = str(row.task)
            paired["protocol"] = str(row.protocol)
            paired["fold"] = int(row.fold)
            paired["input_view"] = str(row.input_view)
            blocks.append(paired)
    if not blocks:
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True, sort=False)


def _prepare_bootstrap(
    paired: pd.DataFrame, metric_column: str
) -> tuple[_PreparedRepresentation, ...]:
    prepared: list[_PreparedRepresentation] = []
    candidate_column = f"{metric_column}_candidate"
    baseline_column = f"{metric_column}_baseline"
    for seed, seed_cell in paired.groupby("representation_seed", sort=True):
        task_cells = [cell for _, cell in seed_cell.groupby("task", sort=True)]
        common_heads = set(task_cells[0]["head_seed"].astype(int).unique())
        for cell in task_cells[1:]:
            common_heads.intersection_update(cell["head_seed"].astype(int).unique())
        heads = np.asarray(sorted(common_heads), dtype=int)
        tasks: list[_PreparedTask] = []
        for task, cell in seed_cell.groupby("task", sort=True):
            working = cell.copy()
            working["_observation"] = (
                working["fold"].astype(str)
                + "::"
                + working["sample_id"].astype(str)
            )
            reference = (
                working.sort_values(["_observation", "head_seed"])
                .drop_duplicates("_observation")
                .set_index("_observation")
                .sort_index()
            )
            observations = reference.index
            candidate = (
                working.pivot(
                    index="_observation", columns="head_seed", values=candidate_column
                )
                .reindex(index=observations, columns=heads)
                .to_numpy(dtype=float)
                .T
            )
            baseline = (
                working.pivot(
                    index="_observation", columns="head_seed", values=baseline_column
                )
                .reindex(index=observations, columns=heads)
                .to_numpy(dtype=float)
                .T
            )
            if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
                raise ValueError(f"task {task} is missing a nested head result")
            group_labels = (
                reference["fold"].astype(str)
                + "::"
                + reference["group_id"].astype(str)
            )
            group_codes, unique_groups = pd.factorize(group_labels, sort=True)
            group_members = tuple(
                np.flatnonzero(group_codes == index)
                for index in range(len(unique_groups))
            )
            group_sizes = np.fromiter(
                (len(members) for members in group_members),
                dtype=int,
                count=len(group_members),
            )
            group_starts = np.concatenate(
                (np.asarray([0], dtype=int), np.cumsum(group_sizes[:-1]))
            )
            tasks.append(
                _PreparedTask(
                    name=str(task),
                    candidate=candidate,
                    baseline=baseline,
                    group_starts=group_starts,
                    group_sizes=group_sizes,
                    flat_group_members=np.concatenate(group_members),
                )
            )
        prepared.append(
            _PreparedRepresentation(seed=int(seed), heads=heads, tasks=tuple(tasks))
        )
    return tuple(prepared)


def _sample_task_indices(task: _PreparedTask, rng: np.random.Generator) -> np.ndarray:
    group_count = len(task.group_sizes)
    selected = rng.integers(0, group_count, size=group_count)
    selected_sizes = task.group_sizes[selected]
    expanded_groups = np.repeat(selected, selected_sizes)
    random_offsets = np.floor(
        rng.random(len(expanded_groups)) * task.group_sizes[expanded_groups]
    ).astype(int)
    return task.flat_group_members[
        task.group_starts[expanded_groups] + random_offsets
    ]


def _task_effects(task: _PreparedTask, indices: np.ndarray) -> np.ndarray:
    candidate = np.mean(task.candidate[:, indices], axis=1)
    baseline = np.mean(task.baseline[:, indices], axis=1)
    if np.any(candidate <= 0.0) or np.any(baseline <= 0.0):
        raise ValueError("sampled property-view task mean is not positive")
    return np.log(candidate / baseline)


def _point_estimate(prepared: tuple[_PreparedRepresentation, ...]) -> float:
    representation_values: list[float] = []
    for representation in prepared:
        task_values = [
            _task_effects(task, np.arange(task.candidate.shape[1], dtype=int))
            for task in representation.tasks
        ]
        representation_values.append(float(np.mean(np.stack(task_values))))
    return float(np.mean(representation_values))


def paired_property_view_hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    metric_column: str = "property_prediction_std",
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> PropertyViewBootstrapResult:
    """Task-equal bootstrap with heads nested inside representation seeds."""

    if paired.empty:
        raise ValueError("paired property-view records are empty")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    prepared = _prepare_bootstrap(paired, metric_column)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=float)
    for draw_index in range(n_resamples):
        representation_values: list[float] = []
        selected_representations = rng.integers(
            0, len(prepared), size=len(prepared)
        )
        for representation_index in selected_representations:
            representation = prepared[int(representation_index)]
            effects = [
                _task_effects(task, _sample_task_indices(task, rng))
                for task in representation.tasks
            ]
            head_indices = rng.integers(
                0, len(representation.heads), size=len(representation.heads)
            )
            representation_values.append(
                float(np.mean(np.stack(effects, axis=0)[:, head_indices]))
            )
        draws[draw_index] = float(np.mean(representation_values))
    estimate = _point_estimate(prepared)
    lower, upper = np.quantile(draws, (0.025, 0.975))
    left = (np.count_nonzero(draws <= 0.0) + 1.0) / (len(draws) + 1.0)
    right = (np.count_nonzero(draws >= 0.0) + 1.0) / (len(draws) + 1.0)
    return PropertyViewBootstrapResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        probability_candidate_better=float(np.mean(draws < 0.0)),
        p_value_two_sided=float(min(1.0, 2.0 * min(left, right))),
        resamples=int(n_resamples),
    )


def _aggregate_row(
    cell: pd.DataFrame,
    *,
    comparison: str,
    candidate: str,
    baseline: str,
    view_family: str,
    metric: str,
    analysis: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    result = paired_property_view_hierarchical_bootstrap(
        cell,
        metric_column=metric,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        "analysis": analysis,
        "comparison": comparison,
        "candidate": candidate,
        "baseline": baseline,
        "view_family": str(view_family),
        "metric": (
            "within_record_prediction_std"
            if metric == "property_prediction_std"
            else metric
        ),
        **result.as_dict(),
        "n_tasks": int(cell["task"].nunique()),
        "n_representation_seeds": int(cell["representation_seed"].nunique()),
        "n_head_seeds": int(cell["head_seed"].nunique()),
        "n_paired_record_rows": int(len(cell)),
        "n_unique_records": int(
            cell[["task", "fold", "sample_id"]].drop_duplicates().shape[0]
        ),
        "tasks": ";".join(sorted(cell["task"].astype(str).unique())),
        "input_view": "structure_only",
        "multiplicity_adjustment": "none_secondary_mechanism_analysis",
    }


def _common_view_cohort(paired: pd.DataFrame) -> pd.DataFrame:
    identity = [
        "representation_seed",
        "head_seed",
        "task",
        "protocol",
        "fold",
        "sample_id",
    ]
    counts = paired.groupby(identity)["view_family"].transform("nunique")
    required_families = paired["view_family"].nunique()
    return paired.loc[counts.eq(required_families)].copy()


def _multi_structure_task_scope(paired: pd.DataFrame) -> pd.DataFrame:
    """Retain tasks with more than one independent held-out structure group."""

    working = paired.copy()
    working["_structure_group"] = (
        working["fold"].astype(str) + "::" + working["group_id"].astype(str)
    )
    group_counts = working.groupby("task")["_structure_group"].nunique()
    retained = set(group_counts.loc[group_counts.gt(1)].index.astype(str))
    return working.loc[working["task"].astype(str).isin(retained)].drop(
        columns="_structure_group"
    )


def run_property_view_comparisons(
    manifest: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the three method contrasts separately for both equivalence axes."""

    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for comparison_index, (comparison, (candidate, baseline)) in enumerate(
        PRIMARY_COMPARISONS.items()
    ):
        paired = load_primary_paired_property_views(manifest, candidate, baseline)
        primary_paired = _multi_structure_task_scope(paired)
        for family_index, (view_family, cell) in enumerate(
            primary_paired.groupby("view_family", sort=True)
        ):
            rows.append(
                _aggregate_row(
                    cell,
                    comparison=comparison,
                    candidate=candidate,
                    baseline=baseline,
                    view_family=str(view_family),
                    metric="property_prediction_std",
                    analysis="multi_structure_tasks",
                    n_resamples=n_resamples,
                    seed=bootstrap_seed + 100 + comparison_index * 10 + family_index,
                )
            )
        for family_index, (view_family, cell) in enumerate(
            paired.groupby("view_family", sort=True)
        ):
            sensitivity_rows.append(
                _aggregate_row(
                    cell,
                    comparison=comparison,
                    candidate=candidate,
                    baseline=baseline,
                    view_family=str(view_family),
                    metric="property_prediction_std",
                    analysis="include_single_structure_pe_i",
                    n_resamples=n_resamples,
                    seed=bootstrap_seed + comparison_index * 10 + family_index,
                )
            )
            sensitivity_rows.append(
                _aggregate_row(
                    cell,
                    comparison=comparison,
                    candidate=candidate,
                    baseline=baseline,
                    view_family=str(view_family),
                    metric="mean_absolute_view_delta",
                    analysis="mean_absolute_delta_all_eligible_tasks",
                    n_resamples=n_resamples,
                    seed=bootstrap_seed + 200 + comparison_index * 10 + family_index,
                )
            )
        common = _common_view_cohort(paired)
        for family_index, (view_family, cell) in enumerate(
            common.groupby("view_family", sort=True)
        ):
            sensitivity_rows.append(
                _aggregate_row(
                    cell,
                    comparison=comparison,
                    candidate=candidate,
                    baseline=baseline,
                    view_family=str(view_family),
                    metric="property_prediction_std",
                    analysis="common_record_cohort_all_eligible_tasks",
                    n_resamples=n_resamples,
                    seed=bootstrap_seed + 300 + comparison_index * 10 + family_index,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(sensitivity_rows)


def write_property_view_analysis_outputs(
    manifest_path: str | Path,
    output: str | Path,
    *,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    progress_every: int = 500,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path).resolve()
    manifest = load_property_view_manifest(manifest_source)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)

    task_metrics, inventory = stream_property_view_task_metrics(
        manifest, progress_every=progress_every
    )
    task_effects = compute_property_view_task_effects(task_metrics)
    aggregates, sensitivities = run_property_view_comparisons(
        manifest,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    atomic_write_csv(task_metrics, destination / "task_metrics.csv")
    atomic_write_csv(task_effects, destination / "task_effects.csv")
    atomic_write_csv(aggregates, destination / "comparison_aggregate.csv")
    atomic_write_csv(sensitivities, destination / "sensitivity_aggregate.csv")
    atomic_write_csv(inventory, destination / "prediction_inventory.csv")

    base_scope = (
        inventory.loc[
            inventory["representation_label"].astype(str).eq("C_fixed")
            & inventory["head_seed"].astype(int).eq(11)
        ]
        .loc[
            :,
            [
                "task",
                "protocol",
                "fold",
                "input_view",
                "test_records",
                "eligible_test_records",
                "status",
            ],
        ]
        .drop_duplicates()
    )
    family_scope = (
        task_metrics.groupby(
            ["task", "protocol", "fold", "input_view", "view_family"],
            as_index=False,
        )
        .agg(
            records_with_generated_views=("n_records", "max"),
            structure_groups_with_generated_views=("n_groups", "max"),
            generated_equivalent_views=("n_equivalent_views", "max"),
        )
    )
    scope = base_scope.merge(
        family_scope,
        on=["task", "protocol", "fold", "input_view"],
        how="left",
    )
    atomic_write_csv(scope, destination / "scope.csv")
    figure_scope = task_effects.loc[
        task_effects["input_view"].astype(str).eq("structure_only")
        & np.isclose(pd.to_numeric(task_effects["label_fraction"]), 1.0)
        & pd.to_numeric(task_effects["subsample_id"], errors="raise").astype(int).eq(0)
    ].copy()
    figure_blocks: list[pd.DataFrame] = []
    for task_name, task_cell in figure_scope.groupby("task", sort=True):
        protocol = _primary_protocol(str(task_name), task_cell["protocol"])
        if protocol is not None:
            figure_blocks.append(
                task_cell.loc[task_cell["protocol"].astype(str).eq(protocol)]
            )
    figure_scope = (
        pd.concat(figure_blocks, ignore_index=True)
        if figure_blocks
        else figure_scope.iloc[0:0]
    )
    figure_scope = figure_scope.loc[figure_scope["n_groups"].astype(int).gt(1)]
    figure_data = (
        figure_scope
        .groupby(
            ["comparison", "candidate", "baseline", "view_family", "task"],
            as_index=False,
        )
        .agg(
            paired_log_ratio=(
                "within_record_prediction_std_paired_log_ratio",
                "mean",
            ),
            representation_seeds=("representation_seed", "nunique"),
            head_seeds=("head_seed", "nunique"),
            paired_record_instances=("n_paired_records", "sum"),
        )
    )
    figure_data["percent_change"] = 100.0 * (
        np.exp(figure_data["paired_log_ratio"]) - 1.0
    )
    atomic_write_csv(figure_data, destination / "figure4d_task_data.csv")

    nonempty = inventory.loc[inventory["status"].eq("nonempty")]
    summary = {
        "source": "record-level summaries of verified equivalent property views",
        "manifest": str(manifest_source),
        "manifest_rows": int(len(manifest)),
        "record_files": int(len(inventory)),
        "nonempty_record_files": int(len(nonempty)),
        "no_eligible_view_files": int(len(inventory) - len(nonempty)),
        "record_rows": int(nonempty["records"].sum()),
        "methods": sorted(task_metrics["method"].astype(str).unique()),
        "tasks_with_eligible_views": sorted(task_metrics["task"].astype(str).unique()),
        "view_families": sorted(task_metrics["view_family"].astype(str).unique()),
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_metric": "within_record_prediction_std",
        "comparison_scope": "full-data structure-only test records; one declared protocol per task; tasks require more than one independent structure group",
        "multiplicity_adjustment": "none; this is a secondary mechanism analysis",
        "sensitivities": [
            "include PE-I as a transparent single-structure task sensitivity",
            "common traversal/periodic-cut record cohort",
            "mean absolute view delta",
        ],
    }
    atomic_write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        destination / "summary.json",
    )
    return summary


__all__ = [
    "PropertyViewBootstrapResult",
    "compute_property_view_task_effects",
    "load_primary_paired_property_views",
    "load_property_view_manifest",
    "paired_property_view_hierarchical_bootstrap",
    "run_property_view_comparisons",
    "select_primary_property_view_manifest",
    "stream_property_view_task_metrics",
    "write_property_view_analysis_outputs",
]
