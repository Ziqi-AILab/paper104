"""Prediction-level statistics for the three Paper104 primary comparisons.

The functions in this module deliberately start from held-out, per-sample
predictions. Aggregate metric files are outputs, never inputs, of the formal
analysis.
"""

from __future__ import annotations

import json
import gc
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from polyfid.analysis.multiple_testing import holm_adjust
from polyfid.evaluation.bootstrap import PRIMARY_COMPARISONS
from polyfid.evaluation.metrics import compute_metric, log_error_ratio
from polyfid.evaluation.predictions import load_predictions


STANDARD_SLICE_COLUMNS = ("split", "label_fraction", "subsample_id", "input_view")

# PredictionRecord permits extra columns. These names are carried through
# when present so distinct evaluation protocols are not silently pooled in the
# task-level tables.
OPTIONAL_CONDITION_COLUMNS = (
    "protocol",
    "evaluation_protocol",
    "split_protocol",
    "feature_view",
    "adaptation",
    "model_variant",
    "checkpoint_fraction",
    "checkpoint_tokens",
    "structural_fidelity_eligible",
    "scope_class",
)

REPORTING_SCOPE_COLUMNS = ("fold", "family_id", "cluster_id")

PAIR_ID_COLUMNS = (
    "sample_id",
    "task",
    "split",
    "representation_seed",
    "head_seed",
    "fold",
    "label_fraction",
    "subsample_id",
    "input_view",
)

RETENTION_TASKS = {"esol", "freesolv", "lipophilicity", "zinc", "zinc_mlm"}

PRIMARY_PREDICTION_COLUMNS = (
    "sample_id",
    "task",
    "split",
    "group_id",
    "y_true",
    "y_pred",
    "method",
    "representation_seed",
    "head_seed",
    "protocol",
    "family_id",
    "cluster_id",
    "fold",
    "label_fraction",
    "subsample_id",
    "input_view",
)

MANIFEST_CELL_COLUMNS = (
    "representation_seed",
    "task",
    "protocol",
    "input_view",
    "fold",
    "label_fraction",
    "subsample_id",
)


@dataclass(frozen=True)
class AggregateBootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    probability_candidate_better: float
    p_value_two_sided: float
    resamples: int
    valid_resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def discover_prediction_files(paths: Iterable[str | Path]) -> list[Path]:
    """Resolve explicit prediction files and directories in stable order."""

    discovered: list[Path] = []
    for value in paths:
        path = Path(value)
        if path.is_dir():
            discovered.extend(path.rglob("*.parquet"))
            discovered.extend(path.rglob("*.jsonl"))
        elif path.suffix in {".parquet", ".jsonl"}:
            discovered.append(path)
        else:
            raise ValueError(f"prediction input is not a parquet/jsonl file or directory: {path}")
    unique = sorted({path.resolve() for path in discovered})
    if not unique:
        raise ValueError("no per-sample prediction files were found")
    return unique


def load_prediction_inputs(paths: Iterable[str | Path]) -> tuple[pd.DataFrame, list[Path]]:
    files = discover_prediction_files(paths)
    blocks: list[pd.DataFrame] = []
    for path in files:
        block = load_predictions(path)
        block["source_path"] = str(path)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True, sort=False), files


def load_prediction_manifest(path: str | Path) -> pd.DataFrame:
    """Read the formal frozen-probe manifest without discovering adjacent files."""

    source = Path(path)
    separator = "\t" if source.suffix.casefold() in {".tsv", ".txt"} else ","
    manifest = pd.read_csv(source, sep=separator)
    required = {
        "output",
        "arm",
        "representation_seed",
        "task",
        "protocol",
        "input_view",
        "fold",
        "label_fraction",
        "subsample_id",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"prediction manifest is missing columns: {missing}")
    if manifest["output"].duplicated().any():
        raise ValueError("prediction manifest contains duplicate output paths")
    if manifest.duplicated(["arm", *MANIFEST_CELL_COLUMNS]).any():
        raise ValueError("prediction manifest contains a duplicate formal run cell")
    return manifest


def infer_task_metric(task: str, overrides: Mapping[str, str] | None = None) -> str:
    """Use the declared Article A metric convention, with explicit overrides."""

    overrides = overrides or {}
    exact = {str(key).casefold(): str(value).casefold() for key, value in overrides.items()}
    name = str(task).casefold()
    if name in exact:
        return exact[name]
    short_name = name.rsplit(":", maxsplit=1)[-1].replace("-", "_")
    if short_name in {"ea", "ip"}:
        return "mae"
    return "rmse"


def parse_metric_overrides(values: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"metric override must have TASK=METRIC form: {value}")
        task, metric = value.split("=", maxsplit=1)
        if not task.strip() or not metric.strip():
            raise ValueError(f"metric override must have TASK=METRIC form: {value}")
        parsed[task.strip()] = metric.strip().casefold()
    return parsed


def _present_condition_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in OPTIONAL_CONDITION_COLUMNS
        if column in frame.columns and frame[column].notna().any()
    ]


def _present_reporting_scope_columns(frame: pd.DataFrame) -> list[str]:
    """Keep evaluation folds and protocol-specific held-out strata visible."""

    columns: list[str] = []
    if "fold" in frame and frame["fold"].notna().any():
        columns.append("fold")
    protocols = (
        frame["protocol"].fillna("").astype(str).str.casefold()
        if "protocol" in frame
        else pd.Series("", index=frame.index)
    )
    if (
        "family_id" in frame
        and frame["family_id"].notna().any()
        and protocols.eq("family_ood").any()
    ):
        columns.append("family_id")
    if (
        "cluster_id" in frame
        and frame["cluster_id"].notna().any()
        and protocols.str.contains("cluster", regex=False).any()
    ):
        columns.append("cluster_id")
    return columns


def compute_task_metrics(
    predictions: pd.DataFrame,
    *,
    metric_overrides: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute raw-unit metrics for each independent representation/head cell."""

    group_columns = [
        "method",
        "representation_seed",
        "head_seed",
        "task",
        *STANDARD_SLICE_COLUMNS,
        *_present_condition_columns(predictions),
        *_present_reporting_scope_columns(predictions),
    ]
    rows: list[dict[str, Any]] = []
    for keys, cell in predictions.groupby(group_columns, dropna=False, sort=True):
        values = dict(zip(group_columns, keys, strict=True))
        metric = infer_task_metric(str(values["task"]), metric_overrides)
        rows.append(
            {
                **values,
                "metric": metric,
                "value": compute_metric(
                    cell["y_true"].to_numpy(dtype=float),
                    cell["y_pred"].to_numpy(dtype=float),
                    metric,
                ),
                "n_rows": int(len(cell)),
                "n_groups": int(cell["group_id"].astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows)


def _validate_manifest_prediction(row: Any, frame: pd.DataFrame) -> None:
    expected_input_view = {
        "structure_only": "valid",
        "structure_plus_metadata": "structure_plus_metadata_secondary",
    }.get(str(row.input_view), str(row.input_view))
    expected = {
        "method": str(row.arm),
        "representation_seed": int(row.representation_seed),
        "task": str(row.task),
        "protocol": str(row.protocol),
        "fold": str(row.fold),
        "subsample_id": int(row.subsample_id),
        "input_view": expected_input_view,
    }
    for column, value in expected.items():
        observed = frame[column]
        if column in {"representation_seed", "subsample_id"}:
            matches = pd.to_numeric(observed, errors="raise").astype(int).eq(int(value))
        else:
            matches = observed.fillna("__none__").astype(str).eq(str(value))
        if not matches.all():
            raise ValueError(
                f"prediction file does not match manifest field {column}={value}"
            )
    if not np.allclose(
        pd.to_numeric(frame["label_fraction"], errors="raise").to_numpy(dtype=float),
        float(row.label_fraction),
    ):
        raise ValueError("prediction file does not match manifest label_fraction")


def stream_formal_summaries_from_manifest(
    manifest: pd.DataFrame,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    progress_every: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize formal prediction files one matched run cell at a time.

    The frozen-probe directory also contains validation-loss tables, so the
    manifest's ``output`` column is the sole input list.  Only the compact
    task metrics, paired effects, and inventory remain resident after each
    run cell.
    """

    metric_rows: list[dict[str, Any]] = []
    effect_blocks: list[pd.DataFrame] = []
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    total = len(manifest)
    processed = 0
    manifest_arms = set(manifest["arm"].astype(str))
    for _, run_rows in manifest.groupby(
        list(MANIFEST_CELL_COLUMNS), dropna=False, sort=True
    ):
        present_arms = set(run_rows["arm"].astype(str))
        for candidate, baseline in PRIMARY_COMPARISONS.values():
            if (
                candidate in manifest_arms
                and baseline in manifest_arms
                and (candidate in present_arms) != (baseline in present_arms)
            ):
                raise ValueError(
                    f"formal run cell is missing one side of {candidate}/{baseline}"
                )
        prediction_blocks: list[pd.DataFrame] = []
        for row in run_rows.itertuples(index=False):
            frame = load_predictions(Path(str(row.output)))
            _validate_manifest_prediction(row, frame)
            prediction_blocks.append(frame)
            # Family and cluster labels identify observations for the raw-row
            # bootstrap.  A reported metric remains one manifest fold/head cell.
            metric_frame = frame.drop(
                columns=["family_id", "cluster_id"], errors="ignore"
            )
            metric_rows.extend(
                compute_task_metrics(
                    metric_frame, metric_overrides=metric_overrides
                ).to_dict(orient="records")
            )
            for (method, task), cell in frame.groupby(
                ["method", "task"], sort=False
            ):
                key = (str(method), str(task))
                values = inventory.setdefault(
                    key,
                    {
                        "rows": 0,
                        "sample_ids": set(),
                        "representation_seeds": set(),
                        "head_seeds": set(),
                    },
                )
                values["rows"] += int(len(cell))
                values["sample_ids"].update(cell["sample_id"].astype(str).unique())
                values["representation_seeds"].update(
                    pd.to_numeric(cell["representation_seed"], errors="raise")
                    .astype(int)
                    .unique()
                )
                values["head_seeds"].update(
                    pd.to_numeric(cell["head_seed"], errors="raise")
                    .astype(int)
                    .unique()
                )
            processed += 1
            if progress_every > 0 and (
                processed % progress_every == 0 or processed == total
            ):
                print(f"streamed_prediction_files={processed}/{total}", flush=True)

        if len(prediction_blocks) > 1:
            effect_input = pd.concat(
                prediction_blocks, ignore_index=True, sort=False
            ).drop(columns=["family_id", "cluster_id"], errors="ignore")
            effects = compute_task_effects(
                effect_input,
                metric_overrides=metric_overrides,
                require_full_coverage=True,
            )
            if not effects.empty:
                effect_blocks.append(effects)

    inventory_rows = [
        {
            "method": method,
            "task": task,
            "rows": int(values["rows"]),
            "samples": int(len(values["sample_ids"])),
            "representation_seeds": int(len(values["representation_seeds"])),
            "head_seeds": int(len(values["head_seeds"])),
        }
        for (method, task), values in sorted(inventory.items())
    ]
    task_effects = (
        pd.concat(effect_blocks, ignore_index=True, sort=False)
        if effect_blocks
        else pd.DataFrame()
    )
    return pd.DataFrame(metric_rows), task_effects, pd.DataFrame(inventory_rows)


def stream_task_metrics_from_manifest(
    manifest: pd.DataFrame,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    progress_every: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_metrics, _, inventory = stream_formal_summaries_from_manifest(
        manifest,
        metric_overrides=metric_overrides,
        progress_every=progress_every,
    )
    return task_metrics, inventory


def _normalise_pair_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            output[column] = None
        if column in {"representation_seed", "head_seed", "subsample_id"}:
            output[column] = pd.to_numeric(output[column], errors="raise").astype(int)
        elif column == "label_fraction":
            output[column] = pd.to_numeric(output[column], errors="raise").astype(float)
        else:
            output[column] = output[column].fillna("__none__").astype(str)
    return output


def pair_method_predictions(
    predictions: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    require_full_coverage: bool = False,
) -> pd.DataFrame:
    """Pair two methods on the exact held-out observation and run identity."""

    optional = _present_condition_columns(predictions)
    keys = [*PAIR_ID_COLUMNS, *optional]
    candidate_rows = _normalise_pair_keys(
        predictions.loc[predictions["method"].eq(candidate)], keys
    )
    baseline_rows = _normalise_pair_keys(
        predictions.loc[predictions["method"].eq(baseline)], keys
    )
    if candidate_rows.empty or baseline_rows.empty:
        return pd.DataFrame()

    def unique_rows(frame: pd.DataFrame, method: str) -> pd.DataFrame:
        frame = frame.drop_duplicates(subset=[*keys, "y_true", "y_pred"])
        if frame.duplicated(keys, keep=False).any():
            raise ValueError(f"{method} has conflicting predictions for the same held-out row")
        return frame

    candidate_rows = unique_rows(candidate_rows, candidate)
    baseline_rows = unique_rows(baseline_rows, baseline)
    metadata = [
        column
        for column in ("group_id", "family_id", "cluster_id")
        if column in candidate_rows and column in baseline_rows
    ]
    left = candidate_rows.loc[:, [*keys, *metadata, "y_true", "y_pred"]].rename(
        columns={"y_true": "y_true_candidate", "y_pred": "y_candidate"}
    )
    right = baseline_rows.loc[:, [*keys, *metadata, "y_true", "y_pred"]].rename(
        columns={
            **{column: f"{column}_baseline" for column in metadata},
            "y_true": "y_true_baseline",
            "y_pred": "y_baseline",
        }
    )
    paired = left.merge(
        right,
        on=keys,
        how="outer" if require_full_coverage else "inner",
        validate="one_to_one",
        indicator=require_full_coverage,
    )
    if paired.empty:
        return paired
    if require_full_coverage and not paired["_merge"].eq("both").all():
        raise ValueError(
            f"{candidate} and {baseline} do not cover the same held-out rows"
        )
    if require_full_coverage:
        paired = paired.drop(columns="_merge")
    if not np.allclose(
        paired["y_true_candidate"].to_numpy(dtype=float),
        paired["y_true_baseline"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(f"{candidate} and {baseline} disagree on held-out labels")
    for column in metadata:
        baseline_column = f"{column}_baseline"
        if not paired[column].fillna("__none__").astype(str).eq(
            paired[baseline_column].fillna("__none__").astype(str)
        ).all():
            raise ValueError(
                f"{candidate} and {baseline} disagree on held-out {column}"
            )
        paired = paired.drop(columns=baseline_column)
    paired["y_true"] = paired.pop("y_true_candidate")
    paired = paired.drop(columns="y_true_baseline")
    paired["candidate"] = candidate
    paired["baseline"] = baseline
    return paired


def compute_task_effects(
    predictions: pd.DataFrame,
    *,
    comparisons: Mapping[str, tuple[str, str]] = PRIMARY_COMPARISONS,
    metric_overrides: Mapping[str, str] | None = None,
    require_full_coverage: bool = False,
) -> pd.DataFrame:
    """Return seed/head-level paired effects while preserving raw task units."""

    rows: list[dict[str, Any]] = []
    for comparison, (candidate, baseline) in comparisons.items():
        paired = pair_method_predictions(
            predictions,
            candidate,
            baseline,
            require_full_coverage=require_full_coverage,
        )
        if paired.empty:
            continue
        group_columns = [
            "representation_seed",
            "head_seed",
            "task",
            *STANDARD_SLICE_COLUMNS,
            *_present_condition_columns(paired),
            *_present_reporting_scope_columns(paired),
        ]
        for keys, cell in paired.groupby(group_columns, dropna=False, sort=True):
            values = dict(zip(group_columns, keys, strict=True))
            metric = infer_task_metric(str(values["task"]), metric_overrides)
            truth = cell["y_true"].to_numpy(dtype=float)
            candidate_value = compute_metric(
                truth, cell["y_candidate"].to_numpy(dtype=float), metric
            )
            baseline_value = compute_metric(
                truth, cell["y_baseline"].to_numpy(dtype=float), metric
            )
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate,
                    "baseline": baseline,
                    **values,
                    "metric": metric,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "paired_log_error_ratio": log_error_ratio(
                        candidate_value, baseline_value, metric
                    ),
                    "n_paired_rows": int(len(cell)),
                    "n_groups": int(cell["group_id"].astype(str).nunique()),
                }
            )
    return pd.DataFrame(rows)


def _metric_effect(cell: pd.DataFrame, metric: str) -> float:
    truth = cell["y_true"].to_numpy(dtype=float)
    candidate = compute_metric(truth, cell["y_candidate"].to_numpy(dtype=float), metric)
    baseline = compute_metric(truth, cell["y_baseline"].to_numpy(dtype=float), metric)
    return log_error_ratio(candidate, baseline, metric)


@dataclass(frozen=True)
class _PreparedTask:
    name: str
    metric: str
    truth: np.ndarray
    candidate: np.ndarray
    baseline: np.ndarray
    group_members: tuple[np.ndarray, ...]
    family_groups: tuple[np.ndarray, ...] | None


@dataclass(frozen=True)
class _PreparedRepresentation:
    seed: int
    heads: np.ndarray
    tasks: tuple[_PreparedTask, ...]


def _sequential_codes(values: Sequence[tuple[Any, ...]]) -> np.ndarray:
    mapping: dict[tuple[Any, ...], int] = {}
    codes = np.empty(len(values), dtype=int)
    for index, value in enumerate(values):
        if value not in mapping:
            mapping[value] = len(mapping)
        codes[index] = mapping[value]
    return codes


def _group_structure(
    groups: np.ndarray,
    families: np.ndarray | None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...] | None]:
    # A monomer pair can occur in more than one held-out family role. Under
    # family-OOD resampling, the statistical group is therefore the pair
    # within its selected family, matching the declared nesting order.
    group_labels = (
        [(str(family), str(group)) for family, group in zip(families, groups, strict=True)]
        if families is not None
        else [(str(value),) for value in groups]
    )
    group_codes = _sequential_codes(group_labels)
    group_members = tuple(np.flatnonzero(group_codes == code) for code in range(group_codes.max() + 1))
    if families is None:
        return group_members, None
    group_family: list[str] = []
    for members in group_members:
        observed = np.unique(families[members].astype(str))
        if observed.size != 1:
            raise ValueError("a structure group spans more than one held-out family")
        group_family.append(str(observed[0]))
    family_codes = _sequential_codes([(value,) for value in group_family])
    family_groups = tuple(
        np.flatnonzero(family_codes == code) for code in range(family_codes.max() + 1)
    )
    return group_members, family_groups


def _prepare_bootstrap(
    paired: pd.DataFrame,
    metric_overrides: Mapping[str, str] | None,
) -> tuple[_PreparedRepresentation, ...]:
    observation_columns = [
        column for column in PAIR_ID_COLUMNS if column != "head_seed" and column in paired
    ] + _present_condition_columns(paired)
    working = paired.copy()
    observation_values = list(
        working.loc[:, observation_columns].itertuples(index=False, name=None)
    )
    working["_observation_id"] = _sequential_codes(observation_values)
    prepared_representations: list[_PreparedRepresentation] = []
    for representation_seed, seed_cell in working.groupby("representation_seed", sort=True):
        task_cells = [cell for _, cell in seed_cell.groupby("task", sort=True)]
        common_heads = set(task_cells[0]["head_seed"].astype(int).unique())
        for task_cell in task_cells[1:]:
            common_heads.intersection_update(task_cell["head_seed"].astype(int).unique())
        heads = np.asarray(sorted(common_heads), dtype=int)
        if heads.size == 0:
            raise ValueError(f"representation seed {representation_seed} has no head shared across tasks")
        prepared_tasks: list[_PreparedTask] = []
        for task, task_cell in seed_cell.groupby("task", sort=True):
            reference = (
                task_cell.sort_values(["_observation_id", "head_seed"])
                .drop_duplicates("_observation_id")
                .set_index("_observation_id")
                .sort_index()
            )
            observation_ids = reference.index.to_numpy(dtype=int)
            candidate = (
                task_cell.pivot(index="_observation_id", columns="head_seed", values="y_candidate")
                .reindex(index=observation_ids, columns=heads)
                .to_numpy(dtype=float)
                .T
            )
            baseline = (
                task_cell.pivot(index="_observation_id", columns="head_seed", values="y_baseline")
                .reindex(index=observation_ids, columns=heads)
                .to_numpy(dtype=float)
                .T
            )
            if not np.isfinite(candidate).all() or not np.isfinite(baseline).all():
                raise ValueError(
                    f"task {task} is missing paired rows for one or more nested head seeds"
                )
            truth = reference["y_true"].to_numpy(dtype=float)
            family_values: np.ndarray | None = None
            family_ood = (
                "protocol" in reference
                and reference["protocol"].fillna("").astype(str).str.casefold().eq(
                    "family_ood"
                ).all()
            )
            if family_ood and "family_id" in reference and reference["family_id"].notna().any():
                family_values = reference["family_id"].fillna("__none__").astype(str).to_numpy()
            elif family_ood and "fold" in reference:
                # Aldeghi's held-out family can be stored as the fold identifier.
                family_values = reference["fold"].fillna("__none__").astype(str).to_numpy()
            group_members, family_groups = _group_structure(
                reference["group_id"].astype(str).to_numpy(), family_values
            )
            prepared_tasks.append(
                _PreparedTask(
                    name=str(task),
                    metric=infer_task_metric(str(task), metric_overrides),
                    truth=truth,
                    candidate=candidate,
                    baseline=baseline,
                    group_members=group_members,
                    family_groups=family_groups,
                )
            )
        prepared_representations.append(
            _PreparedRepresentation(
                seed=int(representation_seed), heads=heads, tasks=tuple(prepared_tasks)
            )
        )
    return tuple(prepared_representations)


def _sample_task_indices(task: _PreparedTask, rng: np.random.Generator) -> np.ndarray:
    selected_groups: list[int] = []
    if task.family_groups is None:
        count = len(task.group_members)
        selected_groups.extend(rng.integers(0, count, size=count).tolist())
    else:
        family_count = len(task.family_groups)
        for family_index in rng.integers(0, family_count, size=family_count):
            family_group_ids = task.family_groups[int(family_index)]
            chosen = family_group_ids[
                rng.integers(0, len(family_group_ids), size=len(family_group_ids))
            ]
            selected_groups.extend(chosen.tolist())
    if all(len(task.group_members[index]) == 1 for index in selected_groups):
        return np.fromiter(
            (int(task.group_members[index][0]) for index in selected_groups),
            dtype=int,
            count=len(selected_groups),
        )
    blocks = []
    for group_index in selected_groups:
        members = task.group_members[group_index]
        blocks.append(members[rng.integers(0, len(members), size=len(members))])
    return np.concatenate(blocks)


def _task_effects_by_head(task: _PreparedTask, indices: np.ndarray) -> np.ndarray:
    truth = task.truth[indices][None, :]
    candidate = task.candidate[:, indices]
    baseline = task.baseline[:, indices]
    if task.metric == "mae":
        candidate_metric = np.mean(np.abs(candidate - truth), axis=1)
        baseline_metric = np.mean(np.abs(baseline - truth), axis=1)
    elif task.metric == "rmse":
        candidate_metric = np.sqrt(np.mean(np.square(candidate - truth), axis=1))
        baseline_metric = np.sqrt(np.mean(np.square(baseline - truth), axis=1))
    else:
        candidate_metric = np.asarray(
            [compute_metric(truth.ravel(), values, task.metric) for values in candidate]
        )
        baseline_metric = np.asarray(
            [compute_metric(truth.ravel(), values, task.metric) for values in baseline]
        )
    return np.asarray(
        [
            log_error_ratio(float(candidate_value), float(baseline_value), task.metric)
            for candidate_value, baseline_value in zip(
                candidate_metric, baseline_metric, strict=True
            )
        ],
        dtype=float,
    )


def _nested_point_estimate(prepared: tuple[_PreparedRepresentation, ...]) -> float:
    representation_values = []
    for representation in prepared:
        task_by_head = [
            _task_effects_by_head(task, np.arange(len(task.truth), dtype=int))
            for task in representation.tasks
        ]
        representation_values.append(float(np.mean(np.stack(task_by_head, axis=0))))
    if not representation_values:
        raise ValueError("no paired representation/head/task cells were available")
    return float(np.mean(representation_values))


def paired_task_equal_hierarchical_bootstrap(
    paired: pd.DataFrame,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> AggregateBootstrapResult:
    """Bootstrap the task-equal paired log error ratio.

    Representation seeds are sampled as independent outer replicates. Head
    seeds are sampled inside each selected representation seed. Within every
    task/head cell, held-out families are sampled only for rows explicitly
    marked ``protocol=family_ood``; all other protocols start at structure
    groups. Rows are then sampled within groups.
    """

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if paired.empty:
        raise ValueError("paired prediction rows are empty")
    rng = np.random.default_rng(seed)
    prepared = _prepare_bootstrap(paired, metric_overrides)
    draws = np.full(n_resamples, np.nan, dtype=float)
    for draw_index in range(n_resamples):
        representation_values: list[float] = []
        selected_representations = rng.integers(0, len(prepared), size=len(prepared))
        for representation_index in selected_representations:
            representation = prepared[int(representation_index)]
            effects_before_head_sampling: list[np.ndarray] = []
            for task in representation.tasks:
                sampled_indices = _sample_task_indices(task, rng)
                try:
                    effects_before_head_sampling.append(
                        _task_effects_by_head(task, sampled_indices)
                    )
                except ValueError:
                    # Some ranking metrics are undefined in one-class draws.
                    continue
            # Head seeds are the final nested layer, after family/group/row
            # resampling has fixed the held-out observations for this draw.
            head_indices = rng.integers(
                0, len(representation.heads), size=len(representation.heads)
            )
            if effects_before_head_sampling:
                task_values = [values[head_indices] for values in effects_before_head_sampling]
                representation_values.append(float(np.mean(np.stack(task_values, axis=0))))
        if representation_values:
            draws[draw_index] = float(np.mean(representation_values))

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise ValueError("all bootstrap draws were undefined")
    estimate = _nested_point_estimate(prepared)
    lower, upper = np.quantile(finite, (0.025, 0.975))
    left = (np.count_nonzero(finite <= 0.0) + 1.0) / (finite.size + 1.0)
    right = (np.count_nonzero(finite >= 0.0) + 1.0) / (finite.size + 1.0)
    return AggregateBootstrapResult(
        estimate=estimate,
        ci_lower=float(lower),
        ci_upper=float(upper),
        probability_candidate_better=float(np.mean(finite < 0.0)),
        p_value_two_sided=float(min(1.0, 2.0 * min(left, right))),
        resamples=int(n_resamples),
        valid_resamples=int(finite.size),
    )


def select_primary_scope(
    paired: pd.DataFrame,
    *,
    label_fraction: float = 1.0,
    subsample_id: int = 0,
    input_view: str = "valid",
    filters: Mapping[str, str] | None = None,
    task_protocols: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    selected = paired.loc[
        np.isclose(paired["label_fraction"].to_numpy(dtype=float), label_fraction)
        & paired["subsample_id"].astype(int).eq(int(subsample_id))
        & paired["input_view"].astype(str).eq(str(input_view))
        & paired["split"].astype(str).eq("test")
    ]
    task_names = selected["task"].astype(str).str.casefold()
    # OpenPoly is an external applicability analysis and molecular tasks measure
    # retention. Both remain in raw task tables but do not enter the main
    # polymer-task aggregate.
    short_task_names = task_names.str.rsplit(":", n=1).str[-1]
    selected = selected.loc[
        ~task_names.str.startswith("openpoly:")
        & ~task_names.isin(RETENTION_TASKS)
        & ~short_task_names.isin(RETENTION_TASKS)
    ]
    for column, value in (filters or {}).items():
        if column not in selected:
            raise ValueError(f"primary aggregate filter column is absent: {column}")
        selected = selected.loc[selected[column].astype(str).eq(str(value))]
    if "protocol" in selected and not selected.empty:
        requested = {str(task).casefold(): str(protocol) for task, protocol in (task_protocols or {}).items()}
        blocks: list[pd.DataFrame] = []
        for task, task_cell in selected.groupby("task", sort=True):
            protocols = sorted(task_cell["protocol"].astype(str).unique())
            task_name = str(task).casefold()
            short_name = task_name.rsplit(":", maxsplit=1)[-1].replace("-", "_")
            chosen = requested.get(task_name, requested.get(short_name))
            if chosen is None and short_name in {"ea", "ip"} and "family_ood" in protocols:
                chosen = "family_ood"
            if chosen is None and "group_ood" in protocols:
                chosen = "group_ood"
            if chosen is None and len(protocols) == 1:
                chosen = protocols[0]
            if chosen is not None and chosen in protocols:
                blocks.append(task_cell.loc[task_cell["protocol"].astype(str).eq(chosen)])
        selected = pd.concat(blocks, ignore_index=True) if blocks else selected.iloc[0:0]
    return selected


def run_primary_analysis(
    predictions: pd.DataFrame,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    primary_label_fraction: float = 1.0,
    primary_subsample_id: int = 0,
    primary_input_view: str = "valid",
    primary_filters: Mapping[str, str] | None = None,
    primary_task_protocols: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for offset, (comparison, (candidate, baseline)) in enumerate(PRIMARY_COMPARISONS.items()):
        paired = pair_method_predictions(predictions, candidate, baseline)
        scoped = (
            select_primary_scope(
                paired,
                label_fraction=primary_label_fraction,
                subsample_id=primary_subsample_id,
                input_view=primary_input_view,
                filters=primary_filters,
                task_protocols=primary_task_protocols,
            )
            if not paired.empty
            else paired
        )
        if scoped.empty:
            unavailable[comparison] = (
                f"no paired {candidate}/{baseline} test predictions in the declared primary scope"
            )
            continue
        result = paired_task_equal_hierarchical_bootstrap(
            scoped,
            metric_overrides=metric_overrides,
            n_resamples=n_resamples,
            seed=bootstrap_seed + offset,
        )
        rows.append(
            {
                "comparison": comparison,
                "candidate": candidate,
                "baseline": baseline,
                **result.as_dict(),
                "n_tasks": int(scoped["task"].nunique()),
                "n_representation_seeds": int(scoped["representation_seed"].nunique()),
                "n_head_seeds": int(scoped["head_seed"].nunique()),
                "n_paired_rows": int(len(scoped)),
                "tasks": ";".join(sorted(scoped["task"].astype(str).unique())),
                "primary_label_fraction": float(primary_label_fraction),
                "primary_subsample_id": int(primary_subsample_id),
                "primary_input_view": str(primary_input_view),
                "task_protocols": ";".join(
                    f"{task}={protocol}"
                    for task, protocol in sorted(
                        scoped.groupby("task")["protocol"].first().astype(str).to_dict().items()
                    )
                )
                if "protocol" in scoped
                else "",
            }
        )
    aggregates = pd.DataFrame(rows)
    if not aggregates.empty:
        adjusted = holm_adjust(
            dict(zip(aggregates["comparison"], aggregates["p_value_two_sided"], strict=True))
        )
        aggregates["holm_adjusted_p"] = aggregates["comparison"].map(adjusted)
    return aggregates, unavailable


def select_primary_manifest_scope(
    manifest: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    task_protocols: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Select only files that can contribute to the declared main aggregate."""

    selected = manifest.loc[manifest["arm"].astype(str).isin([candidate, baseline])].copy()
    selected = selected.loc[
        np.isclose(pd.to_numeric(selected["label_fraction"]), 1.0)
        & pd.to_numeric(selected["subsample_id"], errors="raise").astype(int).eq(0)
        & selected["input_view"].astype(str).eq("structure_only")
    ]
    task_names = selected["task"].astype(str).str.casefold()
    short_names = task_names.str.rsplit(":", n=1).str[-1].str.replace("-", "_", regex=False)
    selected = selected.loc[
        ~task_names.str.startswith("openpoly:")
        & ~task_names.isin(RETENTION_TASKS)
        & ~short_names.isin(RETENTION_TASKS)
    ]
    requested = {
        str(task).casefold(): str(protocol)
        for task, protocol in (task_protocols or {}).items()
    }
    blocks: list[pd.DataFrame] = []
    for task, task_cell in selected.groupby("task", sort=True):
        protocols = sorted(task_cell["protocol"].astype(str).unique())
        task_name = str(task).casefold()
        short_name = task_name.rsplit(":", maxsplit=1)[-1].replace("-", "_")
        chosen = requested.get(task_name, requested.get(short_name))
        if chosen is None and short_name in {"ea", "ip"} and "family_ood" in protocols:
            chosen = "family_ood"
        if chosen is None and "group_ood" in protocols:
            chosen = "group_ood"
        if chosen is None and len(protocols) == 1:
            chosen = protocols[0]
        if chosen is not None and chosen in protocols:
            blocks.append(task_cell.loc[task_cell["protocol"].astype(str).eq(chosen)])
    return pd.concat(blocks, ignore_index=True) if blocks else selected.iloc[0:0]


def _load_primary_prediction_files(
    manifest_rows: pd.DataFrame,
    *,
    extra_columns: Iterable[str] = (),
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*PRIMARY_PREDICTION_COLUMNS, *extra_columns]))
    blocks: list[pd.DataFrame] = []
    for path in manifest_rows["output"].astype(str):
        source = Path(path)
        if source.suffix == ".parquet":
            blocks.append(pd.read_parquet(source, columns=columns))
        else:
            blocks.append(load_predictions(source).loc[:, columns])
    if not blocks:
        return pd.DataFrame(columns=columns)
    return pd.concat(blocks, ignore_index=True, sort=False)


def run_primary_analysis_from_manifest(
    manifest: pd.DataFrame,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    primary_filters: Mapping[str, str] | None = None,
    primary_task_protocols: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run the three bootstraps while holding only one method pair in memory."""

    rows: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    for offset, (comparison, (candidate, baseline)) in enumerate(PRIMARY_COMPARISONS.items()):
        manifest_rows = select_primary_manifest_scope(
            manifest,
            candidate,
            baseline,
            task_protocols=primary_task_protocols,
        )
        if manifest_rows.empty:
            unavailable[comparison] = (
                f"no paired {candidate}/{baseline} prediction files in the declared primary scope"
            )
            continue
        print(
            f"loading_primary_comparison={comparison} files={len(manifest_rows)}",
            flush=True,
        )
        predictions = _load_primary_prediction_files(
            manifest_rows,
            extra_columns=(primary_filters or {}).keys(),
        )
        paired = pair_method_predictions(
            predictions,
            candidate,
            baseline,
            require_full_coverage=True,
        )
        scoped = select_primary_scope(
            paired,
            filters=primary_filters,
            task_protocols=primary_task_protocols,
        )
        if scoped.empty:
            unavailable[comparison] = (
                f"no paired {candidate}/{baseline} test predictions in the declared primary scope"
            )
            del predictions, paired, scoped
            gc.collect()
            continue
        print(
            f"bootstrapping_primary_comparison={comparison} paired_rows={len(scoped)} "
            f"resamples={n_resamples}",
            flush=True,
        )
        result = paired_task_equal_hierarchical_bootstrap(
            scoped,
            metric_overrides=metric_overrides,
            n_resamples=n_resamples,
            seed=bootstrap_seed + offset,
        )
        rows.append(
            {
                "comparison": comparison,
                "candidate": candidate,
                "baseline": baseline,
                **result.as_dict(),
                "n_tasks": int(scoped["task"].nunique()),
                "n_representation_seeds": int(scoped["representation_seed"].nunique()),
                "n_head_seeds": int(scoped["head_seed"].nunique()),
                "n_paired_rows": int(len(scoped)),
                "tasks": ";".join(sorted(scoped["task"].astype(str).unique())),
                "primary_label_fraction": 1.0,
                "primary_subsample_id": 0,
                "primary_input_view": "valid",
                "task_protocols": ";".join(
                    f"{task}={protocol}"
                    for task, protocol in sorted(
                        scoped.groupby("task")["protocol"].first().astype(str).to_dict().items()
                    )
                )
                if "protocol" in scoped
                else "",
            }
        )
        del predictions, paired, scoped
        gc.collect()

    aggregates = pd.DataFrame(rows)
    if not aggregates.empty:
        adjusted = holm_adjust(
            dict(zip(aggregates["comparison"], aggregates["p_value_two_sided"], strict=True))
        )
        aggregates["holm_adjusted_p"] = aggregates["comparison"].map(adjusted)
    return aggregates, unavailable


def write_manifest_analysis_outputs(
    prediction_manifest: str | Path,
    output: str | Path,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    primary_filters: Mapping[str, str] | None = None,
    primary_task_protocols: Mapping[str, str] | None = None,
    progress_every: int = 500,
) -> dict[str, Any]:
    """Write complete formal statistics without loading the full corpus at once."""

    source = Path(prediction_manifest).resolve()
    manifest = load_prediction_manifest(source)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)

    task_metrics, task_effects, inventory = stream_formal_summaries_from_manifest(
        manifest,
        metric_overrides=metric_overrides,
        progress_every=progress_every,
    )
    task_metrics.to_csv(destination / "task_metrics.csv", index=False)
    inventory.to_csv(destination / "prediction_inventory.csv", index=False)

    task_effects.to_csv(destination / "task_effects.csv", index=False)
    aggregates, unavailable = run_primary_analysis_from_manifest(
        manifest,
        metric_overrides=metric_overrides,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        primary_filters=primary_filters,
        primary_task_protocols=primary_task_protocols,
    )
    aggregates.to_csv(destination / "primary_aggregate.csv", index=False)
    summary: dict[str, Any] = {
        "source": "per-sample prediction records listed by the formal manifest",
        "prediction_manifest": str(source),
        "prediction_files": int(len(manifest)),
        "prediction_rows": int(inventory["rows"].sum()),
        "methods": sorted(task_metrics["method"].astype(str).unique()),
        "tasks": sorted(task_metrics["task"].astype(str).unique()),
        "metric_overrides": dict(metric_overrides or {}),
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "primary_filters": dict(primary_filters or {}),
        "primary_task_protocols": dict(primary_task_protocols or {}),
        "completed_primary_comparisons": aggregates["comparison"].tolist()
        if not aggregates.empty
        else [],
        "unavailable_primary_comparisons": unavailable,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def write_analysis_outputs(
    predictions: pd.DataFrame,
    prediction_files: Sequence[Path],
    output: str | Path,
    *,
    metric_overrides: Mapping[str, str] | None = None,
    n_resamples: int = 10_000,
    bootstrap_seed: int = 2026,
    primary_filters: Mapping[str, str] | None = None,
    primary_task_protocols: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    task_metrics = compute_task_metrics(predictions, metric_overrides=metric_overrides)
    task_effects = compute_task_effects(predictions, metric_overrides=metric_overrides)
    aggregates, unavailable = run_primary_analysis(
        predictions,
        metric_overrides=metric_overrides,
        n_resamples=n_resamples,
        bootstrap_seed=bootstrap_seed,
        primary_filters=primary_filters,
        primary_task_protocols=primary_task_protocols,
    )
    task_metrics.to_csv(destination / "task_metrics.csv", index=False)
    task_effects.to_csv(destination / "task_effects.csv", index=False)
    aggregates.to_csv(destination / "primary_aggregate.csv", index=False)
    inventory = (
        predictions.groupby(["method", "task"], dropna=False)
        .agg(
            rows=("sample_id", "size"),
            samples=("sample_id", "nunique"),
            representation_seeds=("representation_seed", "nunique"),
            head_seeds=("head_seed", "nunique"),
        )
        .reset_index()
    )
    inventory.to_csv(destination / "prediction_inventory.csv", index=False)
    summary: dict[str, Any] = {
        "source": "per-sample prediction records",
        "prediction_files": [str(path) for path in prediction_files],
        "prediction_rows": int(len(predictions)),
        "methods": sorted(predictions["method"].astype(str).unique()),
        "tasks": sorted(predictions["task"].astype(str).unique()),
        "metric_overrides": dict(metric_overrides or {}),
        "bootstrap_resamples": int(n_resamples),
        "bootstrap_seed": int(bootstrap_seed),
        "primary_filters": dict(primary_filters or {}),
        "primary_task_protocols": dict(primary_task_protocols or {}),
        "completed_primary_comparisons": aggregates["comparison"].tolist()
        if not aggregates.empty
        else [],
        "unavailable_primary_comparisons": unavailable,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


__all__ = [
    "AggregateBootstrapResult",
    "compute_task_effects",
    "compute_task_metrics",
    "discover_prediction_files",
    "infer_task_metric",
    "load_prediction_manifest",
    "load_prediction_inputs",
    "pair_method_predictions",
    "paired_task_equal_hierarchical_bootstrap",
    "parse_metric_overrides",
    "run_primary_analysis",
    "run_primary_analysis_from_manifest",
    "select_primary_scope",
    "select_primary_manifest_scope",
    "stream_formal_summaries_from_manifest",
    "stream_task_metrics_from_manifest",
    "write_manifest_analysis_outputs",
    "write_analysis_outputs",
]
