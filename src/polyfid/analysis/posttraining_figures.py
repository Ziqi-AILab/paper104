"""Compact descriptive tables and supplementary post-training figures.

These builders consume fold-level metrics derived from prediction records.
They do not replace the seven-arm frozen-probe primary analysis, or estimate
bootstrap intervals from aggregate metrics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from polyfid.analysis.manuscript import latex_escape


SOURCE_GROUPS = ("adaptation", "chemprop", "classical", "transpolymer", "periogt", "grin")
EXTERNAL_GROUPS = ("transpolymer", "periogt", "grin")
CONDITION_COLUMNS = (
    "task", "protocol", "split", "label_fraction", "subsample_id", "input_view", "metric"
)
FOCUS_TASKS = ("EA", "IP", "EGC", "XC")
DETERMINISTIC_METHODS = {
    "baseline_ecfp_ridge", "baseline_rdkit_ridge", "baseline_tanimoto_nn",
    "baseline_nearest_neighbor",
}


def pool_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pool held-out error instances before averaging independent model fits.

    In particular, RMSE is reconstructed from sum of squared errors; averaging
    fold RMSEs would answer a different question. Family-OOD rows can represent
    more than one held-out role, so ``n_rows`` counts prediction instances,
    not necessarily distinct structures across all folds.
    """

    keys = ["source_group", "method", "representation_seed", "head_seed", *CONDITION_COLUMNS]
    rows = []
    for values, cell in metrics.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, values, strict=True))
        metric = str(record["metric"]).casefold()
        if metric not in {"mae", "rmse"}:
            raise ValueError(f"fold pooling supports MAE/RMSE, not {metric}")
        weights = cell["n_rows"].to_numpy(dtype=float)
        errors = cell["value"].to_numpy(dtype=float)
        pooled = float(np.average(errors if metric == "mae" else errors**2, weights=weights))
        rows.append({
            **record,
            "value": pooled if metric == "mae" else float(np.sqrt(pooled)),
            "n_rows": int(weights.sum()),
            "n_folds": int(cell["fold"].nunique()),
            "folds": ";".join(sorted(cell["fold"].astype(str).unique())),
        })
    return pd.DataFrame(rows)


def summarize_replicates(pooled: pd.DataFrame) -> pd.DataFrame:
    """Average heads within representations, then representations equally.

    A single fixed representation has only downstream-fit variability. Ridge
    and nearest-neighbour predictions are deterministic for a fixed split:
    their repeated seed labels are not counted as independent fits.
    """

    keys = ["source_group", "method", *CONDITION_COLUMNS, "folds"]
    rows = []
    for values, cell in pooled.groupby(keys, dropna=False, sort=True):
        record = dict(zip(keys, values, strict=True))
        by_representation = cell.groupby("representation_seed", sort=True)["value"].mean()
        head_counts = cell.groupby("representation_seed")["head_seed"].nunique()
        rep_seeds = cell["representation_seed"].astype(int).unique()
        learned_representations = int(np.count_nonzero(rep_seeds != 0))
        deterministic = str(record["method"]) in DETERMINISTIC_METHODS
        if deterministic:
            repeat_values = np.asarray([float(by_representation.mean())])
            basis = "deterministic fit; no seed SD"
        elif len(by_representation) > 1:
            repeat_values = by_representation.to_numpy(dtype=float)
            basis = "representation-seed means; heads nested"
        else:
            repeat_values = cell["value"].to_numpy(dtype=float)
            basis = "downstream training/head seeds; fixed representation"
        rows.append({
            **record,
            "mean": float(by_representation.mean()),
            "sd": float(np.std(repeat_values, ddof=1)) if len(repeat_values) > 1 else np.nan,
            "sd_basis": basis,
            "n_outer_repeats": int(len(repeat_values)),
            "n_representation_seeds": learned_representations,
            "n_recorded_head_seeds_min": int(head_counts.min()),
            "n_recorded_head_seeds_max": int(head_counts.max()),
            "n_seed_fit_cells": 1 if deterministic else int(len(cell)),
            "n_folds": int(cell["n_folds"].max()),
            "n_prediction_instances_min": int(cell["n_rows"].min()),
            "n_prediction_instances_max": int(cell["n_rows"].max()),
        })
    return pd.DataFrame(rows)


def common_split_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep controlled baseline cells that share the frozen-probe conditions.

    Manifest construction establishes the common split. Matching here keeps
    protocol, input information, learning-curve sample, and actual folds
    visible; external pretrained references remain in their own table.
    """

    candidates = summary.loc[summary["source_group"].isin(("frozen_probe", "chemprop", "classical"))]
    keys = [*CONDITION_COLUMNS, "folds", "n_prediction_instances_min", "n_prediction_instances_max"]
    blocks = []
    for _, cell in candidates.groupby(keys, dropna=False, sort=True):
        groups = set(cell["source_group"])
        if "frozen_probe" in groups and groups.intersection({"chemprop", "classical"}):
            blocks.append(cell)
    return pd.concat(blocks, ignore_index=True) if blocks else candidates.iloc[:0].copy()


def _main_ood_scope(frame: pd.DataFrame, *, include_aggregate: bool = False) -> pd.DataFrame:
    selected = frame.copy()
    if "split" in selected:
        selected = selected.loc[selected["split"].eq("test")]
    if "label_fraction" in selected:
        selected = selected.loc[np.isclose(selected["label_fraction"].astype(float), 1.0)]
    if "subsample_id" in selected:
        selected = selected.loc[selected["subsample_id"].astype(int).eq(0)]
    if "input_view" in selected:
        selected = selected.loc[selected["input_view"].eq("valid")]
    if "task" in selected:
        tasks = selected["task"].astype(str).str.upper()
        allowed = (*FOCUS_TASKS, "AGGREGATE") if include_aggregate else FOCUS_TASKS
        selected = selected.loc[tasks.isin(allowed)]
        if "protocol" in selected:
            expected = selected["task"].astype(str).str.upper().map(
                {"EA": "family_ood", "IP": "family_ood", "EGC": "group_ood", "XC": "group_ood",
                 "AGGREGATE": "task_specific_ood"}
            )
            selected = selected.loc[selected["protocol"].eq(expected)]
    return selected.copy()


def _method_label(method: str) -> str:
    return method.removeprefix("baseline_").replace("_", " ")


def _save(figure: plt.Figure, root: Path, stem: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root / f"{stem}.pdf", root / f"{stem}.png"]
    for path in outputs:
        figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return outputs


def _metric_plot(summary: pd.DataFrame, root: Path, stem: str, title: str) -> list[Path]:
    selected = _main_ood_scope(summary)
    if selected.empty:
        return []
    tasks = [task for task in FOCUS_TASKS if task in set(selected["task"])]
    figure, axes = plt.subplots(1, len(tasks), figsize=(4.7 * len(tasks), 5), squeeze=False)
    for task, axis in zip(tasks, axes.ravel(), strict=True):
        cell = selected.loc[selected["task"].eq(task)].sort_values("method")
        y = np.arange(len(cell))
        for position, row in zip(y, cell.itertuples(index=False), strict=True):
            axis.plot(float(row.mean), position, "o", color="#3B6FB6", markersize=4)
            if np.isfinite(row.sd):
                axis.hlines(position, row.mean - row.sd, row.mean + row.sd, color="#3B6FB6", linewidth=1)
        axis.set_yticks(y, [_method_label(value) for value in cell["method"]])
        axis.invert_yaxis()
        axis.set_title(task)
        axis.set_xlabel(str(cell.iloc[0]["metric"]).upper() + " (original target units)")
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(title + "\n100% labels, structure-only OOD; bars are descriptive SD, not CI", fontsize=11)
    figure.tight_layout()
    return _save(figure, root, stem)


def _effect_plot(frame: pd.DataFrame, root: Path, stem: str, title: str) -> list[Path]:
    if frame.empty:
        return []
    selected = _main_ood_scope(frame, include_aggregate=True)
    if selected.empty:
        return []
    if "candidate" in selected and "baseline" in selected:
        selected = selected.loc[
            selected["candidate"].astype(str).str.endswith(("_lora", "_full"))
            | selected["baseline"].astype(str).str.endswith(("_lora", "_full"))
        ].copy()
    if selected.empty:
        return []
    required = ["estimate", "ci_lower", "ci_upper"]
    selected = selected.loc[np.isfinite(selected[required].to_numpy(dtype=float)).all(axis=1)]
    if selected.empty:
        return []
    sort_keys = [column for column in ("task", "comparison", "candidate", "baseline") if column in selected]
    selected = selected.sort_values(sort_keys)
    labels = []
    for row in selected.to_dict(orient="records"):
        pair = (
            f"{_method_label(str(row['candidate']))} vs {_method_label(str(row['baseline']))}"
            if "candidate" in row and "baseline" in row
            else str(row.get("comparison", "comparison")).replace("_", " ")
        )
        if str(row.get("task", "")).casefold() == "aggregate":
            count = row.get("n_tasks")
            scope = f"{int(count)}-task OOD union" if pd.notna(count) else "Task-specific OOD union"
            labels.append(f"{scope}: {pair}")
        else:
            labels.append(f"{row['task']}: {pair}" if "task" in row else pair)
    y = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(8.2, max(3.3, 0.35 * len(selected) + 1.6)))
    axis.hlines(y, selected["ci_lower"], selected["ci_upper"], color="#3B6FB6", linewidth=1.4)
    axis.plot(selected["estimate"], y, "o", color="#3B6FB6", markersize=4)
    axis.axvline(0.0, color="0.5", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired log error ratio (negative favors candidate); 95% hierarchical CI")
    axis.set_title(title)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return _save(figure, root, stem)


def _latex_table(frame: pd.DataFrame, path: Path, caption: str) -> Path:
    lines = [
        r"\begin{longtable}{llllrr}",
        r"\caption{" + caption + r"}\\",
        r"\toprule",
        r"Task & Protocol & Method & Metric & Mean & Descriptive SD \\",
        r"\midrule\endhead",
    ]
    for row in frame.itertuples(index=False):
        sd = "--" if not np.isfinite(row.sd) else f"{row.sd:.4g}"
        cells = [row.task, row.protocol, _method_label(row.method), row.metric.upper()]
        lines.append(" & ".join([*(latex_escape(value) for value in cells), f"{row.mean:.4g}", sd]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_posttraining_outputs(project_root: str | Path, analysis_root: str | Path) -> list[Path]:
    """Render compact tables/figures without reopening prediction corpora."""

    project = Path(project_root)
    analysis = Path(analysis_root)
    sources = [("frozen_probe", project / "results/statistics/formal_frozen_probes/task_metrics.csv")]
    sources.extend((group, analysis / "metrics" / group / "task_metrics.csv") for group in SOURCE_GROUPS)
    blocks = []
    for group, path in sources:
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["source_group"] = group
        frame["task"] = frame["task"].astype(str).str.upper()
        frame["input_view"] = frame["input_view"].replace({
            "structure_only": "valid", "structure_plus_metadata": "structure_plus_metadata_secondary"
        })
        blocks.append(frame)
    if not blocks:
        return []
    metrics = pd.concat(blocks, ignore_index=True, sort=False)
    pooled = pool_fold_metrics(metrics)
    summary = summarize_replicates(pooled)
    common = common_split_table(summary)
    adaptation = summary.loc[summary["source_group"].eq("adaptation")].copy()
    external = summary.loc[summary["source_group"].isin(EXTERNAL_GROUPS)].copy()
    external["comparison_scope"] = "external model reference; corpus, architecture or input differences prevent a controlled initialization comparison"
    tables = analysis / "tables"
    figures = analysis / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, frame in (
        ("pooled_seed_metrics", pooled), ("all_method_summary", summary),
        ("common_split_baselines", common), ("adaptation_summary", adaptation),
        ("external_reference", external),
    ):
        path = tables / f"{name}.csv"
        frame.to_csv(path, index=False)
        outputs.append(path)
    common_focus = _main_ood_scope(common)
    adaptation_focus = _main_ood_scope(adaptation)
    full_external = external.loc[
        external["split"].eq("test") & np.isclose(external["label_fraction"].astype(float), 1.0)
        & external["subsample_id"].eq(0) & external["input_view"].eq("valid")
    ]
    sd_note = (
        " Fold errors are pooled before head and representation averaging. SD is across "
        "representation means for continued-pretrained models and across downstream training "
        "seeds for fixed representations; deterministic fits have no seed SD."
    )
    for name, frame, caption in (
        ("common_split_baselines", common_focus, r"Descriptive same-split references, structure-only OOD at 100\% labels."),
        ("adaptation_summary", adaptation_focus, r"Downstream adaptation, structure-only OOD at 100\% labels."),
        ("external_reference", full_external, "External model references, reported separately because corpora, architectures or model inputs differ."),
    ):
        if not frame.empty:
            outputs.append(_latex_table(frame, tables / f"{name}.tex", caption + sd_note))
    outputs.extend(_metric_plot(common, figures, "controlled_baselines_ood", "Same-split controlled baseline context"))
    outputs.extend(_metric_plot(adaptation, figures, "adaptation_ood_performance", "Downstream adaptation"))
    for filename, stem, title in (
        ("secondary_aggregate.csv", "adaptation_aggregate_effects", "Secondary adaptation comparisons: task-specific OOD unions"),
        ("task_bootstrap.csv", "adaptation_task_effects", "Secondary adaptation comparisons: task-specific effects"),
    ):
        path = analysis / filename
        if path.is_file():
            outputs.extend(_effect_plot(pd.read_csv(path), figures, stem, title))
    return outputs


__all__ = ["pool_fold_metrics", "summarize_replicates", "common_split_table", "render_posttraining_outputs"]
