"""Finite, secondary confidence intervals needed for the Paper104 draft.

Geometry uses the five formal representations, not the three-seed component
controls. The property bridge uses compact within-record outputs and the same
records for both stability axes and canonical-reference error. Its structure
draws are shared across tasks: a polymer appearing in several datasets is one
global cluster, not several independent task-specific clusters.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from polyfid.io import atomic_save_npz, atomic_write_csv, atomic_write_text


SEEDS = (42, 123, 777, 2025, 2026)
HEADS = (11, 22, 33)
METHODS = ("A_eqs", "A_gcl")
BRIDGE_TASKS = ("EEA", "EGB", "EGC", "EI", "EPS", "NC", "XC")
BRIDGE_METRICS = ("periodic_cut_prediction_sd", "traversal_prediction_sd", "reference_rmse")
GEOMETRY_METRICS = {
    "equivalence_cosine": "anchor_positive_cosine",
    "recall_at_1": "retrieval_correct_at_1",
    "mrr": "reciprocal_rank",
    "topology_margin": "topology_margin",
    "triplet_accuracy": "triplet_correct",
}
POSITIVE_FAMILIES = (
    "alternative_periodic_cut", "randomized_traversal", "reverse_when_non_noop",
)
EDIT_FAMILIES = ("endpoint_relocation", "closure_rewiring")
RECORD_KEYS = ["task", "protocol", "fold", "sample_id"]


def _json(path: Path, value: dict) -> None:
    atomic_write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", path)


def _chunked_draws(
    root: Path, *, n_resamples: int, chunk_size: int, seed: int,
    draw: Callable[[int, np.random.Generator], np.ndarray],
) -> np.ndarray:
    """Resume completed chunks; each chunk has its own deterministic RNG stream."""
    root.mkdir(parents=True, exist_ok=True)
    settings = {"resamples": n_resamples, "chunk_size": chunk_size, "seed": seed}
    settings_path = root / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError(f"bootstrap settings differ from existing run: {root}")
    _json(settings_path, settings)
    blocks = []
    for start in range(0, n_resamples, chunk_size):
        count = min(chunk_size, n_resamples - start)
        path = root / f"draws_{start:05d}.npz"
        if path.exists():
            with np.load(path) as source:
                block = source["draws"]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, start]))
            block = draw(count, rng)
            atomic_save_npz(path, {"draws": block})
        blocks.append(block)
        _json(root / "progress.json", {**settings, "completed": start + count})
        print(f"{root.parent.name}/{root.name}: {start + count}/{n_resamples}", flush=True)
    return np.concatenate(blocks)


def _interval(draws: np.ndarray, estimate: float) -> dict:
    lower, upper = np.quantile(draws, (0.025, 0.975))
    return {"estimate": float(estimate), "ci_lower": float(lower),
            "ci_upper": float(upper), "resamples": len(draws)}


def load_geometry_pair(project_root: Path, filename: str) -> pd.DataFrame:
    root = project_root / "results/metrics/fidelity/relations"
    frames = {
        method: pd.concat([
            pd.read_parquet(root / f"{method}_seed-{seed}" / filename)
            for seed in SEEDS
        ], ignore_index=True)
        for method in METHODS
    }
    controls = filename.startswith("topology_control")
    keys = ["representation_seed", "anchor_sample_id", "positive_sample_id",
            "negative_sample_id", "control_row_id" if controls else "triplet_id",
            "negative_relation" if controls else "positive_relation"]
    columns = ["topology_margin", "triplet_correct"] if controls else list(GEOMETRY_METRICS.values())
    paired = frames[METHODS[0]][keys + columns].merge(
        frames[METHODS[1]][keys + columns], on=keys, how="outer",
        validate="one_to_one", suffixes=("_candidate", "_baseline"), indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError(f"geometry methods have different paired records: {filename}")
    return paired.drop(columns="_merge")


def geometry_bootstrap(
    paired: pd.DataFrame, metrics: dict[str, str], *, n_resamples: int,
    chunk_size: int, seed: int, output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired representation → anchor bootstrap, retaining each anchor's rows."""
    columns = [f"{column}_{method}" for column in metrics.values()
               for method in ("candidate", "baseline")]
    arrays, sizes, point_rows = [], [], []
    for rep, cell in paired.groupby("representation_seed", sort=True):
        grouped = cell.groupby("anchor_sample_id", sort=True)
        arrays.append(grouped[columns].sum().to_numpy(dtype=float))
        sizes.append(grouped.size().to_numpy())
        for metric, column in metrics.items():
            candidate, baseline = (float(cell[f"{column}_{side}"].mean())
                                   for side in ("candidate", "baseline"))
            point_rows.append({"representation_seed": int(rep), "metric": metric,
                               "candidate_value": candidate, "baseline_value": baseline,
                               "difference": candidate - baseline})

    def draw(count: int, rng: np.random.Generator) -> np.ndarray:
        result = np.zeros((count, len(metrics)))
        selected_seeds = rng.integers(len(arrays), size=(count, len(arrays)))
        for slot in range(len(arrays)):
            for rep, values in enumerate(arrays):
                positions = np.flatnonzero(selected_seeds[:, slot] == rep)
                if not len(positions):
                    continue
                indices = rng.integers(len(values), size=(len(positions), len(values)))
                means = values[indices].sum(axis=1) / sizes[rep][indices].sum(axis=1)[:, None]
                result[positions] += means[:, 0::2] - means[:, 1::2]
        return result / len(arrays)

    draws = _chunked_draws(output_root, n_resamples=n_resamples, chunk_size=chunk_size,
                           seed=seed, draw=draw)
    points = pd.DataFrame(point_rows)
    rows = []
    for index, metric in enumerate(metrics):
        cell = points.loc[points.metric.eq(metric)]
        rows.append({"metric": metric,
                     "candidate_value": float(cell.candidate_value.mean()),
                     "baseline_value": float(cell.baseline_value.mean()),
                     **_interval(draws[:, index], cell.difference.mean())})
    return pd.DataFrame(rows), points


def write_geometry_analysis(project_root: Path, output_root: Path, *, n_resamples: int,
                            chunk_size: int, seed: int) -> dict:
    triplets = load_geometry_pair(project_root, "triplet_metrics.parquet")
    controls = load_geometry_pair(project_root, "topology_control_metrics.parquet")
    scopes = [("overall", triplets, GEOMETRY_METRICS)]
    scopes += [(family, triplets.loc[triplets.positive_relation.eq(family)], GEOMETRY_METRICS)
               for family in POSITIVE_FAMILIES]
    edit_metrics = {name: GEOMETRY_METRICS[name] for name in ("topology_margin", "triplet_accuracy")}
    scopes += [(family, controls.loc[controls.negative_relation.eq(family)], edit_metrics)
               for family in EDIT_FAMILIES]
    intervals, points, scope_rows = [], [], []
    for index, (scope, frame, metrics) in enumerate(scopes):
        interval, point = geometry_bootstrap(
            frame, metrics, n_resamples=n_resamples, chunk_size=chunk_size,
            seed=seed + index, output_root=output_root / "draws" / scope,
        )
        info = {"scope": scope, "n_representation_seeds": frame.representation_seed.nunique(),
                "n_anchors": frame.anchor_sample_id.nunique(),
                "n_paired_seed_rows": len(frame),
                "candidate": METHODS[0], "baseline": METHODS[1]}
        intervals.append(interval.assign(**info))
        points.append(point.assign(scope=scope))
        scope_rows.append(info)
    result = pd.concat(intervals, ignore_index=True)
    atomic_write_csv(result, output_root / "intervals.csv")
    atomic_write_csv(pd.concat(points, ignore_index=True), output_root / "per_seed_metrics.csv")
    atomic_write_csv(pd.DataFrame(scope_rows), output_root / "scope.csv")
    summary = {"status": "complete", "intervals": len(result), "resamples": n_resamples,
               "representation_seeds": list(SEEDS), "contrast": "A_eqs minus A_gcl",
               "hierarchy": "representation seed -> anchor (retain nested relation rows)",
               "positive_effect": "higher geometry response; favorable for all five reported metrics",
               "inference_scope": "secondary 95% percentile intervals; no new primary tests or Holm family",
               "three_seed_component_controls": "not included",
               "input_root": str(project_root / "results/metrics/fidelity/relations")}
    _json(output_root / "summary.json", summary)
    return summary


@dataclass
class BridgeData:
    records: pd.DataFrame
    values: np.ndarray  # representation, record, head, metric, candidate/baseline
    seeds: tuple[int, ...]
    heads: tuple[int, ...]
    tasks: tuple[str, ...]
    group_codes: np.ndarray
    task_indices: tuple[np.ndarray, ...]
    repeated_rows: tuple[np.ndarray, ...]


def prepare_bridge(paired: pd.DataFrame) -> BridgeData:
    """Keep one common record cohort across both axes, methods, seeds and heads."""
    seeds = tuple(int(value) for value in sorted(paired.representation_seed.unique()))
    heads = tuple(int(value) for value in sorted(paired.head_seed.unique()))
    paired = paired.copy()
    # A row must occur for every nested fit; an absent view is not imputed as zero.
    fit_count = paired.groupby(RECORD_KEYS).size()
    cohort = fit_count.loc[fit_count.eq(len(seeds) * len(heads))].reset_index()[RECORD_KEYS]
    paired = paired.merge(cohort, on=RECORD_KEYS, how="inner", validate="many_to_one")
    records = paired.sort_values(RECORD_KEYS).drop_duplicates(RECORD_KEYS)[RECORD_KEYS + ["group_id"]]
    records = records.reset_index(drop=True)
    records["record_index"] = np.arange(len(records))
    paired = paired.merge(records[RECORD_KEYS + ["record_index"]], on=RECORD_KEYS, validate="many_to_one")
    values = np.full((len(seeds), len(records), len(heads), 3, 2), np.nan)
    rep_codes = paired.representation_seed.map({seed: i for i, seed in enumerate(seeds)}).to_numpy()
    head_codes = paired.head_seed.map({head: i for i, head in enumerate(heads)}).to_numpy()
    row_codes = paired.record_index.to_numpy()
    for metric, column in enumerate(("periodic_cut_sd", "traversal_sd", "squared_error")):
        for side, suffix in enumerate(("candidate", "baseline")):
            values[rep_codes, row_codes, head_codes, metric, side] = paired[f"{column}_{suffix}"]
    if not np.isfinite(values).all():
        raise ValueError("same-record bridge is missing a paired seed/head prediction")
    group_codes, _ = pd.factorize(records.group_id, sort=True)
    tasks = tuple(sorted(records.task.unique()))
    task_indices = tuple(np.flatnonzero(records.task.eq(task)) for task in tasks)
    repeated_rows = tuple(np.asarray(indices, dtype=int) for indices in
                          records.groupby(["group_id", "task"], sort=True).indices.values()
                          if len(indices) > 1)
    return BridgeData(records, values, seeds, heads, tasks, group_codes, task_indices, repeated_rows)


def load_property_bridge(project_root: Path) -> BridgeData:
    manifest_path = project_root / "runs/manifests/property_views/property_view_prediction_manifest.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t")
    selected = manifest.loc[
        manifest.arm.isin(METHODS) & manifest.representation_seed.isin(SEEDS)
        & manifest.task.str.upper().isin(BRIDGE_TASKS) & manifest.protocol.eq("group_ood")
        & manifest.input_view.eq("structure_only") & np.isclose(manifest.label_fraction, 1.0)
        & manifest.subsample_id.eq(0)
    ]
    blocks = []
    columns = [*RECORD_KEYS, "group_id", "representation_seed", "head_seed", "view_family",
               "property_prediction_std", "reference_y_pred", "y_true", "split"]
    for row in selected.itertuples(index=False):
        for head in HEADS:
            path = Path(row.output_root) / f"head-seed-{head}/equivalent_view_within_record.parquet"
            frame = pd.read_parquet(path, columns=columns)
            frame = frame.loc[frame.split.eq("test")]
            keys = RECORD_KEYS + ["group_id", "representation_seed", "head_seed"]
            cut = frame.loc[frame.view_family.eq("periodic_cut"), keys + [
                "property_prediction_std", "reference_y_pred", "y_true"]]
            traversal = frame.loc[frame.view_family.eq("traversal"), keys + ["property_prediction_std"]]
            cell = cut.merge(traversal, on=keys, validate="one_to_one",
                             suffixes=("_cut", "_traversal"))
            cell = cell.rename(columns={"property_prediction_std_cut": "periodic_cut_sd",
                                        "property_prediction_std_traversal": "traversal_sd"})
            cell["squared_error"] = (cell.reference_y_pred - cell.y_true) ** 2
            cell["method"] = row.arm
            cell["task"] = cell.task.str.upper()
            blocks.append(cell)
    combined = pd.concat(blocks, ignore_index=True)
    keys = RECORD_KEYS + ["representation_seed", "head_seed"]
    paired = combined.loc[combined.method.eq(METHODS[0])].merge(
        combined.loc[combined.method.eq(METHODS[1])], on=keys, how="inner",
        validate="one_to_one", suffixes=("_candidate", "_baseline"),
    )
    if not paired.group_id_candidate.eq(paired.group_id_baseline).all():
        raise ValueError("paired property predictions disagree on global structure groups")
    if not np.allclose(paired.y_true_candidate, paired.y_true_baseline):
        raise ValueError("paired property predictions disagree on target values")
    paired["group_id"] = paired.group_id_candidate
    return prepare_bridge(paired)


def sample_shared_group_weights(data: BridgeData, rng: np.random.Generator) -> np.ndarray:
    """One shared global group draw; resample replicate rows within group/task.

    Tasks are fixed strata, not resampled datasets. A structure's multiplicity
    is identical in every task containing it. Replicate measurements are then
    sampled within that task/structure; heads are sampled only after computing
    each fit's metric. Global group labels are never prefixed with task/fold.
    """
    count = int(data.group_codes.max()) + 1
    group_weights = np.bincount(rng.integers(count, size=count), minlength=count)
    weights = group_weights[data.group_codes].copy()
    for indices in data.repeated_rows:
        n = int(group_weights[data.group_codes[indices[0]]]) * len(indices)
        weights[indices] = rng.multinomial(n, np.full(len(indices), 1.0 / len(indices)))
    return weights


def _bridge_fit_values(data: BridgeData, rep: int, weights: np.ndarray) -> np.ndarray:
    # task, head, metric, candidate/baseline; RMSE is reconstructed from SSE.
    result = np.empty((len(data.tasks), len(data.heads), 3, 2))
    for task, indices in enumerate(data.task_indices):
        denominator = weights[indices].sum()
        if denominator == 0:
            raise ValueError("a global structure draw contains no observations for a task")
        result[task] = np.einsum("r,rhmc->hmc", weights[indices], data.values[rep, indices]) / denominator
    result[:, :, 2, :] = np.sqrt(result[:, :, 2, :])
    return result


def bridge_bootstrap(data: BridgeData, *, n_resamples: int, chunk_size: int,
                     seed: int, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows, point_effects = [], []
    for rep, rep_seed in enumerate(data.seeds):
        fits = _bridge_fit_values(data, rep, np.ones(len(data.records), dtype=int))
        point_effects.append(np.log(fits[..., 0] / fits[..., 1]).mean(axis=1))
        for task, task_name in enumerate(data.tasks):
            for head, head_seed in enumerate(data.heads):
                for metric, metric_name in enumerate(BRIDGE_METRICS):
                    candidate, baseline = fits[task, head, metric]
                    point_rows.append({"representation_seed": rep_seed, "head_seed": head_seed,
                                       "task": task_name, "metric": metric_name,
                                       "candidate_value": candidate, "baseline_value": baseline,
                                       "paired_log_ratio": np.log(candidate / baseline)})
    point = np.mean(point_effects, axis=0)
    point = np.vstack((point, point.mean(axis=0)))

    def draw(count: int, rng: np.random.Generator) -> np.ndarray:
        result = np.empty((count, len(data.tasks) + 1, len(BRIDGE_METRICS)))
        for draw_index in range(count):
            effects = np.zeros((len(data.tasks), len(BRIDGE_METRICS)))
            for rep in rng.integers(len(data.seeds), size=len(data.seeds)):
                fits = _bridge_fit_values(data, rep, sample_shared_group_weights(data, rng))
                selected_heads = rng.integers(len(data.heads), size=len(data.heads))
                head_fits = np.take(fits, selected_heads, axis=1)
                effects += np.log(head_fits[..., 0] / head_fits[..., 1]).mean(axis=1)
            result[draw_index, :-1] = effects / len(data.seeds)
            result[draw_index, -1] = result[draw_index, :-1].mean(axis=0)
        return result

    draws = _chunked_draws(output_root / "draws", n_resamples=n_resamples,
                           chunk_size=chunk_size, seed=seed, draw=draw)
    points = pd.DataFrame(point_rows)
    intervals = []
    for task, task_name in enumerate((*data.tasks, "aggregate")):
        for metric, metric_name in enumerate(BRIDGE_METRICS):
            cell = points.loc[points.task.eq(task_name) & points.metric.eq(metric_name)]
            intervals.append({"task": task_name, "metric": metric_name,
                              "candidate": METHODS[0], "baseline": METHODS[1],
                              "candidate_value": cell.candidate_value.mean(),
                              "baseline_value": cell.baseline_value.mean(),
                              **_interval(draws[:, task, metric], point[task, metric])})
    return pd.DataFrame(intervals), points


def write_property_bridge(project_root: Path, output_root: Path, *, n_resamples: int,
                          chunk_size: int, seed: int) -> dict:
    data = load_property_bridge(project_root)
    scope_rows = []
    for task in (*data.tasks, "aggregate"):
        records = data.records if task == "aggregate" else data.records.loc[data.records.task.eq(task)]
        scope_rows.append({"task": task, "n_records": len(records),
                           "n_global_structure_groups": records.group_id.nunique(),
                           "n_representation_seeds": len(data.seeds), "n_head_seeds": len(data.heads),
                           "protocol": "group_ood", "label_fraction": 1.0,
                           "input_view": "structure_only", "split": "test"})
    atomic_write_csv(data.records.drop(columns="record_index"), output_root / "cohort.csv")
    atomic_write_csv(pd.DataFrame(scope_rows), output_root / "scope.csv")
    intervals, points = bridge_bootstrap(data, n_resamples=n_resamples, chunk_size=chunk_size,
                                         seed=seed, output_root=output_root)
    atomic_write_csv(intervals, output_root / "intervals.csv")
    atomic_write_csv(points, output_root / "per_seed_head_task_metrics.csv")
    summary = {"status": "complete", "intervals": len(intervals), "resamples": n_resamples,
               "n_records": len(data.records), "n_global_structure_groups": data.records.group_id.nunique(),
               "tasks": list(data.tasks), "representation_seeds": list(data.seeds),
               "head_seeds": list(data.heads), "contrast": "log(A_eqs / A_gcl)",
               "negative_effect": "lower within-record SD or lower canonical-reference RMSE",
               "hierarchy": "representation -> globally shared structure group -> row within group/task -> head",
               "estimator": "record-equal mean SD or pooled-fold RMSE per fit; paired log ratios averaged over heads, representations, then tasks",
               "axes": list(BRIDGE_METRICS),
               "inference_scope": "same-record seven-task secondary bridge, not the 12-task primary test; no new Holm family",
               "joint_draws": "same representation, global structure, row and head draws for all three axes and seven tasks",
               "three_seed_component_controls": "not included",
               "manifest": str(project_root / "runs/manifests/property_views/property_view_prediction_manifest.tsv")}
    _json(output_root / "summary.json", summary)
    return summary


def run_draft_closure(project_root: str | Path, output_root: str | Path | None = None,
                      *, stage: str = "all", n_resamples: int = 10_000,
                      chunk_size: int = 100, seed: int = 2026) -> dict:
    project = Path(project_root).resolve()
    output = Path(output_root) if output_root else project / "results/statistics/draft_closure"
    result = {}
    if stage in {"all", "geometry"}:
        result["geometry"] = write_geometry_analysis(project, output / "geometry",
            n_resamples=n_resamples, chunk_size=chunk_size, seed=seed)
    if stage in {"all", "property_bridge"}:
        result["property_bridge"] = write_property_bridge(project, output / "property_bridge",
            n_resamples=n_resamples, chunk_size=chunk_size, seed=seed + 100)
    return result


def render_draft_closure_figures(analysis_root: str | Path, output_root: str | Path) -> list[Path]:
    """Render completed interval CSVs only; never load predictions or bootstrap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    analysis = Path(analysis_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    geometry = pd.read_csv(analysis / "geometry/intervals.csv")
    bridge = pd.read_csv(analysis / "property_bridge/intervals.csv")
    bridge_scope = pd.read_csv(analysis / "property_bridge/scope.csv").set_index("task")
    outputs = []
    family_labels = {
        "overall": "Overall", "alternative_periodic_cut": "Periodic cut",
        "randomized_traversal": "Traversal", "reverse_when_non_noop": "Reverse",
        "endpoint_relocation": "Endpoint relocation", "closure_rewiring": "Closure rewiring",
    }
    metric_labels = {
        "equivalence_cosine": "Equivalence cosine", "recall_at_1": "Recall@1", "mrr": "MRR",
        "topology_margin": "Topology margin", "triplet_accuracy": "Triplet accuracy",
    }
    geometry["plot_scale"] = np.where(geometry.metric.isin(["recall_at_1", "triplet_accuracy"]), 100.0, 1.0)
    geometry["plot_unit"] = np.where(geometry.plot_scale.eq(100.0), "percentage points", "difference")
    for name in ("estimate", "ci_lower", "ci_upper"):
        geometry[f"plot_{name}"] = geometry[name] * geometry.plot_scale
        bridge[f"plot_{name}"] = np.expm1(bridge[name]) * 100.0
    bridge["plot_unit"] = "percent change: 100 * (exp(paired log ratio) - 1)"
    atomic_write_csv(geometry, analysis / "geometry/plot_data.csv")
    atomic_write_csv(bridge, analysis / "property_bridge/plot_data.csv")

    def forest(axis, frame: pd.DataFrame, labels: list[str]) -> None:
        positions = np.arange(len(frame))
        axis.hlines(positions, frame.plot_ci_lower, frame.plot_ci_upper, color="#315F9D", linewidth=1.4)
        axis.scatter(frame.plot_estimate, positions, color="#315F9D", s=25, zorder=3)
        axis.axvline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)

    def save(figure, stem: str) -> None:
        for suffix in ("pdf", "png"):
            path = output / f"{stem}.{suffix}"
            figure.savefig(path, dpi=220, bbox_inches="tight")
            outputs.append(path)
        plt.close(figure)

    figure, axes = plt.subplots(1, 5, figsize=(17.5, 3.6), constrained_layout=True)
    families = ("overall", *POSITIVE_FAMILIES)
    for axis, metric in zip(axes, GEOMETRY_METRICS, strict=True):
        cell = geometry.loc[geometry.metric.eq(metric) & geometry.scope.isin(families)].set_index("scope").loc[list(families)]
        forest(axis, cell, [family_labels[family] for family in families])
        axis.set_title(metric_labels[metric], fontsize=10)
        axis.set_xlabel("A_eqs − A_gcl\n" + str(cell.plot_unit.iloc[0]), fontsize=9)
    figure.suptitle(
        "Geometry responses remain relation-specific\n"
        "Five formal representations; paired representation → anchor 95% intervals; positive favors A_eqs",
        fontsize=11,
    )
    save(figure, "supplement_geometry_paired_intervals")

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), constrained_layout=True)
    for axis, metric in zip(axes, ("topology_margin", "triplet_accuracy"), strict=True):
        cell = geometry.loc[geometry.metric.eq(metric) & geometry.scope.isin(EDIT_FAMILIES)].set_index("scope").loc[list(EDIT_FAMILIES)]
        forest(axis, cell, [family_labels[family] for family in EDIT_FAMILIES])
        axis.set_title(metric_labels[metric], fontsize=10)
        axis.set_xlabel("A_eqs − A_gcl\n" + str(cell.plot_unit.iloc[0]), fontsize=9)
    figure.suptitle("Topology-edit controls | paired 95% intervals, five formal representations", fontsize=11)
    save(figure, "supplement_topology_edit_intervals")

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 5.3), constrained_layout=True)
    tasks = (*BRIDGE_TASKS, "aggregate")
    titles = ("Periodic-cut prediction SD", "Traversal prediction SD", "Reference prediction RMSE")
    for axis, metric, title in zip(axes, BRIDGE_METRICS, titles, strict=True):
        cell = bridge.loc[bridge.metric.eq(metric)].set_index("task").loc[list(tasks)]
        forest(axis, cell, [task if task != "aggregate" else "Task-equal aggregate" for task in tasks])
        axis.axhline(len(tasks) - 1.5, color="0.8", linewidth=0.8)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("A_eqs relative to A_gcl (%)\nnegative = lower SD / error", fontsize=9)
    scope = bridge_scope.loc["aggregate"]
    figure.suptitle(
        "Stability and property error on the same seven-task records\n"
        f"{int(scope.n_records):,} task-records / {int(scope.n_global_structure_groups):,} global structure groups; "
        "joint paired 95% intervals; five representations × three nested heads",
        fontsize=11,
    )
    save(figure, "supplement_same_record_property_bridge")
    return outputs
