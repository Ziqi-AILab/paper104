"""Observed-data builders for the five Paper104 main figures.

The figures follow the paper's argument rather than the order in which files
are produced.  A main figure is written only when every panel needed for its
stated conclusion has a formal input.  Missing inputs are reported in
``figures/availability.json``; engineering smoke and development-seed outputs
are never promoted into paper figures.
"""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from polyfid.analysis.posttraining_figures import pool_fold_metrics
from polyfid.evaluation.bootstrap import PRIMARY_COMPARISONS
from polyfid.io import atomic_write_csv


METHOD_ORDER = ["C", "B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs"]
FORMAL_REPRESENTATION_SEEDS = {42, 123, 777, 2025, 2026}
DEVELOPMENT_SEEDS = {13, 37}
METHOD_LABELS = {
    "C": "C",
    "B0": "B0",
    "A0": "A0",
    "A_att": r"A$_{att}$",
    "A_gcl": r"A$_{gcl}$",
    "A_eqs": r"A$_{eqs}$",
    "B_eqs": r"B$_{eqs}$",
    "matched_random_masking": "matched random",
    "permuted_boundary_masking": "permuted boundary",
}
METHOD_COLORS = {
    "C": "#7A7A7A",
    "B0": "#D47A4A",
    "A0": "#3B6FB6",
    "A_att": "#4E9B78",
    "A_gcl": "#8A68AC",
    "A_eqs": "#C34F63",
    "B_eqs": "#B99435",
    "matched_random_masking": "#6C8EBF",
    "permuted_boundary_masking": "#8F8F8F",
}
COMPARISON_LABELS = {
    "molecular_initialization": "A0 vs B0\nmolecular initialization",
    "boundary_supervision": r"A$_{att}$ vs A0" + "\nboundary supervision",
    "verified_relations": r"A$_{eqs}$ vs A$_{gcl}$" + "\npolymer-relation supervision",
}
RETENTION_TASKS = {"esol", "freesolv", "lipophilicity", "zinc", "zinc_mlm"}
OPENPOLY_HIGHLIGHTS = (
    "tg",
    "bandgap_chain",
    "co2_permeability",
    "refractive_index",
    "crystallization_tendency",
    "tensile_strength",
)


@dataclass(frozen=True)
class FigureOutcome:
    figure: str
    status: str
    source_paths: tuple[str, ...] = ()
    output_paths: tuple[str, ...] = ()
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )


def _save(figure: plt.Figure, output_root: Path, stem: str) -> tuple[str, ...]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = (output_root / f"{stem}.pdf", output_root / f"{stem}.png")
    for path in outputs:
        figure.savefig(path, dpi=300)
    plt.close(figure)
    return tuple(str(path) for path in outputs)


def _not_engineering_output(path: Path) -> bool:
    excluded = ("smoke", "pilot", "dev", "integration")
    return not any(
        part.casefold().startswith(prefix)
        for part in path.parts
        for prefix in excluded
    )


def _formal_paths(root: Path, pattern: str) -> list[Path]:
    return [path for path in sorted(root.rglob(pattern)) if _not_engineering_output(path)]


def _downstream_statistic_paths(results_root: Path, filename: str) -> list[Path]:
    """Select downstream statistics without adjacent property-view tables."""

    declared = results_root / "statistics" / "formal_frozen_probes" / filename
    if declared.is_file():
        return [declared]
    return [
        path
        for path in _formal_paths(results_root, filename)
        if "property_views" not in path.parts
    ]


def _filter_formal_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    if "representation_seed" not in frame:
        return frame
    seeds = pd.to_numeric(frame["representation_seed"], errors="coerce")
    # Explicitly remove the two development seeds.  A fixed reference may use
    # a sentinel seed, so formal filtering is not implemented as a brittle
    # exact allow-list.
    return frame.loc[~seeds.isin(DEVELOPMENT_SEEDS)].copy()


def _read_tables(paths: Iterable[Path]) -> tuple[pd.DataFrame, list[Path]]:
    blocks: list[pd.DataFrame] = []
    used: list[Path] = []
    for path in paths:
        try:
            block = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        except (OSError, ValueError):
            continue
        block["source_path"] = str(path)
        blocks.append(block)
        used.append(path)
    if not blocks:
        return pd.DataFrame(), []
    return pd.concat(blocks, ignore_index=True, sort=False), used


def _relation_tables(
    project_root: Path, *, include_smoke: bool = False
) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((project_root / "data/relations").rglob("pi1m_triplets.parquet"))
    if not include_smoke:
        paths = [path for path in paths if _not_engineering_output(path)]
    frame, used = _read_tables(paths)
    if not frame.empty and "split" not in frame:
        frame["split"] = [Path(value).parent.name for value in frame["source_path"]]
    return frame, used


def _supplemental_relation_tables(project_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = _formal_paths(
        project_root / "data/relations", "supplemental_topology_controls.parquet"
    )
    return _read_tables(paths)


def _typed_relation_tables(project_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = _formal_paths(
        project_root / "data/relations", "typed_relation_registry.parquet"
    )
    return _read_tables(paths)


def _short_label(value: str) -> str:
    return str(value).replace("alternative_periodic_cut", "periodic cut").replace(
        "randomized_traversal", "random traversal"
    ).replace("reverse_when_non_noop", "non-noop reversal").replace("_", " ")


def _draw_structure(axis: plt.Axes, smiles: str, title: str, color: str) -> None:
    """Render an observed structure; fall back to its exact PSMILES string."""

    axis.set_axis_off()
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError("unreadable structure")
        image = Draw.MolToImage(molecule, size=(440, 165), kekulize=True)
        axis.imshow(image)
    except (ImportError, RuntimeError, ValueError):
        axis.text(
            0.5,
            0.55,
            "\n".join(textwrap.wrap(str(smiles), width=30)),
            ha="center",
            va="center",
            family="monospace",
            fontsize=7,
            transform=axis.transAxes,
        )
    axis.set_title(title, color=color, fontsize=8.5, pad=1)


def figure1_relation_benchmark(
    project_root: Path,
    output_root: Path,
    *,
    include_smoke: bool = False,
) -> FigureOutcome:
    """Figure 1: study object, information boundary, examples, and census."""

    frame, sources = _relation_tables(project_root, include_smoke=include_smoke)
    typed, typed_sources = _typed_relation_tables(project_root)
    required = {
        "split",
        "anchor_psmiles",
        "positive_psmiles",
        "negative_psmiles",
        "positive_relation",
        "morgan_similarity",
        "anchor_token_length",
        "negative_token_length",
        "anchor_formula",
        "negative_formula",
        "anchor_atom_count",
        "negative_atom_count",
        "anchor_periodic_identity_representable",
        "anchor_nontrivial_periodic_cut_available",
    }
    typed_required = {
        "split",
        "anchor_id",
        "relation_class",
        "training_eligible",
    }
    if (
        frame.empty
        or not required.issubset(frame.columns)
        or typed.empty
        or not typed_required.issubset(typed.columns)
    ):
        return FigureOutcome(
            "figure1_relation_benchmark",
            "unavailable",
            tuple(str(path) for path in sources),
            message=(
                "formal relation triplets and their typed relation sidecars are "
                "required; smoke relations are excluded"
            ),
        )

    sources = [*sources, *typed_sources]
    preview = any(not _not_engineering_output(path) for path in sources)
    example_pool = frame.loc[
        frame["positive_relation"].astype(str).eq("alternative_periodic_cut")
    ].copy()
    if example_pool.empty:
        example_pool = frame.copy()
    example_pool["_display_length"] = (
        example_pool["anchor_psmiles"].astype(str).str.len()
        + example_pool["positive_psmiles"].astype(str).str.len()
        + example_pool["negative_psmiles"].astype(str).str.len()
    )
    example = example_pool.sort_values(
        ["_display_length", "morgan_similarity"], ascending=[True, False]
    ).iloc[0]

    figure = plt.figure(figsize=(13.2, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, height_ratios=(1.08, 1.0))

    scope_axis = figure.add_subplot(grid[0, 0])
    scope_axis.set_axis_off()
    scope_axis.set_title("a  Study object and information boundary", pad=4)
    scope_axis.text(
        0.02,
        0.88,
        "IN SCOPE",
        color="#25623F",
        fontweight="bold",
        transform=scope_axis.transAxes,
    )
    scope_axis.text(
        0.04,
        0.78,
        "2D  •  linear  •  one component\nexactly two attachment sites (*)",
        va="top",
        linespacing=1.5,
        transform=scope_axis.transAxes,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#E8F3EC", "ec": "#4E9B78"},
    )
    scope_axis.annotate(
        "repeat-unit PSMILES  →  explicit two-cell graph\nalgorithmically checked under this definition",
        xy=(0.04, 0.42),
        xycoords="axes fraction",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.4", "fc": "#EEF3FA", "ec": "#3B6FB6"},
    )
    scope_axis.text(
        0.02,
        0.25,
        "OUTSIDE THE CLAIM",
        color="#8A3C3C",
        fontweight="bold",
        transform=scope_axis.transAxes,
    )
    scope_axis.text(
        0.04,
        0.16,
        "branched/multi-attachment polymers; blends;\n3D conformations; processing-condition semantics",
        va="top",
        fontsize=7.5,
        transform=scope_axis.transAxes,
    )

    example_axis = figure.add_subplot(grid[0, 1:])
    example_axis.set_axis_off()
    example_axis.set_title("b  One observed relation triplet", pad=4)
    positions = ((0.00, 0.10, 0.275, 0.78), (0.365, 0.10, 0.275, 0.78), (0.725, 0.10, 0.275, 0.78))
    labels = (
        ("anchor", example["anchor_psmiles"], "#315F9D"),
        (f"equivalent: {_short_label(example['positive_relation'])}", example["positive_psmiles"], "#377D58"),
        ("matched structural change", example["negative_psmiles"], "#B05252"),
    )
    for bounds, (label, smiles, color) in zip(positions, labels, strict=True):
        inset = example_axis.inset_axes(bounds)
        _draw_structure(inset, str(smiles), label, color)
    example_axis.text(
        0.32,
        0.50,
        "same two-cell periodic\nidentity",
        ha="center",
        va="center",
        fontsize=7,
        color="#377D58",
        zorder=20,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
        transform=example_axis.transAxes,
    )
    example_axis.text(
        0.68,
        0.50,
        "different structural graph\nmatched composition",
        ha="center",
        va="center",
        fontsize=7,
        color="#B05252",
        zorder=20,
        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"},
        transform=example_axis.transAxes,
    )
    example_axis.text(
        0.5,
        0.01,
        f"Observed anchor–negative similarity = {float(example['morgan_similarity']):.2f}",
        ha="center",
        fontsize=7.5,
        transform=example_axis.transAxes,
    )

    count_axis = figure.add_subplot(grid[1, 0])
    relation_classes = (
        "equivalence_preserving",
        "topology_changing",
        "polymer_semantics_inapplicable",
        "unknown_or_ineligible",
    )
    class_labels = (
        "equivalent\n(close)",
        "structural change\n(separate)",
        "semantic scope\n(descriptive)",
        "unknown/ineligible\n(no target)",
    )
    class_colors = ("#4E9B78", "#B05252", "#8A68AC", "#A8A8A8")
    class_counts = (
        typed["relation_class"].astype(str).value_counts().reindex(
            relation_classes, fill_value=0
        )
    )
    positions = np.arange(len(relation_classes))
    count_axis.barh(positions, class_counts.to_numpy(dtype=float), color=class_colors)
    count_axis.set_yticks(positions, class_labels)
    count_axis.invert_yaxis()
    count_axis.set_xlabel("Relation records")
    count_axis.set_title("c  Typed relation contract")
    for position, value in enumerate(class_counts):
        count_axis.text(float(value), position, f"  {int(value):,}", va="center", fontsize=7)

    match_axis = figure.add_subplot(grid[1, 1])
    match_axis.hist(
        frame["morgan_similarity"].dropna(),
        bins=np.linspace(0.60, 1.0, 17),
        color="#4E9B78",
        edgecolor="white",
    )
    match_axis.axvline(0.60, color="black", linestyle="--", linewidth=1)
    token_match = (
        frame["negative_token_length"].to_numpy(dtype=float)
        - frame["anchor_token_length"].to_numpy(dtype=float)
    )
    match_axis.text(
        0.98,
        0.96,
        f"|token difference| ≤ 5: {np.mean(np.abs(token_match) <= 5) * 100:.1f}%",
        ha="right",
        va="top",
        fontsize=7,
        transform=match_axis.transAxes,
    )
    match_axis.set_xlabel("Anchor–negative Morgan similarity")
    match_axis.set_ylabel("Triplets")
    match_axis.set_title("d  Near-negative matching")

    eligibility_axis = figure.add_subplot(grid[1, 2])
    candidate_scope = typed.loc[
        typed["training_eligible"].astype(bool)
        | typed["relation_class"].astype(str).eq("unknown_or_ineligible")
    ].copy()
    anchor_status = candidate_scope.groupby(
        ["split", "anchor_id"], as_index=False
    )["training_eligible"].any()
    accepted = int(anchor_status["training_eligible"].sum())
    examined = int(len(anchor_status))
    observed = np.asarray(
        [
            accepted / examined,
            frame["anchor_nontrivial_periodic_cut_available"].astype(bool).mean(),
            frame["anchor_formula"].astype(str).eq(frame["negative_formula"].astype(str)).mean(),
        ],
        dtype=float,
    )
    labels = [
        "relation eligible\n/ examined anchor",
        "nontrivial cut\n/ accepted anchor",
        "exact formula\nmatch",
    ]
    eligibility_axis.bar(labels, observed * 100, color=["#3B6FB6", "#8A68AC", "#D47A4A"])
    for index, value in enumerate(observed):
        eligibility_axis.text(index, value * 100 + 1.5, f"{value * 100:.1f}%", ha="center", fontsize=7.5)
    eligibility_axis.set_ylim(0, 112)
    eligibility_axis.set_ylabel("Observed records (%)")
    eligibility_axis.set_title("e  Eligibility and composition")
    eligibility_axis.text(
        0.02,
        0.97,
        f"Candidate-search denominator: {examined:,}\nAccepted anchors: {accepted:,}",
        va="top",
        fontsize=6.8,
        transform=eligibility_axis.transAxes,
    )

    title = (
        "Figure 1 | Algorithmically verified relations under an explicit two-cell definition"
    )
    stem = "figure1_relation_benchmark"
    if preview:
        title += " (engineering preview)"
        stem += "_smoke_preview"
    figure.suptitle(title, fontsize=12, fontweight="bold")
    outputs = _save(figure, output_root, stem)
    return FigureOutcome(
        stem,
        "generated",
        tuple(str(path) for path in sources),
        outputs,
        f"{len(frame):,} observed triplets with an explicit four-class relation sidecar",
    )


def _arm_design(config_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    reference = values["reference_arm"]
    rows.append(
        {
            "method": str(reference["id"]),
            "initialization": str(reference["initialization"]),
            "objective": "none",
        }
    )
    rows.extend(
        {
            "method": str(row["id"]),
            "initialization": str(row["initialization"]),
            "objective": str(row["objective"]),
        }
        for row in values["trained_arms"]
    )
    design = pd.DataFrame(rows)
    design["molecular init."] = design["initialization"].eq("molecular_pretrained")
    design["PI1M MLM"] = design["objective"].ne("none")
    design["boundary MLM"] = design["objective"].str.contains("boundary")
    design["generic relation"] = design["objective"].str.contains("generic_contrastive")
    design["verified equivalence"] = design["objective"].str.contains("equivalence_topology")
    design["shared observed negative"] = design["objective"].str.contains(
        "generic_contrastive|equivalence_topology", regex=True
    )
    order = [method for method in METHOD_ORDER if method in set(design["method"])]
    design["_order"] = design["method"].map({value: index for index, value in enumerate(order)})
    return design.sort_values("_order").drop(columns="_order"), values


def figure2_controlled_design(project_root: Path, output_root: Path) -> FigureOutcome:
    """Figure 2: observed relation composition and the configured seven arms."""

    relation, relation_sources = _relation_tables(project_root)
    supplemental, supplemental_sources = _supplemental_relation_tables(project_root)
    config_path = project_root / "configs/pretraining/arms.yaml"
    if relation.empty or not config_path.is_file() or "positive_relation" not in relation:
        return FigureOutcome(
            "figure2_controlled_pretraining",
            "unavailable",
            tuple(str(path) for path in relation_sources),
            message="formal relation composition and the seven-arm configuration are both required",
        )
    try:
        design, _ = _arm_design(config_path)
    except (KeyError, TypeError, yaml.YAMLError, OSError) as error:
        return FigureOutcome(
            "figure2_controlled_pretraining",
            "unavailable",
            (str(config_path),),
            message=f"the seven-arm configuration could not be read: {error}",
        )
    if set(METHOD_ORDER) != set(design["method"]):
        return FigureOutcome(
            "figure2_controlled_pretraining",
            "unavailable",
            (str(config_path),),
            message="the configured method set does not contain the declared seven controlled arms",
        )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.2),
        gridspec_kw={"width_ratios": (0.82, 1.75)},
        constrained_layout=True,
    )
    relation_counts = relation["positive_relation"].astype(str).value_counts()
    labels = [f"positive: {_short_label(value)}" for value in relation_counts.index]
    values = relation_counts.to_numpy(dtype=float).tolist()
    colors = ["#3B6FB6", "#4E9B78", "#D47A4A"][: len(values)]
    if not supplemental.empty and "negative_relation" in supplemental:
        supplemental_counts = supplemental["negative_relation"].astype(str).value_counts()
        labels.extend(f"control: {_short_label(value)}" for value in supplemental_counts.index)
        values.extend(supplemental_counts.to_numpy(dtype=float).tolist())
        colors.extend(["#A76EAE", "#B99435"][: len(supplemental_counts)])
    positions = np.arange(len(values))
    axes[0].barh(positions, values, color=colors)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Observed relation records")
    axes[0].set_title("a  Formal relation composition")
    axes[0].text(
        0.02,
        -0.16,
        "Primary negative: matched real distinct two-cell graph\nSupplemental connection-edit controls remain separate",
        va="top",
        fontsize=7,
        transform=axes[0].transAxes,
        clip_on=False,
    )

    columns = [
        "molecular init.",
        "PI1M MLM",
        "boundary MLM",
        "generic relation",
        "verified equivalence",
        "shared observed negative",
    ]
    matrix = design[columns].to_numpy(dtype=int)
    axes[1].imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#F1F1F1", "#3B6FB6"]), vmin=0, vmax=1, aspect="auto")
    display_columns = [
        "molecular\ninit.", "PI1M\nMLM", "boundary\nMLM",
        "traversal\npositive", "polymer-relation\npositive mixture",
        "shared\nobserved\nnegative",
    ]
    axes[1].set_xticks(np.arange(len(columns)), display_columns)
    axes[1].set_yticks(
        np.arange(len(design)),
        [METHOD_LABELS.get(value, value) for value in design["method"]],
    )
    axes[1].tick_params(axis="both", length=0)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axes[1].text(
                column_index,
                row_index,
                "✓" if matrix[row_index, column_index] else "—",
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] else "#888888",
                fontsize=11,
                fontweight="bold" if matrix[row_index, column_index] else "normal",
            )
    axes[1].set_title("b  Seven-arm controlled design")
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    axes[1].text(
        0.5,
        -0.19,
        "Trained arms: matched split, anchor stream/budget and checkpoint schedule.\n"
        "A_gcl / A_eqs: matched optimizer steps, view count and negative IDs;\n"
        "positive source and relation-family mixture differ. Total FLOPs / wall time are not matched.",
        ha="center",
        va="top",
        fontsize=7.5,
        transform=axes[1].transAxes,
    )
    axes[1].text(
        0.5,
        -0.37,
        "Primary contrasts: A0−B0 (initialization)  |  A_att−A0 (boundary)  |  A_eqs−A_gcl (relation supervision)",
        ha="center",
        va="top",
        fontsize=7.5,
        transform=axes[1].transAxes,
    )
    figure.suptitle(
        "Figure 2 | Seven training arms define controlled scientific contrasts",
        fontsize=12,
        fontweight="bold",
    )
    outputs = _save(figure, output_root, "figure2_controlled_pretraining")
    sources = tuple(
        str(path) for path in [*relation_sources, *supplemental_sources, config_path]
    )
    return FigureOutcome(
        "figure2_controlled_pretraining",
        "generated",
        sources,
        outputs,
        f"seven configured arms; {len(relation):,} primary relation triplets",
    )


def _token_efficiency_records(results_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Read the fixed-mask evaluations at the six declared token checkpoints."""

    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path in _formal_paths(results_root, "metrics.json"):
        if "token_efficiency" not in path.parts:
            continue
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        required = {
            "method",
            "representation_seed",
            "checkpoint_fraction",
            "checkpoint_tokens",
            "token_mean_nll",
        }
        if not required.issubset(entry) or entry["checkpoint_tokens"] is None:
            continue
        method = str(entry["method"])
        seed = int(entry["representation_seed"])
        if method not in {"A0", "B0"}:
            continue
        if seed in DEVELOPMENT_SEEDS:
            continue
        rows.append(
            {
                "method": method,
                "representation_seed": seed,
                "checkpoint_fraction": float(entry["checkpoint_fraction"]),
                "checkpoint_tokens": int(entry["checkpoint_tokens"]),
                "token_mean_nll": float(entry["token_mean_nll"]),
                "source_path": str(path),
            }
        )
        sources.append(path)
    return pd.DataFrame(rows), sources


def _has_complete_token_efficiency_grid(frame: pd.DataFrame) -> bool:
    """Return whether all 60 prespecified A0/B0 checkpoint cells are present."""

    required_columns = {
        "method",
        "representation_seed",
        "checkpoint_fraction",
    }
    if frame.empty or not required_columns.issubset(frame.columns):
        return False
    expected = {
        (method, seed, fraction)
        for method in ("A0", "B0")
        for seed in FORMAL_REPRESENTATION_SEEDS
        for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
    }
    observed = {
        (str(row.method), int(row.representation_seed), float(row.checkpoint_fraction))
        for row in frame.itertuples()
    }
    return len(frame) == len(expected) and observed == expected


def _boundary_summaries(results_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = _formal_paths(results_root, "local_mlm_summary.csv")
    frame, sources = _read_tables(paths)
    if not frame.empty:
        frame = _filter_formal_seeds(frame)
    return frame, sources


def _has_boundary_seed_coverage(
    frame: pd.DataFrame, method: str, expected_seeds: set[int]
) -> bool:
    """Return whether a boundary-MLM method has every declared replicate."""

    if frame.empty or not {"method", "representation_seed"}.issubset(frame.columns):
        return False
    method_seeds = set(
        pd.to_numeric(
            frame.loc[frame["method"].eq(method), "representation_seed"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    return expected_seeds.issubset(method_seeds)


def figure3_initialization_and_boundary(
    results_root: Path, output_root: Path
) -> FigureOutcome:
    """Figure 3: token efficiency and attachment-local effects."""

    training, training_sources = _token_efficiency_records(results_root)
    local, local_sources = _boundary_summaries(results_root)
    local_required = {"method", "representation_seed", "stratum", "mean_nll"}
    primary_boundary_complete = all(
        _has_boundary_seed_coverage(local, method, FORMAL_REPRESENTATION_SEEDS)
        for method in ("A0", "A_att")
    )
    if (
        training.empty
        or not _has_complete_token_efficiency_grid(training)
        or local.empty
        or not local_required.issubset(local.columns)
        or not primary_boundary_complete
    ):
        return FigureOutcome(
            "figure3_boundary_locality",
            "unavailable",
            tuple(str(path) for path in [*training_sources, *local_sources]),
            message=(
                "Figure 3 requires the complete 60-cell formal A0/B0 six-checkpoint "
                "fixed-mask MLM grid and formal "
                "A0/A_att atom-group MLM summaries; development and smoke runs are excluded"
            ),
        )

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), constrained_layout=True)
    for method in ("B0", "A0"):
        cell = training.loc[training["method"].eq(method)]
        for _, run in cell.groupby("representation_seed"):
            ordered = run.sort_values("checkpoint_fraction")
            axes[0].plot(
                ordered["checkpoint_tokens"] / 1e6,
                ordered["token_mean_nll"],
                color=METHOD_COLORS[method],
                alpha=0.24,
                linewidth=0.8,
            )
        summary = (
            cell.groupby("checkpoint_fraction", as_index=False)
            .agg(
                checkpoint_tokens=("checkpoint_tokens", "mean"),
                token_mean_nll=("token_mean_nll", "mean"),
            )
            .sort_values("checkpoint_fraction")
        )
        axes[0].plot(
            summary["checkpoint_tokens"] / 1e6,
            summary["token_mean_nll"],
            marker="o",
            linewidth=2.1,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axes[0].set_xlabel("Observed non-padding tokens (millions)")
    axes[0].set_ylabel("PI1M validation fixed-mask token NLL")
    axes[0].set_title("a  Molecular initialization and token efficiency")
    axes[0].legend(frameon=False)

    # distance_3_plus is a compatibility alias for interior, not independent
    # evidence, so the plot includes the underlying population only once.
    strata = ["distance_0", "distance_1", "distance_2", "interior"]
    local_strata = set(local["stratum"].astype(str))
    available_strata = [value for value in strata if value in local_strata]
    control_seeds = {42, 123, 777}
    candidate_methods = ["A_att"]
    method_seeds = {"A_att": FORMAL_REPRESENTATION_SEEDS}
    candidate_methods.extend(
        method
        for method in ("matched_random_masking", "permuted_boundary_masking")
        if _has_boundary_seed_coverage(local, method, control_seeds)
    )
    method_seeds.update({method: control_seeds for method in candidate_methods[1:]})
    plotted = 0
    for method in candidate_methods:
        candidate = local.loc[
            local["method"].eq(method)
            & local["representation_seed"].isin(method_seeds[method]),
            ["representation_seed", "stratum", "mean_nll"],
        ]
        baseline = local.loc[
            local["method"].eq("A0"),
            ["representation_seed", "stratum", "mean_nll"],
        ]
        paired = candidate.merge(
            baseline,
            on=["representation_seed", "stratum"],
            suffixes=("_candidate", "_a0"),
        )
        if paired.empty:
            continue
        paired["delta_nll"] = paired["mean_nll_candidate"] - paired["mean_nll_a0"]
        summary = paired.groupby("stratum")["delta_nll"].agg(["mean", "sem"]).reindex(available_strata)
        axes[1].errorbar(
            np.arange(len(available_strata)),
            summary["mean"],
            yerr=summary["sem"].fillna(0.0),
            marker="o",
            capsize=3,
            linewidth=1.6,
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
        plotted += 1
    if plotted == 0:
        plt.close(figure)
        return FigureOutcome(
            "figure3_boundary_locality",
            "unavailable",
            tuple(str(path) for path in [*training_sources, *local_sources]),
            message="A_att and A0 local MLM summaries do not share a formal representation seed",
        )
    axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    axes[1].set_xticks(np.arange(len(available_strata)), [value.replace("distance_", "d=").replace("_plus", "+") for value in available_strata], rotation=18)
    axes[1].set_ylabel(r"Masked-token NLL difference from A0 ($\Delta$NLL)")
    axes[1].set_title("b  Attachment-local effect at matched atom masks")
    axes[1].legend(frameon=False, fontsize=7.5)
    figure.suptitle(
        "Figure 3 | Initialization and boundary supervision are tested at their intended scales",
        fontsize=12,
        fontweight="bold",
    )
    outputs = _save(figure, output_root, "figure3_boundary_locality")
    return FigureOutcome(
        "figure3_boundary_locality",
        "generated",
        tuple(str(path) for path in [*training_sources, *local_sources]),
        outputs,
        "validation token efficiency and attachment-stratified atom-group MLM",
    )


def _fidelity_triplets(results_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frame, sources = _read_tables(_formal_paths(results_root, "triplet_metrics.parquet"))
    if frame.empty:
        return frame, sources
    frame = _filter_formal_seeds(frame)
    if "method" in frame:
        # Three-seed mechanism controls are reported separately and must not
        # change the seven-arm main Figure 4 population.
        frame = frame.loc[frame["method"].astype(str).isin(METHOD_ORDER)].copy()
    if "split" in frame and frame["split"].astype(str).eq("test").any():
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    return frame, sources


def figure4_fidelity_tradeoff(results_root: Path, output_root: Path) -> FigureOutcome:
    """Figure 4: equivalence and topology behavior on test structures."""

    frame, sources = _fidelity_triplets(results_root)
    required = {
        "method",
        "representation_seed",
        "positive_relation",
        "anchor_positive_cosine",
        "topology_margin",
        "triplet_correct",
        "retrieval_correct_at_1",
        "reciprocal_rank",
    }
    if frame.empty or not required.issubset(frame.columns):
        return FigureOutcome(
            "figure4_fidelity_tradeoff",
            "unavailable",
            tuple(str(path) for path in sources),
            message="formal test-split triplet metrics for equivalence retrieval and topology discrimination are unavailable",
        )
    methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    if not methods:
        return FigureOutcome(
            "figure4_fidelity_tradeoff",
            "unavailable",
            tuple(str(path) for path in sources),
            message="triplet metrics do not identify a declared training arm",
        )
    per_seed = (
        frame.groupby(["method", "representation_seed"], as_index=False)
        .agg(
            equivalence_cosine=("anchor_positive_cosine", "mean"),
            recall_at_1=("retrieval_correct_at_1", "mean"),
            mrr=("reciprocal_rank", "mean"),
            topology_margin=("topology_margin", "mean"),
            triplet_accuracy=("triplet_correct", "mean"),
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
    x = np.arange(len(methods))
    for metric, marker, label in (
        ("equivalence_cosine", "o", "cosine"),
        ("recall_at_1", "s", "Recall@1"),
        ("mrr", "^", "MRR"),
    ):
        summary = per_seed.groupby("method")[metric].mean().reindex(methods)
        axes[0].plot(x, summary, marker=marker, linewidth=1.4, label=label)
    axes[0].set_xticks(x, [METHOD_LABELS.get(value, value) for value in methods], rotation=25)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Observed metric (reported separately)")
    axes[0].set_title("a  Equivalent-view consistency")
    axes[0].legend(frameon=False, fontsize=7.5)

    for method_index, method in enumerate(methods):
        cell = per_seed.loc[per_seed["method"].eq(method)]
        jitter = np.linspace(-0.08, 0.08, len(cell)) if len(cell) > 1 else np.zeros(len(cell))
        axes[1].scatter(
            method_index + jitter,
            cell["topology_margin"],
            color=METHOD_COLORS.get(method, "#555555"),
            alpha=0.5,
            s=24,
        )
        axes[1].scatter(
            method_index,
            cell["topology_margin"].mean(),
            color=METHOD_COLORS.get(method, "#555555"),
            edgecolor="black",
            linewidth=0.5,
            s=58,
        )
        axes[1].text(
            method_index,
            cell["topology_margin"].mean(),
            f"\n{cell['triplet_accuracy'].mean():.2f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    axes[1].axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, [METHOD_LABELS.get(value, value) for value in methods], rotation=25)
    axes[1].set_ylabel("Topology margin")
    axes[1].set_title("b  Relative topology discrimination\n(numbers: triplet accuracy)")

    family = (
        frame.groupby(["method", "representation_seed", "positive_relation"], as_index=False)["topology_margin"]
        .mean()
        .groupby(["method", "positive_relation"], as_index=False)["topology_margin"]
        .mean()
    )
    families = sorted(family["positive_relation"].astype(str).unique())
    focus = [method for method in ("A_gcl", "A_eqs", "A_att") if method in methods]
    if not focus:
        focus = methods
    for method in focus:
        values = family.loc[family["method"].eq(method)].set_index("positive_relation")["topology_margin"].reindex(families)
        axes[2].plot(
            np.arange(len(families)),
            values,
            marker="o",
            linewidth=1.6,
            color=METHOD_COLORS.get(method),
            label=METHOD_LABELS.get(method, method),
        )
    axes[2].axhline(0.0, color="0.45", linestyle="--", linewidth=1)
    axes[2].set_xticks(np.arange(len(families)), [_short_label(value) for value in families], rotation=22)
    axes[2].set_ylabel("Mean topology margin")
    axes[2].set_title("c  Test-set positive-view families")
    axes[2].legend(frameon=False, fontsize=7.5)

    figure.suptitle(
        "Figure 4 | Equivalent-view agreement and relative topology discrimination are distinct responses",
        fontsize=12,
        fontweight="bold",
    )
    outputs = _save(figure, output_root, "figure4_fidelity_tradeoff")
    return FigureOutcome(
        "figure4_fidelity_tradeoff",
        "generated",
        tuple(str(path) for path in sources),
        outputs,
        f"{len(frame):,} observed test triplets; equivalence and topology metrics remain separate",
    )


def figure4d_property_view_stability(
    statistics_root: str | Path,
    output_root: str | Path,
) -> FigureOutcome:
    """Companion panel: property predictions across legal equivalent views."""

    statistics = Path(statistics_root)
    output = Path(output_root)
    task_path = statistics / "figure4d_task_data.csv"
    aggregate_path = statistics / "comparison_aggregate.csv"
    if not task_path.is_file() or not aggregate_path.is_file():
        return FigureOutcome(
            "figure4d_property_view_stability",
            "unavailable",
            tuple(str(path) for path in (task_path, aggregate_path) if path.is_file()),
            message="record-level property-view task effects and bootstrap aggregates are required",
        )
    task = pd.read_csv(task_path)
    aggregate = pd.read_csv(aggregate_path)
    task_required = {"comparison", "view_family", "task", "paired_log_ratio"}
    aggregate_required = {
        "comparison",
        "view_family",
        "estimate",
        "ci_lower",
        "ci_upper",
    }
    if (
        task.empty
        or aggregate.empty
        or not task_required.issubset(task.columns)
        or not aggregate_required.issubset(aggregate.columns)
    ):
        return FigureOutcome(
            "figure4d_property_view_stability",
            "unavailable",
            (str(task_path), str(aggregate_path)),
            message="property-view statistics do not contain the required comparison fields",
        )

    comparisons = [
        value
        for value in (
            "molecular_initialization",
            "boundary_supervision",
            "verified_relations",
        )
        if value in set(aggregate["comparison"].astype(str))
    ]
    families = [
        value
        for value in ("periodic_cut", "traversal")
        if value in set(aggregate["view_family"].astype(str))
    ]
    if len(comparisons) != 3 or len(families) != 2:
        return FigureOutcome(
            "figure4d_property_view_stability",
            "unavailable",
            (str(task_path), str(aggregate_path)),
            message="all three comparisons and both equivalent-view axes are required",
        )

    _style()
    figure, axes = plt.subplots(
        1, 3, figsize=(13.4, 5.4), sharex=False, sharey=True, constrained_layout=True
    )
    family_colors = {"periodic_cut": "#8A68AC", "traversal": "#3B6FB6"}
    family_labels = {"periodic_cut": "periodic cut", "traversal": "traversal"}
    tasks = sorted(task["task"].astype(str).unique())
    labels = ["task-equal aggregate", *tasks]
    y = np.arange(len(labels), dtype=float)
    offsets = {"periodic_cut": -0.13, "traversal": 0.13}

    for axis, comparison in zip(axes, comparisons, strict=True):
        task_cell = task.loc[task["comparison"].astype(str).eq(comparison)]
        aggregate_cell = aggregate.loc[
            aggregate["comparison"].astype(str).eq(comparison)
        ]
        for family in families:
            color = family_colors[family]
            family_tasks = (
                task_cell.loc[task_cell["view_family"].astype(str).eq(family)]
                .groupby("task")["paired_log_ratio"]
                .mean()
                .reindex(tasks)
            )
            task_percent = 100.0 * (np.exp(family_tasks.to_numpy(dtype=float)) - 1.0)
            axis.scatter(
                task_percent,
                y[1:] + offsets[family],
                s=28,
                color=color,
                alpha=0.82,
                label=family_labels[family],
            )
            aggregate_row = aggregate_cell.loc[
                aggregate_cell["view_family"].astype(str).eq(family)
            ].iloc[0]
            estimate = 100.0 * (np.exp(float(aggregate_row["estimate"])) - 1.0)
            lower = 100.0 * (np.exp(float(aggregate_row["ci_lower"])) - 1.0)
            upper = 100.0 * (np.exp(float(aggregate_row["ci_upper"])) - 1.0)
            axis.errorbar(
                estimate,
                y[0] + offsets[family],
                xerr=np.asarray([[estimate - lower], [upper - estimate]]),
                fmt="D",
                markersize=5,
                capsize=3,
                color=color,
                linewidth=1.3,
            )
        axis.axvline(0.0, color="0.45", linestyle="--", linewidth=1)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Prediction-SD change (%)\nnegative = candidate more stable")
        axis.set_title(COMPARISON_LABELS.get(comparison, comparison))
        axis.grid(axis="x", color="0.9", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(
        "Figure 4d | Property-level invariance is axis-specific and task-dependent",
        fontsize=12,
        fontweight="bold",
    )
    outputs = _save(figure, output, "figure4d_property_view_stability")
    return FigureOutcome(
        "figure4d_property_view_stability",
        "generated",
        (str(task_path), str(aggregate_path)),
        outputs,
        "record-equal prediction standard deviation; traversal and periodic-cut axes remain separate",
    )


def _task_name(value: str) -> str:
    return str(value).casefold().rsplit(":", maxsplit=1)[-1].replace("-", "_").replace(" ", "_")


def _full_test_scope(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _filter_formal_seeds(frame)
    if "split" in selected:
        selected = selected.loc[selected["split"].astype(str).eq("test")]
    if "label_fraction" in selected:
        selected = selected.loc[np.isclose(pd.to_numeric(selected["label_fraction"]), 1.0)]
    if "subsample_id" in selected:
        selected = selected.loc[pd.to_numeric(selected["subsample_id"]).eq(0)]
    if "input_view" in selected:
        selected = selected.loc[selected["input_view"].astype(str).eq("valid")]
    return selected.copy()


def _primary_task_scope(frame: pd.DataFrame) -> pd.DataFrame:
    selected = _full_test_scope(frame)
    if selected.empty:
        return selected
    names = selected["task"].astype(str).map(_task_name)
    selected = selected.loc[
        ~selected["task"].astype(str).str.casefold().str.startswith("openpoly:")
        & ~names.isin(RETENTION_TASKS)
    ]
    if "protocol" not in selected:
        return selected
    blocks: list[pd.DataFrame] = []
    for task, cell in selected.groupby("task", sort=True):
        protocols = set(cell["protocol"].astype(str))
        name = _task_name(task)
        choice = None
        if name in {"ea", "ip"} and "family_ood" in protocols:
            choice = "family_ood"
        elif "group_ood" in protocols:
            choice = "group_ood"
        elif len(protocols) == 1:
            choice = next(iter(protocols))
        if choice is not None:
            blocks.append(cell.loc[cell["protocol"].astype(str).eq(choice)])
    return pd.concat(blocks, ignore_index=True) if blocks else selected.iloc[0:0]


def _learning_curve_scope(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the four tasks with the declared six-point label curves."""

    selected = _filter_formal_seeds(frame)
    if "split" in selected:
        selected = selected.loc[selected["split"].astype(str).eq("test")]
    if "input_view" in selected:
        selected = selected.loc[selected["input_view"].astype(str).eq("valid")]
    names = selected["task"].astype(str).map(_task_name)
    selected = selected.loc[names.isin({"ea", "ip", "egc", "xc"})].copy()
    blocks: list[pd.DataFrame] = []
    for task, cell in selected.groupby("task", sort=True):
        protocols = set(cell["protocol"].astype(str)) if "protocol" in cell else set()
        name = _task_name(task)
        choice = "family_ood" if name in {"ea", "ip"} else "group_ood"
        if choice in protocols:
            blocks.append(cell.loc[cell["protocol"].astype(str).eq(choice)])
    return pd.concat(blocks, ignore_index=True) if blocks else selected.iloc[0:0]


def _method_ranks(task_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = _primary_task_scope(task_metrics)
    selected = selected.loc[selected["method"].isin(METHOD_ORDER)]
    if selected.empty or "value" not in selected:
        return pd.DataFrame()
    pooled = _pool_figure_metrics(selected)
    by_rep = pooled.groupby(["task", "method", "representation_seed"], as_index=False)["value"].mean()
    averaged = by_rep.groupby(["task", "method"], as_index=False)["value"].mean()
    pivot = averaged.pivot(index="task", columns="method", values="value")
    present = [method for method in METHOD_ORDER if method in pivot.columns]
    if len(present) < 2:
        return pd.DataFrame()
    complete = pivot[present].dropna()
    if complete.empty:
        return pd.DataFrame()
    ranks = complete.rank(axis=1, method="average", ascending=True)
    return pd.DataFrame(
        {
            "method": present,
            "mean_rank": ranks.mean(axis=0).reindex(present).to_numpy(),
            "tasks": len(complete),
        }
    ).sort_values("mean_rank")


def _retention_matrix(task_metrics: pd.DataFrame) -> pd.DataFrame:
    selected = _full_test_scope(task_metrics)
    if selected.empty:
        return pd.DataFrame()
    selected["retention_task"] = selected["task"].astype(str).map(_task_name)
    selected = selected.loc[selected["retention_task"].isin(RETENTION_TASKS)]
    if selected.empty or "C" not in set(selected["method"]):
        return pd.DataFrame()
    pooled = _pool_figure_metrics(selected)
    pooled["retention_task"] = pooled["task"].astype(str).map(_task_name)
    by_rep = pooled.groupby(["method", "retention_task", "representation_seed"], as_index=False)["value"].mean()
    means = by_rep.groupby(["method", "retention_task"], as_index=False)["value"].mean()
    table = means.pivot(index="method", columns="retention_task", values="value")
    if "C" not in table.index:
        return pd.DataFrame()
    common_tasks = [task for task in sorted(table.columns) if np.isfinite(table.loc["C", task]) and table.loc["C", task] > 0]
    methods = [method for method in METHOD_ORDER if method in table.index]
    if not common_tasks or len(methods) < 2:
        return pd.DataFrame()
    return np.log(table.loc[methods, common_tasks].divide(table.loc["C", common_tasks], axis=1))


def _openpoly_effects(task_effects: pd.DataFrame) -> pd.DataFrame:
    selected = _full_test_scope(task_effects)
    if selected.empty:
        return selected
    selected = selected.loc[
        selected["task"].astype(str).str.casefold().str.startswith("openpoly:")
        & selected["comparison"].astype(str).eq("verified_relations")
    ].copy()
    if selected.empty:
        return selected
    selected["endpoint"] = selected["task"].astype(str).map(_task_name)
    selected = selected.loc[selected["endpoint"].isin(OPENPOLY_HIGHLIGHTS)]
    return (
        selected.groupby("endpoint", as_index=False)["paired_log_error_ratio"]
        .mean()
        .set_index("endpoint")
        .reindex(OPENPOLY_HIGHLIGHTS)
        .dropna()
        .reset_index()
    )


def _pool_figure_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct held-out MAE/RMSE before any seed/head averaging."""
    return pool_fold_metrics(frame.assign(source_group="frozen_probe"))


def _learning_curve_effects(task_metrics: pd.DataFrame) -> pd.DataFrame:
    """Descriptive four-task curves, preserving the nested repetition weights."""
    selected = _learning_curve_scope(task_metrics)
    if selected.empty:
        return pd.DataFrame()
    pooled = _pool_figure_metrics(selected)
    keys = ["representation_seed", "head_seed", "task", "protocol", "split",
            "label_fraction", "subsample_id", "input_view", "metric"]
    blocks = []
    for comparison, (candidate, baseline) in PRIMARY_COMPARISONS.items():
        paired = pooled.loc[pooled.method.eq(candidate), keys + ["value"]].merge(
            pooled.loc[pooled.method.eq(baseline), keys + ["value"]],
            on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one",
        )
        paired["paired_log_error_ratio"] = np.log(paired.value_candidate / paired.value_baseline)
        # Heads are nested in representations; subsamples and tasks receive
        # equal weight rather than weight proportional to their fold count.
        per_rep = paired.groupby(["task", "label_fraction", "subsample_id", "representation_seed"],
                                 as_index=False).paired_log_error_ratio.mean()
        per_sample = per_rep.groupby(["task", "label_fraction", "subsample_id"],
                                     as_index=False).paired_log_error_ratio.mean()
        per_task = per_sample.groupby(["task", "label_fraction"],
                                     as_index=False).paired_log_error_ratio.mean()
        curve = per_task.groupby("label_fraction", as_index=False).agg(
            paired_log_error_ratio=("paired_log_error_ratio", "mean"), n_tasks=("task", "nunique"))
        blocks.append(curve.assign(comparison=comparison))
    return pd.concat(blocks, ignore_index=True)


def _fidelity_utility_cells(
    task_metrics: pd.DataFrame,
    fidelity_triplets: pd.DataFrame,
) -> pd.DataFrame:
    """Align separate fidelity axes with task-equal downstream error by run.

    This is a descriptive association, not a composite score or a causal
    analysis.  C is one fixed molecular checkpoint (representation seed 0), so
    its task value is broadcast as the common reference for each independently
    pretrained candidate seed.  Equivalence and topology remain separate
    columns.
    """

    downstream = _primary_task_scope(task_metrics)
    required_downstream = {"method", "representation_seed", "task", "value"}
    required_fidelity = {
        "method",
        "representation_seed",
        "anchor_positive_cosine",
        "topology_margin",
    }
    if (
        downstream.empty
        or fidelity_triplets.empty
        or not required_downstream.issubset(downstream.columns)
        or not required_fidelity.issubset(fidelity_triplets.columns)
    ):
        return pd.DataFrame()

    task_values = _pool_figure_metrics(downstream)
    reference_keys = ["task", "protocol", "head_seed", "metric"]
    reference = task_values.loc[task_values.method.eq("C"), reference_keys + ["value"]].rename(
        columns={"value": "reference_value"})
    candidates = task_values.loc[~task_values["method"].eq("C")]
    paired = candidates.merge(
        reference,
        on=reference_keys,
        how="inner",
        validate="many_to_one",
    )
    paired = paired.loc[
        paired["value"].gt(0.0) & paired["reference_value"].gt(0.0)
    ].copy()
    if paired.empty:
        return pd.DataFrame()
    paired["task_log_error_ratio_vs_C"] = np.log(
        paired["value"] / paired["reference_value"]
    )
    per_task = paired.groupby(["method", "representation_seed", "task"], as_index=False)[
        "task_log_error_ratio_vs_C"].mean()
    utility = (
        per_task.groupby(["method", "representation_seed"], as_index=False)
        .agg(
            task_equal_log_error_ratio_vs_C=(
                "task_log_error_ratio_vs_C",
                "mean",
            ),
            downstream_tasks=("task", "nunique"),
        )
    )
    fidelity = (
        fidelity_triplets.groupby(
            ["method", "representation_seed"], as_index=False
        )
        .agg(
            equivalence_cosine=("anchor_positive_cosine", "mean"),
            topology_margin=("topology_margin", "mean"),
        )
    )
    return fidelity.merge(
        utility,
        on=["method", "representation_seed"],
        how="inner",
        validate="one_to_one",
    )


def _scope_counts(results_root: Path) -> tuple[pd.Series, list[Path]]:
    scope, sources = _read_tables(_formal_paths(results_root, "scope.parquet"))
    if scope.empty or not {"sample_id", "repeat_eligible"}.issubset(scope.columns):
        return pd.Series(dtype=int), sources
    # The same evaluation records are repeated across methods; the study-object
    # coverage is counted once per sample.
    scope = scope.drop_duplicates("sample_id")
    counts = scope["repeat_eligible"].astype(bool).value_counts()
    return counts, sources


def figure5_downstream_utility(results_root: Path, output_root: Path) -> FigureOutcome:
    """Figure 5: ranking, low-data effects, retention, and applicability."""

    aggregates, aggregate_sources = _read_tables(
        _downstream_statistic_paths(results_root, "primary_aggregate.csv")
    )
    metrics, metric_sources = _read_tables(
        _downstream_statistic_paths(results_root, "task_metrics.csv")
    )
    scope_counts, scope_sources = _scope_counts(results_root)
    aggregate_required = {"comparison", "estimate", "ci_lower", "ci_upper"}
    metrics_required = {"method", "task", "value"}
    ranks = _method_ranks(metrics) if metrics_required.issubset(metrics.columns) else pd.DataFrame()
    retention = _retention_matrix(metrics) if metrics_required.issubset(metrics.columns) else pd.DataFrame()
    curve = _learning_curve_effects(metrics) if metrics_required.issubset(metrics.columns) else pd.DataFrame()
    fidelity, fidelity_sources = _fidelity_triplets(results_root)
    fidelity_utility = (
        _fidelity_utility_cells(metrics, fidelity)
        if metrics_required.issubset(metrics.columns)
        else pd.DataFrame()
    )
    missing: list[str] = []
    if aggregates.empty or not aggregate_required.issubset(aggregates.columns):
        missing.append("primary aggregate effects")
    if curve.empty:
        missing.append("fold-pooled task-level learning-curve metrics")
    if ranks.empty:
        missing.append("matched primary-task method ranks")
    if retention.empty:
        missing.append("molecular retention metrics relative to C")
    if fidelity_utility.empty:
        missing.append("seed-aligned fidelity and downstream utility records")
    if scope_counts.empty:
        missing.append("repeat-unit eligibility records")
    sources = tuple(
        str(path)
        for path in [
            *aggregate_sources,
            *metric_sources,
            *scope_sources,
            *fidelity_sources,
        ]
    )
    if missing:
        return FigureOutcome(
            "figure5_downstream_utility",
            "unavailable",
            sources,
            message="Figure 5 awaits " + "; ".join(missing),
        )

    aggregates = aggregates.drop_duplicates("comparison", keep="last")
    comparison_order = [
        value
        for value in ("molecular_initialization", "boundary_supervision", "verified_relations")
        if value in set(aggregates["comparison"])
    ]
    if len(comparison_order) != 3:
        return FigureOutcome(
            "figure5_downstream_utility",
            "unavailable",
            sources,
            message="Figure 5 requires all three prespecified primary aggregate comparisons",
        )
    aggregates = aggregates.set_index("comparison").loc[comparison_order].reset_index()

    figure, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)

    rank_axis = axes[0, 0]
    ranks = ranks.sort_values("mean_rank", ascending=False)
    rank_axis.barh(
        [METHOD_LABELS.get(value, value) for value in ranks["method"]],
        ranks["mean_rank"],
        color=[METHOD_COLORS.get(value, "#777777") for value in ranks["method"]],
    )
    rank_axis.set_xlabel("Mean rank (lower is better)")
    rank_axis.set_title(f"a  Matched primary-task ranking (n={int(ranks['tasks'].iloc[0])})")

    effect_axis = axes[0, 1]
    positions = np.arange(len(aggregates))
    estimates = aggregates["estimate"].to_numpy(dtype=float)
    effect_axis.errorbar(
        estimates,
        positions,
        xerr=np.vstack(
            [
                estimates - aggregates["ci_lower"].to_numpy(dtype=float),
                aggregates["ci_upper"].to_numpy(dtype=float) - estimates,
            ]
        ),
        fmt="o",
        color="#315F9D",
        capsize=3,
    )
    effect_axis.axvline(0.0, color="0.4", linestyle="--", linewidth=1)
    effect_axis.set_yticks(
        positions,
        [COMPARISON_LABELS.get(value, value) for value in aggregates["comparison"]],
    )
    effect_axis.invert_yaxis()
    effect_axis.set_xlabel("Task-equal paired log error ratio")
    effect_axis.set_title("b  Prespecified downstream effects")

    low_axis = axes[0, 2]
    for comparison in comparison_order:
        cell = curve.loc[curve["comparison"].eq(comparison)].sort_values("label_fraction")
        low_axis.plot(
            cell["label_fraction"] * 100,
            cell["paired_log_error_ratio"],
            marker="o",
            label=COMPARISON_LABELS.get(comparison, comparison).split("\n")[0],
        )
    low_axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    low_axis.set_xscale("log")
    low_axis.set_xlabel("Training labels (%)")
    low_axis.set_ylabel("Task-equal paired log error ratio")
    low_axis.set_title("c  Four-task low-label behavior (descriptive)")
    low_axis.legend(frameon=False, fontsize=7)

    retention_axis = axes[1, 0]
    image = retention_axis.imshow(retention.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto")
    retention_axis.set_xticks(np.arange(len(retention.columns)), [value.replace("_", " ") for value in retention.columns], rotation=28)
    retention_axis.set_yticks(np.arange(len(retention.index)), [METHOD_LABELS.get(value, value) for value in retention.index])
    retention_axis.set_title("d  Molecular retention relative to C")
    retention_axis.set_xlabel("log(error / C error); 0 means unchanged error")
    figure.colorbar(image, ax=retention_axis, fraction=0.045, pad=0.03)

    association_axis = axes[1, 1]
    association_axis.set_axis_off()
    association_axis.set_title(
        "e  Fidelity–utility association (descriptive)", pad=6
    )
    association_panels = (
        association_axis.inset_axes((0.00, 0.08, 0.47, 0.80)),
        association_axis.inset_axes((0.53, 0.08, 0.47, 0.80)),
    )
    for axis, metric, label in (
        (association_panels[0], "equivalence_cosine", "Equivalence cosine"),
        (association_panels[1], "topology_margin", "Topology margin"),
    ):
        for method, cell in fidelity_utility.groupby("method", sort=True):
            axis.scatter(
                cell[metric],
                cell["task_equal_log_error_ratio_vs_C"],
                color=METHOD_COLORS.get(str(method), "#777777"),
                label=METHOD_LABELS.get(str(method), str(method)),
                alpha=0.75,
                s=25,
            )
        axis.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
        axis.set_xlabel(label, fontsize=7)
        axis.tick_params(labelsize=6.5)
        correlation = fidelity_utility[[metric, "task_equal_log_error_ratio_vs_C"]].corr(
            method="spearman"
        ).iloc[0, 1]
        axis.set_title(rf"$\rho_s$={correlation:.2f}", fontsize=7.5)
    association_panels[0].set_ylabel(
        "Task-equal log(error / C)", fontsize=7
    )

    scope_axis = axes[1, 2]
    eligible = int(scope_counts.get(True, 0))
    unavailable = int(scope_counts.get(False, 0))
    total = eligible + unavailable
    scope_axis.barh([0], [eligible], color="#4E9B78", label="eligible")
    scope_axis.barh([0], [unavailable], left=[eligible], color="#D0D0D0", label="outside/unsupported")
    scope_axis.set_yticks([])
    scope_axis.set_xlabel("Relation-test cohort records")
    scope_axis.set_title("f  Scope of the relation-test cohort")
    scope_axis.legend(frameon=False, fontsize=7, loc="lower center", ncol=2)
    scope_axis.text(
        0.5,
        0.73,
        f"Repeat-unit eligible: {eligible:,}/{total:,} ({eligible / total * 100:.1f}%)",
        ha="center",
        transform=scope_axis.transAxes,
    )
    scope_axis.text(
        0.5,
        0.33,
        "Selected 2D linear, one-component,\ndual-attachment repeat units.\nNot a downstream-wide eligibility estimate;\nsemantic controls are not equivalent views.",
        ha="center",
        va="center",
        fontsize=7.5,
        transform=scope_axis.transAxes,
    )

    figure.suptitle(
        "Figure 5 | Downstream utility is reported with its data regime, retention cost, and scope",
        fontsize=12,
        fontweight="bold",
    )
    outputs = _save(figure, output_root, "figure5_downstream_utility")
    atomic_write_csv(ranks, output_root / "figure5_method_ranks.csv")
    atomic_write_csv(curve, output_root / "figure5_learning_curve_effects.csv")
    atomic_write_csv(retention.reset_index(), output_root / "figure5_retention.csv")
    atomic_write_csv(fidelity_utility, output_root / "figure5_fidelity_utility.csv")
    return FigureOutcome(
        "figure5_downstream_utility",
        "generated",
        sources,
        outputs,
        "fold-pooled MAE/RMSE; heads nested within representations; original primary CIs unchanged; low-label curves and fidelity–utility associations descriptive",
    )


def make_all_figures(
    project_root: str | Path,
    results_root: str | Path,
    output_root: str | Path,
    *,
    include_smoke: bool = False,
) -> list[FigureOutcome]:
    _style()
    project = Path(project_root).resolve()
    results = Path(results_root)
    if not results.is_absolute():
        results = project / results
    output = Path(output_root)
    if not output.is_absolute():
        output = project / output
    outcomes = [
        figure1_relation_benchmark(project, output, include_smoke=include_smoke),
        figure2_controlled_design(project, output),
        figure3_initialization_and_boundary(results, output),
        figure4_fidelity_tradeoff(results, output),
        figure5_downstream_utility(results, output),
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "availability.json").write_text(
        json.dumps([outcome.as_dict() for outcome in outcomes], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return outcomes


__all__ = [
    "FigureOutcome",
    "figure4d_property_view_stability",
    "make_all_figures",
]
