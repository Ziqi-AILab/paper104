"""Equivalent-view consistency and applicability-control construction."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from polyfid.chemistry import (
    StructureError,
    assess_repeat_eligibility,
    canonicalize_smiles,
    generate_equivalent_periodic_cuts,
    generate_randomized_smiles,
    parse_repeat_unit,
    renumber_atoms,
)

from .fidelity import row_cosine_similarity

try:
    from rdkit import Chem, rdBase
except ModuleNotFoundError:  # pragma: no cover - declared project dependency
    Chem = None  # type: ignore[assignment]
    rdBase = None  # type: ignore[assignment]


EQUIVALENCE_FAMILIES = (
    "randomized_smiles",
    "alternative_periodic_cut",
    "atom_order_permutation",
)
APPLICABILITY_FAMILIES = ("valid", "h_capped", "marker_substituted")


def _unique_non_noop(values: Sequence[str], reference: str) -> list[str]:
    seen = {reference}
    output: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            output.append(text)
    return output


def generate_atom_order_permutations(
    text: str, *, count: int, seed: int
) -> list[str]:
    """Renumber RDKit atoms explicitly and serialize the graph non-canonically."""

    if Chem is None:
        raise RuntimeError("RDKit is required for atom-order permutations")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        raise StructureError("RDKit could not parse the structure", code="invalid_smiles")
    reference_identity = canonicalize_smiles(text)
    candidates: list[str] = []
    for view_index in range(count):
        order = list(range(molecule.GetNumAtoms()))
        random.Random(seed + 15485863 * view_index).shuffle(order)
        candidate = Chem.MolToSmiles(
            renumber_atoms(molecule, order), canonical=False, isomericSmiles=True
        )
        if canonicalize_smiles(candidate) != reference_identity:
            raise StructureError(
                "atom permutation changed molecular identity",
                code="atom_permutation_changed_identity",
            )
        candidates.append(candidate)
    return _unique_non_noop(candidates, text)


def _remove_markers(text: str) -> str | None:
    if Chem is None:
        raise RuntimeError("RDKit is required for semantic controls")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    editable = Chem.RWMol(molecule)
    marker_indices = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0
    ]
    for index in sorted(marker_indices, reverse=True):
        editable.RemoveAtom(index)
    candidate = editable.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(candidate)
    except (ValueError, RuntimeError):
        return None
    return Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)


def _substitute_markers(text: str) -> str | None:
    """Replace attachment markers by carbon atoms for a readable scope control."""

    if Chem is None:
        raise RuntimeError("RDKit is required for semantic controls")
    molecule = Chem.MolFromSmiles(text)
    if molecule is None:
        return None
    candidate = Chem.Mol(molecule)
    for atom in candidate.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(6)
            atom.SetIsotope(0)
            atom.SetAtomMapNum(0)
            atom.SetNoImplicit(False)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(candidate)
    except (ValueError, RuntimeError):
        return None
    return Chem.MolToSmiles(candidate, canonical=True, isomericSmiles=True)


def build_invariance_views(
    records: pd.DataFrame,
    *,
    sample_id_column: str = "sample_id",
    text_column: str = "canonical_psmiles",
    randomized_count: int = 10,
    periodic_count: int = 10,
    atom_permutation_count: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build verified equivalent views and separate applicability controls."""

    view_rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []
    scope_rows: list[dict[str, object]] = []
    for row_index, row in records.reset_index(drop=True).iterrows():
        sample_id = str(row[sample_id_column])
        text = str(row[text_column])
        assessment = assess_repeat_eligibility(text)
        scope_rows.append(
            {
                "sample_id": sample_id,
                "text": text,
                "repeat_eligible": bool(assessment.eligible),
                "eligibility_reason": assessment.reason,
            }
        )
        if not assessment.eligible:
            continue
        unit = parse_repeat_unit(text)
        view_seed = int(seed) + row_index * 104729
        randomized = _unique_non_noop(
            generate_randomized_smiles(
                text, count=randomized_count, seed=view_seed
            ),
            text,
        )
        periodic = _unique_non_noop(
            [
                view.psmiles
                for view in generate_equivalent_periodic_cuts(
                    unit, max_views=periodic_count, seed=view_seed
                )
                if view.transformation != "reference"
            ],
            text,
        )
        permutations = generate_atom_order_permutations(
            text, count=atom_permutation_count, seed=view_seed
        )
        for family, values in (
            ("randomized_smiles", randomized),
            ("alternative_periodic_cut", periodic),
            ("atom_order_permutation", permutations),
        ):
            for view_index, value in enumerate(values):
                view_rows.append(
                    {
                        "sample_id": sample_id,
                        "view_id": f"{sample_id}:{family}:{view_index}",
                        "view_family": family,
                        "view_index": view_index,
                        "reference_text": text,
                        "view_text": value,
                        "canonical_identity": unit.canonical_identity,
                        "analysis_role": "equivalence_consistency",
                    }
                )

        controls = {
            "valid": text,
            "h_capped": _remove_markers(text),
            "marker_substituted": _substitute_markers(text),
        }
        for family, value in controls.items():
            if value:
                semantic_rows.append(
                    {
                        "sample_id": sample_id,
                        "control_id": f"{sample_id}:{family}",
                        "control_family": family,
                        "reference_text": text,
                        "control_text": value,
                        "analysis_role": "applicability_only",
                    }
                )

    view_columns = [
        "sample_id",
        "view_id",
        "view_family",
        "view_index",
        "reference_text",
        "view_text",
        "canonical_identity",
        "analysis_role",
    ]
    semantic_columns = [
        "sample_id",
        "control_id",
        "control_family",
        "reference_text",
        "control_text",
        "analysis_role",
    ]
    return (
        pd.DataFrame(view_rows, columns=view_columns),
        pd.DataFrame(semantic_rows, columns=semantic_columns),
        pd.DataFrame(scope_rows),
    )


def score_view_consistency(
    view_rows: pd.DataFrame,
    reference_embeddings: Mapping[int, np.ndarray],
    view_embeddings: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-view cosine values and family/layer means."""

    rows: list[pd.DataFrame] = []
    for layer in reference_embeddings:
        block = view_rows.copy()
        block["layer"] = int(layer)
        block["reference_view_cosine"] = row_cosine_similarity(
            reference_embeddings[layer], view_embeddings[layer]
        )
        rows.append(block)
    details = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if details.empty:
        return details, pd.DataFrame(
            columns=["view_family", "layer", "n_views", "n_samples", "mean_cosine"]
        )
    summary = (
        details.groupby(["view_family", "layer"], as_index=False)
        .agg(
            n_views=("view_id", "size"),
            n_samples=("sample_id", "nunique"),
            mean_cosine=("reference_view_cosine", "mean"),
        )
        .sort_values(["view_family", "layer"])
        .reset_index(drop=True)
    )
    return details, summary


def score_applicability_controls(
    control_rows: pd.DataFrame,
    reference_embeddings: Mapping[int, np.ndarray],
    control_embeddings: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe semantic controls without treating them as equivalent views."""

    rows: list[pd.DataFrame] = []
    for layer in reference_embeddings:
        block = control_rows.copy()
        block["layer"] = int(layer)
        block["reference_control_cosine"] = row_cosine_similarity(
            reference_embeddings[layer], control_embeddings[layer]
        )
        rows.append(block)
    details = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if details.empty:
        return details, pd.DataFrame(
            columns=["control_family", "layer", "n_controls", "n_samples", "mean_cosine"]
        )
    summary = (
        details.groupby(["control_family", "layer"], as_index=False)
        .agg(
            n_controls=("control_id", "size"),
            n_samples=("sample_id", "nunique"),
            mean_cosine=("reference_control_cosine", "mean"),
        )
        .sort_values(["control_family", "layer"])
        .reset_index(drop=True)
    )
    return details, summary


def score_layer_drift(
    sample_ids: Sequence[str],
    current: Mapping[int, np.ndarray],
    reference: Mapping[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare aligned checkpoints at each requested hidden layer."""

    rows: list[pd.DataFrame] = []
    for layer in current:
        block = pd.DataFrame(
            {
                "sample_id": np.asarray(sample_ids, dtype=str),
                "layer": int(layer),
                "checkpoint_cosine": row_cosine_similarity(
                    reference[layer], current[layer]
                ),
            }
        )
        rows.append(block)
    details = pd.concat(rows, ignore_index=True)
    summary = (
        details.groupby("layer", as_index=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            mean_checkpoint_cosine=("checkpoint_cosine", "mean"),
        )
        .sort_values("layer")
        .reset_index(drop=True)
    )
    return details, summary


__all__ = [
    "APPLICABILITY_FAMILIES",
    "EQUIVALENCE_FAMILIES",
    "build_invariance_views",
    "generate_atom_order_permutations",
    "score_applicability_controls",
    "score_layer_drift",
    "score_view_consistency",
]
