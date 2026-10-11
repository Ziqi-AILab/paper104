"""Same-split ECFP/RDKit, tree, and nearest-neighbour references."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def normalize_structure_text(text: str) -> str:
    """Convert Paper104's component text to one RDKit-readable SMILES.

    A controlled structure view can contain multiple ``|``-separated
    components.  The metadata-bearing PE-I view additionally appends fields
    after ``$``.  Molecular baselines consume the same chemical components as
    the neural models; metadata is encoded separately below.
    """

    value = str(text).strip()
    if "|" not in value and "$" not in value:
        return value
    components: list[str] = []
    for section in value.split("|"):
        structure = section.split("$", 1)[0].strip()
        if structure and Chem.MolFromSmiles(structure) is not None:
            components.append(structure)
    if not components:
        raise ValueError("structure text does not contain an RDKit-readable component")
    return ".".join(components)


def structure_smiles(frame: Any) -> np.ndarray:
    """Return one disconnected-component SMILES per evaluation row."""

    output: list[str] = []
    has_sections = "canonical_structure_sections" in frame
    for row in frame.to_dict(orient="records"):
        sections = row.get("canonical_structure_sections") if has_sections else None
        if sections is not None and not isinstance(sections, str):
            candidates = [str(value) for value in sections if str(value).strip()]
            if candidates:
                output.append(".".join(candidates))
                continue
        output.append(normalize_structure_text(str(row["model_text"])))
    return np.asarray(output, dtype=object)


def _metadata_sections(value: Any) -> tuple[tuple[str, ...], ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(tuple(str(token) for token in section) for section in value)


@dataclass(frozen=True)
class MetadataFeatureEncoder:
    """Explicit train-only positional encoding for the PE-I metadata fields."""

    positions: tuple[tuple[int, int], ...]
    categories: tuple[tuple[str, ...], ...]

    @classmethod
    def fit(cls, values: Sequence[Any]) -> "MetadataFeatureEncoder":
        observed: dict[tuple[int, int], set[str]] = {}
        for value in values:
            for section_index, section in enumerate(_metadata_sections(value)):
                for field_index, token in enumerate(section):
                    try:
                        float(token)
                    except ValueError:
                        observed.setdefault((section_index, field_index), set()).add(token)
                    else:
                        observed.setdefault((section_index, field_index), set())
        positions = tuple(sorted(observed))
        return cls(
            positions=positions,
            categories=tuple(tuple(sorted(observed[position])) for position in positions),
        )

    @property
    def feature_count(self) -> int:
        # Each position has a numeric value and a numeric-present indicator,
        # followed by its observed train categories.
        return sum(2 + len(values) for values in self.categories)

    def transform(self, values: Sequence[Any]) -> np.ndarray:
        offsets: list[int] = []
        offset = 0
        category_maps: list[dict[str, int]] = []
        for categories in self.categories:
            offsets.append(offset)
            category_maps.append({value: index for index, value in enumerate(categories)})
            offset += 2 + len(categories)
        output = np.zeros((len(values), offset), dtype=np.float32)
        position_index = {position: index for index, position in enumerate(self.positions)}
        for row_index, value in enumerate(values):
            for section_index, section in enumerate(_metadata_sections(value)):
                for field_index, token in enumerate(section):
                    local_index = position_index.get((section_index, field_index))
                    if local_index is None:
                        continue
                    base = offsets[local_index]
                    try:
                        output[row_index, base] = float(token)
                        output[row_index, base + 1] = 1.0
                    except ValueError:
                        category_index = category_maps[local_index].get(token)
                        if category_index is not None:
                            output[row_index, base + 2 + category_index] = 1.0
        return output


def metadata_features(
    train_values: Sequence[Any],
    *other_values: Sequence[Any],
) -> tuple[MetadataFeatureEncoder, tuple[np.ndarray, ...]]:
    """Fit metadata columns on training rows and transform all requested roles."""

    encoder = MetadataFeatureEncoder.fit(train_values)
    arrays = (encoder.transform(train_values),) + tuple(
        encoder.transform(values) for values in other_values
    )
    return encoder, arrays


def ecfp_features(
    smiles: Sequence[str],
    *,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, includeChirality=True)
    output = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(normalize_structure_text(str(text)))
        if molecule is None:
            raise ValueError(f"invalid structure at row {row}")
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), output[row])
    return output


RDKIT_2D_DESCRIPTORS = tuple(
    (name, function)
    for name, function in Descriptors._descList
    if name != "Ipc"
)


def rdkit_2d_features(smiles: Sequence[str]) -> np.ndarray:
    """Compute the standard finite RDKit 2D descriptor vector.

    ``Ipc`` is omitted because its magnitude can dominate the numerical scale
    for larger repeat units.  Remaining non-finite descriptor values are
    replaced after calculation and the downstream estimator standardizes the
    continuous columns using training data only.
    """

    output = np.empty((len(smiles), len(RDKIT_2D_DESCRIPTORS)), dtype=np.float32)
    for row, text in enumerate(smiles):
        molecule = Chem.MolFromSmiles(normalize_structure_text(str(text)))
        if molecule is None:
            raise ValueError(f"invalid structure at row {row}")
        values: list[float] = []
        for _, function in RDKIT_2D_DESCRIPTORS:
            try:
                values.append(float(function(molecule)))
            except (RuntimeError, ValueError, ZeroDivisionError):
                # A small subset of charge-dependent RDKit descriptors does
                # not define values for the ``*`` attachment atom in PSMILES.
                # Keep the remaining valid 2D descriptors and map only those
                # observed undefined entries to zero below.
                values.append(float("nan"))
        output[row] = values
    return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)


def train_classical(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    estimator: str = "ridge",
    seed: int = 0,
    task_type: str = "regression",
) -> np.ndarray:
    if task_type not in {"regression", "binary"}:
        raise ValueError("task_type must be regression or binary")
    if estimator == "ridge":
        if task_type == "regression":
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
            )
    elif estimator == "xgboost":
        if task_type == "regression":
            from xgboost import XGBRegressor

            model = XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=8,
            )
        else:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=8,
                eval_metric="logloss",
            )
    else:
        raise ValueError("estimator must be ridge or xgboost")
    model.fit(train_x, train_y)
    if task_type == "binary":
        return np.asarray(model.predict_proba(test_x)[:, 1], dtype=float)
    return np.asarray(model.predict(test_x), dtype=float)


def nearest_neighbor_predictions(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    metric: str = "tanimoto",
    block_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict from the closest training row without materializing a huge matrix.

    Tanimoto is used for the structure-only ECFP view.  PE-I's secondary
    structure-plus-metadata reference uses Euclidean distance after scaling
    every combined feature on training rows only.
    """

    test = np.asarray(test_features, dtype=np.float32)
    train = np.asarray(train_features, dtype=np.float32)
    if metric == "euclidean":
        center = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale < 1e-8] = 1.0
        train = (train - center) / scale
        test = (test - center) / scale
    elif metric != "tanimoto":
        raise ValueError("nearest-neighbour metric must be tanimoto or euclidean")

    best_indices = np.empty(len(test), dtype=np.int64)
    best_scores = np.empty(len(test), dtype=np.float32)
    for start in range(0, len(test), block_size):
        block = test[start : start + block_size]
        if metric == "tanimoto":
            intersections = block @ train.T
            unions = (
                block.sum(axis=1, keepdims=True)
                + train.sum(axis=1)[None, :]
                - intersections
            )
            scores = intersections / np.maximum(unions, 1.0)
            local_indices = np.argmax(scores, axis=1)
            local_scores = scores[np.arange(len(block)), local_indices]
        else:
            distances = (
                np.sum(np.square(block), axis=1, keepdims=True)
                + np.sum(np.square(train), axis=1)[None, :]
                - 2.0 * (block @ train.T)
            )
            distances = np.maximum(distances, 0.0)
            local_indices = np.argmin(distances, axis=1)
            local_scores = -np.sqrt(distances[np.arange(len(block)), local_indices])
        best_indices[start : start + len(block)] = local_indices
        best_scores[start : start + len(block)] = local_scores
    return np.asarray(train_targets)[best_indices], best_scores


__all__ = [
    "RDKIT_2D_DESCRIPTORS",
    "MetadataFeatureEncoder",
    "ecfp_features",
    "metadata_features",
    "nearest_neighbor_predictions",
    "normalize_structure_text",
    "rdkit_2d_features",
    "structure_smiles",
    "train_classical",
]
