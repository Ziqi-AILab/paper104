"""Chemprop 1.6.1 adapter using the project's declared split tables.

Chemprop 1.6.1 imports an RDKit helper which still looks for a pandas symbol
that moved in pandas 2.3.  The small compatibility shim below only restores
that import alias; model and data behavior remain Chemprop's official code.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
from contextlib import contextmanager
from functools import wraps
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from polyfid.baselines.classical import structure_smiles
from polyfid.baselines.restoration import (
    verify_prediction_roundtrip,
    write_retained_description,
)


def apply_pandas_rdkit_compat() -> bool:
    """Restore the pandas 2.3 symbol expected by RDKit PandasTools."""

    import pandas.io.formats.format as pandas_format
    import pandas.io.formats.printing as pandas_printing

    if hasattr(pandas_format, "get_adjustment"):
        return False
    pandas_format.get_adjustment = pandas_printing.get_adjustment
    return True


def apply_sklearn_metrics_compat() -> bool:
    """Bridge Chemprop 1.6.1 to scikit-learn versions without ``squared``.

    Chemprop keeps a module-local reference to ``mean_squared_error`` and uses
    the former ``squared=`` keyword for its RMSE metric.  Recent scikit-learn
    exposes ``root_mean_squared_error`` instead.  Replacing only Chemprop's
    local reference preserves its requested MSE/RMSE behavior.
    """

    from sklearn.metrics import mean_squared_error, root_mean_squared_error

    if "squared" in inspect.signature(mean_squared_error).parameters:
        return False

    from chemprop.train import metrics as chemprop_metrics

    def compatible_mean_squared_error(
        y_true: Any,
        y_pred: Any,
        *,
        sample_weight: Any = None,
        multioutput: Any = "uniform_average",
        squared: bool = True,
    ) -> Any:
        metric = mean_squared_error if squared else root_mean_squared_error
        return metric(
            y_true,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput,
        )

    chemprop_metrics.mean_squared_error = compatible_mean_squared_error
    return True


@contextmanager
def chemprop_torch_load_compat():
    """Use the PyTorch 2.5 loading default expected by Chemprop 1.6.1.

    Chemprop stores an ``argparse.Namespace`` beside its model weights and
    reloads that file without a ``weights_only`` argument.  PyTorch 2.6 changed
    the default, so keep Chemprop's existing checkpoint format working only
    while its training routine is active.
    """

    import torch

    original_load = torch.load

    @wraps(original_load)
    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


@dataclass(frozen=True)
class ChempropProbe:
    available: bool
    version: str | None
    compatible_version: bool
    detail: str


def probe_chemprop() -> ChempropProbe:
    try:
        version = importlib.metadata.version("chemprop")
    except importlib.metadata.PackageNotFoundError:
        return ChempropProbe(False, None, False, "chemprop is not installed")
    apply_pandas_rdkit_compat()
    try:
        import chemprop  # noqa: F401
    except Exception as exc:
        return ChempropProbe(False, version, version == "1.6.1", f"import failed: {exc}")
    return ChempropProbe(
        True,
        version,
        version == "1.6.1",
        "official Chemprop import is available",
    )


@dataclass(frozen=True)
class ChempropFiles:
    train: Path
    validation: Path
    test: Path
    train_features: Path | None = None
    validation_features: Path | None = None
    test_features: Path | None = None


def prepare_chemprop_files(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    additional_features: np.ndarray | None = None,
) -> ChempropFiles:
    """Write ordered Chemprop CSVs from an already group-safe evaluation frame."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    feature_paths: dict[str, Path | None] = {}
    if additional_features is not None:
        additional_features = np.asarray(additional_features, dtype=np.float32)
        if additional_features.ndim != 2 or len(additional_features) != len(frame):
            raise ValueError("Chemprop additional features must align with evaluation rows")
    smiles = structure_smiles(frame)
    for split in ("train", "validation", "test"):
        mask = frame["split"].eq(split).to_numpy()
        subset = frame.loc[mask, ["sample_id", "label"]].copy()
        if subset.empty:
            raise ValueError(f"Chemprop split {split!r} is empty")
        table = subset.rename(columns={"label": "target"})
        table.insert(1, "smiles", smiles[mask])
        path = destination / f"{split}.csv"
        table.to_csv(path, index=False)
        paths[split] = path
        if additional_features is None:
            feature_paths[split] = None
        else:
            feature_path = destination / f"{split}_features.npz"
            np.savez_compressed(feature_path, features=additional_features[mask])
            feature_paths[split] = feature_path
    return ChempropFiles(
        paths["train"],
        paths["validation"],
        paths["test"],
        feature_paths["train"],
        feature_paths["validation"],
        feature_paths["test"],
    )


def chemprop_train_arguments(
    files: ChempropFiles,
    *,
    save_dir: str | Path,
    seed: int,
    task_type: str = "regression",
    epochs: int = 30,
    batch_size: int = 50,
    num_workers: int = 8,
    cuda: bool = True,
) -> list[str]:
    if task_type not in {"regression", "binary"}:
        raise ValueError("task_type must be regression or binary")
    arguments = [
        "--data_path",
        str(files.train),
        "--separate_val_path",
        str(files.validation),
        "--separate_test_path",
        str(files.test),
        "--dataset_type",
        "classification" if task_type == "binary" else "regression",
        "--smiles_columns",
        "smiles",
        "--target_columns",
        "target",
        "--save_dir",
        str(save_dir),
        "--seed",
        str(seed),
        "--pytorch_seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--metric",
        "auc" if task_type == "binary" else "rmse",
        "--save_preds",
        "--quiet",
    ]
    if not cuda:
        arguments.append("--no_cuda")
    if files.train_features is not None:
        arguments.extend(
            [
                "--features_path",
                str(files.train_features),
                "--separate_val_features_path",
                str(files.validation_features),
                "--separate_test_features_path",
                str(files.test_features),
            ]
        )
    return arguments


def run_official_chemprop_train(
    arguments: Sequence[str], *, require_cuda: bool = True
) -> None:
    """Invoke the installed Chemprop 1.6.1 parser and training function."""

    probe = probe_chemprop()
    if not probe.available:
        raise RuntimeError(probe.detail)
    if not probe.compatible_version:
        raise RuntimeError(f"Paper104 expects Chemprop 1.6.1, found {probe.version}")
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("formal Chemprop training must run in an A800 GPU job")
    apply_pandas_rdkit_compat()
    apply_sklearn_metrics_compat()
    from chemprop.args import TrainArgs
    from chemprop.train import cross_validate, run_training

    args = TrainArgs().parse_args(list(arguments))
    with chemprop_torch_load_compat():
        cross_validate(args=args, train_func=run_training)


def locate_test_predictions(save_dir: str | Path) -> Path:
    root = Path(save_dir)
    candidates = (root / "test_preds.csv", root / "fold_0/test_preds.csv")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Chemprop did not produce test predictions under {root}")


def check_retained_chemprop(
    save_dir: str | Path,
    files: ChempropFiles,
    reference_predictions: np.ndarray,
    *,
    batch_size: int,
    num_workers: int,
) -> None:
    """Reload Chemprop's selected checkpoint, including its official scalers."""

    from chemprop.args import PredictArgs
    from chemprop.train import make_predictions
    from chemprop.utils import load_args, load_scalers

    root = Path(save_dir)
    checkpoint = root / "fold_0/model_0/model.pt"
    arguments = [
        "--test_path", str(files.test),
        "--preds_path", str(root / "reloaded_test_predictions.csv"),
        "--checkpoint_path", str(checkpoint),
        "--smiles_columns", "smiles",
        "--batch_size", str(batch_size),
        "--num_workers", str(num_workers),
    ]
    if files.test_features is not None:
        arguments.extend(["--features_path", str(files.test_features)])
    with chemprop_torch_load_compat():
        config = load_args(str(checkpoint))
        scalers = load_scalers(str(checkpoint))
        predicted = make_predictions(args=PredictArgs().parse_args(arguments))
    scaler = scalers[0]
    write_retained_description(
        root,
        config=config.as_dict(),
        target_scaler={
            "kind": "chemprop_embedded_standard_scaler",
            "mean": 0.0 if scaler is None else float(np.asarray(scaler.means).reshape(-1)[0]),
            "std": 1.0 if scaler is None else float(np.asarray(scaler.stds).reshape(-1)[0]),
            "source_checkpoint": str(checkpoint),
        },
    )
    verify_prediction_roundtrip(
        reference_predictions, predicted, root / "reload_check.json",
    )


def read_chemprop_predictions(path: str | Path, *, expected_rows: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if len(frame) != int(expected_rows):
        raise ValueError(
            f"Chemprop returned {len(frame)} rows for {expected_rows} held-out samples"
        )
    if "target" in frame:
        values = frame["target"]
    else:
        numeric = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if not str(column).endswith("_uncal_var")
        ]
        if len(numeric) != 1:
            raise ValueError(f"cannot identify Chemprop prediction column: {list(frame.columns)}")
        values = frame[numeric[0]]
    predictions = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("Chemprop prediction output contains non-finite values")
    return predictions


def write_chemprop_run_description(
    path: str | Path,
    *,
    probe: ChempropProbe,
    arguments: Sequence[str],
    files: ChempropFiles,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "chemprop": asdict(probe),
        "arguments": list(arguments),
        "train_path": str(files.train),
        "validation_path": str(files.validation),
        "test_path": str(files.test),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "ChempropFiles",
    "ChempropProbe",
    "apply_pandas_rdkit_compat",
    "apply_sklearn_metrics_compat",
    "chemprop_torch_load_compat",
    "chemprop_train_arguments",
    "locate_test_predictions",
    "prepare_chemprop_files",
    "probe_chemprop",
    "read_chemprop_predictions",
    "run_official_chemprop_train",
    "write_chemprop_run_description",
]
