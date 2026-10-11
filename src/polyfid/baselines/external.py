"""Same-split adapters for upstream polymer baselines.

The adapter boundary is deliberately small: Paper104 owns split membership and
row identifiers, while each upstream model owns its representation and
training code.  A model is eligible for the common result table only after it
returns one finite prediction for every held-out row of the selected fold.
"""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

from polyfid.records import PredictionRecord


class ExternalBaselineAdapter(Protocol):
    name: str

    def prepare(self, frame: pd.DataFrame, output_dir: str | Path) -> Sequence[Path]: ...

    def prediction_records(
        self, frame: pd.DataFrame, prediction_path: str | Path
    ) -> Sequence[PredictionRecord]: ...


@dataclass(frozen=True)
class ExternalBaselineSpec:
    name: str
    directory: str
    required_files: tuple[str, ...]
    required_modules: tuple[str, ...]
    evaluation_scope: str
    reporting_table: str
    runner_status: str


@dataclass(frozen=True)
class ExternalAssetStatus:
    name: str
    expected_path: str
    asset_present: bool
    required_files_present: bool
    runtime_dependencies_present: bool
    execution_supported: bool
    evaluation_scope: str
    reporting_table: str
    integration_status: str
    missing_runtime_modules: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


TRANSPOLYMER_STANDARD_TASKS = frozenset(
    {"egc", "egb", "eea", "ei", "xc", "eps", "nc"}
)
TRANSPOLYMER_SUPPLEMENTAL_VOCAB_TASKS = frozenset(
    {"opv", "pe-ii", "pe_i", "pe-i", "pe_i_structure", "pe_i_metadata"}
)
LINEAR_REPEAT_TASKS = frozenset({"egc", "egb", "eea", "ei", "xc", "eps", "nc"})


EXTERNAL_BASELINE_SPECS: tuple[ExternalBaselineSpec, ...] = (
    ExternalBaselineSpec(
        name="TransPolymer",
        directory="transpolymer",
        required_files=(
            "source/PolymerSmilesTokenization.py",
            "checkpoint/config.json",
            "checkpoint/pytorch_model.bin",
            "tokenizer/roberta-base/vocab.json",
            "tokenizer/roberta-base/merges.txt",
        ),
        required_modules=("torch", "transformers", "sklearn"),
        evaluation_scope=(
            "same-split external reference for Egc, Egb, Eea, Ei, Xc, EPS, and Nc; "
            "its upstream pretraining corpus differs from the controlled arms"
        ),
        reporting_table="same_split_external_reference",
        runner_status="validated thin runner",
    ),
    ExternalBaselineSpec(
        name="PerioGT",
        directory="periogt",
        required_files=(
            "source/PerioGT_common/scripts/prepare_ft_dataset.py",
            "source/PerioGT_common/models/light.py",
            "source/PerioGT_common/datasets/pretrain/scaler_all.pkl",
            "checkpoint/pretrained_ckpt/pretrained/light/base.pth",
        ),
        required_modules=("torch", "dgl", "dgllife", "mordred"),
        evaluation_scope="same-split linear dual-attachment repeat-unit tasks",
        reporting_table="same_split_external_reference",
        runner_status="validated thin runner",
    ),
    ExternalBaselineSpec(
        name="GRIN",
        directory="grin",
        required_files=(
            "source/torch_molecule/predictor/grin/modeling_grin.py",
            "source/torch_molecule/predictor/grin/model.py",
        ),
        required_modules=("torch", "torch_geometric", "CombineMols"),
        evaluation_scope="same-split linear dual-attachment repeat-unit tasks",
        reporting_table="same_split_external_reference",
        runner_status="validated thin runner using native PyTorch max reduction",
    ),
    ExternalBaselineSpec(
        name="wD-MPNN",
        directory="wdmpnn",
        required_files=(),
        required_modules=("torch", "torch_geometric"),
        evaluation_scope="Aldeghi EA/IP only, where weighted connection rules exist",
        reporting_table="same_split_external_reference",
        runner_status="no vetted upstream implementation in the shared asset root",
    ),
    ExternalBaselineSpec(
        name="PoCo",
        directory="poco",
        required_files=(),
        required_modules=("torch",),
        evaluation_scope="independent reference until a compatible official runner exists",
        reporting_table="literature_reference",
        runner_status="official compatible source or checkpoint not present",
    ),
)

EXPECTED_EXTERNAL_ASSETS = {
    spec.name: spec.directory for spec in EXTERNAL_BASELINE_SPECS
}


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def detect_external_assets(model_root: str | Path) -> tuple[ExternalAssetStatus, ...]:
    """Report scientific execution readiness, not just directory presence."""

    root = Path(model_root) / "external_baselines"
    statuses: list[ExternalAssetStatus] = []
    for spec in EXTERNAL_BASELINE_SPECS:
        expected = root / spec.directory
        present = expected.is_dir() and any(expected.iterdir())
        files_present = bool(spec.required_files) and all(
            (expected / relative).is_file() for relative in spec.required_files
        )
        missing_modules = tuple(
            module for module in spec.required_modules if not _module_present(module)
        )
        runtime_present = not missing_modules
        supported = (
            spec.name in {"TransPolymer", "PerioGT", "GRIN"}
            and files_present
            and runtime_present
        )
        if supported:
            status = "ready for the declared same-split tasks and formal A800 execution"
        elif not present:
            status = spec.runner_status
        elif not files_present:
            status = f"asset directory exists but required model files are incomplete; {spec.runner_status}"
        elif missing_modules:
            status = (
                "source/model assets are present, but the active Paper104 environment lacks "
                + ", ".join(missing_modules)
            )
        else:
            status = spec.runner_status
        statuses.append(
            ExternalAssetStatus(
                name=spec.name,
                expected_path=str(expected),
                asset_present=present,
                required_files_present=files_present,
                runtime_dependencies_present=runtime_present,
                execution_supported=supported,
                evaluation_scope=spec.evaluation_scope,
                reporting_table=spec.reporting_table,
                integration_status=status,
                missing_runtime_modules=missing_modules,
            )
        )
    return tuple(statuses)


def _sequence_value(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def graph_model_text(row: dict[str, Any]) -> str:
    """Select a structure-only PSMILES input for graph baselines."""

    for column in ("canonical_structure_sections", "component_canonical_psmiles"):
        values = _sequence_value(row.get(column))
        if values:
            return ".".join(values)
    return str(row.get("model_text", row.get("canonical_text", ""))).split("|", 1)[0]


def external_model_text(frame: pd.DataFrame, baseline: str, task: str) -> pd.Series:
    """Return the upstream model input without changing row membership."""

    key = baseline.casefold().replace("_", "-")
    task_key = task.casefold().replace("_", "-")
    if key == "transpolymer":
        if task_key in TRANSPOLYMER_SUPPLEMENTAL_VOCAB_TASKS:
            raise ValueError(
                "this TransPolymer task requires an upstream supplemental vocabulary "
                "that is not present as usable content in the shared assets"
            )
        if task_key not in TRANSPOLYMER_STANDARD_TASKS:
            raise ValueError(f"TransPolymer has no declared compatible adapter for task {task!r}")
        return frame["model_text"].astype(str)
    if key in {"periogt", "grin"}:
        if task_key not in LINEAR_REPEAT_TASKS:
            raise ValueError(
                f"{baseline} is limited to the declared linear repeat-unit task scope"
            )
        values = pd.Series(
            [graph_model_text(row) for row in frame.to_dict(orient="records")],
            index=frame.index,
        )
        if not values.str.count(r"\*").eq(2).all():
            raise ValueError(
                f"{baseline} periodic input needs exactly two attachment markers per row"
            )
        return values
    if key in {"wd-mpnn", "wdmpnn"}:
        if task.casefold() not in {"ea", "ip"} or "raw_text" not in frame:
            raise ValueError("wD-MPNN is only compatible with weighted Aldeghi EA/IP rows")
        values = frame["raw_text"].astype(str)
        if not values.str.contains("|", regex=False).all():
            raise ValueError("Aldeghi weighted connection metadata is unavailable")
        return values
    if key == "poco":
        raise ValueError("a compatible official PoCo implementation is not available locally")
    raise ValueError(f"unknown external baseline {baseline!r}")


def materialize_external_fold(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    baseline: str,
    task: str,
) -> tuple[Path, Path, Path]:
    """Write exact project train/validation/test rows for one upstream runner."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_text = external_model_text(frame, baseline, task)
    outputs: list[Path] = []
    for split in ("train", "validation", "test"):
        selected = frame.loc[frame["split"].eq(split)].copy()
        table = pd.DataFrame(
            {
                "sample_id": selected["sample_id"].astype(str),
                "smiles": model_text.loc[selected.index].astype(str),
                "value": selected["label"].astype(float),
                "split": split,
            }
        )
        output = destination / f"{split}.csv"
        table.to_csv(output, index=False)
        outputs.append(output)
    return tuple(outputs)  # type: ignore[return-value]


def read_external_predictions(
    frame: pd.DataFrame,
    prediction_path: str | Path,
    *,
    baseline: str,
    task: str,
    protocol: str,
    fold: int,
    seed: int,
    representation_path: str | None = None,
) -> list[PredictionRecord]:
    """Convert upstream test predictions to the common row-level schema."""

    source = Path(prediction_path)
    supplied = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    if not {"sample_id", "prediction"}.issubset(supplied.columns):
        raise ValueError("external predictions need sample_id and prediction columns")
    if "split" in supplied:
        supplied = supplied.loc[supplied["split"].astype(str).eq("test")].copy()
    if supplied["sample_id"].astype(str).duplicated().any():
        raise ValueError("external predictions contain duplicate test sample IDs")
    test = frame.loc[frame["split"].eq("test")].copy()
    values = supplied.set_index(supplied["sample_id"].astype(str))["prediction"]
    expected_ids = test["sample_id"].astype(str)
    missing = expected_ids.loc[~expected_ids.isin(values.index)]
    if not missing.empty:
        raise ValueError(f"external predictions are missing {len(missing)} held-out rows")
    predictions = values.loc[expected_ids].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise ValueError("external predictions contain non-finite values")

    records: list[PredictionRecord] = []
    for row, prediction in zip(test.to_dict(orient="records"), predictions, strict=True):
        eligibility = row.get("structural_fidelity_eligible")
        if eligibility is not None and pd.isna(eligibility):
            eligibility = None
        records.append(
            PredictionRecord(
                sample_id=str(row["sample_id"]),
                task=str(task),
                split="test",
                group_id=str(row["group_id"]),
                y_true=float(row["label"]),
                y_pred=float(prediction),
                method=baseline.casefold().replace("-", "_"),
                representation_seed=0,
                head_seed=int(seed),
                protocol=str(protocol),
                family_id=None if pd.isna(row.get("family_id")) else str(row.get("family_id")),
                cluster_id=None if pd.isna(row.get("cluster_id")) else str(row.get("cluster_id")),
                fold=str(fold),
                representation_path=representation_path,
                structural_fidelity_eligible=(
                    None if eligibility is None else bool(eligibility)
                ),
                scope_class=(
                    None if pd.isna(row.get("scope_class")) else str(row.get("scope_class"))
                ),
            )
        )
    return records


__all__ = [
    "EXPECTED_EXTERNAL_ASSETS",
    "EXTERNAL_BASELINE_SPECS",
    "ExternalAssetStatus",
    "ExternalBaselineAdapter",
    "detect_external_assets",
    "external_model_text",
    "graph_model_text",
    "materialize_external_fold",
    "read_external_predictions",
]
