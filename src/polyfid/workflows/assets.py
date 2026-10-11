"""Import already prepared Article B assets into shared project namespaces."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from polyfid.paths import ProjectPaths


@dataclass(frozen=True)
class AssetCopy:
    name: str
    source: Path
    destination: Path


def declared_assets(paths: ProjectPaths) -> tuple[AssetCopy, ...]:
    if paths.legacy_asset_root is None:
        raise ValueError(
            "legacy_asset_root must be declared in configs/paths.yaml for the "
            "one-time read-only asset import"
        )
    source_root = paths.legacy_asset_root
    data = paths.shared_data_root
    models = paths.shared_model_root
    return (
        AssetCopy("pi1m_raw", source_root / "data/raw/pi1m_upstream", data / "raw/pi1m"),
        AssetCopy("aldeghi_raw", source_root / "data/raw/aldeghi_upstream", data / "raw/aldeghi_vipea"),
        AssetCopy("transpolymer_raw", source_root / "data/raw/transpolymer", data / "raw/transpolymer/exports"),
        AssetCopy("transpolymer_upstream", source_root / "data/raw/transpolymer_upstream", data / "raw/transpolymer/upstream"),
        AssetCopy("openpoly_raw", source_root / "data/raw/openpoly", data / "raw/openpoly"),
        AssetCopy("molecular_retention_raw", source_root / "data/raw/moleculenet", data / "raw/molecular_retention"),
        AssetCopy("pi1m_standardized", source_root / "data/processed/pretraining/pi1m", data / "standardized/v1/pi1m"),
        AssetCopy("zinc20_standardized", source_root / "data/processed/pretraining/zinc20", data / "standardized/v1/zinc20"),
        AssetCopy("aldeghi_standardized", source_root / "data/processed/downstream/aldeghi", data / "standardized/v1/aldeghi_vipea"),
        AssetCopy("transpolymer_standardized", source_root / "data/processed/downstream/transpolymer", data / "standardized/v1/transpolymer"),
        AssetCopy("openpoly_standardized", source_root / "data/processed/external/openpoly", data / "standardized/v1/openpoly"),
        AssetCopy("molecular_retention_standardized", source_root / "data/processed/retention/moleculenet", data / "standardized/v1/molecular_retention"),
        AssetCopy("protected_repeat_groups", source_root / "data/processed/splits/protected_repeat_groups.txt", data / "standardized/v1/protected_repeat_groups.txt"),
        AssetCopy("downstream_metadata", source_root / "data/processed/downstream.metadata.json", data / "standardized/v1/downstream.metadata.json"),
        AssetCopy("chemberta_100m_mlm", source_root / "artifacts/models/chemberta_100m_mlm", models / "chemberta_100m_mlm"),
    )


def import_existing_assets(paths: ProjectPaths) -> list[dict[str, str]]:
    """Copy each reusable source once and leave existing destinations untouched."""

    outcomes: list[dict[str, str]] = []
    for asset in declared_assets(paths):
        if asset.destination.exists():
            outcomes.append({"name": asset.name, "status": "existing", "path": str(asset.destination)})
            continue
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        if asset.source.is_dir():
            shutil.copytree(asset.source, asset.destination)
        else:
            shutil.copy2(asset.source, asset.destination)
        outcomes.append({"name": asset.name, "status": "copied", "path": str(asset.destination)})
    for destination in (
        paths.checkpoint_root / "dev",
        paths.checkpoint_root / "pretraining",
        paths.checkpoint_root / "downstream",
        paths.checkpoint_root / "cross_backbone",
        paths.shared_model_root / "external_baselines",
    ):
        destination.mkdir(parents=True, exist_ok=True)
    return outcomes


__all__ = ["AssetCopy", "declared_assets", "import_existing_assets"]
