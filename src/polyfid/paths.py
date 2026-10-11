"""Central project and shared-asset paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    shared_data_root: Path
    shared_model_root: Path
    checkpoint_root: Path
    legacy_asset_root: Path | None = None
    formal_openpoly: bool = False
    exclusion_root: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProjectPaths":
        source = Path(path)
        values: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
        legacy_root = values.get("legacy_asset_root")
        formal_retained = values.get("formal_retained", {})
        return cls(
            project_root=Path(values["project_root"]).expanduser().resolve(),
            shared_data_root=Path(values["shared_data_root"]).expanduser().resolve(),
            shared_model_root=Path(values["shared_model_root"]).expanduser().resolve(),
            checkpoint_root=Path(values["checkpoint_root"]).expanduser().resolve(),
            legacy_asset_root=(
                Path(legacy_root).expanduser().resolve()
                if legacy_root is not None
                else None
            ),
            formal_openpoly=bool(formal_retained.get("openpoly", False)),
            exclusion_root=(
                Path(values["exclusion_root"]).expanduser().resolve()
                if values.get("exclusion_root") else None
            ),
        )

    def project_path(self, *parts: str) -> Path:
        return self.project_root.joinpath(*parts)

    def exclusion_path(self, name: str) -> Path:
        """Locate the externally supplied frozen scientific exclusion lists."""
        root = self.exclusion_root or self.project_root / "assets/exclusions"
        return root / name

    @property
    def dataset_root(self) -> Path:
        return self.shared_data_root

    @property
    def model_root(self) -> Path:
        return self.shared_model_root


DEFAULT_PATH_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "paths.yaml"


def load_paths(path: str | Path = DEFAULT_PATH_CONFIG) -> ProjectPaths:
    return ProjectPaths.from_yaml(path)


__all__ = ["DEFAULT_PATH_CONFIG", "ProjectPaths", "load_paths"]
