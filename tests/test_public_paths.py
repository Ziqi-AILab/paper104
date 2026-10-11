from pathlib import Path

from polyfid.paths import ProjectPaths


def test_external_exclusion_root_preserves_other_project_paths(tmp_path: Path) -> None:
    roots = ProjectPaths(tmp_path, tmp_path / "data", tmp_path / "models", tmp_path / "weights",
                         exclusion_root=tmp_path / "frozen_lists")
    assert roots.exclusion_path("protected_repeat_groups.txt") == tmp_path / "frozen_lists/protected_repeat_groups.txt"
    assert roots.checkpoint_root == tmp_path / "weights"


def test_default_external_exclusion_location(tmp_path: Path) -> None:
    roots = ProjectPaths(tmp_path, tmp_path / "data", tmp_path / "models", tmp_path / "weights")
    assert roots.exclusion_path("protected_repeat_groups_periodic.txt") == tmp_path / "assets/exclusions/protected_repeat_groups_periodic.txt"
