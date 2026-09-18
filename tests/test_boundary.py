from __future__ import annotations

from pathlib import Path

from preprocessing.boundary import _existing_boundary_path


def test_existing_boundary_path_none_when_unset(tmp_config):
    tmp_config.boundary_path = None
    assert _existing_boundary_path(tmp_config) is None


def test_existing_boundary_path_finds_relative_file(tmp_config, tmp_path: Path):
    shp = tmp_path / "unit_test_boundary.shp"
    shp.write_text("placeholder")
    tmp_config.boundary_path = Path("unit_test_boundary.shp")
    assert _existing_boundary_path(tmp_config) == shp


def test_existing_boundary_path_missing_file_returns_none(tmp_config):
    tmp_config.boundary_path = tmp_config.project_root / "missing" / "county.shp"
    assert _existing_boundary_path(tmp_config) is None
