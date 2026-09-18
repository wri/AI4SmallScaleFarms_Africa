from __future__ import annotations

from pathlib import Path

from src.config import PipelineConfig
from tests.conftest import write_yaml


def test_slug_and_derived_paths(tmp_path: Path):
    cfg = PipelineConfig(area_name="Nakuru County", project_root=tmp_path)
    assert cfg.slug == "nakuru_county"
    assert cfg.model_path == tmp_path / "data" / "models" / "nakuru_county_rf_best_model.pkl"
    assert cfg.merged_features_path.name == "nakuru_county_merged_features.csv"
    assert cfg.feature_asset_id.endswith("nakuru_county_pixel_features_for_rf")


def test_gaul_lookup_prefers_explicit_name():
    cfg = PipelineConfig(area_name="Nyandarua_smoke", gaul_name="Nyandarua")
    assert cfg.gaul_lookup_name == "Nyandarua"


def test_is_binary_default_and_multiclass():
    assert PipelineConfig().is_binary is True
    assert PipelineConfig(label_mode="binary").is_binary is True
    assert PipelineConfig(label_mode="MULTICLASS").is_binary is False


def test_reference_date_falls_back_to_start():
    cfg = PipelineConfig(start_date="2024-03-01", refdate=None)
    assert cfg.reference_date == "2024-03-01"
    cfg.refdate = "2024-04-01"
    assert cfg.reference_date == "2024-04-01"


def test_from_dict_ignores_unknown_keys_and_coerces_paths():
    cfg = PipelineConfig.from_dict(
        {
            "area_name": "Kitui",
            "survey_geojson": "data/raw/kitui.geojson",
            "not_a_field": 123,
        }
    )
    assert cfg.area_name == "Kitui"
    assert isinstance(cfg.survey_geojson, Path)
    assert not hasattr(cfg, "not_a_field")


def test_from_yaml_and_to_dict(tmp_path: Path):
    path = write_yaml(
        tmp_path / "cfg.yaml",
        {"area_name": "Nakuru", "label_mode": "multiclass", "min_class_count": 10},
    )
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.area_name == "Nakuru"
    assert cfg.is_binary is False
    dumped = cfg.to_dict()
    assert dumped["area_name"] == "Nakuru"
    assert isinstance(dumped["project_root"], str)


def test_resolve_path_relative_and_absolute(tmp_path: Path):
    cfg = PipelineConfig(project_root=tmp_path)
    assert cfg.resolve_path(None) is None
    rel = cfg.resolve_path("data/raw/x.geojson")
    assert rel == tmp_path / "data" / "raw" / "x.geojson"
    abs_path = tmp_path / "abs.geojson"
    assert cfg.resolve_path(abs_path) == abs_path


def test_ensure_dirs_creates_data_tree(tmp_path: Path):
    cfg = PipelineConfig(project_root=tmp_path)
    cfg.ensure_dirs()
    for d in (cfg.raw_dir, cfg.interim_dir, cfg.processed_dir, cfg.outputs_dir, cfg.models_dir):
        assert d.is_dir()


def test_custom_feature_asset_id():
    cfg = PipelineConfig(ee_feature_asset="projects/x/assets/custom")
    assert cfg.feature_asset_id == "projects/x/assets/custom"
