from __future__ import annotations

import pytest
from shapely.geometry import Point

from preprocessing.survey import (
    add_binary_label,
    add_multiclass_label,
    add_plot_area,
    add_target_label,
    class_legend,
    is_point_survey,
    normalize_crop_name,
    other_class_breakdown,
    prepare_survey,
    sample_geometry_mode,
)
from src.config import PipelineConfig

gpd = pytest.importorskip("geopandas")


def test_normalize_crop_name_aliases_and_empty():
    aliases = {"whear": "wheat", "ovacodo": "avocado", "potato": "potatoes"}
    assert normalize_crop_name("Whear", aliases) == "wheat"
    assert normalize_crop_name("Ovacodo", aliases) == "avocado"
    assert normalize_crop_name("Potato", aliases) == "potatoes"
    assert normalize_crop_name("Maize", aliases) == "maize"
    assert normalize_crop_name(None, aliases) is None
    assert normalize_crop_name("nan", aliases) is None
    assert normalize_crop_name("  ", aliases) is None


def test_binary_label_is_case_insensitive(binary_points_gdf, tmp_config):
    tmp_config.label_mode = "binary"
    tmp_config.target_column = "maize_pos"
    labelled = add_binary_label(binary_points_gdf, tmp_config)
    assert labelled["maize_pos"].tolist() == [1, 1, 0, 0]


def test_binary_label_uses_any_crop_column(binary_points_gdf, tmp_config):
    tmp_config.target_crop_names = ["Beans"]
    labelled = add_binary_label(binary_points_gdf, tmp_config)
    assert labelled["maize_pos"].tolist() == [0, 1, 0, 0]


def test_binary_label_missing_columns_raises(tmp_config):
    gdf = gpd.GeoDataFrame({"fid": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    with pytest.raises(KeyError, match="crop columns"):
        add_binary_label(gdf, tmp_config)


def test_multiclass_collapses_rare_classes_and_aliases(multiclass_points_gdf, tmp_config):
    tmp_config.label_mode = "multiclass"
    tmp_config.target_column = "crop_class_code"
    tmp_config.min_class_count = 10
    labelled = add_multiclass_label(multiclass_points_gdf, tmp_config)
    counts = labelled["crop_class"].value_counts().to_dict()
    assert counts["maize"] == 12
    assert counts["wheat"] == 13  # Wheat + Whear
    assert counts["potatoes"] == 13  # Potatoes + Potato
    assert counts["grass"] == 12  # Grass + Grasa
    assert "avocado" not in set(labelled["crop_class"])
    assert counts["other"] == 5  # 3 avocado + 2 beans
    codes, names = class_legend(labelled, tmp_config)
    assert names[0] == "maize"
    assert names[-1] == "other"
    assert codes == list(range(1, len(names) + 1))
    breakdown = other_class_breakdown(labelled, tmp_config)
    assert breakdown["avocado"] == 3
    assert breakdown["beans"] == 2


def test_multiclass_missing_source_column_raises(tmp_config):
    tmp_config.label_mode = "multiclass"
    gdf = gpd.GeoDataFrame({"fid": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326")
    with pytest.raises(KeyError, match="crop_a"):
        add_multiclass_label(gdf, tmp_config)


def test_add_target_label_dispatches_on_label_mode(binary_points_gdf, tmp_config):
    tmp_config.label_mode = "binary"
    out = add_target_label(binary_points_gdf, tmp_config)
    assert "maize_pos" in out.columns
    tmp_config.label_mode = "multiclass"
    tmp_config.target_column = "crop_class_code"
    tmp_config.min_class_count = 1
    out = add_target_label(binary_points_gdf, tmp_config)
    assert "crop_class" in out.columns


def test_point_vs_polygon_geometry(binary_points_gdf, polygon_gdf, tmp_config):
    assert is_point_survey(binary_points_gdf) is True
    assert is_point_survey(polygon_gdf) is False
    tmp_config.sample_geometry = "auto"
    assert sample_geometry_mode(binary_points_gdf, tmp_config) == "point"
    assert sample_geometry_mode(polygon_gdf, tmp_config) == "polygon"
    tmp_config.sample_geometry = "polygon"
    assert sample_geometry_mode(binary_points_gdf, tmp_config) == "polygon"


def test_plot_area_skipped_for_points_computed_for_polygons(binary_points_gdf, polygon_gdf, tmp_config):
    points = add_plot_area(binary_points_gdf, tmp_config)
    assert "plot_area_acres" not in points.columns
    polys = add_plot_area(polygon_gdf, tmp_config)
    assert (polys["plot_area_acres"] > 0).all()


def test_prepare_survey_writes_geojson_and_float_fid(tmp_path, binary_points_gdf, tmp_config):
    path = tmp_path / "survey.geojson"
    binary_points_gdf.to_file(path, driver="GeoJSON")
    tmp_config.survey_geojson = path
    tmp_config.label_mode = "binary"
    prepared = prepare_survey(tmp_config)
    assert prepared["fid"].dtype == "float64"
    assert "maize_pos" in prepared.columns


def test_prepare_survey_requires_path(tmp_config):
    tmp_config.survey_geojson = None
    with pytest.raises(ValueError, match="survey file"):
        prepare_survey(tmp_config)
