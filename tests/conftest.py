"""Shared fixtures for pipeline unit tests (no Earth Engine)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import Point, box

from src.config import PipelineConfig

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


@pytest.fixture
def tmp_config(tmp_path: Path) -> PipelineConfig:
    """Config rooted in a temp directory so tests never write to ``data/``."""
    cfg = PipelineConfig(
        area_name="UnitTest",
        project_root=tmp_path,
        reuse_ee_features=False,
        cv_folds=2,
        test_size=0.25,
        random_state=0,
        rf_param_grid={
            "rf__n_estimators": [10],
            "rf__criterion": ["gini"],
            "rf__max_depth": [3],
            "rf__min_samples_split": [2],
            "rf__random_state": [0],
        },
        feature_columns=["precipitation", "elevation", "slope"],
    )
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def binary_points_gdf():
    if gpd is None:
        pytest.skip("geopandas is required")
    records = [
        {"fid": 1, "crop_a": "Maize", "crop_b": None},
        {"fid": 2, "crop_a": "maize", "crop_b": "Beans"},
        {"fid": 3, "crop_a": "Wheat", "crop_b": None},
        {"fid": 4, "crop_a": "Potatoes", "crop_b": None},
    ]
    geom = [Point(36.4 + i * 0.01, -0.2) for i in range(len(records))]
    return gpd.GeoDataFrame(records, geometry=geom, crs="EPSG:4326")


@pytest.fixture
def multiclass_points_gdf():
    if gpd is None:
        pytest.skip("geopandas is required")
    crops = (
        ["Maize"] * 12
        + ["Wheat"] * 12
        + ["Potatoes"] * 12
        + ["Grass"] * 11
        + ["Canola"] * 11
        + ["Coffee"] * 11
        + ["Ovacodo"] * 3
        + ["Beans"] * 2
        + ["Whear"]
        + ["Potato"]
        + ["Grasa"]
    )
    records = [{"fid": i + 1, "crop_a": crop} for i, crop in enumerate(crops)]
    geom = [Point(36.0 + (i % 20) * 0.01, -0.5 + (i // 20) * 0.01) for i in range(len(records))]
    return gpd.GeoDataFrame(records, geometry=geom, crs="EPSG:4326")


@pytest.fixture
def polygon_gdf():
    if gpd is None:
        pytest.skip("geopandas is required")
    poly = box(36.40, -0.30, 36.401, -0.299)
    return gpd.GeoDataFrame(
        {"fid": [1.0], "crop_a": ["Maize"], "crop_b": [None]},
        geometry=[poly],
        crs="EPSG:4326",
    )


def write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload))
    return path


def write_geotiff(path: Path, arr: np.ndarray, names=None, tags=None, dtype=None):
    import rasterio
    from rasterio.transform import from_origin

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    profile = {
        "driver": "GTiff",
        "height": int(arr.shape[1]),
        "width": int(arr.shape[2]),
        "count": int(arr.shape[0]),
        "dtype": dtype or arr.dtype,
        "crs": "EPSG:4326",
        "transform": from_origin(36.0, 0.0, 0.001, 0.001),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(profile["dtype"]))
        if names:
            for i, name in enumerate(names, start=1):
                if i <= dst.count:
                    dst.set_band_description(i, str(name))
        if tags:
            dst.update_tags(**tags)
    return path
