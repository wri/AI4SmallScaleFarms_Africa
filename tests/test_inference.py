from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier

from models.inference import (
    _bbox_intersects,
    _estimate_geotiff_bytes,
    _expected_feature_names,
    _http_error_text,
    _is_rate_limit_error,
    _tile_bboxes,
    _tile_stem,
    _validate_feature_raster,
    predict_probability_raster,
)
from tests.conftest import write_geotiff


class _FakeSrc:
    def __init__(self, count, descriptions=None, n_features=None):
        self.count = count
        self.descriptions = descriptions


class _FakeModel:
    def __init__(self, names, n_features=None):
        self.feature_names_in_ = np.array(names)
        self.n_features_in_ = n_features if n_features is not None else len(names)


def test_tile_bboxes_covers_extent_without_gaps():
    tiles = _tile_bboxes(36.0, -0.2, 36.16, -0.04, 0.08)
    west, south, east, north = tiles[0]
    assert west == pytest.approx(36.0)
    assert south == pytest.approx(-0.2)
    assert east == pytest.approx(36.08)
    assert north == pytest.approx(-0.12)
    west = min(t[0] for t in tiles)
    east = max(t[2] for t in tiles)
    south = min(t[1] for t in tiles)
    north = max(t[3] for t in tiles)
    assert west == pytest.approx(36.0)
    assert east == pytest.approx(36.16)
    assert south == pytest.approx(-0.2)
    assert north == pytest.approx(-0.04)


def test_tile_bboxes_single_cell_when_extent_tiny():
    assert _tile_bboxes(1.0, 2.0, 1.0, 2.0, 0.08) == [(1.0, 2.0, 1.0, 2.0)]


def test_estimate_geotiff_bytes_positive():
    n = _estimate_geotiff_bytes(36.0, -0.1, 36.1, 0.0, 30, 26)
    assert n > 0


def test_tile_stem_replaces_minus_sign():
    assert _tile_stem(36.1, -0.2, 36.2, -0.1).startswith("tile_36.1000_m0.2000_")


def test_bbox_intersects_polygon():
    aoi = box(36.0, -0.2, 36.2, 0.0)
    assert _bbox_intersects(aoi, 36.05, -0.15, 36.10, -0.10) is True
    assert _bbox_intersects(aoi, 37.0, 1.0, 37.1, 1.1) is False


def test_rate_limit_and_http_error_text():
    assert _is_rate_limit_error(RuntimeError("HTTP 429"))
    assert not _is_rate_limit_error(RuntimeError("not found"))
    assert "boom" in _http_error_text(RuntimeError("boom"))


def test_validate_feature_raster_band_count_and_names(tmp_config):
    model = _FakeModel(["precipitation", "elevation", "slope"])
    _validate_feature_raster(_FakeSrc(3, ["precipitation", "elevation", "slope"]), model, tmp_config)
    with pytest.raises(ValueError, match="expects 3"):
        _validate_feature_raster(_FakeSrc(2, ["precipitation", "elevation"]), model, tmp_config)
    with pytest.raises(ValueError, match="band names"):
        _validate_feature_raster(
            _FakeSrc(3, ["slope", "elevation", "precipitation"]), model, tmp_config
        )


def test_expected_feature_names_fallback(tmp_config):
    class Bare:
        pass

    tmp_config.feature_columns = ["a", "b"]
    assert _expected_feature_names(Bare(), tmp_config) == ["a", "b"]


def test_predict_probability_raster_binary_and_multiclass(tmp_config):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "precipitation": rng.normal(10, 1, 40),
        "elevation": rng.normal(1800, 20, 40),
        "slope": rng.normal(5, 1, 40),
    })
    y_bin = pd.Series([0] * 20 + [1] * 20)
    model = RandomForestClassifier(n_estimators=8, random_state=0)
    model.fit(X, y_bin)

    arr = np.stack(
        [
            np.full((4, 4), 10.0, dtype="float32"),
            np.full((4, 4), 1800.0, dtype="float32"),
            np.full((4, 4), 5.0, dtype="float32"),
        ]
    )
    arr[0, 0, 0] = np.nan
    feature_tif = write_geotiff(
        tmp_config.feature_image_path, arr, names=list(X.columns)
    )

    tmp_config.label_mode = "binary"
    prob_path = predict_probability_raster(model, tmp_config, feature_tif)
    import rasterio

    with rasterio.open(prob_path) as src:
        assert src.count == 1
        data = src.read(1)
        assert np.isnan(data[0, 0])
        assert np.isfinite(data[1, 1])
        assert 0.0 <= float(np.nanmax(data)) <= 1.0

    y_multi = pd.Series(([1, 2, 3] * 14)[:40])
    multi = RandomForestClassifier(n_estimators=8, random_state=0)
    multi.fit(X, y_multi)
    multi.class_names_ = ["maize", "wheat", "potatoes"]
    tmp_config.label_mode = "multiclass"
    multi_path = predict_probability_raster(
        multi, tmp_config, feature_tif, out_path=tmp_config.outputs_dir / "multi_prob.tif"
    )
    with rasterio.open(multi_path) as src:
        assert src.count == 3
        assert list(src.descriptions) == ["maize", "wheat", "potatoes"]
        tags = src.tags()
        assert tags["label_mode"] == "multiclass"
        assert tags["class_codes"] == "1,2,3"
        stack = src.read()
        finite = np.isfinite(stack[:, 1, 1])
        assert finite.all()
        np.testing.assert_allclose(stack[:, 1, 1].sum(), 1.0, atol=1e-5)
