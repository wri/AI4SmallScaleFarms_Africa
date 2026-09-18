from __future__ import annotations

import numpy as np

from models.postprocess import (
    _classified_from_probability_stack,
    _non_cropland_mask,
    apply_cropland_mask,
    threshold_probability,
)
from tests.conftest import write_geotiff


def test_non_cropland_mask_binary_classes_and_percent(tmp_config):
    binary = np.array([[1, 0], [1, 0]], dtype="float32")
    assert _non_cropland_mask(binary, tmp_config).tolist() == [[False, True], [False, True]]

    gfsad = np.array([[2, 1], [6, 0]], dtype="float32")
    mask = _non_cropland_mask(gfsad, tmp_config)
    assert mask.tolist() == [[False, True], [False, True]]

    tmp_config.cropland_classes = []
    tmp_config.cropland_threshold = 40
    percent = np.array([[50, 10], [40, 39]], dtype="float32")
    mask = _non_cropland_mask(percent, tmp_config)
    assert mask.tolist() == [[False, True], [False, True]]

    empty = np.array([[np.nan, np.nan]])
    assert _non_cropland_mask(empty, tmp_config).all()


def test_classified_from_probability_stack_argmax_and_nodata():
    probs = np.zeros((3, 2, 2), dtype="float32")
    probs[0, 0, 0] = 0.7
    probs[1, 0, 0] = 0.2
    probs[2, 0, 0] = 0.1
    probs[2, 0, 1] = 0.9
    probs[:, 1, 1] = np.nan
    classified = _classified_from_probability_stack(probs, [1, 2, 3])
    assert classified[0, 0] == 1
    assert classified[0, 1] == 3
    assert classified[1, 1] == 0


def test_threshold_probability_binary(tmp_config):
    probs = np.array([[[0.2, 0.8], [np.nan, 0.5]]], dtype="float32")
    path = write_geotiff(tmp_config.probability_map_path, probs, names=["P(Maize)"])
    tmp_config.label_mode = "binary"
    tmp_config.probability_threshold = 0.5
    out = threshold_probability(tmp_config, path)
    import rasterio

    with rasterio.open(out) as src:
        arr = src.read(1)
    assert arr.tolist() == [[0, 1], [0, 1]]


def test_threshold_probability_multiclass(tmp_config):
    stack = np.zeros((3, 2, 2), dtype="float32")
    stack[0, 0, 0] = 0.9
    stack[1, 0, 1] = 0.8
    stack[2, 1, 0] = 0.7
    path = write_geotiff(
        tmp_config.probability_map_path,
        stack,
        names=["maize", "wheat", "other"],
        tags={"class_codes": "1,2,3", "class_names": "maize,wheat,other"},
    )
    tmp_config.label_mode = "multiclass"
    out = threshold_probability(tmp_config, path)
    import rasterio

    with rasterio.open(out) as src:
        arr = src.read(1)
        assert src.tags()["class_names"] == "maize,wheat,other"
    assert arr[0, 0] == 1
    assert arr[0, 1] == 2
    assert arr[1, 0] == 3


def test_apply_cropland_mask_binary_and_multiclass(tmp_config):
    import rasterio

    tmp_config.label_mode = "binary"
    probs = np.array([[[0.9, 0.9], [0.1, 0.9]]], dtype="float32")
    crop = np.array([[1, 0], [1, 1]], dtype="uint8")
    p_path = write_geotiff(tmp_config.probability_map_path, probs)
    c_path = write_geotiff(tmp_config.cropland_map_path, crop[np.newaxis, ...], dtype="uint8")
    out = apply_cropland_mask(tmp_config, p_path, c_path)
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert arr.tolist() == [[1, 0], [0, 1]]

    tmp_config.label_mode = "multiclass"
    stack = np.zeros((2, 2, 2), dtype="float32")
    stack[0] = 0.9
    stack[1] = 0.1
    p_path = write_geotiff(
        tmp_config.outputs_dir / "mc.tif",
        stack,
        names=["maize", "wheat"],
        tags={"class_codes": "1,2", "class_names": "maize,wheat"},
    )
    out = apply_cropland_mask(tmp_config, p_path, c_path, tmp_config.outputs_dir / "mc_class.tif")
    with rasterio.open(out) as src:
        arr = src.read(1)
    assert arr[0, 0] == 1
    assert arr[0, 1] == 0
