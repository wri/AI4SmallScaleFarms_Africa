from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from preprocessing import features as feat


def test_rate_and_compute_limit_detection():
    assert feat._is_rate_limit(RuntimeError("HTTP 429 rate limit"))
    assert feat._is_rate_limit(RuntimeError("User request limit exceeded"))
    assert not feat._is_rate_limit(RuntimeError("User memory limit exceeded"))
    assert feat._is_compute_limit(RuntimeError("User memory limit exceeded"))
    assert feat._is_compute_limit(RuntimeError("Computation timed out"))
    assert not feat._is_compute_limit(RuntimeError("something else"))


def test_sampled_value_prefers_band_name_then_mean_then_first():
    assert feat._sampled_value({"elevation": 1800, "mean": 1}, "elevation") == 1800
    assert feat._sampled_value({"mean": 12.5}, "precipitation") == 12.5
    assert feat._sampled_value({"first": 9}, "precipitation") == 9
    assert feat._sampled_value({"fid": 1}, "precipitation") is None


def test_join_and_merge_features():
    survey = pd.DataFrame({"fid": [1.0, 2.0], "maize_pos": [1, 0]})
    precip = pd.DataFrame({"fid": [1.0, 2.0], "precipitation": [10.0, 12.0]})
    terrain = pd.DataFrame({"fid": [1.0, 2.0], "elevation": [1800, 1900]})
    merged = feat.merge_features(survey, [precip, terrain])
    assert list(merged.columns) == ["fid", "maize_pos", "precipitation", "elevation"]
    assert len(merged) == 2
    skipped = feat.merge_features(survey, [pd.DataFrame(), None])
    assert list(skipped.columns) == ["fid", "maize_pos"]


def test_build_xy_keeps_present_columns_and_drops_missing(tmp_config, capsys):
    tmp_config.feature_columns = ["precipitation", "elevation", "missing_col"]
    tmp_config.target_column = "maize_pos"
    merged = pd.DataFrame(
        {
            "precipitation": [1.0, 2.0],
            "elevation": [100.0, 200.0],
            "maize_pos": [0, 1],
        }
    )
    X, y = feat.build_xy(merged, tmp_config)
    assert list(X.columns) == ["precipitation", "elevation"]
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert y.tolist() == [0, 1]
    assert "missing_col" in capsys.readouterr().out


def test_select_harmonic_columns():
    df = pd.DataFrame(
        {
            "fid": [1],
            "GCVI_mean": [1.0],
            "GCVI_r2": [0.8],
            "NDVI_mean": [0.4],
            "other": [9],
        }
    )
    out = feat.select_harmonic_columns(df, ["GCVI"])
    assert list(out.columns) == ["fid", "GCVI_mean", "GCVI_r2"]


def test_concat_band_batches(tmp_path: Path):
    pd.DataFrame({"fid": [1, 2], "GCVI_mean": [0.1, 0.2]}).to_csv(tmp_path / "gcvi.csv", index=False)
    pd.DataFrame({"fid": [1, 2], "NDVI_mean": [0.3, 0.4]}).to_csv(tmp_path / "ndvi_batch_1.csv", index=False)
    out = tmp_path / "joined.csv"
    merged = feat.concat_band_batches(tmp_path, ["GCVI", "NDVI"], out_path=out)
    assert out.exists()
    assert list(merged.columns) == ["fid", "GCVI_mean", "NDVI_mean"]
    empty = feat.concat_band_batches(tmp_path, ["NBR1"])
    assert empty.empty


def test_concat_band_batches_requires_id(tmp_path: Path):
    pd.DataFrame({"GCVI_mean": [0.1]}).to_csv(tmp_path / "gcvi.csv", index=False)
    with pytest.raises(KeyError, match="fid"):
        feat.concat_band_batches(tmp_path, ["GCVI"])


def test_correlation_pruned_columns_drops_highly_correlated():
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [10.0, 1.0, 8.0, 2.0],
        }
    )
    kept = feat.correlation_pruned_columns(df, threshold=0.85)
    assert "a" in kept
    assert "b" not in kept
    assert "c" in kept


def test_cached_csv_roundtrip(tmp_path: Path):
    path = tmp_path / "x.csv"
    assert feat._cached_csv(path, enabled=True) is None
    pd.DataFrame({"fid": [1]}).to_csv(path, index=False)
    cached = feat._cached_csv(path, enabled=True)
    assert cached is not None and len(cached) == 1
    assert feat._cached_csv(path, enabled=False) is None
