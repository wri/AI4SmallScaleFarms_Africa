from __future__ import annotations

import pandas as pd

from preprocessing.timeseries import create_lists_for_extract, daily_mean, sample_lists


def test_sample_lists_keeps_every_nth_item():
    assert sample_lists([0, 1, 2, 3, 4, 5], 2) == [0, 2, 4]
    assert sample_lists(["a", "b", "c"], 1) == ["a", "b", "c"]


def test_daily_mean_aggregates_timestamps():
    df = pd.DataFrame(
        {
            "datetime": ["2024-03-01 10:00:00", "2024-03-01 16:00:00", "2024-03-02 10:00:00"],
            "GCVI": [1.0, 3.0, 5.0],
        }
    )
    daily = daily_mean(df, "GCVI")
    assert list(daily["GCVI"]) == [2.0, 5.0]
    assert "date_ordinal" in daily.columns


def test_create_lists_for_extract_rewrites_band_suffixes():
    df = pd.DataFrame(
        {
            "datetime": ["2024-03-01 00:00:00"],
            "band_id": ["20240301T073641_GCVI"],
        }
    )
    times, bands = create_lists_for_extract(df, "_GCVI", "_NDVI")
    assert times == ["2024-03-01 00:00:00"]
    assert bands == ["20240301T073641_NDVI"]
