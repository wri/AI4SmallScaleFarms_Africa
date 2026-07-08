"""Feature engineering: harmonic coefficients, terrain, precipitation, merge.

Generalises notebook cells 50-82. Produces the per-plot feature table used to
train the classifier. Two ways to obtain the harmonic-regression features are
offered:

* ``extract_harmonic_features`` -- computes coefficients per plot in-notebook
  via ``reduceRegions`` and ``getInfo`` (fine for hundreds of plots).
* ``export_harmonic_features_to_drive`` + ``concat_band_batches`` -- the batch
  Drive-export workflow for very large surveys that would otherwise time out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import ee
import numpy as np
import pandas as pd

from src.config import PipelineConfig


# ---------------------------------------------------------------------------
# Harmonic regression coefficients per plot
# ---------------------------------------------------------------------------
def _reduce_to_df(
    image: ee.Image, plots: ee.FeatureCollection, id_field: str, scale: int
) -> pd.DataFrame:
    """Reduce an image over each plot and return properties as a DataFrame."""
    reduced = image.reduceRegions(
        reducer=ee.Reducer.mean(), collection=plots, scale=scale
    )
    features = reduced.getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in features])


def extract_harmonic_features(
    imgcoll: ee.ImageCollection,
    plots: ee.FeatureCollection,
    config: PipelineConfig,
    bands: Optional[Sequence[str]] = None,
    id_field: str = "fid",
    batch_size: int = 8,
) -> pd.DataFrame:
    """Compute harmonic-regression coefficients per plot for each band.

    Iterates band-by-band (and plot-batch-by-batch) to stay within EE limits,
    then joins the per-band tables on ``id_field``.
    """
    import harmonics

    bands = list(bands or config.regression_bands)
    n = plots.size().getInfo()
    plots_list = plots.toList(n)

    merged: Optional[pd.DataFrame] = None
    for band in bands:
        coef_img = harmonics.run_std_regressions(
            imgcoll.select([band]), [band], refdate=config.reference_date
        )
        band_frames: List[pd.DataFrame] = []
        for i in range(0, n, batch_size):
            batch = ee.FeatureCollection(plots_list.slice(i, i + batch_size))
            band_frames.append(_reduce_to_df(coef_img, batch, id_field, config.scale))
        band_df = pd.concat(band_frames, ignore_index=True)
        keep = [c for c in band_df.columns if c.startswith(f"{band}_") or c == id_field]
        band_df = band_df[keep]
        merged = band_df if merged is None else merged.merge(band_df, on=id_field, how="inner")
    return merged if merged is not None else pd.DataFrame()


def export_harmonic_features_to_drive(
    imgcoll: ee.ImageCollection,
    plots: ee.FeatureCollection,
    config: PipelineConfig,
    bands: Optional[Sequence[str]] = None,
    folder: str = "GEE_Exports_All_Bands",
    batch_size: int = 5,
) -> Dict[str, dict]:
    """Start batched Drive exports of harmonic coefficients (large surveys).

    Returns a dict of task-id -> metadata. Download the resulting CSVs and
    recombine with ``concat_band_batches``.
    """
    import harmonics

    bands = list(bands or config.regression_bands)
    n = plots.size().getInfo()
    plots_list = plots.toList(n)
    tasks: Dict[str, dict] = {}
    for band in bands:
        coef_img = harmonics.run_std_regressions(
            imgcoll.select([band]), [band], refdate=config.reference_date
        )
        for i in range(0, n, batch_size):
            batch_no = i // batch_size + 1
            batch = ee.FeatureCollection(plots_list.slice(i, i + batch_size))
            results = coef_img.reduceRegions(
                reducer=ee.Reducer.mean(), collection=batch, scale=config.scale
            )
            task = ee.batch.Export.table.toDrive(
                collection=results,
                description=f"{band}_Batch_{batch_no}",
                folder=folder,
                fileNamePrefix=f"{band.lower()}_batch_{batch_no}",
                fileFormat="CSV",
            )
            task.start()
            tasks[task.id] = {"band": band, "batch_number": batch_no}
            print(f"Export started for {band} - Batch {batch_no} (Task {task.id})")
    return tasks


def concat_band_batches(folder: str | Path, bands: Sequence[str], id_field: str = "fid") -> pd.DataFrame:
    """Concatenate per-band batch CSVs and merge them side-by-side on ``id_field``."""
    folder = Path(folder)
    merged: Optional[pd.DataFrame] = None
    for band in bands:
        files = sorted(folder.glob(f"{band.lower()}_batch_*.csv"))
        if not files:
            print(f"No files found for {band}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if id_field not in df.columns:
            raise KeyError(f"'{id_field}' column not found in {band} files")
        cols = [c for c in df.columns if c != id_field]
        df = df[[id_field] + cols]
        merged = df if merged is None else merged.merge(
            df, on=id_field, how="inner", suffixes=("", f"_{band.lower()}")
        )
    return merged if merged is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Terrain features
# ---------------------------------------------------------------------------
def extract_terrain_features(
    plots: ee.FeatureCollection, config: PipelineConfig, id_field: str = "fid"
) -> pd.DataFrame:
    """Mean elevation, slope and aspect per plot from SRTM."""
    srtm = ee.Image(config.srtm_asset)
    elevation = srtm.select("elevation")
    measures = {
        "elevation": elevation,
        "slope": ee.Terrain.slope(elevation),
        "aspect": ee.Terrain.aspect(elevation),
    }
    out: Optional[pd.DataFrame] = None
    for label, img in measures.items():
        means = img.reduceRegions(
            reducer=ee.Reducer.mean(), collection=plots, scale=config.scale
        )
        data = [
            {id_field: f["properties"][id_field], label: f["properties"].get("mean")}
            for f in means.getInfo()["features"]
        ]
        df = pd.DataFrame(data)
        out = df if out is None else out.merge(df, on=id_field)
    return out if out is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Precipitation features
# ---------------------------------------------------------------------------
def extract_precipitation(
    plots: ee.FeatureCollection, config: PipelineConfig, id_field: str = "fid"
) -> pd.DataFrame:
    """Total (summed) precipitation over the season, averaged per plot."""
    start = config.precip_start_date or config.start_date
    end = config.precip_end_date or config.end_date
    precip = (
        ee.ImageCollection(config.precip_dataset)
        .filter(ee.Filter.date(start, end))
        .select("precipitation")
    )
    tot = precip.reduce(ee.Reducer.sum()).reduceRegions(
        reducer=ee.Reducer.mean(), collection=plots, scale=config.scale
    )
    data = [
        {id_field: f["properties"][id_field], "precipitation": f["properties"].get("mean")}
        for f in tot.getInfo()["features"]
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Merge + selection
# ---------------------------------------------------------------------------
def select_harmonic_columns(df: pd.DataFrame, bands: Sequence[str], id_field: str = "fid") -> pd.DataFrame:
    """Keep only harmonic coefficient columns for the requested bands + id."""
    keep = [id_field] + [c for c in df.columns if any(c.startswith(f"{b}_") for b in bands)]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def merge_features(
    survey_gdf,
    feature_frames: Sequence[pd.DataFrame],
    id_field: str = "fid",
) -> pd.DataFrame:
    """Merge the survey table with all feature tables on ``id_field``."""
    merged = survey_gdf
    for frame in feature_frames:
        if frame is None or frame.empty:
            continue
        merged = merged.merge(frame, on=id_field)
    return merged


def correlation_pruned_columns(
    df: pd.DataFrame, threshold: float = 0.85
) -> List[str]:
    """Return columns that survive a pairwise-correlation filter."""
    numerical = df.select_dtypes(include=["number"])
    corr = numerical.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return [c for c in numerical.columns if c not in to_drop]


def build_xy(merged: pd.DataFrame, config: PipelineConfig):
    """Split the merged table into feature matrix ``X`` and target ``y``."""
    feature_cols = [c for c in config.feature_columns if c in merged.columns]
    missing = [c for c in config.feature_columns if c not in merged.columns]
    if missing:
        print(f"Warning: missing feature columns dropped from X: {missing}")
    X = merged[feature_cols]
    y = merged[config.target_column]
    return X, y
