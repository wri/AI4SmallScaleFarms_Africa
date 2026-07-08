"""Load and prepare survey/training vector data.

Generalises notebook cells 0-12: read the survey GeoJSON, build a binary
target label for the crop(s) of interest, and compute per-plot area.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd

from src.config import PipelineConfig

# 1 square metre in acres.
SQM_TO_ACRES = 0.0002471054


def load_survey(config: PipelineConfig, path: Optional[str | Path] = None) -> gpd.GeoDataFrame:
    """Read the survey vector file and reproject to the working CRS."""
    path = path or config.survey_geojson
    if path is None:
        raise ValueError("No survey file provided. Set `survey_geojson` in the config.")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=config.working_epsg)
    return gdf.to_crs(epsg=config.working_epsg)


def add_target_label(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    """Add a binary target column: 1 if any crop column names the target crop.

    Only the crop columns that actually exist in the data are used, so the same
    code works across survey schemas with differing numbers of crop fields.
    """
    gdf = gdf.copy()
    crop_cols = [c for c in config.crop_columns if c in gdf.columns]
    if not crop_cols:
        raise KeyError(
            f"None of the crop columns {config.crop_columns} were found in the survey data. "
            f"Available columns: {list(gdf.columns)}"
        )
    gdf[config.target_column] = (
        gdf.loc[:, crop_cols].isin(config.target_crop_names).sum(axis=1)
    )
    # Clamp to a strict 0/1 label (a plot with the crop in multiple columns is still 1).
    gdf[config.target_column] = (gdf[config.target_column] > 0).astype(int)
    return gdf


def add_plot_area(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    """Compute per-plot area in acres via a planar (projected) CRS."""
    gdf = gdf.copy()
    projected = gdf.to_crs(epsg=config.area_calc_epsg)
    gdf["plot_area_acres"] = projected.area * SQM_TO_ACRES
    return gdf


def prepare_survey(config: PipelineConfig, path: Optional[str | Path] = None) -> gpd.GeoDataFrame:
    """Full survey preparation: load -> label -> area, returned in working CRS.

    Ensures ``fid`` is float so it can be joined against the EE-derived feature
    tables (which return float ids).
    """
    gdf = load_survey(config, path)
    gdf = add_target_label(gdf, config)
    gdf = add_plot_area(gdf, config)
    if "fid" in gdf.columns:
        gdf["fid"] = gdf["fid"].astype("float64")
    return gdf.to_crs(epsg=config.working_epsg)
