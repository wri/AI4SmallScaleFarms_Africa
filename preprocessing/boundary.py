"""Derive the study-area boundary and inference AOI.

Generalises notebook cell 8 (GAUL admin lookup) and cell 93 (rectangular AOI).
Instead of hardcoding "Nyandarua" the admin name comes from the config, so any
GAUL level-2 (or other admin) area can be selected.
"""

from __future__ import annotations

from typing import Optional, Sequence

import ee
import geemap
import geopandas as gpd

from src.config import PipelineConfig


def get_admin_boundary(config: PipelineConfig) -> ee.Feature:
    """Return the study-area boundary as an ``ee.Feature`` from a GAUL lookup."""
    fc = ee.FeatureCollection(config.gaul_dataset).filter(
        ee.Filter.eq(config.gaul_name_field, config.area_name)
    )
    return ee.Feature(fc.geometry())


def boundary_from_survey(gdf: gpd.GeoDataFrame) -> ee.Feature:
    """Fallback: derive an EE boundary from the survey extent (convex hull)."""
    hull = gdf.to_crs(epsg=4326).union_all().convex_hull
    return ee.Feature(geemap.geopandas_to_ee(
        gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
    ).geometry())


def get_aoi(config: PipelineConfig, boundary: Optional[ee.Feature] = None) -> ee.Geometry:
    """Return the inference AOI geometry.

    Uses ``config.aoi_bbox`` when provided, otherwise the bounds of the admin
    boundary. This is the region over which the wall-to-wall feature image is
    exported and classified.
    """
    if config.aoi_bbox is not None:
        return ee.Geometry.Rectangle(list(config.aoi_bbox))
    boundary = boundary or get_admin_boundary(config)
    return boundary.geometry().bounds()


def survey_to_ee(gdf: gpd.GeoDataFrame, columns: Optional[Sequence[str]] = None) -> ee.FeatureCollection:
    """Convert (a subset of) the survey GeoDataFrame to an EE FeatureCollection."""
    if columns is not None:
        keep = [c for c in columns if c in gdf.columns]
        gdf = gdf.loc[:, keep]
    return geemap.geopandas_to_ee(gdf)
