"""Derive the study-area boundary and inference AOI.

Generalises notebook cell 8 (GAUL admin lookup) and cell 93 (rectangular AOI).
Instead of hardcoding "Nyandarua" the admin name comes from the config, so any
GAUL level-2 (or other admin) area can be selected. A local shapefile / GeoJSON
(``config.boundary_path``) is used when the file exists; otherwise the pipeline
falls back to GAUL filtered by ``config.gaul_lookup_name``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import ee
import geemap
import geopandas as gpd

from src.config import PipelineConfig


def boundary_from_vector(path: str | Path, working_epsg: int = 4326) -> ee.Feature:
    """Dissolve a local admin vector into a single EE Feature."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=working_epsg)
    gdf = gdf.to_crs(epsg=working_epsg)
    union = gdf.union_all()
    dissolved = gpd.GeoDataFrame({"id": [0]}, geometry=[union], crs=f"EPSG:{working_epsg}")
    fc = geemap.geopandas_to_ee(dissolved)
    return ee.Feature(fc.first())


def _existing_boundary_path(config: PipelineConfig) -> Optional[Path]:
    """Return the local boundary file if it exists, else ``None``.

    Tries the path as given (cwd-relative) and resolved against ``project_root``.
    """
    if config.boundary_path is None:
        return None
    given = Path(config.boundary_path).expanduser()
    candidates = [given]
    resolved = config.resolve_path(given)
    if resolved is not None and resolved not in candidates:
        candidates.append(resolved)
    for path in candidates:
        if path.exists():
            return path
    return None


def _boundary_from_gaul(config: PipelineConfig) -> ee.Feature:
    """Notebook cell 8: FAO GAUL level-2 feature filtered by ADM name."""
    name = config.gaul_lookup_name
    print(
        f"Using GAUL boundary -> {config.gaul_dataset} "
        f"{config.gaul_name_field}={name!r}"
    )
    fc = ee.FeatureCollection(config.gaul_dataset).filter(
        ee.Filter.eq(config.gaul_name_field, name)
    )
    n = fc.size().getInfo()
    if not n:
        raise ValueError(
            f"No GAUL feature with {config.gaul_name_field}={name!r} in "
            f"{config.gaul_dataset}. Set `gaul_name` to the ADM2 name "
            f"(notebook cell 8 uses 'Nyandarua')."
        )
    return ee.Feature(fc.geometry())


def get_admin_boundary(config: PipelineConfig) -> ee.Feature:
    """Return the study-area boundary as an ``ee.Feature``.

    Prefers a local shapefile / GeoJSON when ``boundary_path`` exists on disk.
    If the path is unset or the file is missing, falls back to a GAUL ADM2
    lookup using ``gaul_name`` (or ``area_name`` when ``gaul_name`` is unset).
    """
    local = _existing_boundary_path(config)
    if local is not None:
        print(f"Using local admin boundary -> {local}")
        return boundary_from_vector(local, config.working_epsg)
    if config.boundary_path is not None:
        print(
            f"Local boundary not found at {config.boundary_path}; "
            "falling back to GAUL"
        )
    return _boundary_from_gaul(config)


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
        rect = ee.Geometry.Rectangle(list(config.aoi_bbox))
        boundary = boundary or get_admin_boundary(config)
        return rect.intersection(boundary.geometry(), 1)
    boundary = boundary or get_admin_boundary(config)
    return boundary.geometry().bounds()


def survey_to_ee(gdf: gpd.GeoDataFrame, columns: Optional[Sequence[str]] = None) -> ee.FeatureCollection:
    """Convert (a subset of) the survey GeoDataFrame to an EE FeatureCollection."""
    if columns is not None:
        keep = [c for c in columns if c in gdf.columns]
        gdf = gdf.loc[:, keep]
    return geemap.geopandas_to_ee(gdf)
