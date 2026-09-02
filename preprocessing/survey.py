"""Load and prepare survey/training vector data.

Generalises notebook cells 0-12: read the survey GeoJSON, build a target
label (binary maize vs other, or multiclass crop_a), and compute per-plot
area for polygon surveys.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from src.config import PipelineConfig

# 1 square metre in acres.
SQM_TO_ACRES = 0.0002471054

_POINT_TYPES = {"point", "multipoint"}


def load_survey(config: PipelineConfig, path: Optional[str | Path] = None) -> gpd.GeoDataFrame:
    """Read the survey vector file and reproject to the working CRS."""
    path = path or config.survey_geojson
    if path is None:
        raise ValueError("No survey file provided. Set `survey_geojson` in the config.")
    path = config.resolve_path(path) or path
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=config.working_epsg)
    return gdf.to_crs(epsg=config.working_epsg)


def is_point_survey(gdf: gpd.GeoDataFrame) -> bool:
    """True when every geometry is a Point or MultiPoint."""
    types = {str(t).lower() for t in gdf.geom_type.unique()}
    return bool(types) and types <= _POINT_TYPES


def sample_geometry_mode(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> str:
    """``point`` (one 30 m pixel at the GPS) or ``polygon`` (zonal mean)."""
    mode = str(config.sample_geometry or "auto").strip().lower()
    if mode in {"point", "polygon"}:
        return mode
    return "point" if is_point_survey(gdf) else "polygon"


def normalize_crop_name(value, aliases: Optional[dict] = None) -> Optional[str]:
    """Lowercase, strip, and apply the typo / synonym map."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower().replace("_", " ")
    if text in {"", "none", "nan", "<na>"}:
        return None
    mapping = {str(k).strip().lower(): str(v).strip().lower() for k, v in (aliases or {}).items()}
    return mapping.get(text, text)


def add_binary_label(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    """Add a binary target: 1 if any crop column names the target crop."""
    gdf = gdf.copy()
    crop_cols = [c for c in config.crop_columns if c in gdf.columns]
    if not crop_cols:
        raise KeyError(
            f"None of the crop columns {config.crop_columns} were found in the survey data. "
            f"Available columns: {list(gdf.columns)}"
        )
    targets = {n.lower() for n in config.target_crop_names}
    matched = gdf.loc[:, crop_cols].apply(
        lambda col: col.astype(str).str.strip().str.lower().isin(targets)
    )
    gdf[config.target_column] = matched.any(axis=1).astype(int)
    return gdf


def _ordered_class_names(kept: List[str], config: PipelineConfig) -> List[str]:
    """Preferred crops first, then any other kept names, then ``other`` last."""
    preferred = [n.strip().lower() for n in (config.preferred_class_order or [])]
    other = config.other_class_name.strip().lower()
    ordered: List[str] = []
    for name in preferred:
        if name in kept and name not in ordered:
            ordered.append(name)
    extras = sorted(n for n in kept if n not in ordered and n != other)
    ordered.extend(extras)
    if other in kept:
        ordered.append(other)
    return ordered


def add_multiclass_label(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    """Label from ``label_column`` (crop_a): clean aliases, collapse rare classes.

    Writes:
    - ``raw_label_column``: normalised crop name before collapsing
    - ``class_name_column``: collapsed class name (maize, wheat, ..., other)
    - ``target_column``: integer codes 1..N (0 is reserved for nodata on maps)
    """
    gdf = gdf.copy()
    source = config.label_column
    if source not in gdf.columns:
        raise KeyError(
            f"Multiclass label column {source!r} was not found in the survey data. "
            f"Available columns: {list(gdf.columns)}"
        )

    raw = gdf[source].map(lambda v: normalize_crop_name(v, config.crop_aliases))
    raw_col = config.raw_label_column
    name_col = config.class_name_column
    gdf[raw_col] = raw

    counts = Counter(v for v in raw if v is not None)
    min_count = max(int(config.min_class_count), 1)
    other = config.other_class_name.strip().lower()
    collapsed = []
    for value in raw:
        if value is None or counts[value] < min_count:
            collapsed.append(other)
        else:
            collapsed.append(value)
    gdf[name_col] = collapsed

    kept = sorted(set(collapsed))
    ordered = _ordered_class_names(kept, config)
    code_map = {name: i + 1 for i, name in enumerate(ordered)}
    gdf[config.target_column] = gdf[name_col].map(code_map).astype(int)

    print("Multiclass labels (after alias cleanup and rare-class collapse):")
    for name in ordered:
        n = int((gdf[name_col] == name).sum())
        print(f"  {code_map[name]:2d}  {name:12s}  n={n}")
    other_counts = Counter(v for v, c in zip(raw, collapsed) if c == other and v is not None)
    if other_counts:
        print(f"  {other!r} breakdown (n < {min_count}):")
        for crop, n in other_counts.most_common():
            print(f"      {n:3d}  {crop}")
    return gdf


def add_target_label(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    """Add the configured target column (binary or multiclass)."""
    if config.is_binary:
        return add_binary_label(gdf, config)
    return add_multiclass_label(gdf, config)


def add_plot_area(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> gpd.GeoDataFrame:
    """Compute per-plot area in acres via a planar (projected) CRS.

    Point surveys skip this (area would be zero).
    """
    if is_point_survey(gdf):
        return gdf
    gdf = gdf.copy()
    projected = gdf.to_crs(epsg=config.area_calc_epsg)
    gdf["plot_area_acres"] = projected.area * SQM_TO_ACRES
    return gdf


def other_class_breakdown(gdf: gpd.GeoDataFrame, config: PipelineConfig) -> Dict[str, int]:
    """Counts of original (normalised) crop names that were folded into ``other``."""
    raw_col = config.raw_label_column
    name_col = config.class_name_column
    if raw_col not in gdf.columns or name_col not in gdf.columns:
        return {}
    other = config.other_class_name.strip().lower()
    subset = gdf[gdf[name_col] == other]
    return subset[raw_col].dropna().value_counts().to_dict()


def class_legend(gdf: pd.DataFrame, config: PipelineConfig) -> Optional[Tuple[List[int], List[str]]]:
    """Integer codes and names aligned with the trained label space."""
    if config.is_binary:
        crop = config.target_crop_names[0] if config.target_crop_names else "target"
        return [0, 1], ["other", crop]
    name_col = config.class_name_column
    code_col = config.target_column
    if name_col not in gdf.columns or code_col not in gdf.columns:
        return None
    mapping = gdf.groupby(code_col)[name_col].first().sort_index()
    codes = [int(c) for c in mapping.index.tolist()]
    names = [str(n) for n in mapping.tolist()]
    return codes, names


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
