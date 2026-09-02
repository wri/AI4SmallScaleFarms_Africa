"""Feature engineering: harmonic coefficients, terrain, precipitation, merge.

Generalises notebook cells 50-82. Produces the per-plot feature table used to
train the classifier. Harmonic coefficients are fetched from Earth Engine in
band batches, written as CSVs under ``data/interim/``, then concatenated into
``data/processed/<area>_harmonic_features.csv`` — no Google Drive step.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import ee
import numpy as np
import pandas as pd

from src.config import PipelineConfig

_RATE_LIMIT_MARKERS = (
    "user request limit",
    "quota exceeded",
    "rate limit",
    "concurrent request",
    "limit exceeded",
    "429",
)
_COMPUTE_LIMIT_MARKERS = (
    "memory",
    "timed out",
    "timeout",
    "computation exceeded",
    "too many concurrent aggregations",
    "user memory",
)


def _ee_msg(exc: BaseException) -> str:
    return str(exc).lower()


def _is_compute_limit(exc: BaseException) -> bool:
    msg = _ee_msg(exc)
    return any(marker in msg for marker in _COMPUTE_LIMIT_MARKERS)


def _is_rate_limit(exc: BaseException) -> bool:
    msg = _ee_msg(exc)
    if "memory" in msg:
        return False
    return any(marker in msg for marker in _RATE_LIMIT_MARKERS)


def _cached_csv(path: Path, enabled: bool) -> Optional[pd.DataFrame]:
    if enabled and path.exists():
        print(f"Reusing cached features -> {path}")
        return pd.read_csv(path)
    return None


def _image_reducer(sample_geometry: str) -> ee.Reducer:
    """Zonal mean inside polygons; the 30 m pixel under a GPS point."""
    if sample_geometry == "point":
        return ee.Reducer.first()
    return ee.Reducer.mean()


def _sampled_value(props: dict, band_name: str):
    """Value from reduceRegions (mean/first) or sampleRegions (band name)."""
    if band_name in props and props[band_name] is not None:
        return props[band_name]
    for key in ("mean", "first"):
        if key in props:
            return props[key]
    return None


# ---------------------------------------------------------------------------
# Harmonic regression coefficients per plot
# ---------------------------------------------------------------------------
def _reduce_to_df(
    image: ee.Image,
    plots: ee.FeatureCollection,
    scale: int,
    tile_scale: int,
    retries: int = 5,
    sample_geometry: str = "polygon",
) -> pd.DataFrame:
    """Reduce an image over each labelled geometry and return properties as a DataFrame."""
    reduced = image.reduceRegions(
        reducer=_image_reducer(sample_geometry),
        collection=plots,
        scale=scale,
        tileScale=tile_scale,
    )
    delay = 8.0
    last_exc: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            features = reduced.getInfo()["features"]
            return pd.DataFrame([f["properties"] for f in features])
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc) and attempt < retries - 1:
                print(f"    EE user/rate limit ({exc}); retry in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("reduceRegions failed")  # pragma: no cover


def _reduce_range(
    image: ee.Image,
    plots_list: ee.List,
    start: int,
    end: int,
    scale: int,
    tile_scale: int,
    max_tile_scale: int = 16,
    sample_geometry: str = "polygon",
) -> pd.DataFrame:
    """Reduce plots ``[start, end)``, splitting on Earth Engine compute limits."""
    n = end - start
    batch = ee.FeatureCollection(plots_list.slice(start, end))
    try:
        print(f"    plots {start + 1}-{end} (n={n}, tileScale={tile_scale})")
        return _reduce_to_df(
            image, batch, scale, tile_scale, sample_geometry=sample_geometry
        )
    except Exception as exc:
        if _is_compute_limit(exc) and n > 1:
            mid = start + n // 2
            next_ts = min(max(tile_scale * 2, 4), max_tile_scale)
            print(
                f"    compute limit on {n} plots ({exc}); "
                f"splitting {start}:{end} with tileScale={next_ts}"
            )
            left = _reduce_range(
                image, plots_list, start, mid, scale, next_ts, max_tile_scale,
                sample_geometry=sample_geometry,
            )
            right = _reduce_range(
                image, plots_list, mid, end, scale, next_ts, max_tile_scale,
                sample_geometry=sample_geometry,
            )
            return pd.concat([left, right], ignore_index=True)
        raise


def extract_harmonic_features(
    imgcoll: ee.ImageCollection,
    plots: ee.FeatureCollection,
    config: PipelineConfig,
    bands: Optional[Sequence[str]] = None,
    id_field: str = "fid",
    batch_size: Optional[int] = None,
    aoi: Optional[ee.Geometry] = None,
    sample_geometry: str = "polygon",
) -> pd.DataFrame:
    """Sample county-wide harmonic coefficients at each training location.

    Polygons use a zonal mean; points use the 30 m pixel under the GPS
    (``ee.Reducer.first``). The coefficient image is county-wide (clipped to
    ``aoi`` when given, never to the labelled geometries).
    Per-band CSVs under ``config.harmonic_band_dir`` are reused on resume.
    """
    cached = _cached_csv(config.harmonic_features_path, config.reuse_ee_features)
    if cached is not None:
        return cached

    from src.ee_utils import import_harmonics

    harmonics = import_harmonics()

    bands = list(bands or config.regression_bands)
    tile_scale = max(int(config.harmonic_tile_scale), 1)
    chunk = batch_size if batch_size is not None else config.harmonic_batch_size
    band_dir = config.harmonic_band_dir
    band_dir.mkdir(parents=True, exist_ok=True)

    cached_bands: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    for band in bands:
        band_csv = band_dir / f"{band.lower()}.csv"
        if config.reuse_ee_features and band_csv.exists():
            print(f"  {band}: reuse {band_csv}")
            cached_bands[band] = pd.read_csv(band_csv)
        else:
            missing.append(band)

    n: Optional[int] = None
    plots_list: Optional[ee.List] = None
    if missing:
        n = int(plots.size().getInfo())
        if chunk is None or chunk <= 0 or chunk >= n:
            chunk_n = n
        else:
            chunk_n = int(chunk)
        n_batches = (n + chunk_n - 1) // chunk_n
        how = "pixel-at-point (first)" if sample_geometry == "point" else "polygon zonal mean"
        print(
            f"Harmonic features: {n} locations ({how}), {len(bands)} bands "
            f"({len(missing)} to fetch), {n_batches} request(s)/band, "
            f"tileScale={tile_scale}"
        )
        imgcoll = imgcoll.select(missing)
        plots_list = plots.toList(n)

    merged: Optional[pd.DataFrame] = None
    for band in bands:
        if band in cached_bands:
            band_df = cached_bands[band]
        else:
            assert n is not None and plots_list is not None
            t0 = time.perf_counter()
            coef_img = harmonics.run_std_regressions(
                imgcoll.select([band]), [band],
                refdate=config.reference_date, nharmonics=config.n_harmonics,
            )
            if aoi is not None:
                coef_img = coef_img.clip(aoi)
            if chunk is None or chunk <= 0 or chunk >= n:
                step = n
            else:
                step = int(chunk)
            frames = [
                _reduce_range(
                    coef_img, plots_list, i, min(i + step, n),
                    config.scale, tile_scale, sample_geometry=sample_geometry,
                )
                for i in range(0, n, step)
            ]
            band_df = pd.concat(frames, ignore_index=True)
            keep = [c for c in band_df.columns if c.startswith(f"{band}_") or c == id_field]
            band_df = band_df[keep]
            band_csv = band_dir / f"{band.lower()}.csv"
            band_df.to_csv(band_csv, index=False)
            print(f"  {band}: {time.perf_counter() - t0:.1f}s -> {band_csv}")
        merged = band_df if merged is None else merged.merge(band_df, on=id_field, how="inner")

    if merged is None:
        return pd.DataFrame()
    out = config.harmonic_features_path
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"Concatenated harmonic features -> {out} ({merged.shape})")
    return merged


def export_harmonic_features_to_drive(
    imgcoll: ee.ImageCollection,
    plots: ee.FeatureCollection,
    config: PipelineConfig,
    bands: Optional[Sequence[str]] = None,
    folder: str = "GEE_Exports_All_Bands",
    batch_size: int = 5,
    sample_geometry: str = "polygon",
) -> Dict[str, dict]:
    """Start batched Drive exports of harmonic coefficients (large surveys).

    Returns a dict of task-id -> metadata. Download the resulting CSVs and
    recombine with ``concat_band_batches``.
    """
    from src.ee_utils import import_harmonics

    harmonics = import_harmonics()

    bands = list(bands or config.regression_bands)
    n = plots.size().getInfo()
    plots_list = plots.toList(n)
    tasks: Dict[str, dict] = {}
    for band in bands:
        coef_img = harmonics.run_std_regressions(
            imgcoll.select([band]), [band],
            refdate=config.reference_date, nharmonics=config.n_harmonics,
        )
        for i in range(0, n, batch_size):
            batch_no = i // batch_size + 1
            batch = ee.FeatureCollection(plots_list.slice(i, i + batch_size))
            results = coef_img.reduceRegions(
                reducer=_image_reducer(sample_geometry),
                collection=batch,
                scale=config.scale,
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


def concat_band_batches(
    folder: str | Path,
    bands: Sequence[str],
    id_field: str = "fid",
    out_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Concatenate per-band CSVs (``{band}.csv`` or ``{band}_batch_*.csv``) on ``id_field``."""
    folder = Path(folder)
    merged: Optional[pd.DataFrame] = None
    for band in bands:
        prefix = band.lower()
        files = sorted(folder.glob(f"{prefix}_batch_*.csv")) or sorted(folder.glob(f"{prefix}.csv"))
        if not files:
            print(f"No files found for {band}")
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if id_field not in df.columns:
            raise KeyError(f"'{id_field}' column not found in {band} files")
        cols = [c for c in df.columns if c != id_field]
        df = df[[id_field] + cols]
        merged = df if merged is None else merged.merge(df, on=id_field, how="inner", suffixes=("", f"_{band.lower()}"))
    if merged is None:
        return pd.DataFrame()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)
    return merged


# ---------------------------------------------------------------------------
# Terrain features
# ---------------------------------------------------------------------------
def extract_terrain_features(
    plots: ee.FeatureCollection,
    config: PipelineConfig,
    id_field: str = "fid",
    sample_geometry: str = "polygon",
) -> pd.DataFrame:
    """Elevation, slope and aspect per labelled geometry from SRTM."""
    cached = _cached_csv(config.terrain_features_path, config.reuse_ee_features)
    if cached is not None:
        return cached
    srtm = ee.Image(config.srtm_asset)
    elevation = srtm.select("elevation")
    measures = {
        "elevation": elevation,
        "slope": ee.Terrain.slope(elevation),
        "aspect": ee.Terrain.aspect(elevation),
    }
    reducer = _image_reducer(sample_geometry)
    out: Optional[pd.DataFrame] = None
    for label, img in measures.items():
        sampled = img.reduceRegions(
            reducer=reducer, collection=plots, scale=config.scale
        )
        data = [
            {
                id_field: f["properties"][id_field],
                label: _sampled_value(f["properties"], label),
            }
            for f in sampled.getInfo()["features"]
        ]
        df = pd.DataFrame(data)
        out = df if out is None else out.merge(df, on=id_field)
    if out is None:
        return pd.DataFrame()
    path = config.terrain_features_path
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Terrain features -> {path} ({out.shape})")
    return out


# ---------------------------------------------------------------------------
# Precipitation features
# ---------------------------------------------------------------------------
def precip_image(
    config: PipelineConfig, aoi: Optional[ee.Geometry] = None
) -> ee.Image:
    """Seasonal CHIRPS image. Notebook inference (cell 97) uses temporal mean."""
    start = config.precip_start_date or config.start_date
    end = config.precip_end_date or config.end_date
    coll = (
        ee.ImageCollection(config.precip_dataset)
        .filter(ee.Filter.date(start, end))
        .select("precipitation")
    )
    if aoi is not None:
        coll = coll.filterBounds(aoi)
    reducer = (config.precip_reducer or "mean").lower()
    img = coll.mean() if reducer == "mean" else coll.sum()
    return img.rename("precipitation")


def extract_precipitation(
    plots: ee.FeatureCollection,
    config: PipelineConfig,
    id_field: str = "fid",
    sample_geometry: str = "polygon",
) -> pd.DataFrame:
    """Seasonal precipitation per labelled geometry (temporal then spatial)."""
    cached = _cached_csv(config.precip_features_path, config.reuse_ee_features)
    if cached is not None:
        return cached
    tot = precip_image(config).reduceRegions(
        reducer=_image_reducer(sample_geometry),
        collection=plots,
        scale=config.scale,
    )
    data = [
        {
            id_field: f["properties"][id_field],
            "precipitation": _sampled_value(f["properties"], "precipitation"),
        }
        for f in tot.getInfo()["features"]
    ]
    df = pd.DataFrame(data)
    path = config.precip_features_path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Precipitation features -> {path} ({df.shape})")
    return df


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
