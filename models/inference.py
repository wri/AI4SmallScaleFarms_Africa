"""Wall-to-wall inference over an AOI.

Generalises notebook cells 92-101: build a pixel-level feature image for the
whole AOI, export it as a GeoTIFF, then apply the trained classifier to produce
a per-pixel probability raster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import ee
import numpy as np

from src.config import PipelineConfig
from preprocessing.features import precip_image
from preprocessing.sentinel import get_masked_collection


def _aoi_bounds(aoi: ee.Geometry) -> tuple[float, float, float, float]:
    coords = aoi.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), min(lats), max(lons), max(lats)


def _tile_bboxes(
    west: float, south: float, east: float, north: float, tile_deg: float
) -> list[tuple[float, float, float, float]]:
    tiles = []
    y = south
    while y < north - 1e-9:
        y2 = min(y + tile_deg, north)
        x = west
        while x < east - 1e-9:
            x2 = min(x + tile_deg, east)
            tiles.append((x, y, x2, y2))
            x = x2
        y = y2
    return tiles or [(west, south, east, north)]


def _get_download_url(image: ee.Image, region: ee.Geometry, config: PipelineConfig) -> str:
    return image.getDownloadURL(
        {
            "scale": config.scale,
            "crs": f"EPSG:{config.working_epsg}",
            "region": region,
            "format": "GEO_TIFF",
        }
    )


def _retrieve_url(url: str, path: Path) -> Path:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Download failed: {path}")
    return path


def _mosaic_geotiffs(tile_paths: list[Path], out_path: Path) -> Path:
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, transform = merge(srcs)
        profile = srcs[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for src in srcs:
            src.close()
    return out_path


def download_ee_geotiff(
    image: ee.Image,
    aoi: ee.Geometry,
    path: str | Path,
    config: PipelineConfig,
) -> Path:
    """Download an EE image to a local GeoTIFF, tiling if the AOI is too large."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        url = _get_download_url(image, aoi, config)
        print(f"Downloading -> {path}")
        _retrieve_url(url, path)
        print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
        return path
    except Exception as exc:
        print(f"Whole-AOI download failed ({exc}); tiling at {config.export_tile_deg} deg")

    west, south, east, north = _aoi_bounds(aoi)
    bboxes = _tile_bboxes(west, south, east, north, config.export_tile_deg)
    print(f"Downloading {len(bboxes)} tiles")
    tile_dir = path.parent / f"{path.stem}_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[Path] = []
    for i, (w, s, e, n) in enumerate(bboxes, start=1):
        tile_geom = ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)
        clipped = image.clip(tile_geom)
        tile_path = tile_dir / f"tile_{i:03d}.tif"
        print(f"  tile {i}/{len(bboxes)} [{w:.4f},{s:.4f},{e:.4f},{n:.4f}]")
        url = _get_download_url(clipped, tile_geom, config)
        _retrieve_url(url, tile_path)
        tile_paths.append(tile_path)
    _mosaic_geotiffs(tile_paths, path)
    print(f"Mosaicked {len(tile_paths)} tiles -> {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def build_feature_image(
    aoi: ee.Geometry, config: PipelineConfig, bands: Optional[Sequence[str]] = None
) -> ee.Image:
    """Build the multi-band feature image matching the training features.

    Combines per-band harmonic-regression coefficients, terrain (elevation,
    slope, aspect) and seasonal precipitation, then selects exactly the
    ``config.feature_columns`` so the band order matches training.
    """
    import harmonics

    bands = list(bands or config.regression_bands)
    coll = get_masked_collection(aoi, config, bands=config.bands).select(bands)

    coef_images = [
        harmonics.run_std_regressions(
            coll.select([band]), [band],
            refdate=config.reference_date, nharmonics=config.n_harmonics,
        )
        for band in bands
    ]
    s2_feature_img = ee.Image.cat(coef_images)

    srtm = ee.Image(config.srtm_asset)
    terrain = ee.Terrain.products(srtm).select(["elevation", "slope", "aspect"])

    precip = precip_image(config, aoi=aoi)

    feature_img = s2_feature_img.addBands(terrain).addBands(precip).clip(aoi)
    # Select in training order without a getInfo() round-trip (that call would
    # force EE to materialise band names and is slow on large AOIs).
    return feature_img.select(list(config.feature_columns))


def export_feature_image_to_drive(
    feature_img: ee.Image,
    aoi: ee.Geometry,
    config: PipelineConfig,
    folder: str = "crop_mapping",
) -> ee.batch.Task:
    """Start a Drive export of the feature image as a GeoTIFF."""
    task = ee.batch.Export.image.toDrive(
        image=feature_img.toFloat(),
        description=f"{config.slug}_pixel_features_for_rf",
        folder=folder,
        fileNamePrefix=f"{config.slug}_pixel_features_for_rf",
        region=aoi,
        scale=config.scale,
        maxPixels=int(1e13),
        fileFormat="GeoTIFF",
    )
    task.start()
    print("Export task started:", task.id)
    return task


def export_feature_image_local(
    feature_img: ee.Image,
    aoi: ee.Geometry,
    config: PipelineConfig,
    path: Optional[str | Path] = None,
) -> Path:
    """Download the AOI feature image as a GeoTIFF into ``data/outputs/``.

    Tries a single ``getDownloadURL`` first; county-scale AOIs are tiled and
    mosaicked when that request exceeds Earth Engine's download limit.
    """
    return download_ee_geotiff(
        feature_img.toFloat(), aoi, Path(path or config.feature_image_path), config
    )


def predict_probability_raster(
    model,
    config: PipelineConfig,
    feature_tif: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    """Apply ``model`` to a feature GeoTIFF, writing a probability raster.

    Reads the exported feature image, reshapes to one row per pixel, predicts
    the positive-class probability for finite pixels, and writes a single-band
    float32 GeoTIFF (NaN where inputs were incomplete).
    """
    import rasterio

    feature_tif = Path(feature_tif or config.feature_image_path)
    out_path = Path(out_path or config.probability_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_tif) as src:
        arr = src.read()  # (bands, height, width)
        profile = src.profile
    n_bands, height, width = arr.shape

    X_pixels = arr.reshape(n_bands, height * width).T
    valid = np.all(np.isfinite(X_pixels), axis=1)

    probs = np.full((height * width,), np.nan, dtype="float32")
    if valid.any():
        probs[valid] = model.predict_proba(X_pixels[valid])[:, 1]
    prob_raster = probs.reshape(height, width)

    profile.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob_raster.astype("float32"), 1)
    return out_path
