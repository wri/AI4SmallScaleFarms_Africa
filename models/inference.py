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
from preprocessing.sentinel import get_masked_collection


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

    start = config.precip_start_date or config.start_date
    end = config.precip_end_date or config.end_date
    precip = (
        ee.ImageCollection(config.precip_dataset)
        .filterDate(start, end)
        .filterBounds(aoi)
        .select("precipitation")
        .mean()
        .rename("precipitation")
    )

    feature_img = s2_feature_img.addBands(terrain).addBands(precip).clip(aoi)
    select_cols = [c for c in config.feature_columns if c in feature_img.bandNames().getInfo()]
    return feature_img.select(select_cols)


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
