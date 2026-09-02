"""Wall-to-wall inference over an AOI.

County-wide harmonic + vegetation-index features are computed on Earth Engine.
Interactive ``getDownloadURL`` cannot hold that graph in memory, so county
exports are materialized with a batch ``Export.image.toAsset`` first, then
downloaded as GeoTIFF tiles that intersect the county polygon (not its
envelope). Small AOIs (smoke tests) still download in one request.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import ee
import numpy as np

from src.config import PipelineConfig
from preprocessing.features import precip_image
from preprocessing.sentinel import get_masked_collection

_EE_DOWNLOAD_LIMIT_BYTES = 50_331_648


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


def _estimate_geotiff_bytes(
    west: float, south: float, east: float, north: float, scale: int, n_bands: int
) -> int:
    m_per_deg = 111_320.0
    width_px = max((east - west) * m_per_deg / scale, 1.0)
    height_px = max((north - south) * m_per_deg / scale, 1.0)
    return int(width_px * height_px * n_bands * 4)


def _http_error_text(exc: BaseException) -> str:
    import urllib.error

    if not isinstance(exc, urllib.error.HTTPError):
        return str(exc)
    body = exc.read().decode("utf-8", errors="replace").strip()
    body = " ".join(body.split())
    if len(body) > 400:
        body = body[:400] + "..."
    return f"HTTP {exc.code} {exc.reason}: {body or '(empty body)'}"


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
    import urllib.error
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, str(path))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(_http_error_text(exc)) from exc
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Download failed: {path}")
    return path


def _set_band_descriptions(path: Path, names: Sequence[str]) -> None:
    import rasterio

    if not names:
        return
    with rasterio.open(path, "r+") as dst:
        for i, name in enumerate(names, start=1):
            if i <= dst.count:
                dst.set_band_description(i, str(name))


def _mosaic_geotiffs(
    tile_paths: list[Path], out_path: Path, band_names: Optional[Sequence[str]] = None
) -> Path:
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
            if band_names:
                for i, name in enumerate(band_names, start=1):
                    if i <= dst.count:
                        dst.set_band_description(i, str(name))
    finally:
        for src in srcs:
            src.close()
    return out_path


def _tile_stem(west: float, south: float, east: float, north: float) -> str:
    return f"tile_{west:.4f}_{south:.4f}_{east:.4f}_{north:.4f}".replace("-", "m")


def _aoi_shapely(aoi: ee.Geometry):
    from shapely.geometry import shape

    info = aoi.getInfo()
    if info.get("type") == "GeometryCollection":
        geojson = {"type": "GeometryCollection", "geometries": info.get("geometries", [])}
    else:
        geojson = {"type": info.get("type"), "coordinates": info.get("coordinates")}
    return shape(geojson)


def _bbox_intersects(aoi_shape, west: float, south: float, east: float, north: float) -> bool:
    from shapely.geometry import box

    return aoi_shape.intersects(box(west, south, east, north))


def _is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in ("429", "rate limit", "user request limit", "quota"))


def _asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def _wait_for_task(task: ee.batch.Task, poll_s: int) -> None:
    while True:
        status = task.status()
        state = status.get("state")
        extra = status.get("error_message") or status.get("description") or ""
        print(f"  EE batch task {task.id}: {state}" + (f" ({extra})" if extra and state != "COMPLETED" else ""))
        if state == "COMPLETED":
            return
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"Earth Engine export {task.id} {state}: "
                f"{status.get('error_message') or status}"
            )
        time.sleep(max(poll_s, 15))


def materialize_feature_image(
    image: ee.Image, aoi: ee.Geometry, config: PipelineConfig
) -> ee.Image:
    """Run county-wide harmonic compute as a batch task, return the stored image.

    Interactive downloads of this graph hit ``User memory limit exceeded``.
    Batch ``Export.image.toAsset`` has a 12-hour window and much more memory.
    """
    asset_id = config.feature_asset_id
    if config.reuse_ee_features and _asset_exists(asset_id):
        print(f"Reusing EE asset {asset_id}")
        return ee.Image(asset_id)

    if _asset_exists(asset_id):
        print(f"Deleting existing EE asset {asset_id}")
        ee.data.deleteAsset(asset_id)

    print(f"Batch-exporting county-wide features -> {asset_id}")
    print("This is one Earth Engine batch task; it can take a while.")
    parent = "/".join(asset_id.split("/")[:-1])
    if not _asset_exists(parent):
        try:
            ee.data.createAsset({"type": "Folder"}, parent)
        except Exception:
            pass
    task = ee.batch.Export.image.toAsset(
        image=image.toFloat(),
        description=f"{config.slug}_pixel_features_for_rf",
        assetId=asset_id,
        region=aoi,
        scale=config.scale,
        crs=f"EPSG:{config.working_epsg}",
        maxPixels=int(1e13),
    )
    try:
        task.start()
    except Exception as exc:
        raise RuntimeError(
            f"Could not start Export.image.toAsset to {asset_id} ({exc}). "
            "Create an Assets root for this Cloud project in the Earth Engine "
            "Code Editor, or re-run with --drive."
        ) from exc
    _wait_for_task(task, config.export_poll_seconds)
    return ee.Image(asset_id)


def download_ee_geotiff(
    image: ee.Image,
    aoi: ee.Geometry,
    path: str | Path,
    config: PipelineConfig,
    n_bands: int = 1,
    band_names: Optional[Sequence[str]] = None,
) -> Path:
    """Download an EE image to a local GeoTIFF.

    Small AOIs use a single ``getDownloadURL``. Larger AOIs are tiled; only
    tiles that intersect the AOI polygon are requested. Existing tile files
    are reused.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export_img = image.reproject(crs=f"EPSG:{config.working_epsg}", scale=config.scale)
    west, south, east, north = _aoi_bounds(aoi)
    estimated = _estimate_geotiff_bytes(west, south, east, north, config.scale, n_bands)

    if estimated <= int(_EE_DOWNLOAD_LIMIT_BYTES * 0.85):
        try:
            print(f"Downloading -> {path}")
            url = _get_download_url(export_img.clip(aoi), aoi, config)
            _retrieve_url(url, path)
            _set_band_descriptions(path, band_names or [])
            print(f"Saved {path} ({path.stat().st_size / 1e6:.2f} MB)")
            return path
        except Exception as exc:
            print(f"Whole-AOI download failed ({exc}); tiling at {config.export_tile_deg} deg")
    else:
        print(
            f"AOI ~{estimated / 1e6:.0f} MB exceeds EE's 50 MB download cap; "
            f"tiling at {config.export_tile_deg} deg"
        )

    bboxes = _tile_bboxes(west, south, east, north, config.export_tile_deg)
    aoi_shape = _aoi_shapely(aoi)
    bboxes = [b for b in bboxes if _bbox_intersects(aoi_shape, *b)]
    print(f"Downloading {len(bboxes)} tiles that intersect the AOI")
    tile_dir = path.parent / f"{path.stem}_tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_paths: list[Path] = []
    for i, (w, s, e, n) in enumerate(bboxes, start=1):
        tile_path = tile_dir / f"{_tile_stem(w, s, e, n)}.tif"
        print(f"  tile {i}/{len(bboxes)} [{w:.4f},{s:.4f},{e:.4f},{n:.4f}]")
        if tile_path.exists() and tile_path.stat().st_size > 0:
            print(f"    reuse {tile_path.name} ({tile_path.stat().st_size / 1e6:.2f} MB)")
            tile_paths.append(tile_path)
            continue
        tile_geom = ee.Geometry.Rectangle([w, s, e, n], proj="EPSG:4326", geodesic=False)
        region = tile_geom.intersection(aoi, 1)
        delay = 8.0
        for attempt in range(4):
            try:
                url = _get_download_url(export_img.clip(region), tile_geom, config)
                _retrieve_url(url, tile_path)
                print(f"    saved {tile_path.name} ({tile_path.stat().st_size / 1e6:.2f} MB)")
                break
            except Exception as exc:
                print(f"    attempt {attempt + 1}/4 failed ({exc})")
                if _is_rate_limit_error(exc) and attempt < 3:
                    time.sleep(delay)
                    delay = min(delay * 2, 60.0)
                    continue
                raise
        tile_paths.append(tile_path)
    if not tile_paths:
        raise RuntimeError(f"No tiles downloaded for {path}")
    _mosaic_geotiffs(tile_paths, path, band_names=band_names)
    print(f"Mosaicked {len(tile_paths)} tiles -> {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def build_feature_image(
    aoi: ee.Geometry, config: PipelineConfig, bands: Optional[Sequence[str]] = None
) -> ee.Image:
    """County-wide feature image: VI harmonics, terrain, precipitation.

    Vegetation-index bands (and RDED4) are fit with harmonic regression over
    the county polygon. Seasonal ``{band}_mean`` comes from that same reducer.
    The result is clipped to the AOI polygon and ordered as
    ``config.feature_columns``.
    """
    from src.ee_utils import import_harmonics

    harmonics = import_harmonics()

    bands = list(bands or config.regression_bands)
    coll = get_masked_collection(aoi, config, bands=bands).select(bands)

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
        crs=f"EPSG:{config.working_epsg}",
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
    """Write the AOI feature image to ``data/outputs/`` as a GeoTIFF.

    County-scale graphs are materialized to an Earth Engine asset first, then
    tiled locally. Only tiles that intersect the county polygon are downloaded.
    """
    dest = Path(path or config.feature_image_path)
    if config.reuse_ee_features and dest.exists() and dest.stat().st_size > 0:
        print(f"Reusing cached feature GeoTIFF -> {dest}")
        return dest

    img = feature_img.toFloat()
    west, south, east, north = _aoi_bounds(aoi)
    estimated = _estimate_geotiff_bytes(
        west, south, east, north, config.scale, len(config.feature_columns)
    )
    if estimated > int(_EE_DOWNLOAD_LIMIT_BYTES * 0.85):
        img = materialize_feature_image(img, aoi, config)

    return download_ee_geotiff(
        img,
        aoi,
        dest,
        config,
        n_bands=len(config.feature_columns),
        band_names=list(config.feature_columns),
    )


def _expected_feature_names(model, config: PipelineConfig) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names]
    return list(config.feature_columns)


def _validate_feature_raster(src, model, config: PipelineConfig) -> None:
    """Fail loudly if the GeoTIFF does not match the trained model."""
    expected = _expected_feature_names(model, config)
    n_expected = int(getattr(model, "n_features_in_", len(expected)))
    if src.count != n_expected:
        raise ValueError(
            f"Feature GeoTIFF has {src.count} bands but the model expects "
            f"{n_expected}: {expected}. Rebuild the feature image with the same "
            f"feature_columns, n_harmonics and season used in training."
        )
    descriptions = [d or "" for d in (src.descriptions or ())]
    if any(descriptions) and list(descriptions[:n_expected]) != expected:
        raise ValueError(
            f"Feature GeoTIFF band names {list(descriptions[:n_expected])} do not "
            f"match the trained model {expected}. Band order must match training; "
            f"do not reuse a GeoTIFF built with a different feature recipe."
        )
    if not any(descriptions):
        print(
            "Warning: feature GeoTIFF has no band names; assuming band order "
            "matches the trained model."
        )


def predict_probability_raster(
    model,
    config: PipelineConfig,
    feature_tif: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    """Apply ``model`` to a feature GeoTIFF, writing a probability raster.

    Binary models write a single-band positive-class probability (Nyandarua).
    Multiclass models write one band per class, named and tagged so the
    classified map can be built with argmax. Incomplete pixels are NaN.
    """
    import rasterio

    from models.train import model_class_names

    feature_tif = Path(feature_tif or config.feature_image_path)
    out_path = Path(out_path or config.probability_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_tif) as src:
        _validate_feature_raster(src, model, config)
        arr = src.read()  # (bands, height, width)
        profile = src.profile
    n_bands, height, width = arr.shape

    X_pixels = arr.reshape(n_bands, height * width).T
    valid = np.all(np.isfinite(X_pixels), axis=1)

    class_names = model_class_names(model, config)
    class_codes = [int(c) for c in model.classes_]
    binary = config.is_binary

    if binary:
        probs = np.full((height * width,), np.nan, dtype="float32")
        if valid.any():
            # Positive class is index 1 when classes_ is [0, 1] / [other, maize].
            proba = model.predict_proba(X_pixels[valid])
            pos = 1 if proba.shape[1] > 1 else 0
            probs[valid] = proba[:, pos]
        raster = probs.reshape(1, height, width)
        band_names = [f"P({class_names[-1] if class_names else 'positive'})"]
        out_codes, out_names = class_codes, class_names
    else:
        n_classes = len(model.classes_)
        stack = np.full((n_classes, height * width), np.nan, dtype="float32")
        if valid.any():
            stack[:, valid] = model.predict_proba(X_pixels[valid]).T
        raster = stack.reshape(n_classes, height, width)
        band_names = [str(n) for n in class_names]
        out_codes, out_names = class_codes, class_names

    profile.update(count=raster.shape[0], dtype="float32", nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(raster.astype("float32"))
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(
            label_mode="binary" if binary else "multiclass",
            class_codes=",".join(str(c) for c in out_codes),
            class_names=",".join(out_names),
        )
    return out_path
