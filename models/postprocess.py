"""Post-processing: cropland masking, thresholding and visualisation.

Generalises notebook cells 102-111: convert the probability raster into a
final binary crop map by masking to cropland (GFSAD from Earth Engine) and
applying a probability threshold, plus helpers to visualise the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import ee
import numpy as np

from src.config import PipelineConfig


def build_cropland_image(aoi: ee.Geometry, config: PipelineConfig) -> ee.Image:
    """Binary cropland mask from GFSAD (notebook cell 108).

    ``USGS/GFSAD1000_V1`` is an Image (cell 108 used ImageCollection, which
    fails). Classes 2-6 are cropland; 0=water, 1=non-cropland. The 40% cutoff
    in cell 111 applies only to percent-style rasters (``cropland_classes``
    empty).
    """
    img = ee.Image(config.cropland_dataset).select(config.cropland_band)
    classes = list(config.cropland_classes or [])
    if classes:
        mask = img.remap(classes, [1] * len(classes), 0)
    else:
        mask = img.gte(config.cropland_threshold)
    return mask.rename("cropland").clip(aoi).toUint8()


def export_cropland_mask_local(
    aoi: ee.Geometry,
    config: PipelineConfig,
    path: Optional[str | Path] = None,
) -> Path:
    """Download the AOI cropland mask as a GeoTIFF into ``data/outputs/``."""
    from models.inference import download_ee_geotiff

    img = build_cropland_image(aoi, config)
    return download_ee_geotiff(img, aoi, Path(path or config.cropland_map_path), config)


def export_cropland_mask_to_drive(
    aoi: ee.Geometry,
    config: PipelineConfig,
    folder: str = "crop_mapping",
) -> ee.batch.Task:
    """Start a Drive export of the cropland mask (county-scale AOIs)."""
    task = ee.batch.Export.image.toDrive(
        image=build_cropland_image(aoi, config),
        description=f"{config.slug}_cropland_mask",
        folder=folder,
        fileNamePrefix=f"{config.slug}_cropland_mask",
        region=aoi,
        scale=config.scale,
        maxPixels=int(1e13),
        fileFormat="GeoTIFF",
    )
    task.start()
    print("Cropland-mask export task started:", task.id)
    return task


def _align_to_reference(src_path: Path, ref_profile: dict, height: int, width: int) -> np.ndarray:
    """Read ``src_path`` and resample it onto the reference raster grid."""
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(src_path) as src:
        data = src.read(1)
        if (
            src.height == height
            and src.width == width
            and src.transform == ref_profile["transform"]
            and src.crs == ref_profile.get("crs")
        ):
            return data
        aligned = np.zeros((height, width), dtype=data.dtype)
        reproject(
            source=data,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.nearest,
        )
        return aligned


def _non_cropland_mask(cropland: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """True where the pixel should be removed as non-cropland."""
    finite = np.isfinite(cropland)
    values = cropland[finite]
    if values.size == 0:
        return np.ones(cropland.shape, dtype=bool)
    unique = set(np.unique(values).tolist())
    if unique.issubset({0, 1}):
        return cropland == 0
    classes = list(config.cropland_classes or [])
    if classes:
        return ~np.isin(cropland, classes)
    return cropland < config.cropland_threshold


def threshold_probability(
    config: PipelineConfig,
    probability_tif: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> Path:
    """Binarise the probability raster at ``threshold`` into a classified map."""
    import rasterio

    probability_tif = Path(probability_tif or config.probability_map_path)
    out_path = Path(out_path or config.classified_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = config.probability_threshold if threshold is None else threshold

    with rasterio.open(probability_tif) as src:
        probs = src.read(1)
        profile = src.profile

    classified = np.where(np.isnan(probs), 0, (probs >= threshold).astype("uint8"))
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(classified.astype("uint8"), 1)
    return out_path


def apply_cropland_mask(
    config: PipelineConfig,
    probability_tif: Optional[str | Path] = None,
    cropland_tif: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    """Mask the probability raster to cropland then threshold to a final map.

    ``cropland_tif`` is resampled onto the probability grid when needed.
    Binary 0/1 masks (from ``build_cropland_image``) drop zeros; GFSAD class
    rasters keep ``cropland_classes``; percent rasters keep values at or
    above ``cropland_threshold`` (notebook cell 111). If no cropland file is
    provided, only the probability threshold is applied.
    """
    import rasterio

    probability_tif = Path(probability_tif or config.probability_map_path)
    out_path = Path(out_path or config.classified_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(probability_tif) as src:
        probs = src.read(1)
        profile = src.profile
        height, width = src.height, src.width

    masked = probs.copy()
    if cropland_tif is not None:
        cropland = _align_to_reference(Path(cropland_tif), profile, height, width)
        masked[_non_cropland_mask(cropland, config)] = np.nan

    classified = np.where(
        np.isnan(masked), 0, (masked >= config.probability_threshold).astype("uint8")
    )
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(classified.astype("uint8"), 1)
    return out_path


def plot_maps(
    feature_tif: str | Path,
    probability_tif: str | Path,
    titles: Tuple[str, str] = ("Feature (band 1)", "Predicted probability"),
    save_path: Optional[str | Path] = None,
):
    """Side-by-side plot of a feature band and the probability raster."""
    import matplotlib.pyplot as plt
    import rasterio

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    with rasterio.open(feature_tif) as fsrc:
        ax1.imshow(fsrc.read(1), cmap="Greens")
    ax1.set_title(titles[0])
    with rasterio.open(probability_tif) as psrc:
        im = ax2.imshow(psrc.read(1), cmap="Wistia")
    ax2.set_title(titles[1])
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Four-panel pipeline figure
# ---------------------------------------------------------------------------
_FEATURE_BAND_PREF = ("GCVI_mean", "NDVI_mean", "GCVI_constant", "precipitation")


def _load_boundary_gdf(config: PipelineConfig):
    import geopandas as gpd

    path = config.resolve_path(config.boundary_path)
    if path is not None and path.exists():
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=config.working_epsg)
        gdf = gdf.to_crs(epsg=config.working_epsg)
        union = gdf.union_all()
        return gpd.GeoDataFrame({"id": [0]}, geometry=[union], crs=gdf.crs)
    return None


def _read_raster(path: Path):
    import rasterio
    from rasterio.plot import plotting_extent

    with rasterio.open(path) as src:
        data = src.read()
        extent = plotting_extent(src)
        crs = src.crs
        descriptions = list(src.descriptions) if src.descriptions else []
    return data, extent, crs, descriptions


def _feature_band(data, descriptions, config: PipelineConfig) -> Tuple[np.ndarray, str]:
    names = [d for d in descriptions if d] or list(config.feature_columns)
    n_bands = data.shape[0]
    names = names[:n_bands]
    for preferred in _FEATURE_BAND_PREF:
        if preferred in names:
            idx = names.index(preferred)
            return data[idx], preferred
    return data[0], names[0] if names else "band 1"


def _panel_extent(config: PipelineConfig, raster_extent, boundary_gdf):
    if config.aoi_bbox is not None:
        w, s, e, n = config.aoi_bbox
        pad = max((e - w), (n - s)) * 0.08
        return (w - pad, e + pad, s - pad, n + pad)
    if raster_extent is not None:
        return raster_extent
    if boundary_gdf is not None and not boundary_gdf.empty:
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
        return (minx, maxx, miny, maxy)
    return None


def _style_ax(ax, title: str, extent=None):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _placeholder(ax, message: str):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="0.4",
            transform=ax.transAxes, wrap=True)


def plot_pipeline_stages(
    config: PipelineConfig,
    save_path: Optional[str | Path] = None,
    feature_band: Optional[str] = None,
    dpi: int = 200,
) -> Optional[Path]:
    """Render a 4-panel PNG: survey, feature layer, probability, classified map.

    Missing rasters become labelled placeholders so the figure can be produced
    as soon as the survey exists. Notebook cell 107 used Greens / Wistia; the
    classified panel is the cropland-masked map from cell 111.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from preprocessing.survey import prepare_survey

    save_path = Path(save_path or config.pipeline_figure_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    survey = prepare_survey(config)
    crop = config.target_crop_names[0] if config.target_crop_names else "target crop"
    boundary = _load_boundary_gdf(config)

    feature_arr = prob_arr = class_arr = None
    feature_ext = prob_ext = class_ext = None
    feature_label = "Feature"

    if config.feature_image_path.exists():
        data, feature_ext, _, descriptions = _read_raster(config.feature_image_path)
        if feature_band and feature_band in (list(descriptions) or config.feature_columns):
            names = [d for d in descriptions if d] or list(config.feature_columns)
            feature_arr = data[names.index(feature_band)]
            feature_label = feature_band
        else:
            feature_arr, feature_label = _feature_band(data, descriptions, config)
    if config.probability_map_path.exists():
        data, prob_ext, _, _ = _read_raster(config.probability_map_path)
        prob_arr = data[0]
    if config.classified_map_path.exists():
        data, class_ext, _, _ = _read_raster(config.classified_map_path)
        class_arr = data[0]

    extent = _panel_extent(config, feature_ext or prob_ext or class_ext, boundary)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    ax_survey, ax_feat, ax_prob, ax_class = axes.ravel()
    fig.suptitle(
        f"{config.gaul_lookup_name} {crop} mapping  ·  survey → features → probability → post-process",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # 1. Survey in boundary
    if boundary is not None:
        boundary.plot(ax=ax_survey, facecolor="#e8efe4", edgecolor="#4a5d3a", linewidth=1.2)
    positives = survey[survey[config.target_column] == 1]
    negatives = survey[survey[config.target_column] != 1]
    if not negatives.empty:
        negatives.plot(ax=ax_survey, facecolor="#c4a574", edgecolor="#6b4f2a",
                       linewidth=0.4, alpha=0.9)
    if not positives.empty:
        positives.plot(ax=ax_survey, facecolor="#2f6b3a", edgecolor="#14351c",
                       linewidth=0.4, alpha=0.9)
    if config.aoi_bbox is not None:
        w, s, e, n = config.aoi_bbox
        ax_survey.add_patch(Rectangle((w, s), e - w, n - s, fill=False,
                                      linestyle="--", edgecolor="#c0392b", linewidth=1.2))
    legend_items = [
        Patch(facecolor="#e8efe4", edgecolor="#4a5d3a", label="County boundary"),
        Patch(facecolor="#2f6b3a", edgecolor="#14351c", label=f"{crop} (survey)"),
        Patch(facecolor="#c4a574", edgecolor="#6b4f2a", label="Other crop (survey)"),
    ]
    if config.aoi_bbox is not None:
        legend_items.append(Line2D([0], [0], color="#c0392b", linestyle="--", label="Inference AOI"))
    ax_survey.legend(handles=legend_items, loc="lower left", fontsize=8, framealpha=0.92)
    _style_ax(
        ax_survey,
        f"1. Survey plots in boundary  (n={len(survey)}, {crop}={int(survey[config.target_column].sum())})",
        extent,
    )

    # 2. Feature layer
    if feature_arr is not None:
        finite = feature_arr[np.isfinite(feature_arr)]
        vmin, vmax = (np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)) if finite.size else (None, None)
        im = ax_feat.imshow(
            np.ma.masked_invalid(feature_arr), cmap="Greens", origin="upper",
            extent=feature_ext, vmin=vmin, vmax=vmax,
        )
        if boundary is not None:
            boundary.boundary.plot(ax=ax_feat, color="#4a5d3a", linewidth=0.8)
        fig.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
        _style_ax(ax_feat, f"2. Feature layer  ({feature_label})", extent)
    else:
        _placeholder(ax_feat, "Feature GeoTIFF not produced yet\n(run export-features / all)")
        ax_feat.set_title("2. Feature layer")

    # 3. Probability
    if prob_arr is not None:
        im = ax_prob.imshow(
            np.ma.masked_invalid(prob_arr), cmap="Wistia", origin="upper",
            extent=prob_ext, vmin=0, vmax=1,
        )
        if boundary is not None:
            boundary.boundary.plot(ax=ax_prob, color="#4a5d3a", linewidth=0.8)
        fig.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04, label=f"P({crop})")
        _style_ax(ax_prob, "3. Predicted probability", extent)
    else:
        _placeholder(ax_prob, "Probability raster not produced yet\n(run infer / all)")
        ax_prob.set_title("3. Predicted probability")

    # 4. Post-processed classified map
    if class_arr is not None:
        cmap = ListedColormap(["#f4f1ea", "#d4a017"])
        ax_class.imshow(
            class_arr, cmap=cmap, origin="upper", extent=class_ext, vmin=0, vmax=1,
        )
        if boundary is not None:
            boundary.boundary.plot(ax=ax_class, color="#4a5d3a", linewidth=0.8)
        ax_class.legend(
            handles=[
                Patch(facecolor="#f4f1ea", edgecolor="#ccc", label="Other / masked"),
                Patch(facecolor="#d4a017", edgecolor="#8a6a00",
                      label=f"{crop} (p ≥ {config.probability_threshold})"),
            ],
            loc="lower left", fontsize=8, framealpha=0.92,
        )
        _style_ax(
            ax_class,
            "4. Post-processed map  (cropland-masked, thresholded)",
            extent,
        )
    else:
        _placeholder(ax_class, "Classified map not produced yet\n(run infer / all)")
        ax_class.set_title("4. Post-processed map")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
