"""Post-processing: cropland masking, thresholding and visualisation.

Generalises notebook cells 102-111: convert the probability raster into a
final binary crop map by masking to cropland (GFSAD from Earth Engine) and
applying a probability threshold, plus helpers to visualise the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

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


def _legend_from_probability(probability_tif: Path, config: PipelineConfig):
    """Class codes/names stored as GeoTIFF tags, else metrics JSON."""
    import json
    import rasterio

    with rasterio.open(probability_tif) as src:
        tags = src.tags()
        count = src.count
        descriptions = [d or "" for d in (src.descriptions or ())]
    codes_raw = tags.get("class_codes") or ""
    names_raw = tags.get("class_names") or ""
    if codes_raw and names_raw:
        codes = [int(c) for c in codes_raw.split(",") if c]
        names = [n for n in names_raw.split(",") if n]
        if len(codes) == len(names):
            return codes, names
    if any(descriptions) and count > 1:
        return list(range(1, count + 1)), [d for d in descriptions]
    if config.metrics_path.exists():
        with open(config.metrics_path) as fh:
            payload = json.load(fh)
        codes = payload.get("class_codes")
        names = payload.get("class_names")
        if codes and names:
            return [int(c) for c in codes], [str(n) for n in names]
    return None, None


def _classified_from_probability_stack(
    probs: np.ndarray, codes: Sequence[int]
) -> np.ndarray:
    """Argmax over class probability bands -> integer class codes. 0 = nodata."""
    n_classes, height, width = probs.shape
    if not codes or len(codes) != n_classes:
        codes = list(range(1, n_classes + 1))
    valid = np.all(np.isfinite(probs), axis=0)
    pred_idx = np.argmax(probs, axis=0)
    classified = np.zeros((height, width), dtype="uint8")
    code_arr = np.array(codes, dtype="uint8")
    classified[valid] = code_arr[pred_idx[valid]]
    return classified


def threshold_probability(
    config: PipelineConfig,
    probability_tif: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
    threshold: Optional[float] = None,
) -> Path:
    """Turn the probability raster into a classified map.

    Binary: threshold ``P(positive)``. Multiclass: argmax over class bands.
    """
    import rasterio

    probability_tif = Path(probability_tif or config.probability_map_path)
    out_path = Path(out_path or config.classified_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    threshold = config.probability_threshold if threshold is None else threshold

    with rasterio.open(probability_tif) as src:
        probs = src.read()
        profile = src.profile
        n_bands = src.count

    if config.is_binary or n_bands == 1:
        band = probs[0]
        classified = np.where(np.isnan(band), 0, (band >= threshold).astype("uint8"))
    else:
        codes, _ = _legend_from_probability(probability_tif, config)
        if not codes or len(codes) != n_bands:
            codes = list(range(1, n_bands + 1))
        classified = _classified_from_probability_stack(probs, codes)

    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(classified.astype("uint8"), 1)
        dst.update_tags(**_class_tags(probability_tif, config))
    return out_path


def _class_tags(probability_tif: Path, config: PipelineConfig) -> dict:
    codes, names = _legend_from_probability(probability_tif, config)
    tags = {"label_mode": "binary" if config.is_binary else "multiclass"}
    if codes and names:
        tags["class_codes"] = ",".join(str(c) for c in codes)
        tags["class_names"] = ",".join(names)
    return tags


def apply_cropland_mask(
    config: PipelineConfig,
    probability_tif: Optional[str | Path] = None,
    cropland_tif: Optional[str | Path] = None,
    out_path: Optional[str | Path] = None,
) -> Path:
    """Mask predictions to cropland then write the classified map.

    Binary: drop non-cropland, then threshold probability.
    Multiclass: argmax class codes, then set non-cropland pixels to 0.
    """
    import rasterio

    probability_tif = Path(probability_tif or config.probability_map_path)
    out_path = Path(out_path or config.classified_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(probability_tif) as src:
        probs = src.read()
        profile = src.profile
        height, width = src.height, src.width
        n_bands = src.count

    non_crop = None
    if cropland_tif is not None:
        cropland = _align_to_reference(Path(cropland_tif), profile, height, width)
        non_crop = _non_cropland_mask(cropland, config)

    if config.is_binary or n_bands == 1:
        masked = probs[0].copy()
        if non_crop is not None:
            masked[non_crop] = np.nan
        classified = np.where(
            np.isnan(masked), 0, (masked >= config.probability_threshold).astype("uint8")
        )
    else:
        codes, _ = _legend_from_probability(probability_tif, config)
        if not codes or len(codes) != n_bands:
            codes = list(range(1, n_bands + 1))
        classified = _classified_from_probability_stack(probs, codes)
        if non_crop is not None:
            classified[non_crop] = 0

    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(classified.astype("uint8"), 1)
        dst.update_tags(**_class_tags(probability_tif, config))
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
_MASKED_COLOR = "#f4f1ea"
_CLASS_COLORS = {
    "maize": "#F2C14E",
    "wheat": "#E07A3D",
    "potatoes": "#8B5A2B",
    "grass": "#5B8C3E",
    "canola": "#C5D86D",
    "coffee": "#6B3A2A",
    "other": "#7D8A8C",
}
_TAB10 = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _class_color(name: str, index: int = 0) -> str:
    key = str(name).strip().lower()
    if key in _CLASS_COLORS:
        return _CLASS_COLORS[key]
    return _TAB10[index % len(_TAB10)]


def _class_cmap(names: Sequence[str]):
    from matplotlib.colors import ListedColormap

    colors = [_MASKED_COLOR] + [_class_color(n, i) for i, n in enumerate(names)]
    return ListedColormap(colors)


def _plot_class_raster(ax, arr, extent, names: Sequence[str]):
    n = max(len(names), 1)
    ax.imshow(
        arr, cmap=_class_cmap(names), origin="upper", extent=extent,
        vmin=0, vmax=n, interpolation="nearest",
    )


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
    """Render a 4-panel PNG: survey, feature layer, prediction, classified map.

    Missing rasters become labelled placeholders so the figure can be produced
    as soon as the survey exists. Binary runs keep the maize probability panel;
    multiclass runs show the argmax class map and name rare crops folded into
    ``other`` on the survey panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    from preprocessing.survey import (
        class_legend,
        is_point_survey,
        other_class_breakdown,
        prepare_survey,
    )

    save_path = Path(save_path or config.pipeline_figure_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    survey = prepare_survey(config)
    crop = config.target_crop_names[0] if config.target_crop_names else "target crop"
    boundary = _load_boundary_gdf(config)
    legend = class_legend(survey, config)
    class_codes, class_names = legend if legend else ([0, 1], ["other", crop])
    map_names = [crop] if config.is_binary else list(class_names)
    point_survey = is_point_survey(survey)

    feature_arr = class_arr = None
    prob_stack = None
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
        prob_stack, prob_ext, _, _ = _read_raster(config.probability_map_path)
        tag_codes, tag_names = _legend_from_probability(config.probability_map_path, config)
        if tag_names:
            map_names = list(tag_names)
        if tag_codes:
            class_codes = list(tag_codes)
    if config.classified_map_path.exists():
        data, class_ext, _, _ = _read_raster(config.classified_map_path)
        class_arr = data[0]

    extent = _panel_extent(config, feature_ext or prob_ext or class_ext, boundary)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    ax_survey, ax_feat, ax_prob, ax_class = axes.ravel()
    if config.is_binary:
        fig.suptitle(
            f"{config.gaul_lookup_name} {crop} mapping  ·  survey → features → probability → post-process",
            fontsize=13, fontweight="bold", y=0.98,
        )
    else:
        fig.suptitle(
            f"{config.gaul_lookup_name} crop-type mapping  ·  survey → features → class map → post-process",
            fontsize=13, fontweight="bold", y=0.98,
        )

    # 1. Survey in boundary
    if boundary is not None:
        boundary.plot(ax=ax_survey, facecolor="#e8efe4", edgecolor="#4a5d3a", linewidth=1.2)

    legend_items = [
        Patch(facecolor="#e8efe4", edgecolor="#4a5d3a", label="County boundary"),
    ]
    if config.is_binary:
        positives = survey[survey[config.target_column] == 1]
        negatives = survey[survey[config.target_column] != 1]
        plot_kw = dict(markersize=18, linewidth=0.4, alpha=0.9) if point_survey else dict(linewidth=0.4, alpha=0.9)
        if not negatives.empty:
            negatives.plot(
                ax=ax_survey, facecolor="#c4a574", edgecolor="#6b4f2a", **plot_kw
            )
        if not positives.empty:
            positives.plot(
                ax=ax_survey, facecolor="#2f6b3a", edgecolor="#14351c", **plot_kw
            )
        legend_items.extend([
            Patch(facecolor="#2f6b3a", edgecolor="#14351c", label=f"{crop} (survey)"),
            Patch(facecolor="#c4a574", edgecolor="#6b4f2a", label="Other crop (survey)"),
        ])
        survey_title = (
            f"1. Survey {'points' if point_survey else 'plots'} in boundary  "
            f"(n={len(survey)}, {crop}={int(survey[config.target_column].sum())})"
        )
    else:
        name_col = config.class_name_column
        for i, name in enumerate(class_names):
            subset = survey[survey[name_col] == name]
            if subset.empty:
                continue
            color = _class_color(name, i)
            n = len(subset)
            if point_survey:
                subset.plot(
                    ax=ax_survey, color=color, markersize=18,
                    edgecolor="#333333", linewidth=0.3, alpha=0.9,
                )
            else:
                subset.plot(
                    ax=ax_survey, facecolor=color, edgecolor="#333333",
                    linewidth=0.4, alpha=0.9,
                )
            legend_items.append(
                Patch(facecolor=color, edgecolor="#333333", label=f"{name} (n={n})")
            )
        survey_title = (
            f"1. Survey {'points' if point_survey else 'plots'} in boundary  (n={len(survey)})"
        )
        breakdown = other_class_breakdown(survey, config)
        if breakdown:
            lines = [f"Other (n < {config.min_class_count}):"]
            lines.extend(f"  {crop_name}: {count}" for crop_name, count in
                         sorted(breakdown.items(), key=lambda kv: (-kv[1], kv[0])))
            ax_survey.text(
                0.98, 0.98, "\n".join(lines), transform=ax_survey.transAxes,
                va="top", ha="right", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#ccc"),
            )

    if config.aoi_bbox is not None:
        w, s, e, n = config.aoi_bbox
        ax_survey.add_patch(Rectangle((w, s), e - w, n - s, fill=False,
                                      linestyle="--", edgecolor="#c0392b", linewidth=1.2))
        legend_items.append(Line2D([0], [0], color="#c0392b", linestyle="--", label="Inference AOI"))
    ax_survey.legend(handles=legend_items, loc="lower left", fontsize=7, framealpha=0.92)
    _style_ax(ax_survey, survey_title, extent)

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

    # 3. Probability (binary) or predicted class before cropland mask
    if prob_stack is not None:
        if config.is_binary or prob_stack.shape[0] == 1:
            im = ax_prob.imshow(
                np.ma.masked_invalid(prob_stack[0]), cmap="Wistia", origin="upper",
                extent=prob_ext, vmin=0, vmax=1,
            )
            if boundary is not None:
                boundary.boundary.plot(ax=ax_prob, color="#4a5d3a", linewidth=0.8)
            fig.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04, label=f"P({crop})")
            _style_ax(ax_prob, "3. Predicted probability", extent)
        else:
            pred = _classified_from_probability_stack(prob_stack, class_codes)
            _plot_class_raster(ax_prob, pred, prob_ext, map_names)
            if boundary is not None:
                boundary.boundary.plot(ax=ax_prob, color="#4a5d3a", linewidth=0.8)
            ax_prob.legend(
                handles=[Patch(facecolor=_MASKED_COLOR, edgecolor="#ccc", label="No data")]
                + [Patch(facecolor=_class_color(n, i), edgecolor="#333", label=n) for i, n in enumerate(map_names)],
                loc="lower left", fontsize=7, framealpha=0.92,
            )
            _style_ax(ax_prob, "3. Predicted class  (pre-mask; probabilities stored as bands)", extent)
    else:
        _placeholder(ax_prob, "Probability raster not produced yet\n(run infer / all)")
        ax_prob.set_title("3. Predicted probability")

    # 4. Post-processed classified map
    if class_arr is not None:
        if config.is_binary:
            cmap = ListedColormap(["#f4f1ea", "#d4a017"])
            ax_class.imshow(
                class_arr, cmap=cmap, origin="upper", extent=class_ext, vmin=0, vmax=1,
            )
            class_handles = [
                Patch(facecolor="#f4f1ea", edgecolor="#ccc", label="Other / masked"),
                Patch(facecolor="#d4a017", edgecolor="#8a6a00",
                      label=f"{crop} (p ≥ {config.probability_threshold})"),
            ]
            class_title = "4. Post-processed map  (cropland-masked, thresholded)"
        else:
            _plot_class_raster(ax_class, class_arr, class_ext, map_names)
            class_handles = [Patch(facecolor=_MASKED_COLOR, edgecolor="#ccc", label="Non-cropland / nodata")]
            class_handles.extend(
                Patch(facecolor=_class_color(n, i), edgecolor="#333", label=n)
                for i, n in enumerate(map_names)
            )
            class_title = "4. Post-processed map  (cropland-masked class)"
        if boundary is not None:
            boundary.boundary.plot(ax=ax_class, color="#4a5d3a", linewidth=0.8)
        ax_class.legend(handles=class_handles, loc="lower left", fontsize=7, framealpha=0.92)
        _style_ax(ax_class, class_title, extent)
    else:
        _placeholder(ax_class, "Classified map not produced yet\n(run infer / all)")
        ax_class.set_title("4. Post-processed map")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path
