"""Post-processing: cropland masking, thresholding and visualisation.

Generalises notebook cells 102-111: convert the probability raster into a
final binary crop map by masking to cropland and applying a probability
threshold, plus helpers to visualise the result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.config import PipelineConfig


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

    ``cropland_tif`` should be a raster aligned to the probability raster where
    values below ``config.cropland_threshold`` are treated as non-cropland and
    removed. If not provided, only the probability threshold is applied.
    """
    import rasterio

    probability_tif = Path(probability_tif or config.probability_map_path)
    out_path = Path(out_path or config.classified_map_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(probability_tif) as src:
        probs = src.read(1)
        profile = src.profile

    masked = probs.copy()
    if cropland_tif is not None:
        with rasterio.open(cropland_tif) as csrc:
            cropland = csrc.read(1)
        non_cropland = cropland < config.cropland_threshold
        masked[non_cropland] = np.nan

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
