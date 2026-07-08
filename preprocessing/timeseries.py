"""Per-pixel time-series extraction and harmonic-regression fitting.

Generalises notebook cells 22-49: pulling a band's value at a point across an
ImageCollection, and fitting/plotting a harmonic-regression smoothing of the
resulting time series. These utilities are point-based diagnostics used to
inspect the temporal signal before the wall-to-wall feature engineering.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional, Tuple

import ee
import numpy as np
import pandas as pd

from src.config import PipelineConfig


# ---------------------------------------------------------------------------
# List helpers over an ImageCollection
# ---------------------------------------------------------------------------
def create_timestamp_list(imgcoll: ee.ImageCollection) -> List[str]:
    """List of formatted timestamps for every image in the collection."""
    return (
        ee.List(imgcoll.aggregate_array("system:time_start"))
        .map(lambda t: ee.Date(t).format("Y-MM-dd HH:mm:ss"))
        .getInfo()
    )


def create_band_list(imgcoll: ee.ImageCollection, suffix: str) -> List[str]:
    """List of per-image band ids (with ``suffix``) as produced by ``toBands()``."""
    band_list = ee.List(imgcoll.aggregate_array("system:index")).getInfo()
    return [band + suffix for band in band_list]


def sample_lists(items: list, every_x: int) -> list:
    """Down-sample a list to every ``every_x``-th item, preserving order."""
    return items[0::every_x]


def extract_pixel_vals_dates(
    imgcoll: ee.ImageCollection,
    band: str,
    ee_point: ee.Geometry,
    band_list: Optional[List[str]] = None,
    time_list: Optional[List[str]] = None,
    sample: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Extract values of ``band`` at ``ee_point`` for each unmasked image.

    Returns a DataFrame with columns ``[datetime, <band>, band_id]``.
    """
    def log(*args):
        if verbose:
            print(*args)

    log("selecting band of interest", band, "from imagecollection")
    band_coll = imgcoll.select(band)

    log("filtering for only images containing point")
    point_coll = band_coll.filterBounds(ee_point)

    if not band_list:
        suffix = "_" + band
        log("adding suffix", suffix, "to band name and creating list of bands")
        band_list = create_band_list(point_coll, suffix)
    log(f"number of ids in list: {len(band_list)} | first 5: {band_list[0:5]}")

    if not time_list:
        log("creating list of timestamps")
        time_list = create_timestamp_list(point_coll)
    log(f"number of dates in list: {len(time_list)} | first 5: {time_list[0:5]}")

    if sample:
        band_list = sample_lists(band_list, sample)
        time_list = sample_lists(time_list, sample)

    point_img = point_coll.toBands()
    rows = []
    for idx, band_id in enumerate(band_list):
        img = point_img.select(band_id)
        if img.mask().reduceRegion(ee.Reducer.first(), ee_point).getInfo()[band_id] != 0:
            time_start = time_list[idx]
            log("getting pixel values for img from", time_start, "...")
            pixel_val = img.reduceRegion(ee.Reducer.first(), ee_point).getInfo()[band_id]
            rows.append({"datetime": time_start, band: pixel_val, "band_id": band_id})
        else:
            log("masked pixel, moving on to next image")
    return pd.DataFrame(rows, columns=["datetime", band, "band_id"])


def create_lists_for_extract(
    df: pd.DataFrame, suffix_to_remove: str, suffix_to_add: str
) -> Tuple[List[str], List[str]]:
    """Rebuild time/band lists from a previously extracted DataFrame."""
    time_list = df["datetime"].to_list()
    band_list = df["band_id"].to_list()
    band_list = [b.replace(suffix_to_remove, "") for b in band_list]
    band_list = [b + suffix_to_add for b in band_list]
    return time_list, band_list


# ---------------------------------------------------------------------------
# Harmonic regression (per pixel / point)
# ---------------------------------------------------------------------------
def run_harmonic_regression(
    imgcoll: ee.ImageCollection,
    bands: List[str],
    config: PipelineConfig,
    refdate: Optional[str] = None,
) -> ee.Image:
    """Fit harmonic regression, returning an image of coefficients per band."""
    import harmonics

    return harmonics.run_std_regressions(
        imgcoll.select(bands), bands, refdate=refdate or config.reference_date
    )


def fit_harmonics(
    coef_img: ee.Image,
    imgcoll: ee.ImageCollection,
    bands: List[str],
    config: PipelineConfig,
    refdate: Optional[str] = None,
) -> ee.ImageCollection:
    """Produce smoothed values by fitting the harmonic model to a collection."""
    import harmonics

    return harmonics.fit_harmonics(
        coef_img,
        imgcoll,
        omega=1,
        nharmonics=config.n_harmonics,
        bands=bands,
        refdate=refdate or config.reference_date,
    )


# ---------------------------------------------------------------------------
# Aggregation / plotting helpers
# ---------------------------------------------------------------------------
def daily_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Collapse a timestamped series to a daily mean with an ordinal date."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date")[value_col].mean().reset_index()
    daily["date_ordinal"] = pd.to_datetime(daily["date"]).apply(lambda d: d.toordinal())
    return daily


def plot_harmonic_fit(
    observed: pd.DataFrame,
    fitted: pd.DataFrame,
    value_col: str,
    fitted_col: str,
    save_path: Optional[str] = None,
):
    """Overlay observed points with a cubic-interpolated harmonic-fit line."""
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    x = fitted["date_ordinal"]
    y = fitted[fitted_col]
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    xnew = np.linspace(x.min(), x.max(), 500)
    f = interp1d(x, y, kind="cubic")
    y_smooth = f(xnew)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, c="teal", s=50, label=f"Smoothed {value_col} Points")
    ax.plot(xnew, y_smooth, c="teal", linewidth=3, label=f"Smoothed {value_col} Line")
    ax.scatter(observed["date_ordinal"], observed[value_col], c="sienna", s=50,
               label=f"Original {value_col} Points")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(value_col, fontsize=12)
    ax.set_title(f"Harmonic Fitted {value_col} and Observed {value_col}", fontsize=15)
    date_labels = [date.fromordinal(int(i)) for i in ax.get_xticks() if i >= 1]
    ax.set_xticklabels(date_labels, rotation=45, fontsize=10)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax
