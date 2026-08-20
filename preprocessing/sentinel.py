"""Sentinel-2 L2A acquisition, cloud masking and band selection.

Generalises notebook cells 13-21. Relies on the Azzari et al. ``eetc`` tools,
which must be importable (see ``src.ee_utils.add_eetc_to_path``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import ee

from src.config import PipelineConfig


def get_s2_collection(
    geometry: ee.Geometry,
    config: PipelineConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    add_vis: bool = True,
) -> ee.ImageCollection:
    """Build a Sentinel-2 SR ImageCollection with vegetation indices.

    Imported lazily so that ``eetc`` only needs to be on ``sys.path`` at call
    time (after ``add_eetc_to_path`` has run).
    """
    import gee_tools.datasources.sentinel2_2a as s2_2a

    start = ee.Date(start_date or config.start_date)
    end = ee.Date(end_date or config.end_date)
    ds = s2_2a.Sentinel2SR(
        geometry,
        start_date=start,
        end_date=end,
        addVIs=add_vis,
        addCloudMasks=False,
    )
    asset = config.s2_sr_asset
    if ds.name != asset:
        print(f"Switching Sentinel-2 collection {ds.name} -> {asset}")
        ds.name = asset
        ds.build_img_coll(addVIs=add_vis, addCloudMasks=False)
    return ds.get_img_coll()


def mask_clouds_sr(img: ee.Image, bandnames: Sequence[str]) -> ee.Image:
    """Mask cloudy/shadow pixels using the Scene Classification Layer (SCL).

    Keeps only vegetation (SCL == 4) and bare-soil (SCL == 5) pixels. The
    original notebook used Python's ``or`` on EE objects (a latent bug); this
    uses the correct server-side ``.Or`` combinator.
    """
    scl = img.select(["SCL"])
    clear = scl.eq(4).Or(scl.eq(5))
    return img.updateMask(clear).select(list(bandnames))


def get_masked_collection(
    geometry: ee.Geometry,
    config: PipelineConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    bands: Optional[Sequence[str]] = None,
) -> ee.ImageCollection:
    """Return a cloud-masked collection restricted to the bands of interest."""
    bands = list(bands or config.bands)
    coll = get_s2_collection(geometry, config, start_date, end_date, add_vis=True)
    return coll.map(lambda img: mask_clouds_sr(img, bands))
