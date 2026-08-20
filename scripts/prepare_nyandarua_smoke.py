"""Build the small Ndaragwa survey subset used by config.nyandarua_smoke.yaml."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/raw/Nyandarua/Nyandarua_Labels/Nyandarua.geojson"
DEST = ROOT / "data/raw/Nyandarua/nyandarua_smoke_subset.geojson"
# Matches aoi_bbox in config.nyandarua_smoke.yaml
BBOX = (36.47, 0.07, 36.51, 0.10)


def main() -> Path:
    gdf = gpd.read_file(SRC)
    subset = gdf[gdf.intersects(box(*BBOX))].copy()
    if subset.empty:
        raise RuntimeError(f"No survey plots intersect smoke bbox {BBOX}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    subset.to_file(DEST, driver="GeoJSON")
    crop_cols = [c for c in ("crop_a", "crop_b", "crop_c", "crop_d", "crop_e") if c in subset.columns]
    maize = subset[crop_cols].apply(
        lambda col: col.astype(str).str.strip().str.lower().eq("maize")
    ).any(axis=1)
    print(
        f"Wrote {DEST}  n={len(subset)}  maize={int(maize.sum())}  "
        f"other={int((~maize).sum())}  bbox={list(BBOX)}"
    )
    return DEST


if __name__ == "__main__":
    main()
