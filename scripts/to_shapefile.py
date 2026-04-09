#!/usr/bin/env python3
"""Convert polygonized output (Parquet, GeoJSON, GPKG, etc.) to ESRI Shapefile.

Use this on any product of the polygonization step (e.g. ftw inference polygonize).
Shapefile column names are truncated to 10 characters; attribute data is preserved.

Usage:
  python scripts/to_shapefile.py polygons.parquet
  python scripts/to_shapefile.py polygons.parquet -o fields.shp
  python scripts/to_shapefile.py polygons.geojson -o fields.shp -f
"""
import argparse
import sys
from pathlib import Path

import geopandas as gpd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Parquet, GeoJSON, GPKG (or other vector formats) to Shapefile."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file (e.g. .parquet, .geojson, .gpkg from polygonize).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output .shp path. Default: same basename as input with .shp next to input.",
    )
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Overwrite output if it exists.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out = args.out
    if out is None:
        out = args.input.parent / f"{args.input.stem}.shp"
    out = Path(out)
    if out.suffix.lower() != ".shp":
        out = out.with_suffix(".shp")

    if out.exists() and not args.overwrite:
        print(f"Output exists (use -f to overwrite): {out}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    suffix = args.input.suffix.lower()
    if suffix == ".parquet":
        gdf = gpd.read_parquet(args.input)
    else:
        gdf = gpd.read_file(args.input)

    gdf.to_file(out, driver="ESRI Shapefile")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
