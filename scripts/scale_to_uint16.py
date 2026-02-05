#!/usr/bin/env python3
"""Scale a uint8 [0-255] TIFF to uint16 [0-10000] to match training input.

Training chips are scaled with scale_factor = 10000/255 so that inference
with default preprocess (divide by 3000) sees the same value range.
Use this script on 0-255 imagery before inference if you prefer not to use
--input_scale 255.

Usage:
  python scripts/scale_to_uint16.py input.tif
  python scripts/scale_to_uint16.py input.tif -o output.tif
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio


SCALE_FACTOR = 10000.0 / 255.0  # 0-255 -> 0-10000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale uint8 [0-255] TIFF to uint16 [0-10000] (training-compatible)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input GeoTIFF (e.g. 0-255 uint8).",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path. Default: <input_stem>_scaled.tif next to input.",
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
        out = args.input.parent / f"{args.input.stem}_scaled.tif"
    out = Path(out)

    if out.exists() and not args.overwrite:
        print(f"Output exists (use -f to overwrite): {out}", file=sys.stderr)
        sys.exit(1)

    out.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.input) as src:
        data = src.read()
        data_scaled = (
            data.astype(np.float32) * SCALE_FACTOR
        ).clip(0, 10000).astype(np.uint16)

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint16",
            count=src.count,
            compress="deflate",
        )

        with rasterio.open(out, "w", **profile) as dst:
            dst.write(data_scaled)

    print(f"Wrote: {out}")
    print(f"  Scale: 0-255 -> 0-10000 (factor {SCALE_FACTOR:.2f})")


if __name__ == "__main__":
    main()
