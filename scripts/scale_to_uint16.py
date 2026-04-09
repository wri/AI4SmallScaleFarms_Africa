#!/usr/bin/env python3
"""Scale a uint8 [0-255] TIFF to uint16 [0-10000] to match training input.

Training chips are scaled with scale_factor = 10000/255 so that inference
with default preprocess (divide by 3000) sees the same value range.
Use this script on 0-255 imagery before inference if you prefer not to use
--input_scale 255.

Processes in blocks to avoid loading the whole image into memory (safe for
large rasters like ksa_bands.tif).

Usage:
  python scripts/scale_to_uint16.py input.tif
  python scripts/scale_to_uint16.py input.tif -o output.tif
  python scripts/scale_to_uint16.py input.tif --blocksize 2048  # tune if OOM
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


SCALE_FACTOR = 10000.0 / 255.0  # 0-255 -> 0-10000

# Default block size (pixels per side). One block uses ~blocksize^2 * count * 2 bytes.
DEFAULT_BLOCKSIZE = 2048


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
    parser.add_argument(
        "--blocksize",
        type=int,
        default=DEFAULT_BLOCKSIZE,
        metavar="N",
        help=f"Process in N×N blocks to limit memory (default {DEFAULT_BLOCKSIZE}). Lower if OOM.",
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
    bs = max(1, args.blocksize)

    with rasterio.open(args.input) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint16",
            count=src.count,
            compress="deflate",
        )

        with rasterio.open(out, "w", **profile) as dst:
            for ji, window in dst.block_windows(1):
                # Read this block (may be smaller at edges)
                data = src.read(window=window)
                data_scaled = (
                    data.astype(np.float32) * SCALE_FACTOR
                ).clip(0, 10000).astype(np.uint16)
                dst.write(data_scaled, window=window)

    print(f"Wrote: {out}")
    print(f"  Scale: 0-255 -> 0-10000 (factor {SCALE_FACTOR:.2f})")


if __name__ == "__main__":
    main()
