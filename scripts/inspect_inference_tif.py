#!/usr/bin/env python3
"""Inspect an inference output TIFF to debug polygonization (e.g. 0 polygons).

Usage:
  python scripts/inspect_inference_tif.py inference_output.tif
  python scripts/inspect_inference_tif.py inference_output.tif --input path/to/input_image.tif
"""
import argparse
import sys
import numpy as np
import rasterio
from pathlib import Path


def inspect_raster(path: Path, label: str = "Raster") -> None:
    """Print shape, CRS, bands, unique values, and value stats."""
    with rasterio.open(path) as src:
        data = src.read()
        h, w = src.height, src.width
        nbands = src.count
        crs = src.crs
        transform = src.transform
        pixel_width = abs(transform.a)
        pixel_height = abs(transform.e)
        area_per_pixel = pixel_width * pixel_height

    print(f"=== {label}: {path} ===")
    print(f"Shape: {h} x {w} (H x W)")
    print(f"Bands: {nbands}")
    print(f"CRS: {crs}")
    if crs:
        print(f"Pixel size: {pixel_width:.4f} x {pixel_height:.4f} (W x H) [{crs.linear_units}]")
        print(f"Area per pixel: {area_per_pixel:.4f} [{crs.linear_units}²]")
    print()

    for b in range(nbands):
        band = data[b]
        uniq, counts = np.unique(band, return_counts=True)
        total = band.size
        print(f"Band {b + 1}: dtype={band.dtype}")
        print(f"  Unique values (sample): {uniq.tolist()[:20]}{'...' if len(uniq) > 20 else ''}")
        print(f"  Min: {np.min(band)}, Max: {np.max(band)}, Mean: {np.mean(band):.2f}")
        for v, c in zip(uniq[:15], counts[:15]):
            pct = 100.0 * c / total
            print(f"    value {v}: {c} pixels ({pct:.2f}%)")
        if len(uniq) > 15:
            print(f"    ... and {len(uniq) - 15} more values")
        print()
    return data, nbands, area_per_pixel, crs


def main(path: str, input_path: str | None = None) -> None:
    path = Path(path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    # Optional: inspect input image used for inference
    if input_path:
        input_path = Path(input_path)
        if input_path.exists():
            inspect_raster(input_path, "Input image (used for inference)")
            print("Note: Inference expects reflectance-like values (e.g. 0–10000); preprocess divides by 3000.")
            print("      Two-window models (fcsiam*) expect 2*C bands (two dates stacked).\n")
        else:
            print(f"Input file not found: {input_path}\n")

    data, nbands, area_per_pixel, crs = inspect_raster(path, "Inference output TIFF")

    # Polygonize expects single band with values 0=background, 1=field, 2=boundary
    # and only outputs polygons for value 1, filtering by min_size (default 500 m²).
    if nbands == 1:
        band = data[0]
        n1 = int(np.sum(band == 1))
        n2 = int(np.sum(band == 2))
        n0 = int(np.sum(band == 0))
        print("Polygonize interpretation (single-band):")
        print(f"  Pixels with value 1 (field): {n1} -> these become polygons")
        print(f"  Pixels with value 2 (boundary): {n2}")
        print(f"  Pixels with value 0 (background): {n0}")
        if n1 == 0:
            print()
            print("  -> No polygons: there are no pixels with value 1.")
            print("     The model predicted only background (0) for this scene.")
            print("     Possible causes: wrong input bands/windows, wrong value range, or scene has no fields.")
        else:
            # Rough lower bound: 500 m² at area_per_pixel m²/pixel
            min_pixels = 500 / area_per_pixel
            print(f"  min_size=500 {crs.linear_units}² requires at least ~{min_pixels:.0f} connected pixels per polygon.")
            print(f"  If all value-1 regions are smaller than that, you get 0 polygons.")
    else:
        print("Polygonize: multi-band mode (softmax) not checked here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect inference output TIFF and optionally input image.")
    parser.add_argument(
        "inference_tif",
        nargs="?",
        default="inference_output_prue_b7_spot.tif",
        help="Path to inference output TIFF",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        help="Path to input image used for inference (to check bands and value range)",
    )
    args = parser.parse_args()
    main(args.inference_tif, args.input)
