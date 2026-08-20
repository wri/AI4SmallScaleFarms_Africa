"""Render a 4-panel PNG of the crop-type mapping pipeline.

Panels: survey plots in the county boundary, a selected feature layer
(GCVI mean by default), predicted probability, and the cropland-masked
classified map.

Example::

    python scripts/visualize_pipeline.py --config config/config.nyandarua_smoke.yaml
    python scripts/visualize_pipeline.py --config config/config.nyandarua.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig  # noqa: E402
from models.postprocess import plot_pipeline_stages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save a PNG of survey → feature → probability → post-processed map"
    )
    parser.add_argument("--config", required=True, help="Path to a YAML pipeline config")
    parser.add_argument(
        "--out",
        help="Output PNG path (default: data/outputs/<area>_pipeline_stages.png)",
    )
    parser.add_argument(
        "--feature-band",
        help="Feature-image band to draw (default: GCVI_mean if present)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    config = PipelineConfig.from_yaml(args.config)
    config.ensure_dirs()
    path = plot_pipeline_stages(
        config,
        save_path=args.out,
        feature_band=args.feature_band,
        dpi=args.dpi,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
