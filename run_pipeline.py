"""CLI entrypoint for the crop-type-mapping pipeline.

Examples
--------
Build features + train a model for the area described in a config file::

    python run_pipeline.py train --config config.yaml

Kick off the AOI feature-image export (async, runs on Earth Engine)::

    python run_pipeline.py export-features --config config.yaml

Once the GeoTIFF has been downloaded into data/outputs/, classify it::

    python run_pipeline.py infer --config config.yaml \
        --feature-tif data/outputs/nyandarua_pixel_features_for_rf.tif
"""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.pipeline import CropTypePipeline


def _load_config(args) -> PipelineConfig:
    if args.config:
        return PipelineConfig.from_yaml(args.config)
    return PipelineConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop-type-mapping pipeline")
    parser.add_argument("stage", choices=["features", "train", "export-features", "infer", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument("--survey", help="Override survey GeoJSON path")
    parser.add_argument("--feature-tif", help="Path to the exported AOI feature GeoTIFF (infer stage)")
    args = parser.parse_args()

    config = _load_config(args)
    pipeline = CropTypePipeline(config)
    pipeline.setup_ee()

    if args.stage == "features":
        pipeline.build_feature_table(args.survey)
    elif args.stage == "train":
        pipeline.run_training(survey_path=args.survey)
    elif args.stage == "export-features":
        pipeline.export_inference_features()
    elif args.stage == "infer":
        pipeline.run_inference(feature_tif=args.feature_tif)
    elif args.stage == "all":
        pipeline.run_training(survey_path=args.survey)
        pipeline.export_inference_features()
        print("Feature-image export started. Re-run `infer` once the GeoTIFF is downloaded.")


if __name__ == "__main__":
    main()
