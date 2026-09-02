"""CLI entrypoint for the crop-type-mapping pipeline.

Examples
--------
End-to-end local run (features, train, GeoTIFF, classify)::

    python run_pipeline.py all --config config/config.nyandarua_smoke.yaml

Build the per-plot feature table and train::

    python run_pipeline.py train --config config.yaml

Download the AOI feature GeoTIFF into data/outputs/ then classify::

    python run_pipeline.py infer --config config.yaml

County-wide features are batch-exported to an Earth Engine asset, then tiled
locally (only tiles that intersect the county polygon)::

    python run_pipeline.py export-features --config config/config.nyandarua.yaml

Train a new multiclass model on Nakuru point labels (do not reuse the
Nyandarua maize pickle)::

    python run_pipeline.py train --config config/config.nakuru.yaml

Apply a model trained in one area to another area's feature image (same
label mode and feature recipe)::

    python run_pipeline.py infer --config config/config.nyandarua.yaml \\
        --model data/models/nyandarua_rf_best_model.pkl
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
    parser.add_argument(
        "stage",
        choices=["features", "train", "export-features", "infer", "all", "smoke"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--config", help="Path to a YAML config file")
    parser.add_argument("--survey", help="Override survey GeoJSON path")
    parser.add_argument(
        "--model",
        help="Path to a trained Random Forest .pkl (infer). Use this to apply "
        "a model trained in one area to features from another.",
    )
    parser.add_argument("--feature-tif", help="Path to an existing AOI feature GeoTIFF (infer stage)")
    parser.add_argument(
        "--drive",
        action="store_true",
        help="Export the AOI GeoTIFF to Google Drive instead of downloading it locally",
    )
    args = parser.parse_args()
    local = not args.drive

    config = _load_config(args)
    pipeline = CropTypePipeline(config)
    pipeline.setup_ee()

    if args.stage == "features":
        pipeline.build_feature_table(args.survey)
        pipeline.print_timings()
    elif args.stage == "train":
        pipeline.run_training(survey_path=args.survey)
        pipeline.print_timings()
    elif args.stage == "export-features":
        pipeline.export_inference_features(local=local)
        pipeline.print_timings()
    elif args.stage == "infer":
        pipeline.run_inference(feature_tif=args.feature_tif, model_path=args.model)
        pipeline.print_timings()
    elif args.stage == "all":
        pipeline.run_all(survey_path=args.survey, local=local)
    elif args.stage == "smoke":
        pipeline.run_smoke(survey_path=args.survey)


if __name__ == "__main__":
    main()
