"""End-to-end crop-type-mapping pipeline orchestration.

Wires the preprocessing, model and post-processing modules into a single,
area-agnostic flow driven entirely by a :class:`~src.config.PipelineConfig`.

Typical usage::

    from src.config import PipelineConfig
    from src.pipeline import CropTypePipeline

    config = PipelineConfig.from_yaml("config.yaml")
    pipeline = CropTypePipeline(config)
    pipeline.run_training()      # survey -> features -> trained model
    pipeline.run_inference()     # AOI feature image -> probability -> map

Because Earth Engine batch exports (feature-table CSVs and the AOI feature
GeoTIFF) run asynchronously on Google's servers, the pipeline is split into
resumable stages rather than one monolithic call.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.config import PipelineConfig
from src.ee_utils import add_eetc_to_path, initialize_ee

from preprocessing import survey as survey_mod
from preprocessing import boundary as boundary_mod
from preprocessing import sentinel as sentinel_mod
from preprocessing import features as feat_mod
from models import train as train_mod
from models import inference as infer_mod
from models import postprocess as post_mod


class CropTypePipeline:
    """Coordinates the full workflow for a single configured area."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        config.ensure_dirs()

    # ------------------------------------------------------------------
    def setup_ee(self) -> None:
        """Initialise Earth Engine and make the eetc tools importable."""
        add_eetc_to_path(self.config)
        initialize_ee(self.config)

    # ------------------------------------------------------------------
    def build_feature_table(self, survey_path: Optional[str] = None) -> pd.DataFrame:
        """Survey + EE features -> merged per-plot table (saved to processed/).

        Stages: prepare survey labels/areas, derive the boundary, pull a
        cloud-masked S2 collection, then extract harmonic, terrain and
        precipitation features and merge them all on ``fid``.
        """
        cfg = self.config

        survey_gdf = survey_mod.prepare_survey(cfg, survey_path)
        boundary = boundary_mod.get_admin_boundary(cfg)
        geometry = boundary.geometry()

        coll = sentinel_mod.get_masked_collection(geometry, cfg)
        plots = boundary_mod.survey_to_ee(survey_gdf, columns=["fid", "geometry"])

        harmonic_df = feat_mod.extract_harmonic_features(coll, plots, cfg)
        terrain_df = feat_mod.extract_terrain_features(plots, cfg)
        precip_df = feat_mod.extract_precipitation(plots, cfg)

        merged = feat_mod.merge_features(
            survey_gdf.drop(columns=["geometry"]),
            [precip_df, harmonic_df, terrain_df],
        )
        merged.to_csv(cfg.merged_features_path, index=False)
        print(f"Saved merged feature table -> {cfg.merged_features_path} ({merged.shape})")
        return merged

    # ------------------------------------------------------------------
    def run_training(
        self, merged: Optional[pd.DataFrame] = None, survey_path: Optional[str] = None
    ):
        """Train the classifier from a merged table (built if not supplied)."""
        cfg = self.config
        if merged is None:
            if cfg.merged_features_path.exists():
                merged = pd.read_csv(cfg.merged_features_path)
            else:
                merged = self.build_feature_table(survey_path)

        X, y = feat_mod.build_xy(merged, cfg)
        model, metrics = train_mod.train(X, y, cfg)
        print("Training metrics:", metrics)
        return model, metrics

    # ------------------------------------------------------------------
    def export_inference_features(self):
        """Start the Drive export of the AOI feature image (async task)."""
        cfg = self.config
        boundary = boundary_mod.get_admin_boundary(cfg)
        aoi = boundary_mod.get_aoi(cfg, boundary)
        feature_img = infer_mod.build_feature_image(aoi, cfg)
        return infer_mod.export_feature_image_to_drive(feature_img, aoi, cfg)

    # ------------------------------------------------------------------
    def run_inference(self, model=None, feature_tif: Optional[str] = None):
        """Apply the trained model to the exported feature GeoTIFF.

        Produces a probability raster then a thresholded/cropland-masked
        classified map, both written to data/outputs/.
        """
        cfg = self.config
        model = model or train_mod.load_model(cfg)
        prob_path = infer_mod.predict_probability_raster(model, cfg, feature_tif)
        print(f"Probability raster -> {prob_path}")
        class_path = post_mod.threshold_probability(cfg, prob_path)
        print(f"Classified map -> {class_path}")
        return prob_path, class_path
