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

Because Earth Engine computes features on Google's servers, the slow county-wide
step is a **batch** ``Export.image.toAsset`` (12-hour limit, high memory).
Interactive ``getDownloadURL`` is only used afterwards, on the stored image,
and only for tiles that intersect the county polygon. Training samples that
raster at labelled geometries (polygon zonal mean, or the 30 m pixel under a
GPS point); the raster is never masked to those geometries.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional
import shutil
import time

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
        self.timings: dict = {}
        config.ensure_dirs()

    @contextmanager
    def _timed(self, label: str):
        print(f"[start] {label}")
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.timings[label] = elapsed
        print(f"[done]  {label}: {elapsed:.1f}s")

    def print_timings(self) -> None:
        if not self.timings:
            return
        print("\n=== Stage timings ===")
        for name, elapsed in sorted(self.timings.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {elapsed:8.1f}s  {name}")
        print(f"  {sum(self.timings.values()):8.1f}s  TOTAL (sum of timed stages)")

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

        with self._timed("prepare_survey"):
            survey_gdf = survey_mod.prepare_survey(cfg, survey_path)
            sample_geometry = survey_mod.sample_geometry_mode(survey_gdf, cfg)
            if cfg.is_binary:
                print(
                    f"Survey {sample_geometry}s: {len(survey_gdf)}  "
                    f"target positives: {int(survey_gdf[cfg.target_column].sum())}"
                )
            else:
                print(f"Survey {sample_geometry}s: {len(survey_gdf)}  multiclass")
            if sample_geometry == "point":
                print(f"Sampling Earth Engine at {cfg.scale} m (one pixel per GPS point)")
            else:
                print(f"Sampling Earth Engine as polygon zonal means at {cfg.scale} m")

        with self._timed("admin_boundary"):
            boundary = boundary_mod.get_admin_boundary(cfg)
            county = boundary.geometry()

        with self._timed("survey_to_ee"):
            plots = boundary_mod.survey_to_ee(survey_gdf, columns=["fid", "geometry"])

        with self._timed("sentinel_collection"):
            # County-wide collection, vegetation-index bands only. Labelled
            # geometries are sampled later; the image itself is not masked
            # to those plots or points.
            coll = sentinel_mod.get_masked_collection(
                county, cfg, bands=cfg.regression_bands
            )

        with self._timed("extract_harmonic_features"):
            harmonic_df = feat_mod.extract_harmonic_features(
                coll, plots, cfg, aoi=county, sample_geometry=sample_geometry
            )
        with self._timed("extract_terrain_features"):
            terrain_df = feat_mod.extract_terrain_features(
                plots, cfg, sample_geometry=sample_geometry
            )
        with self._timed("extract_precipitation"):
            precip_df = feat_mod.extract_precipitation(
                plots, cfg, sample_geometry=sample_geometry
            )

        with self._timed("merge_features"):
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
        legend = survey_mod.class_legend(merged, cfg)
        class_codes, class_names = legend if legend else (None, None)
        with self._timed("train_random_forest"):
            model, metrics = train_mod.train(
                X, y, cfg, class_names=class_names, class_codes=class_codes
            )
        print("Training metrics:", metrics)
        return model, metrics

    # ------------------------------------------------------------------
    def export_inference_features(self, local: bool = True):
        """Export the AOI feature image and cropland mask to ``data/outputs/``.

        ``local=True`` (default) materializes the county-wide feature image to
        an Earth Engine asset, then downloads GeoTIFF tiles that intersect the
        county polygon. ``local=False`` starts Drive export tasks instead.
        """
        cfg = self.config
        with self._timed("build_feature_image"):
            boundary = boundary_mod.get_admin_boundary(cfg)
            aoi = boundary_mod.get_aoi(cfg, boundary)
            feature_img = infer_mod.build_feature_image(aoi, cfg)
        if local:
            with self._timed("export_feature_image_local"):
                feature_path = infer_mod.export_feature_image_local(feature_img, aoi, cfg)
            with self._timed("export_cropland_mask_local"):
                post_mod.export_cropland_mask_local(aoi, cfg)
            return feature_path
        with self._timed("export_feature_image_to_drive"):
            feature_task = infer_mod.export_feature_image_to_drive(feature_img, aoi, cfg)
        with self._timed("export_cropland_mask_to_drive"):
            post_mod.export_cropland_mask_to_drive(aoi, cfg)
        return feature_task

    # ------------------------------------------------------------------
    def run_inference(
        self,
        model=None,
        feature_tif: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """Apply the trained model to the AOI feature GeoTIFF.

        If the feature image is not already on disk, it is built on Earth Engine
        and downloaded into ``data/outputs/`` first so inference is end-to-end.
        ``model_path`` (CLI ``--model``) selects a pickle trained in another area.
        """
        cfg = self.config
        model = model or train_mod.load_model(cfg, path=model_path)
        if feature_tif is None and not cfg.feature_image_path.exists():
            print("Feature GeoTIFF not found; extracting and downloading locally")
            self.export_inference_features(local=True)

        with self._timed("predict_probability_raster"):
            prob_path = infer_mod.predict_probability_raster(model, cfg, feature_tif)
        print(f"Probability raster -> {prob_path}")

        cropland_path = cfg.cropland_map_path if cfg.cropland_map_path.exists() else None
        if cropland_path is None:
            try:
                with self._timed("export_cropland_mask_local"):
                    boundary = boundary_mod.get_admin_boundary(cfg)
                    aoi = boundary_mod.get_aoi(cfg, boundary)
                    cropland_path = post_mod.export_cropland_mask_local(aoi, cfg)
            except Exception as exc:
                print(f"Cropland mask download failed ({exc}); classifying without it")
                cropland_path = None

        with self._timed("apply_cropland_mask"):
            class_path = post_mod.apply_cropland_mask(cfg, prob_path, cropland_path)
        print(f"Classified map -> {class_path}")
        with self._timed("plot_pipeline_stages"):
            fig_path = post_mod.plot_pipeline_stages(cfg)
        if fig_path:
            print(f"Pipeline figure -> {fig_path}")
        return prob_path, class_path

    def run_all(self, survey_path: Optional[str] = None, local: bool = True):
        """Full local run: features -> train -> GeoTIFF export -> infer."""
        self.run_training(survey_path=survey_path)
        if local or not self.config.feature_image_path.exists():
            self.export_inference_features(local=local)
        if local or self.config.feature_image_path.exists():
            result = self.run_inference()
        else:
            print("Drive export started. Re-run `infer` once the GeoTIFF is in data/outputs/.")
            result = None
        self.print_timings()
        return result

    def run_smoke(self, survey_path: Optional[str] = None):
        """End-to-end run for a small AOI: features -> train -> local export -> infer."""
        cfg = self.config
        for path in (
            cfg.merged_features_path,
            cfg.harmonic_features_path,
            cfg.terrain_features_path,
            cfg.precip_features_path,
            cfg.model_path,
            cfg.metrics_path,
            cfg.feature_image_path,
            cfg.probability_map_path,
            cfg.classified_map_path,
            cfg.cropland_map_path,
            cfg.pipeline_figure_path,
        ):
            if path.exists():
                path.unlink()
                print(f"Removed previous artefact {path}")
        if cfg.harmonic_band_dir.exists():
            shutil.rmtree(cfg.harmonic_band_dir)
            print(f"Removed previous artefact {cfg.harmonic_band_dir}")
        return self.run_all(survey_path=survey_path, local=True)
