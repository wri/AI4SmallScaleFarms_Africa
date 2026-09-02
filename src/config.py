"""Central configuration for the crop-type-mapping pipeline.

Everything that is specific to an *area of interest* (AOI) lives here so the
rest of the pipeline can stay generic. Point the pipeline at a new region by
editing a YAML file (see ``config.example.yaml``) or by constructing a
``PipelineConfig`` in code -- no changes to the processing modules are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml is optional at import time
    yaml = None


# Vegetation indexes (plus red-edge RDED4) used for mapping. Optical bands
# needed to *compute* these are still loaded inside eetc, then dropped.
DEFAULT_BANDS: List[str] = ["RDED4", "GCVI", "NBR1", "NDTI", "NDVI", "SNDVI"]

# Bands on which harmonic regression coefficients are computed as ML features.
DEFAULT_REGRESSION_BANDS: List[str] = ["RDED4", "GCVI", "NBR1", "NDTI", "NDVI", "SNDVI"]

# Typo / synonym map applied to crop labels before counting and collapsing.
DEFAULT_CROP_ALIASES: dict = {
    "ovacodo": "avocado",
    "ovacado": "avocado",
    "whear": "wheat",
    "potato": "potatoes",
    "grasa": "grass",
    "cyprus": "cypress",
    "pyrethrumpyrethrum": "pyrethrum",
    "brocolli": "broccoli",
}

# Display / raster code order for named classes that survive min_class_count.
# Remaining frequent classes are appended, then ``other``.
DEFAULT_PREFERRED_CLASS_ORDER: List[str] = [
    "maize", "wheat", "potatoes", "grass", "canola", "coffee",
]

# Final feature columns fed to the classifier (must exist in the merged table
# and in the inference feature image).
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "precipitation",
    "GCVI_constant", "GCVI_cos1", "GCVI_count", "GCVI_mean",
    "GCVI_r2", "GCVI_rmse", "GCVI_variance",
    "NBR1_constant", "NBR1_cos1", "NBR1_r2", "NBR1_rmse", "NBR1_variance",
    "NDTI_cos1", "NDTI_mean", "NDTI_rmse", "NDTI_variance",
    "NDVI_constant",
    "RDED4_constant", "RDED4_mean", "RDED4_r2", "RDED4_rmse", "RDED4_variance",
    "elevation", "slope", "aspect",
]


@dataclass
class PipelineConfig:
    """All area-specific and run-specific settings for the pipeline.

    Attributes are grouped by pipeline stage. Anything region specific
    (``area_name``, ``survey_geojson``, ``aoi_bbox`` ...) should be overridden
    per run; the sensible geospatial defaults rarely need changing.
    """

    # --- Identity / area of interest --------------------------------------
    area_name: str = "Nyandarua"
    # GAUL admin lookup used to derive the study-area boundary from Earth Engine.
    gaul_dataset: str = "FAO/GAUL/2015/level2"
    gaul_name_field: str = "ADM2_NAME"
    # ADM2 (or other) name passed to the GAUL filter. Defaults to ``area_name``.
    # Set this when ``area_name`` is a run label (e.g. "Nyandarua_smoke") rather
    # than the GAUL name ("Nyandarua") — notebook cell 8 uses ADM2_NAME=Nyandarua.
    gaul_name: Optional[str] = None
    # Optional explicit inference bounding box [west, south, east, north].
    # If None, the pipeline derives it from the admin boundary bounds.
    aoi_bbox: Optional[Sequence[float]] = None
    # Optional local admin-boundary vector (GeoJSON / shapefile). Used when the
    # file exists; otherwise the pipeline falls back to the GAUL name lookup.
    boundary_path: Optional[Path] = None

    # --- Target label -----------------------------------------------------
    # ``binary``: any of ``crop_columns`` matching ``target_crop_names`` -> 0/1.
    # ``multiclass``: cleaned ``label_column`` (default crop_a); rare classes
    # collapse to ``other_class_name`` when count < ``min_class_count``.
    label_mode: str = "binary"
    target_crop_names: List[str] = field(default_factory=lambda: ["Maize"])
    crop_columns: List[str] = field(
        default_factory=lambda: ["crop_a", "crop_b", "crop_c", "crop_d", "crop_e"]
    )
    target_column: str = "maize_pos"
    label_column: str = "crop_a"
    min_class_count: int = 10
    other_class_name: str = "other"
    crop_aliases: dict = field(default_factory=lambda: dict(DEFAULT_CROP_ALIASES))
    preferred_class_order: List[str] = field(
        default_factory=lambda: list(DEFAULT_PREFERRED_CLASS_ORDER)
    )
    # String label after collapse (multiclass only); integer codes go in ``target_column``.
    class_name_column: str = "crop_class"
    raw_label_column: str = "crop_label_raw"

    # --- Growing season / imagery -----------------------------------------
    start_date: str = "2024-03-01"
    end_date: str = "2024-08-31"
    # Reference date for harmonic regression (defaults to start_date if None).
    refdate: Optional[str] = None
    n_harmonics: int = 2
    # Sentinel-2 L2A surface-reflectance collection. S2_SR is deprecated.
    s2_sr_asset: str = "COPERNICUS/S2_SR_HARMONIZED"

    # --- CRS / geometry ---------------------------------------------------
    # CRS used only for planar area computation (default: UTM 36N / Kenya).
    area_calc_epsg: int = 32636
    # Working CRS for all analysis / EE interaction.
    working_epsg: int = 4326
    # Nominal resolution (metres) for reduceRegions / exports / point sampling.
    scale: int = 30
    # How training locations are sampled from EE images.
    # ``auto``: points -> one pixel at the GPS (Reducer.first); polygons -> zonal mean.
    sample_geometry: str = "auto"

    # --- Bands / features -------------------------------------------------
    bands: List[str] = field(default_factory=lambda: list(DEFAULT_BANDS))
    regression_bands: List[str] = field(default_factory=lambda: list(DEFAULT_REGRESSION_BANDS))
    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))

    # --- Precipitation ----------------------------------------------------
    precip_dataset: str = "UCSB-CHG/CHIRPS/DAILY"
    # Temporal reducer over the season. Notebook inference (cell 97) uses mean.
    precip_reducer: str = "mean"
    # If None, precipitation is aggregated over [start_date, end_date].
    precip_start_date: Optional[str] = None
    precip_end_date: Optional[str] = None

    # --- Terrain ----------------------------------------------------------
    srtm_asset: str = "USGS/SRTMGL1_003"

    # --- Cropland mask (post-processing) ----------------------------------
    # Notebook cell 108: USGS GFSAD 1 km cropland (Image, band "landcover").
    # Classes 2-6 are cropland; 0=water, 1=non-cropland.
    cropland_dataset: str = "USGS/GFSAD1000_V1"
    cropland_band: str = "landcover"
    cropland_classes: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6])
    # Used only when cropland_classes is empty (percent-style rasters, cell 111).
    cropland_threshold: float = 40.0
    probability_threshold: float = 0.5

    # --- Model ------------------------------------------------------------
    test_size: float = 0.20
    random_state: int = 100
    cv_folds: int = 10
    # Passed to RandomForestClassifier (e.g. ``balanced``). None = sklearn default.
    class_weight: Optional[str] = None
    rf_param_grid: dict = field(
        default_factory=lambda: {
            "rf__n_estimators": [100, 1000, 5000],
            "rf__criterion": ["gini", "entropy"],
            "rf__max_depth": [1, 3, 5],
            "rf__min_samples_split": [2, 5, 10],
            "rf__random_state": [10],
        }
    )

    # --- Paths ------------------------------------------------------------
    # Project root (directory that contains data/, preprocessing/, src/).
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    # Path to the cloned Azzari et al. eetc repo (https://github.com/shrutijain90/eetc).
    eetc_path: Optional[Path] = None
    # Survey/training vector file (GeoJSON / shapefile) with crop labels.
    survey_geojson: Optional[Path] = None

    # --- Earth Engine auth / request sizing --------------------------------
    ee_project: Optional[str] = 'land-use-project-505711'
    # 
    ee_service_account_key: Optional[str] = None
    # High-volume endpoint is for many small downloads, not heavy compute.
    ee_high_volume: bool = False
    # Plots per interactive harmonic request. None = all plots in one call.
    harmonic_batch_size: Optional[int] = None
    # EE ``tileScale`` for ``reduceRegions`` (4-16 uses less memory per tile).
    harmonic_tile_scale: int = 4
    # Reuse CSVs / EE assets / local GeoTIFFs instead of re-running Earth Engine.
    reuse_ee_features: bool = True
    # Optional EE asset id for the county-wide feature image. Defaults to
    # projects/<ee_project>/assets/<slug>_pixel_features_for_rf.
    ee_feature_asset: Optional[str] = None
    # Poll interval while a batch Export.image.toAsset is running.
    export_poll_seconds: int = 45
    # Tile width/height (degrees) when pulling a *materialized* GeoTIFF locally.
    # Tiles that do not intersect the county polygon are skipped.
    export_tile_deg: float = 0.08
    export_min_tile_deg: float = 0.02

    # ---------------------------------------------------------------------
    # Derived paths (data/ layout). Created on demand by ``ensure_dirs``.
    # ---------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def models_dir(self) -> Path:
        """Fitted pickles and metrics live under ``data/models/`` (not the Python package)."""
        return self.data_dir / "models"

    @property
    def reference_date(self) -> str:
        return self.refdate or self.start_date

    @property
    def slug(self) -> str:
        """Filesystem-friendly area identifier used for output filenames."""
        return self.area_name.strip().lower().replace(" ", "_")

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.slug}_rf_best_model.pkl"

    @property
    def metrics_path(self) -> Path:
        return self.models_dir / f"{self.slug}_rf_metrics.json"

    @property
    def metrics_index_path(self) -> Path:
        """One-row-per-area table for comparing trained models."""
        return self.models_dir / "model_comparison.csv"

    @property
    def merged_features_path(self) -> Path:
        return self.processed_dir / f"{self.slug}_merged_features.csv"

    @property
    def harmonic_features_path(self) -> Path:
        return self.processed_dir / f"{self.slug}_harmonic_features.csv"

    @property
    def terrain_features_path(self) -> Path:
        return self.processed_dir / f"{self.slug}_terrain_features.csv"

    @property
    def precip_features_path(self) -> Path:
        return self.processed_dir / f"{self.slug}_precipitation.csv"

    @property
    def harmonic_band_dir(self) -> Path:
        """Per-band harmonic CSVs (local stand-in for the old Drive batches)."""
        return self.interim_dir / f"{self.slug}_harmonic_bands"

    @property
    def feature_image_path(self) -> Path:
        return self.outputs_dir / f"{self.slug}_pixel_features_for_rf.tif"

    @property
    def feature_asset_id(self) -> str:
        """Earth Engine asset used to materialize the county-wide feature image."""
        if self.ee_feature_asset:
            return self.ee_feature_asset
        project = self.ee_project or "earthengine-legacy"
        return f"projects/{project}/assets/{self.slug}_pixel_features_for_rf"

    @property
    def probability_map_path(self) -> Path:
        return self.outputs_dir / f"{self.slug}_probability_map.tif"

    @property
    def classified_map_path(self) -> Path:
        return self.outputs_dir / f"{self.slug}_classified_map.tif"

    @property
    def cropland_map_path(self) -> Path:
        return self.outputs_dir / f"{self.slug}_cropland_mask.tif"

    @property
    def pipeline_figure_path(self) -> Path:
        return self.outputs_dir / f"{self.slug}_pipeline_stages.png"

    @property
    def gaul_lookup_name(self) -> str:
        """GAUL ADM name: ``gaul_name`` if set, otherwise ``area_name``."""
        return (self.gaul_name or self.area_name).strip()

    @property
    def is_binary(self) -> bool:
        """True for the Nyandarua-style maize vs other classifier."""
        return str(self.label_mode or "binary").strip().lower() == "binary"

    def resolve_path(self, path: Optional[str | Path]) -> Optional[Path]:
        """Resolve a config path against ``project_root`` when it is relative."""
        if path is None:
            return None
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.project_root / p
        return p

    def ensure_dirs(self) -> None:
        """Create the data/ directory tree if missing."""
        for d in (
            self.raw_dir, self.interim_dir, self.processed_dir,
            self.outputs_dir, self.models_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # (De)serialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a config from a YAML file (unknown keys are ignored)."""
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML configs. `pip install pyyaml`.")
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "PipelineConfig":
        valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {}
        for key, value in raw.items():
            if key not in valid:
                continue
            if key in {"project_root", "eetc_path", "survey_geojson", "boundary_path"} and value is not None:
                value = Path(value).expanduser()
            kwargs[key] = value
        return cls(**kwargs)

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("project_root", "eetc_path", "survey_geojson", "boundary_path"):
            if d.get(key) is not None:
                d[key] = str(d[key])
        return d
