# Data directory

Layout used by the pipeline (paths are configured in `src/config.py`):

- `raw/` — inputs you provide, e.g. the survey/training vector file
  (`Nyandarua.geojson`) with crop-label columns.
- `interim/` — intermediate artefacts (e.g. point time-series CSVs, per-band
  batch exports downloaded from Google Drive).
- `processed/` — the merged per-plot feature table used for training
  (`<area>_merged_features.csv`).
- `outputs/` — model outputs: the AOI feature GeoTIFF, probability raster and
  final classified crop map.

These folders are created automatically by `PipelineConfig.ensure_dirs()`.
Only `raw/` needs to be populated by hand.
