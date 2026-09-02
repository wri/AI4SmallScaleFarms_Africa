# Crop Type Mapping Pipeline

An area-agnostic pipeline for mapping crops from Sentinel-2 imagery. Labelled survey plots (or GPS points) are turned into harmonic, terrain and rainfall features, a Random Forest is trained, and the model is applied wall-to-wall across a county.

The workflow follows [The Crop Type Mapping Overview](https://learn.geo4.dev/Satellite%20Crop%20Mapping.html).
Point the same code at a new county or crop by editing a YAML config — no module changes are required.

Earth Engine computes the satellite features. **All results are written locally** under `data/` (CSVs, model pickles, GeoTIFFs). Google Drive export is optional (`--drive`) and is not required.

```
labelled survey  →  cloud-masked Sentinel-2  →  harmonic / terrain / CHIRPS
                 →  Random Forest  →  probability raster  →  cropland-masked map
```

## Project structure

```
AI4SmallScaleFarms_Africa/
├── run_pipeline.py                   # CLI: features | train | export-features | infer | all | smoke
├── config/
│   ├── config.example.yaml           # template for a new area
│   ├── config.nyandarua_smoke.yaml   # binary maize, ~4 km Ndaragwa test
│   ├── config.nyandarua.yaml         # binary maize, full Nyandarua (polygons)
│   └── config.nakuru.yaml            # multiclass crops, Nakuru (GPS points)
├── scripts/
│   ├── prepare_nyandarua_smoke.py    # clip Nyandarua plots to the smoke bbox
│   └── visualize_pipeline.py         # 4-panel PNG of pipeline stages
├── src/
│   ├── config.py                     # PipelineConfig (all area-specific settings)
│   ├── ee_utils.py                   # Earth Engine init + eetc path
│   └── pipeline.py                   # CropTypePipeline orchestrator
├── preprocessing/
│   ├── survey.py                     # load labels (binary or multiclass), plot area
│   ├── boundary.py                   # local shapefile or GAUL ADM2 fallback
│   ├── sentinel.py                   # Sentinel-2 + SCL cloud mask
│   ├── timeseries.py                 # optional point-level diagnostics (not in CLI)
│   └── features.py                   # harmonics, terrain, CHIRPS, merge
├── models/
│   ├── train.py                      # split, grid search, persist RF
│   ├── inference.py                  # AOI feature image + probability raster
│   └── postprocess.py                # GFSAD cropland mask, classify, figure
└── data/
    ├── raw/                          # survey GeoJSON + optional county shapefile
    ├── interim/                      # per-band harmonic CSVs
    ├── processed/                    # concatenated + merged feature tables
    ├── models/                       # RF pickles + metrics JSON / comparison CSV
    └── outputs/                      # GeoTIFFs, classified map, PNG figure
```

## Setup

1. Install dependencies (use the `gee` conda environment if you already have one):

```bash
pip install -r requirements.txt
```

2. Clone the Azzari et al. Earth Engine tools into the project root (Sentinel-2
   datasources and harmonic regression):

```bash
git clone https://github.com/shrutijain90/eetc.git
```

3. Authenticate Earth Engine once, and set `ee_project` in your YAML to a
   Google Cloud / Earth Engine project you can write assets to:

```bash
earthengine authenticate
```

4. Put labelled survey vectors (and an optional county shapefile) under
   `data/raw/` and choose a config below.

## Input data

Each county run needs a **survey GeoJSON** (`survey_geojson`). A local **county boundary** (`boundary_path`) is optional: if the file is missing, the pipeline uses FAO GAUL level-2 filtered by `gaul_name`.

Survey features must include:

| Field | Role |
|---|---|
| `geometry` | Plot **polygon** (zonal mean) or GPS **point** (one 30 m pixel). Mixes are not supported. |
| `fid` | Unique integer plot ID. Used to join Earth Engine features. |
| `crop_a` … `crop_e` | Crop names as text. Unused slots may be `null`. Matching is case-insensitive. |

Optional attributes (`Farming_type`, `sub-county`, `county`,
`date_of_collection`, `crop_growth_stage`) are carried through to the training
table but are not required.

Sampling is controlled by `sample_geometry`:

- `auto` (default) — polygons → zonal mean; points → the 30 m pixel under the GPS
- `point` / `polygon` — force one mode

Nyandarua uses plot polygons. Nakuru uses GPS points with the same attribute
schema. Do not train a polygon-trained model on a point survey (or the reverse)
without regenerating features.

`gaul_name` is the FAO GAUL ADM2 name (`Nyandarua`, `Nakuru`). `area_name` is
only a run label used in output filenames (for example `Nyandarua_smoke`).

Optional `aoi_bbox: [west, south, east, north]` clips imagery and inference to
a rectangle intersected with the county polygon.

## Binary vs multiclass classification

The same pipeline supports both modes. The switch is `label_mode` in the YAML.
**Train and infer with the same mode.** A binary pickle cannot be applied to a
multiclass feature run, and vice versa. Use `--model` only when the pickle and
the config share the same `label_mode` and `feature_columns`.

### Binary (`label_mode: binary`)

Default. Any of `crop_a`…`crop_e` matching `target_crop_names` becomes 1;
everything else is 0. The target column is `target_column` (default
`maize_pos`).

- Probability GeoTIFF: **one band** — P(target crop)
- Classified map: cropland mask, then `probability_threshold` (default 0.5)
- Metrics: accuracy, precision, recall, F1 on the positive class

Nyandarua maize vs other:

```yaml
label_mode: binary
target_crop_names: ["Maize"]
crop_columns: ["crop_a", "crop_b", "crop_c", "crop_d", "crop_e"]
target_column: maize_pos
probability_threshold: 0.5
```

To map a different single crop, change `target_crop_names` (and usually
`target_column` / `area_name`) and retrain.

### Multiclass (`label_mode: multiclass`)

The source column is `label_column` (default `crop_a`). Names are lowercased
and passed through `crop_aliases` (typos such as `whear` → `wheat`). Classes
with fewer than `min_class_count` samples are collapsed to
`other_class_name` (`other`). Integer codes are 1…N; 0 is nodata on maps.

- Probability GeoTIFF: **one band per class**, named and tagged
- Classified map: cropland mask, then **argmax** over class probabilities
  (`probability_threshold` is unused)
- Metrics: accuracy plus macro / weighted precision, recall, F1, and per-class
  scores. Set `class_weight: balanced` when classes are imbalanced.

Nakuru crop types:

```yaml
label_mode: multiclass
label_column: crop_a
target_column: crop_class_code
min_class_count: 10
other_class_name: other
class_weight: balanced
sample_geometry: point
```

`preferred_class_order` controls raster code order for named crops that survive
the count filter; remaining frequent classes are appended, then `other`.

## Configs

| File | Mode | What it runs |
|---|---|---|
| `config/config.nyandarua_smoke.yaml` | binary | Small Ndaragwa box (~4 km, 14 plots). Fast Earth Engine check. |
| `config/config.nyandarua.yaml` | binary | Full Nyandarua county (~161 plot polygons). |
| `config/config.nakuru.yaml` | multiclass | Full Nakuru county (GPS points; rare `crop_a` classes → `other`). |
| `config/config.example.yaml` | binary | Template for a new area / crop. |

## Running the pipeline

From the project root, with Earth Engine authenticated:

```bash
python run_pipeline.py <stage> --config <yaml> [options]
```

### Stages

| Stage | What it does |
|---|---|
| `features` | Load survey labels, extract Earth Engine harmonics / terrain / CHIRPS, write `data/processed/<area>_merged_features.csv`. |
| `train` | Reuse that table if it exists, otherwise build it, then fit the Random Forest. |
| `export-features` | Materialize the county-wide feature image to an Earth Engine asset, download intersecting GeoTIFF tiles, write the GFSAD cropland mask. |
| `infer` | Apply the trained model to the feature GeoTIFF; write probability + classified maps and the 4-panel PNG. Builds the GeoTIFF locally if it is missing. |
| `all` | `train` → local `export-features` → `infer`. |
| `smoke` | Delete previous artefacts for this `area_name`, then `all`. Use with the smoke config. |

Stages are resumable. `train` will not re-hit Earth Engine if
`data/processed/<area>_merged_features.csv` already exists.
`reuse_ee_features: true` (default) also reuses per-band CSVs and an existing
Earth Engine feature asset.

### CLI options

| Option | Applies to | Purpose |
|---|---|---|
| `--config PATH` | all stages | YAML config. Required in practice; without it the Nyandarua defaults in `src/config.py` are used. |
| `--survey PATH` | `features`, `train`, `all`, `smoke` | Override `survey_geojson`. |
| `--model PATH` | `infer` | Use a pickle trained in another area (same `label_mode` and feature recipe). |
| `--feature-tif PATH` | `infer` | Use an existing AOI feature GeoTIFF instead of `data/outputs/<area>_pixel_features_for_rf.tif`. |
| `--drive` | `export-features`, `all` | Start Google Drive export tasks instead of downloading GeoTIFFs locally. Re-run `infer` once the file is in `data/outputs/`. |

Downloads are **local by default**. County-scale GeoTIFFs are tiled at
`export_tile_deg` (0.08°) and mosaicked if a single `getDownloadURL` is too
large.

### Binary maize — Nyandarua

```bash
# Recommended first run: small AOI, rebuilds artefacts, hits Earth Engine
python run_pipeline.py smoke --config config/config.nyandarua_smoke.yaml

# Full county
python run_pipeline.py all --config config/config.nyandarua.yaml
```

Or stage by stage:

```bash
python run_pipeline.py features --config config/config.nyandarua.yaml
python run_pipeline.py train --config config/config.nyandarua.yaml
python run_pipeline.py export-features --config config/config.nyandarua.yaml
python run_pipeline.py infer --config config/config.nyandarua.yaml
```

### Multiclass crops — Nakuru

Train a **new** model on Nakuru labels. Do not pass the Nyandarua maize pickle.

```bash
python run_pipeline.py train --config config/config.nakuru.yaml
python run_pipeline.py export-features --config config/config.nakuru.yaml
python run_pipeline.py infer --config config/config.nakuru.yaml

# or end-to-end
python run_pipeline.py all --config config/config.nakuru.yaml
```

### Cross-area inference

Apply a model from one area to another area's feature image only when both
runs use the same label mode and the same `feature_columns`:

```bash
python run_pipeline.py infer --config config/config.nyandarua.yaml \
    --model data/models/nyandarua_rf_best_model.pkl
```

### From Python

```python
from src.config import PipelineConfig
from src.pipeline import CropTypePipeline

config = PipelineConfig.from_yaml("config/config.nyandarua.yaml")
pipeline = CropTypePipeline(config)
pipeline.setup_ee()
pipeline.run_all()
```

A smoke test on the Ndaragwa box (14 plots) completed end-to-end against live
Earth Engine in about 29 minutes. Almost all of that was harmonic extraction
and the feature GeoTIFF download. Full-county runs are much longer.

Smoke-test accuracy on a handful of hold-out plots is not a county-scale
metric. Use `config.nyandarua.yaml` or `config.nakuru.yaml` for evaluation.

## Pipeline stages (detail)

1. **Survey** (`preprocessing/survey.py`) — load GeoJSON; build a binary or
   multiclass target; plot area in the configured UTM zone (polygons only);
   cast `fid` to float.
2. **Boundary** (`preprocessing/boundary.py`) — local shapefile if present,
   otherwise GAUL ADM2.
3. **Sentinel-2** (`preprocessing/sentinel.py`) — L2A (`COPERNICUS/S2_SR_HARMONIZED`)
   for `[start_date, end_date]`, SCL classes 4 (vegetation) or 5 (bare soil).
4. **Per-plot features** (`preprocessing/features.py`) — Earth Engine
   `reduceRegions` (polygon mean) or `reduceRegions` with `first` (point),
   written locally:
   - harmonics (`n_harmonics: 2`) per band → `data/interim/<area>_harmonic_bands/*.csv`,
     concatenated to `data/processed/<area>_harmonic_features.csv`
   - SRTM elevation / slope / aspect → `..._terrain_features.csv`
   - CHIRPS seasonal **mean** → `..._precipitation.csv`
   - merge on `fid` → `data/processed/<area>_merged_features.csv`
5. **Train** (`models/train.py`) — stratified split, Random Forest grid search,
   pickle + metrics JSON under `data/models/`.
6. **Inference image** (`models/inference.py`) — the same feature recipe at
   every 30 m pixel → `data/outputs/<area>_pixel_features_for_rf.tif`.
7. **Post-process** (`models/postprocess.py`) — USGS GFSAD1000 cropland
   (classes 2–6), then threshold (binary) or argmax (multiclass).

`preprocessing/timeseries.py` is diagnostic only (GCVI curves at a point) and
is not called by the orchestrator.

## Visualisation

After inference (or any time the artefacts exist):

```bash
python scripts/visualize_pipeline.py --config config/config.nyandarua.yaml
python scripts/visualize_pipeline.py --config config/config.nakuru.yaml
```

Writes `data/outputs/<area>_pipeline_stages.png`:

1. Survey locations in the county boundary
2. Feature layer (`GCVI_mean` by default; override with `--feature-band`)
3. Predicted probability (binary) or argmax class map (multiclass)
4. Cropland-masked classified map

`all` / `infer` / `smoke` also write this PNG automatically. Optional:
`--out path.png`.

To rebuild the 14-plot Nyandarua smoke subset:

```bash
python scripts/prepare_nyandarua_smoke.py
```

## Artefacts

Filenames use the slug of `area_name` (lowercase, spaces → `_`).

| Path | Contents |
|---|---|
| `data/interim/<area>_harmonic_bands/` | One CSV per regression band |
| `data/processed/<area>_harmonic_features.csv` | Concatenated harmonics |
| `data/processed/<area>_terrain_features.csv` | Elevation, slope, aspect |
| `data/processed/<area>_precipitation.csv` | Seasonal CHIRPS mean |
| `data/processed/<area>_merged_features.csv` | Training table (survey + features) |
| `data/models/<area>_rf_best_model.pkl` | Fitted Random Forest |
| `data/models/<area>_rf_metrics.json` | Test / CV scores (per-class for multiclass) |
| `data/models/model_comparison.csv` | One row per trained area |
| `data/outputs/<area>_pixel_features_for_rf.tif` | Wall-to-wall feature stack |
| `data/outputs/<area>_cropland_mask.tif` | GFSAD binary cropland |
| `data/outputs/<area>_probability_map.tif` | P(target) or one band per class |
| `data/outputs/<area>_classified_map.tif` | Final crop map |
| `data/outputs/<area>_pipeline_stages.png` | Four-panel summary figure |

## Adapting to a new area

Copy `config/config.example.yaml` and change only the YAML.

Binary sorghum example:

```yaml
area_name: "Kitui"
gaul_name: "Kitui"
label_mode: binary
target_crop_names: ["Sorghum"]
target_column: sorghum_pos
survey_geojson: "data/raw/Kitui.geojson"
# boundary_path: "data/raw/Kitui/Kitui.shp"   # optional; else GAUL
start_date: "2023-10-01"
end_date: "2024-03-31"
area_calc_epsg: 32637              # UTM zone for planar plot area
```

Multiclass example (same pattern as Nakuru):

```yaml
area_name: "Kitui"
gaul_name: "Kitui"
label_mode: multiclass
label_column: crop_a
target_column: crop_class_code
min_class_count: 10
class_weight: balanced
survey_geojson: "data/raw/Kitui.geojson"
sample_geometry: auto
```

## Notes

- Feature-image band order follows `config.feature_columns` so it matches the
  trained model. Do not mix pickles across configs with different feature lists.
- CHIRPS uses temporal **mean** on both the training table and the inference
  image so precipitation is on the same scale.
- Override the Sentinel-2 collection with `s2_sr_asset` if needed; the default
  is `COPERNICUS/S2_SR_HARMONIZED`.
- County-wide features are batch-exported to
  `projects/<ee_project>/assets/<area>_pixel_features_for_rf`, then tiled
  locally. Tiles that miss the county polygon are skipped.
