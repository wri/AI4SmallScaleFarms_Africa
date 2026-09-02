# Crop Type Mapping Pipeline

An end-to-end, **area-agnostic** pipeline for crop-type classification from
Sentinel-2 satellite imagery. It generalises the original
`Nyandarua Maize Crop Mapping.ipynb` notebook so the same code can be pointed
at *any* region and *any* target crop by editing a YAML config.

The workflow: labelled survey plots → cloud-masked Sentinel-2 time series →
harmonic, terrain and precipitation features → Random Forest → wall-to-wall
probability map → cropland mask and threshold → classified crop map.

Earth Engine computes the satellite features; **all results are written
locally** under `data/` (per-band CSVs, concatenated tables, GeoTIFFs). Google
Drive export is optional (`--drive`), not required.

## Project structure

```
AI4SmallScaleFarms_Africa/
├── run_pipeline.py
├── config/
│   ├── config.example.yaml           # template for a new area
│   ├── config.nyandarua_smoke.yaml   # ~4 km Ndaragwa end-to-end test
│   └── config.nyandarua.yaml         # full Nyandarua county
├── scripts/
│   ├── prepare_nyandarua_smoke.py    # clip survey plots to the smoke bbox
│   └── visualize_pipeline.py         # 4-panel PNG of pipeline stages
├── src/
│   ├── config.py                     # PipelineConfig (all area-specific settings)
│   ├── ee_utils.py                   # Earth Engine init + eetc path
│   └── pipeline.py                   # CropTypePipeline orchestrator
├── preprocessing/
│   ├── survey.py                     # load plots, maize label, plot area
│   ├── boundary.py                   # local shapefile or GAUL ADM2 fallback
│   ├── sentinel.py                   # Sentinel-2 + SCL cloud mask
│   ├── timeseries.py                 # optional point-level diagnostics
│   └── features.py                   # harmonics, terrain, CHIRPS, merge
├── models/
│   ├── train.py                      # split, grid search, persist RF
│   ├── inference.py                  # AOI feature image + probability raster
│   └── postprocess.py                # GFSAD cropland mask, threshold, figure
└── data/
    ├── raw/                          # survey GeoJSON + optional county shapefile
    ├── interim/                      # per-band harmonic CSVs
    ├── processed/                    # concatenated + merged feature tables
    ├── models/                       # RF pickles + metrics JSON / comparison CSV
    └── outputs/                      # GeoTIFFs, classified map, PNG figure
```

## Setup

1. Install dependencies (use the `gee` conda env if you already have one):

```bash
pip install -r requirements.txt
```

2. Clone the Azzari et al. Earth Engine tools into the project root (Sentinel-2
   datasources and harmonic regression):

```bash
git clone https://github.com/shrutijain90/eetc.git
```

3. Authenticate Earth Engine once:

```bash
earthengine authenticate
```

4. Put labelled survey vectors in `data/raw/` and pick a config (see below).

## Configs

| File | What it runs |
|---|---|
| `config/config.nyandarua_smoke.yaml` | Small Ndaragwa box (~4 km, 14 plots). Fast EE check. |
| `config/config.nyandarua.yaml` | Full Nyandarua county (~161 plots). Production. |
| `config/config.example.yaml` | Template for a new area / crop. |

Set `ee_project` to your Google Cloud / Earth Engine project. `gaul_name` is
the FAO GAUL ADM2 name (`Nyandarua`); `area_name` can be a run label such as
`Nyandarua_smoke` used in output filenames.

### Boundary fallback

`boundary.py` uses a local county shapefile when `boundary_path` exists on
disk. If the path is unset or the file is missing, it falls back to
`FAO/GAUL/2015/level2` filtered by `gaul_name` (notebook cell 8). Smoke and
county configs both set `gaul_name: "Nyandarua"` so the GAUL lookup stays
correct even when `area_name` is a run label.

Optional `aoi_bbox: [west, south, east, north]` clips imagery and inference to
a rectangle intersected with the county polygon.

## Running

From the project root, with Earth Engine authenticated:

```bash
# Recommended first run: small-area smoke test (rebuilds artefacts, hits EE)
python run_pipeline.py smoke --config config/config.nyandarua_smoke.yaml

# Full Nyandarua county (local CSVs + GeoTIFFs, no Drive)
python run_pipeline.py all --config config/config.nyandarua.yaml
```

Stages (resumable; `train` reuses `data/processed/<area>_merged_features.csv`
if it already exists):

```bash
python run_pipeline.py features --config config/config.nyandarua.yaml
python run_pipeline.py train --config config/config.nyandarua.yaml
python run_pipeline.py export-features --config config/config.nyandarua.yaml
python run_pipeline.py infer --config config/config.nyandarua.yaml
```

- `features` — survey labels + EE harmonics / terrain / CHIRPS → processed CSVs
- `train` — build the table if needed, then fit the Random Forest
- `export-features` — wall-to-wall feature GeoTIFF + GFSAD mask into `data/outputs/`
- `infer` — probability + classified map; builds the GeoTIFF locally if missing
- `all` — train → local export → infer → 4-panel PNG
- `smoke` — delete previous artefacts, then `all` on the smoke config

Downloads are **local by default**. County-scale GeoTIFFs are tiled at
`export_tile_deg` (0.08°) and mosaicked if a single `getDownloadURL` is too
large. Pass `--drive` only if you want an asynchronous Google Drive export.

From Python:

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
(~24 min) and the feature GeoTIFF download (~5 min). Full-county runs are
much longer, especially the wall-to-wall GeoTIFF.

## Data flow (Nyandarua)

1. **Survey** (`preprocessing/survey.py`) — load GeoJSON, binary `maize_pos`
   from `crop_a`…`crop_e` (case-insensitive), plot area in UTM 36N, float `fid`.
2. **Boundary** (`preprocessing/boundary.py`) — local shapefile, else GAUL.
3. **Sentinel-2** (`preprocessing/sentinel.py`) — L2A for the configured
   season, SCL classes 4 (vegetation) or 5 (bare soil). Uses server-side `.Or`
   (the notebook’s Python `or` was a latent bug).
4. **Per-plot features** (`preprocessing/features.py`) — Earth Engine
   `reduceRegions` + `getInfo`, written locally:
   - harmonics (`n_harmonics: 2`) per band → `data/interim/<area>_harmonic_bands/*.csv`
     then concatenated to `data/processed/<area>_harmonic_features.csv`
   - SRTM elevation / slope / aspect → `..._terrain_features.csv`
   - CHIRPS seasonal **mean** (notebook inference cell 97) → `..._precipitation.csv`
   - merge on `fid` → `data/processed/<area>_merged_features.csv`
5. **Train** (`models/train.py`) — stratified split, Random Forest grid search,
   `data/models/<area>_rf_best_model.pkl` plus metrics JSON.
6. **Inference image** (`models/inference.py`) — same feature recipe at every
   30 m pixel → `data/outputs/<area>_pixel_features_for_rf.tif`.
7. **Post-process** (`models/postprocess.py`) — GFSAD1000 cropland (classes
   2–6; the notebook’s `ImageCollection` load was incorrect, GFSAD is an
   `Image`) then probability ≥ 0.5 → classified map.

`preprocessing/timeseries.py` is diagnostic only (GCVI curves at a point) and
is not called by the orchestrator.

## Visualisation

After inference (or any time the artefacts exist):

```bash
python scripts/visualize_pipeline.py --config config/config.nyandarua_smoke.yaml
python scripts/visualize_pipeline.py --config config/config.nyandarua.yaml
```

Writes `data/outputs/<area>_pipeline_stages.png`:

1. Survey plots in the county boundary (maize vs other; dashed AOI if set)
2. Feature layer (`GCVI_mean` by default; override with `--feature-band`)
3. Predicted maize probability
4. Cropland-masked, thresholded classified map

`all` / `infer` / `smoke` also write this PNG automatically. Optional:
`--out path.png`.

To rebuild the 14-plot smoke survey subset:

```bash
python scripts/prepare_nyandarua_smoke.py
```

## Artefacts

| Path | Contents |
|---|---|
| `data/interim/<area>_harmonic_bands/` | One CSV per regression band |
| `data/processed/<area>_harmonic_features.csv` | Concatenated harmonics |
| `data/processed/<area>_terrain_features.csv` | Elevation, slope, aspect |
| `data/processed/<area>_precipitation.csv` | Seasonal CHIRPS mean |
| `data/processed/<area>_merged_features.csv` | Training table (survey + features) |
| `data/models/<area>_rf_best_model.pkl` | Fitted Random Forest |
| `data/models/<area>_rf_metrics.json` | Test/CV accuracy, precision, recall, F1 |
| `data/models/model_comparison.csv` | One row per trained area |
| `data/outputs/<area>_pixel_features_for_rf.tif` | Wall-to-wall feature stack |
| `data/outputs/<area>_cropland_mask.tif` | GFSAD binary cropland |
| `data/outputs/<area>_probability_map.tif` | P(target crop) |
| `data/outputs/<area>_classified_map.tif` | Final binary crop map |
| `data/outputs/<area>_pipeline_stages.png` | Four-panel summary figure |

## Adapting to a new area

Everything region-specific lives in `PipelineConfig`. Copy
`config/config.example.yaml` and change only the YAML — no module edits:

```yaml
area_name: "Kitui"
gaul_name: "Kitui"                 # GAUL ADM2 name if area_name is a run label
target_crop_names: ["Sorghum"]
survey_geojson: "data/raw/Kitui.geojson"
# boundary_path: "data/raw/Kitui/Kitui.shp"   # optional; else GAUL
start_date: "2023-10-01"
end_date: "2024-03-31"
area_calc_epsg: 32637              # UTM zone for the new region
```

## Notes

- Feature-image band order follows `config.feature_columns` so it matches the
  trained model (notebook cell 79 / 98).
- CHIRPS uses temporal **mean** on both the training table and the inference
  image so the precipitation feature is on the same scale.
- `eetc` is pinned to `COPERNICUS/S2_SR_HARMONIZED` (the old `S2_SR` collection
  is deprecated). Override with `s2_sr_asset` in the YAML if needed.
- Smoke-test RF accuracy on 4 hold-out plots is not a county-scale metric;
  use `config.nyandarua.yaml` (full survey + default grid) for real evaluation.
