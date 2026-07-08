# Crop Type Mapping Pipeline

An end-to-end, **area-agnostic** pipeline for crop-type classification from
Sentinel-2 satellite imagery. It generalises the original `Nyandarua_Maize`
notebook so the same code can be pointed at *any* region and *any* target crop
by editing a single configuration file.

The workflow: prepare labelled survey plots → pull cloud-masked Sentinel-2 time
series → engineer harmonic-regression, terrain and precipitation features →
train a Random Forest → apply it wall-to-wall over the area of interest → mask
to cropland and threshold into a final crop map.

## Project structure

```
Crop_Type_Mapping_all/
├── run_pipeline.py          # CLI entrypoint
├── config.example.yaml      # copy to config.yaml and edit per area
├── requirements.txt
├── src/
│   ├── config.py            # PipelineConfig: all area-specific settings
│   ├── ee_utils.py          # Earth Engine init + eetc path helper
│   └── pipeline.py          # CropTypePipeline orchestrator
├── preprocessing/
│   ├── survey.py            # load survey, build target label, plot area
│   ├── boundary.py          # GAUL boundary + inference AOI
│   ├── sentinel.py          # Sentinel-2 acquisition + cloud masking
│   ├── timeseries.py        # point time-series extraction + harmonic fit
│   └── features.py          # harmonic / terrain / precip features + merge
├── models/
│   ├── train.py             # split, grid search, fit, persist
│   ├── inference.py         # AOI feature image + probability raster
│   └── postprocess.py       # cropland mask + threshold + plots
└── data/
    ├── raw/                 # your input survey GeoJSON (populate this)
    ├── interim/             # intermediate CSVs / batch exports
    ├── processed/           # merged feature table
    └── outputs/             # feature GeoTIFF, probability + classified maps
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Clone the Azzari et al. Earth Engine tools (used for Sentinel-2 datasources
   and harmonic regression) into the project root:

```bash
git clone https://github.com/shrutijain90/eetc.git
```

3. Authenticate Earth Engine (once):

```bash
earthengine authenticate
```

4. Copy and edit the config for your area:

```bash
cp config.example.yaml config.yaml
```

Put your labelled survey vector file in `data/raw/` and point
`survey_geojson` at it. Set `area_name` to the GAUL ADM2 name of your region
and `target_crop_names` to the crop(s) you want to map.

## Running

The pipeline is split into resumable stages because Earth Engine exports run
asynchronously on Google's servers.

```bash
# 1. Build the per-plot feature table and train the model
python run_pipeline.py train --config config.yaml

# 2. Start the wall-to-wall AOI feature-image export (async EE task)
python run_pipeline.py export-features --config config.yaml

# 3. Once the GeoTIFF is downloaded into data/outputs/, classify it
python run_pipeline.py infer --config config.yaml \
    --feature-tif data/outputs/<area>_pixel_features_for_rf.tif
```

Or drive it from Python:

```python
from src.config import PipelineConfig
from src.pipeline import CropTypePipeline

config = PipelineConfig.from_yaml("config.yaml")
pipeline = CropTypePipeline(config)
pipeline.setup_ee()

model, metrics = pipeline.run_training()   # features + training
pipeline.export_inference_features()       # start AOI export
# ... wait for the export, download the GeoTIFF to data/outputs/ ...
pipeline.run_inference()                   # probability + classified maps
```

## Adapting to a new area

Everything region-specific lives in `PipelineConfig`. To map, say, sorghum in a
different county you only change the config — no module edits:

```yaml
area_name: "Kitui"
target_crop_names: ["Sorghum"]
survey_geojson: "data/raw/Kitui.geojson"
start_date: "2023-10-01"
end_date: "2024-03-31"
area_calc_epsg: 32637   # UTM zone appropriate for the new region
```

## Notes

- The cloud mask now uses the correct server-side `SCL.eq(4).Or(SCL.eq(5))`
  combinator (the notebook's Python `or` was a latent bug).
- Feature order for inference is aligned to the training feature columns via
  `config.feature_columns`, so exported band order matches the model.
- For very large surveys use `features.export_harmonic_features_to_drive` +
  `features.concat_band_batches` instead of the in-memory extraction.
