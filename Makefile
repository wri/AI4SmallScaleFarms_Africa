# Makefile for FTW baselines: training, testing, inference, polygonize, and export to shapefile
#
# Override variables on the command line or in the environment, e.g.:
#   make train CONFIG=configs/prue_efnet_b7_finetune_config_spot.yaml
#   make inference INPUT_IMAGE=tests/kenya_tile_32636_00049_-0011_20190916.tif MODEL=path/to/checkpoint.ckpt

# --- Training ---
CONFIG       ?= configs/prue_efnet_b7_finetune_config_spot.yaml
CKPT_PATH    ?=  # optional: path to checkpoint to resume from

# --- Testing ---
TEST_MODEL   ?= $(CKPT_PATH)  # path to .ckpt for evaluation
TEST_COUNTRIES ?= kenya_spot_processed
DATA_DIR     ?= ./data/ftw
TEST_OUT     ?= metrics.csv
GPU          ?= 0
IOU_THRESHOLD ?= 0.5
TEST_ON_3    ?=  # set to 1 for 3-class eval: make test TEST_ON_3=1
MODEL_3_CLASSES ?=  # set to 1 if model predicts 3 classes
TEMPORAL_OPTIONS ?= random_window
SINGLE_WINDOW_SUBDIR ?= scaled
SINGLE_WINDOW_CHANNELS ?= 3

# --- Inference ---
INPUT_IMAGE  ?= tests/kenya_tile_32636_00049_-0011_20190916.tif
MODEL        ?=  # required: model name from registry OR path to .ckpt
INFERENCE_OUT ?=  # default: inference.<basename of INPUT_IMAGE>
INPUT_SCALE  ?= 255  # use 255 for 0-255 uint8 imagery; leave empty for reflectance-like 0-10000
GPU_INF      ?= -1  # -1 = CPU
OVERWRITE    ?=  # set to 1 to pass --overwrite

# --- Polygonize ---
POLYGONIZE_INPUT ?= $(INFERENCE_OUT)  # usually the inference output .tif
POLYGONS_OUT ?=  # default: input name with .parquet extension
SIMPLIFY     ?= 15
MIN_SIZE     ?= 500

# --- Scale uint8 [0-255] to uint16 [0-10000] (training-compatible) ---
SCALE_INPUT  ?= tests/kenya_tile_32636_00049_-0011_20190916.tif
SCALE_OUT    ?=  # default: <stem>_scaled.tif next to SCALE_INPUT

# --- Convert to Shapefile ---
SHP_INPUT    ?=  # parquet or geojson file to convert
SHP_OUT      ?=  # output .shp path (optional; default: same basename as SHP_INPUT)

.PHONY: help train test inference polygonize to-shp run-all scale-image scale-test-image

help:
	@echo "FTW baselines Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  train        Train a model (set CONFIG=..., optionally CKPT_PATH=...)"
	@echo "  test        Run evaluation (set TEST_MODEL=path/to.ckpt, TEST_COUNTRIES=...)"
	@echo "  inference   Run inference on an image (set INPUT_IMAGE=..., MODEL=...)"
	@echo "  polygonize  Polygonize inference .tif (set POLYGONIZE_INPUT=..., optionally POLYGONS_OUT=...)"
	@echo "  to-shp      Convert parquet or geojson to shapefile (set SHP_INPUT=..., optionally SHP_OUT=...)"
	@echo "  scale-image Scale uint8 [0-255] TIFF to uint16 [0-10000] (set SCALE_INPUT=..., optionally SCALE_OUT=...)"
	@echo "  scale-test-image  Scale the Kenya test tile to uint16 (no args)."
	@echo "  run-all     inference -> polygonize (set INPUT_IMAGE=..., MODEL=...)"
	@echo ""
	@echo "Examples:"
	@echo "  make train CONFIG=configs/prue_efnet_b7_finetune_config_spot.yaml"
	@echo "  make test TEST_MODEL=PRUE_EFNET_B7_finetune_spot/lightning_logs/version_6/checkpoints/epoch=283-val_loss=0.02.ckpt TEST_COUNTRIES=kenya_spot_processed TEST_ON_3=1"
	@echo "  make inference INPUT_IMAGE=scene.tif MODEL=path/to.ckpt INPUT_SCALE=255 OVERWRITE=1"
	@echo "  make polygonize POLYGONIZE_INPUT=inference_scene.tif POLYGONS_OUT=scene_polygons.parquet OVERWRITE=1"
	@echo "  make to-shp SHP_INPUT=scene_polygons.parquet SHP_OUT=scene_polygons.shp"
	@echo "  make scale-test-image   # scale tests/kenya_tile_*.tif to uint16"
	@echo "  make inference INPUT_IMAGE=tests/kenya_tile_32636_00049_-0011_20190916_scaled.tif MODEL=path/to.ckpt  # no --input_scale"

# --- Scale uint8 to uint16 [0-10000] (same as training chips) ---
_scale_out = $(if $(SCALE_OUT),-o $(SCALE_OUT),)
_scale_overwrite = $(if $(OVERWRITE),--overwrite,)
scale-image:
	@if [ -z "$(SCALE_INPUT)" ]; then echo "Set SCALE_INPUT=path/to/image.tif"; exit 1; fi
	uv run python scripts/scale_to_uint16.py $(SCALE_INPUT) $(_scale_out) $(_scale_overwrite)

# Scale the Kenya test tile; output: tests/kenya_tile_32636_00049_-0011_20190916_scaled.tif
scale-test-image:
	$(MAKE) scale-image SCALE_INPUT=tests/kenya_tile_32636_00049_-0011_20190916.tif

# --- Training ---
train:
	ftw model fit --config $(CONFIG) $(if $(CKPT_PATH),--ckpt_path $(CKPT_PATH),)

# --- Testing ---
_test_flags = $(if $(TEST_ON_3),--test_on_3_classes,) $(if $(MODEL_3_CLASSES),--model_predicts_3_classes,) \
	--temporal_options $(TEMPORAL_OPTIONS) $(if $(SINGLE_WINDOW_SUBDIR),--single_window_subdir $(SINGLE_WINDOW_SUBDIR),) \
	$(if $(SINGLE_WINDOW_CHANNELS),--single_window_channels $(SINGLE_WINDOW_CHANNELS),)
test:
	@if [ -z "$(TEST_MODEL)" ]; then echo "Set TEST_MODEL=path/to/checkpoint.ckpt"; exit 1; fi
	ftw model test --model "$(TEST_MODEL)" --countries $(TEST_COUNTRIES) --dir $(DATA_DIR) \
		--gpu $(GPU) --iou_threshold $(IOU_THRESHOLD) --out $(TEST_OUT) $(_test_flags)

# --- Inference ---
_inf_out = $(if $(INFERENCE_OUT),--out $(INFERENCE_OUT),)
_inf_scale = $(if $(INPUT_SCALE),--input_scale $(INPUT_SCALE),)
_inf_overwrite = $(if $(OVERWRITE),--overwrite,)
inference:
	@if [ -z "$(MODEL)" ]; then echo "Set MODEL=model_name_or_path/to/checkpoint.ckpt"; exit 1; fi
	ftw inference run $(INPUT_IMAGE) --model $(MODEL) --gpu $(GPU_INF) $(_inf_out) $(_inf_scale) $(_inf_overwrite)

# --- Polygonize ---
_poly_out = $(if $(POLYGONS_OUT),--out $(POLYGONS_OUT),)
_poly_overwrite = $(if $(OVERWRITE),--overwrite,)
polygonize:
	ftw inference polygonize $(POLYGONIZE_INPUT) $(_poly_out) --simplify $(SIMPLIFY) --min_size $(MIN_SIZE) $(_poly_overwrite)

# --- Convert parquet or geojson to shapefile ---
_shp_out := $(or $(SHP_OUT),$(basename $(SHP_INPUT)).shp)
to-shp:
	@if [ -z "$(SHP_INPUT)" ]; then echo "Set SHP_INPUT=path/to/file.parquet or .geojson"; exit 1; fi
	uv run python -c "import geopandas as gpd; gpd.read_file('$(SHP_INPUT)').to_file('$(_shp_out)')"
	@echo "Wrote $(_shp_out)"

# --- Convenience: inference then polygonize (set INPUT_IMAGE, MODEL) ---
# ftw default out is dirname(INPUT)/inference.basename(INPUT); we use same pattern
_run_inf_out := $(or $(INFERENCE_OUT),inference.$(notdir $(INPUT_IMAGE)))
_run_poly_out := $(or $(POLYGONS_OUT),$(basename $(_run_inf_out)).parquet)
run-all:
	@if [ -z "$(INPUT_IMAGE)" ]; then echo "Set INPUT_IMAGE=path/to/image.tif"; exit 1; fi
	@if [ -z "$(MODEL)" ]; then echo "Set MODEL=model_name_or_path/to/checkpoint.ckpt"; exit 1; fi
	$(MAKE) inference INPUT_IMAGE=$(INPUT_IMAGE) MODEL=$(MODEL) INFERENCE_OUT=$(_run_inf_out) INPUT_SCALE=$(INPUT_SCALE) GPU_INF=$(GPU_INF) OVERWRITE=$(OVERWRITE)
	$(MAKE) polygonize POLYGONIZE_INPUT=$(_run_inf_out) POLYGONS_OUT=$(_run_poly_out) SIMPLIFY=$(SIMPLIFY) MIN_SIZE=$(MIN_SIZE) OVERWRITE=$(OVERWRITE)
