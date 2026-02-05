# FTW Training Workflow Documentation

## Overview

This document provides a comprehensive guide to the Fields of The World (FTW) training workflow, covering the complete pipeline from data preprocessing to model testing. The FTW training system is built on PyTorch Lightning and uses a modular architecture for semantic segmentation of agricultural field boundaries.

---

## Table of Contents

1. [Data Preprocessing](#1-data-preprocessing)
2. [Dataset Loading and Configuration](#2-dataset-loading-and-configuration)
3. [Data Augmentation](#3-data-augmentation)
4. [Model Architecture](#4-model-architecture)
5. [Loss Functions](#5-loss-functions)
6. [Training Process](#6-training-process)
7. [Validation](#7-validation)
8. [Model Testing](#8-model-testing)
9. [Configuration Files](#9-configuration-files)
10. [Metrics and Evaluation](#10-metrics-and-evaluation)

---

## 1. Data Preprocessing

### 1.1 Dataset Structure

The FTW dataset is organized by country, with each country containing:

- **Sentinel-2 Imagery**: Two temporal windows (Window A and Window B)
  - `s2_images/window_a/`: Early season imagery
  - `s2_images/window_b/`: Late season imagery
- **Label Masks**:
  - `label_masks/semantic_2class/`: Binary masks (background, field)
  - `label_masks/semantic_3class/`: Three-class masks (background, field, boundary)
  - `label_masks/instance/`: Instance segmentation masks
- **Metadata**: `chips_{country}.parquet` containing split information (train/val/test)

### 1.2 Temporal Options

The dataset supports multiple temporal fusion strategies:

- **`stacked`**: Concatenates Window B and Window A (8 channels: 4 bands × 2 windows)
- **`windowA`**: Uses only Window A (4 channels)
- **`windowB`**: Uses only Window B (4 channels)
- **`median`**: Computes pixel-wise median across both windows (4 channels)
- **`rgb`**: Uses RGB bands from both windows (6 channels: 3 bands × 2 windows)
- **`random_window`**: Randomly selects one window per sample (4 channels)

### 1.3 Data Normalization

Images are normalized using one of two approaches:

1. **Fixed Normalization** (default):
   - Images divided by 3000 (typical Sentinel-2 L2A radiance scale)
   - Mean: `[0, 0, 0, ...]` (zeros)
   - Std: `[3000, 3000, 3000, ...]` (per channel)

2. **Random Divisor Normalization** (augmentation):
   - Training batches divided by random scalar in range [1500, 4500]
   - Applied only during training to improve generalization
   - Validation/test uses fixed 3000 normalization

### 1.4 Data Splits

The dataset is split into:
- **Training set**: Used for model optimization
- **Validation set**: Used for hyperparameter tuning and early stopping
- **Test set**: Used for final model evaluation

Splits are defined in the `chips_{country}.parquet` files and can be filtered by country.

---

## 2. Dataset Loading and Configuration

### 2.1 FTW Dataset Class

The `FTW` dataset class (`ftw_tools.training.datasets.FTW`) handles:

- **File Discovery**: Scans country directories for matching image/mask pairs
- **Data Loading**: Reads Sentinel-2 GeoTIFFs and label masks using Rasterio
- **Temporal Fusion**: Combines temporal windows according to `temporal_options`
- **Sample Filtering**: Supports ignoring specific samples via `ignore_sample_fn`
- **Boundary Loading**: Optional loading of 3-class masks with boundaries

**Key Parameters**:
- `root`: Root directory containing country folders
- `countries`: List of countries to load
- `split`: Dataset split ("train", "val", "test")
- `temporal_options`: Temporal fusion strategy
- `load_boundaries`: Whether to load 3-class masks
- `num_samples`: Limit number of samples (-1 for all)

### 2.2 FTWDataModule

The `FTWDataModule` (PyTorch Lightning) manages:

- **Dataset Initialization**: Creates train/val/test datasets
- **DataLoader Configuration**: Batch size, workers, shuffling
- **Augmentation Pipeline**: Applies augmentations during training
- **Country Selection**: Separate country lists for train/val/test

**Key Parameters**:
- `batch_size`: Mini-batch size (default: 64)
- `num_workers`: Data loading workers (default: 0)
- `train_countries`: Countries for training
- `val_countries`: Countries for validation
- `test_countries`: Countries for testing
- `temporal_options`: Temporal fusion strategy
- `preprocess_aug`: Enable random divisor normalization
- `brightness_aug`: Enable brightness augmentation
- `resize_aug`: Enable random resized crop
- `random_shuffle`: Enable random channel shuffle

---

## 3. Data Augmentation

### 3.1 Training Augmentations

The training augmentation pipeline includes:

1. **Normalization**:
   - Fixed: `K.Normalize(mean=[0,0,...], std=[3000,3000,...])`
   - Random: `randomDivisorNormalize` (divides by random [1500, 4500])

2. **Geometric Augmentations**:
   - `RandomRotation`: 90-degree rotations (p=0.5)
   - `RandomHorizontalFlip`: Horizontal flipping (p=0.5)
   - `RandomVerticalFlip`: Vertical flipping (p=0.5)

3. **Photometric Augmentations**:
   - `RandomSharpness`: Sharpness adjustment (p=0.5)
   - `RandomBrightness`: Brightness adjustment (p=0.5, range [0.5, 1.5]) - optional

4. **Spatial Augmentations**:
   - `RandomResizedCrop`: Random crop and resize (p=0.5, scale [0.3, 0.9]) - optional

5. **Channel Augmentations**:
   - `randomChannelShuffle`: Shuffles temporal windows (p=0.5) - optional

6. **Resize**:
   - `Resize`: Upsampling by `resize_factor` (if specified)

### 3.2 Validation/Test Augmentations

Only normalization is applied (no random augmentations):
- `K.Normalize(mean=[0,0,...], std=[3000,3000,...])`

### 3.3 Augmentation Implementation

Augmentations are applied using Kornia's `AugmentationSequential` in the `on_after_batch_transfer` hook, ensuring they run on the correct device (CPU/GPU).

---

## 4. Model Architecture

### 4.1 Supported Models

The training system supports multiple segmentation architectures:

1. **U-Net** (`unet`):
   - Encoder-decoder architecture
   - Configurable backbone encoder
   - Standard U-Net decoder

2. **DeepLabV3+** (`deeplabv3+`):
   - Atrous spatial pyramid pooling
   - Encoder-decoder structure

3. **UPerNet** (`upernet`):
   - Unified Perceptual Parsing Network
   - Multi-scale feature extraction

4. **Segformer** (`segformer`):
   - Transformer-based architecture
   - Efficient attention mechanisms

5. **DPT** (`dpt`):
   - Dense Prediction Transformer
   - Vision transformer backbone

6. **FCN** (`fcn`):
   - Fully Convolutional Network
   - Simple baseline architecture

7. **FCSiam Models**:
   - `fcsiamdiff`: Siamese difference network
   - `fcsiamconc`: Siamese concatenation network
   - `fcsiamavg`: Siamese average network

### 4.2 Backbone Encoders

Models support various backbone encoders from:
- **timm**: PyTorch Image Models (e.g., `efficientnet-b3`, `resnet50`)
- **smp**: Segmentation Models PyTorch encoders

### 4.3 Model Configuration

**Key Parameters**:
- `model`: Architecture name
- `backbone`: Encoder backbone
- `weights`: Pretrained weights (True/False/path)
- `in_channels`: Input channels (4, 6, or 8 depending on temporal option)
- `num_classes`: Output classes (2 or 3)
- `freeze_backbone`: Freeze encoder weights
- `freeze_decoder`: Freeze decoder weights
- `patch_weights`: Custom weight patching for multi-temporal inputs

### 4.4 Weight Initialization

- **ImageNet Pretrained**: Standard encoders initialized with ImageNet weights
- **Custom Patching**: For multi-temporal inputs, first conv layer weights are patched to handle 8-channel input by duplicating RGB weights

---

## 5. Loss Functions

### 5.1 Available Loss Functions

1. **Cross-Entropy** (`ce`):
   - Standard multi-class cross-entropy
   - Supports class weights and ignore index

2. **Jaccard Loss** (`jaccard`):
   - IoU-based loss for segmentation
   - Handles ignore index

3. **Focal Loss** (`focal`):
   - Addresses class imbalance
   - Normalized variant

4. **Dice Loss** (`dice`):
   - Dice coefficient-based loss

5. **Tversky Loss** (`tversky`):
   - Generalization of Dice loss
   - Configurable alpha/beta parameters

6. **Combined Losses**:
   - `ce+dice`: Cross-entropy + Dice
   - `logcoshdice`: Log-cosh Dice loss
   - `logcoshdice+ce`: Log-cosh Dice + Cross-entropy
   - `ftnmt`: Fractal Tanimoto loss
   - `ce+ftnmt`: Cross-entropy + Fractal Tanimoto
   - `tversky_ce`: Tversky Focal + Cross-entropy

7. **Specialized Losses**:
   - `pixel_weighted_ce`: Pixel-weighted cross-entropy with Gaussian blur
   - `localtversky`: Locally weighted Tversky focal loss

### 5.2 Loss Configuration

**Key Parameters**:
- `loss`: Loss function name
- `class_weights`: Per-class weighting (list or None)
- `ignore_index`: Class index to ignore (e.g., 3 for unknown)
- `pixel_weight_scale`: Scale for pixel-weighted losses

### 5.3 Edge Agreement Loss

Optional edge agreement loss (`edge_agreement_loss`) focuses learning on boundary pixels by masking non-edge pixels to the "unknown" class before loss computation.

---

## 6. Training Process

### 6.1 Training Loop

The training process follows PyTorch Lightning's standard workflow:

1. **Initialization**:
   - Model, loss, and metrics are configured
   - Optimizer and learning rate scheduler are set up
   - DataModule prepares train/val datasets

2. **Training Step** (`training_step`):
   - Forward pass through model
   - Loss computation
   - Metric updates (precision, recall, IoU per class)
   - Loss logging

3. **Validation Step** (`validation_step`):
   - Forward pass on validation data
   - Loss and metric computation
   - Object-level metric accumulation (TP, FP, FN)
   - Corner consensus metric (spatial consistency)
   - Visualization logging (first 10 batches)

4. **Epoch End**:
   - Aggregate metrics across batches
   - Log per-class metrics
   - Compute object-level precision/recall/F1
   - Learning rate logging

### 6.2 Optimizer Configuration

- **Optimizer**: AdamW with AMSGrad
- **Learning Rate**: Configurable (default: 1e-3)
- **Scheduler**: Cosine Annealing LR
  - `T_max`: Patience parameter
  - `eta_min`: Minimum learning rate (1e-6)

### 6.3 Training Configuration

**Key Parameters**:
- `max_epochs`: Maximum training epochs
- `log_every_n_steps`: Logging frequency
- `accelerator`: Hardware accelerator ("gpu", "cpu", "mps")
- `devices`: Device IDs
- `monitor`: Metric to monitor for checkpointing (default: `val_loss`)

### 6.4 Checkpointing

Model checkpoints are saved based on:
- **Best Model**: Lowest validation loss (if `save_top_k > 0`)
- **Last Model**: Final epoch state (if `save_last=True`)
- **Filename Pattern**: `{epoch}-{val_loss:.2f}`

---

## 7. Validation

### 7.1 Validation Metrics

**Pixel-Level Metrics** (per class):
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **IoU (Jaccard Index)**: Intersection over Union

**Object-Level Metrics**:
- **Object Precision**: Field detection precision
- **Object Recall**: Field detection recall
- **Object F1**: Harmonic mean of precision and recall
- **IoU Threshold**: 0.5 for matching predictions to ground truth

**Spatial Consistency Metrics**:
- **Corner Consensus**: Measures prediction consistency across corner crops
  - Tests model on 128×128 corner patches with 64-pixel padding
  - Computes agreement between center and corner predictions

### 7.2 Validation Visualization

During validation, the first 10 batches are visualized:
- Input images (Window A and/or Window B)
- Ground truth masks
- Model predictions
- Logged to TensorBoard/MLflow

### 7.3 Validation Logging

All metrics are logged to:
- **TensorBoard**: Scalars and images
- **MLflow**: If configured
- **Console**: Progress bar with key metrics

---

## 8. Model Testing

### 8.1 Test Command

The test command (`ftw model test`) evaluates trained models on test sets:

```bash
ftw model test \
    --model <checkpoint_path> \
    --countries <country_list> \
    --dir <dataset_root> \
    --gpu <gpu_id> \
    --out <output_csv>
```

### 8.2 Test Process

1. **Model Loading**:
   - Load checkpoint from file
   - Move model to specified device (GPU/CPU)
   - Set model to evaluation mode

2. **Data Loading**:
   - Create test dataset with specified countries
   - Configure temporal options and normalization
   - Create DataLoader with appropriate batch size

3. **Inference**:
   - Process batches through model
   - Apply optional resize factor (upsample for inference)
   - Handle FCSiam models (temporal rearrangement)
   - Convert 3-class predictions to 2-class if needed

4. **Metric Computation**:
   - Pixel-level metrics (IoU, precision, recall)
   - Object-level metrics (TP, FP, FN, precision, recall, F1)
   - Bootstrap confidence intervals (optional)

### 8.3 Bootstrap Confidence Intervals

Optional bootstrap sampling provides 95% confidence intervals:
- Samples with replacement at the sample level
- Computes metrics for each bootstrap sample
- Reports percentile-based confidence intervals

### 8.4 Test Output

Results are saved to CSV with columns:
- `train_checkpoint`: Model checkpoint path
- `countries`: Test countries
- `pixel_level_iou`: Field class IoU
- `pixel_level_precision`: Field class precision
- `pixel_level_recall`: Field class recall
- `object_level_precision`: Object detection precision
- `object_level_recall`: Object detection recall
- `object_level_f1`: Object detection F1 score
- Confidence intervals (if bootstrap enabled)

---

## 9. Configuration Files

### 9.1 Configuration Structure

Training is configured via YAML files with three main sections:

```yaml
trainer:
  max_epochs: 100
  log_every_n_steps: 10
  accelerator: "gpu"
  default_root_dir: "logs/experiment_name"
  devices: [0]
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        mode: min
        save_top_k: 1
        save_last: true

model:
  class_path: ftw_tools.training.trainers.CustomSemanticSegmentationTask
  init_args:
    loss: "jaccard"
    model: "unet"
    backbone: "efficientnet-b3"
    weights: true
    in_channels: 8
    num_classes: 3
    lr: 1e-3
    patience: 100

data:
  class_path: ftw_tools.training.datamodules.FTWDataModule
  init_args:
    batch_size: 32
    num_workers: 8
    train_countries: ["france", "belgium"]
    val_countries: ["france"]
    test_countries: ["france"]
  dict_kwargs:
    root: "./data/ftw"
    load_boundaries: true
    temporal_options: "stacked"

seed_everything: 42
```

### 9.2 Training Execution

Training is executed via CLI:

```bash
# Train from scratch
ftw model fit --config configs/example_config.yaml

# Resume from checkpoint
ftw model fit --config configs/example_config.yaml --ckpt_path <checkpoint.ckpt>
```

### 9.3 Configuration Parameters

**Trainer Parameters**:
- `max_epochs`: Training duration
- `accelerator`: Hardware ("gpu", "cpu", "mps")
- `devices`: GPU IDs
- `default_root_dir`: Logging directory

**Model Parameters**:
- `loss`: Loss function name
- `model`: Architecture name
- `backbone`: Encoder backbone
- `in_channels`: Input channels (4, 6, or 8)
- `num_classes`: Output classes (2 or 3)
- `lr`: Learning rate
- `patience`: Cosine annealing period

**Data Parameters**:
- `batch_size`: Mini-batch size
- `num_workers`: Data loading workers
- `train_countries`: Training countries
- `val_countries`: Validation countries
- `test_countries`: Test countries
- `root`: Dataset root directory
- `load_boundaries`: Use 3-class masks
- `temporal_options`: Temporal fusion strategy


---

## 10. Metrics and Evaluation

### 10.1 Pixel-Level Metrics

Computed per class using TorchMetrics:

- **MulticlassJaccardIndex**: Intersection over Union
- **MulticlassPrecision**: Precision per class
- **MulticlassRecall**: Recall per class

**Class Definitions**:
- Class 0: Background
- Class 1: Field
- Class 2: Boundary (3-class models only)
- Class 3: Unknown/Ignore

### 10.2 Object-Level Metrics

Computed using polygon-based matching:

1. **Polygon Extraction**: Convert masks to polygons using Rasterio
2. **IoU Matching**: Match predictions to ground truth using IoU threshold (0.5)
3. **Metric Computation**:
   - **True Positives (TP)**: Correctly detected fields
   - **False Positives (FP)**: Incorrectly detected fields
   - **False Negatives (FN)**: Missed fields

**Formulas**:
- Object Precision = TP / (TP + FP)
- Object Recall = TP / (TP + FN)
- Object F1 = 2 × (Precision × Recall) / (Precision + Recall)

### 10.3 Spatial Consistency Metrics

**Corner Consensus**:
- Tests model on 128×128 corner crops with 64-pixel padding
- Measures agreement between center and corner predictions
- Higher values indicate better spatial consistency

### 10.4 Metric Logging

Metrics are logged at multiple levels:

1. **Per-Class Metrics**: Logged separately for each class
2. **Macro-Averaged Metrics**: Average across classes
3. **Object-Level Metrics**: Field detection performance
4. **Visualizations**: Sample predictions logged as images

### 10.5 Evaluation Best Practices

1. **Use Test Set Only for Final Evaluation**: Keep test set untouched during development
2. **Monitor Validation Metrics**: Use validation set for hyperparameter tuning
3. **Report Multiple Metrics**: Both pixel-level and object-level metrics provide complementary insights
4. **Bootstrap Confidence Intervals**: Use for robust statistical reporting
5. **Visual Inspection**: Review logged visualizations to identify failure modes

---

## Summary

The FTW training workflow provides a comprehensive pipeline for semantic segmentation of agricultural field boundaries:

1. **Data Preprocessing**: Flexible temporal fusion and normalization strategies
2. **Augmentation**: Rich augmentation pipeline for improved generalization
3. **Model Architecture**: Support for multiple state-of-the-art architectures
4. **Loss Functions**: Diverse loss functions for different training scenarios
5. **Training**: PyTorch Lightning-based training with comprehensive logging
6. **Validation**: Multi-level metrics (pixel, object, spatial consistency)
7. **Testing**: Robust evaluation with bootstrap confidence intervals

The modular design allows researchers to easily experiment with different configurations while maintaining reproducibility and best practices.

---

## References

- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
- **Segmentation Models PyTorch**: https://github.com/qubvel/segmentation_models.pytorch
- **Kornia**: https://kornia.readthedocs.io/
- **TorchMetrics**: https://torchmetrics.readthedocs.io/
- **FTW Dataset**: https://fieldsofthe.world/

