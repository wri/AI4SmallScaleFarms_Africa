# Deep Explanation of Configuration Parameters

This guide explains each parameter in the training configuration file and how it affects model behavior, training dynamics, and performance.

---

## Table of Contents

1. [Trainer Parameters](#trainer-parameters)
2. [Model Architecture Parameters](#model-architecture-parameters)
3. [Loss Function Parameters](#loss-function-parameters)
4. [Data Loading Parameters](#data-loading-parameters)
5. [Optimization Parameters](#optimization-parameters)

---

## Trainer Parameters

### `trainer.max_epochs`
**Type:** Integer  
**Example:** `100`

**What it does:**
- Defines the maximum number of complete passes through the entire training dataset
- One epoch = one full iteration over all training samples
- Training stops when this limit is reached OR when early stopping triggers (if configured)

**Impact on model:**
- **Too low (< 50)**: Model may not converge, underfitting
- **Too high (> 500)**: Risk of overfitting, wasted compute time
- **Typical range**: 50-200 epochs for field delineation tasks
- The model learns progressively: early epochs learn basic patterns, later epochs refine boundaries

**Relation to other parameters:**
- Works with `patience` to control learning rate scheduling
- Early stopping callbacks can terminate training before `max_epochs` if validation loss plateaus

---

### `trainer.log_every_n_steps`
**Type:** Integer  
**Example:** `10`

**What it does:**
- Controls how frequently training metrics are logged to TensorBoard/console
- Logs metrics every N training steps (batches)
- One step = processing one batch of images

**Impact on model:**
- **No direct impact on model performance** - purely for monitoring
- Lower values (5-10): More frequent updates, better real-time monitoring, higher I/O overhead
- Higher values (50-100): Less frequent updates, lower overhead, may miss short-term trends

**Why it matters:**
- Helps detect training issues early (loss spikes, NaN values)
- Allows real-time monitoring of convergence
- Useful for debugging data loading or augmentation issues

---

### `trainer.accelerator`
**Type:** String  
**Example:** `"gpu"`

**What it does:**
- Specifies the hardware accelerator for training
- Options: `"gpu"`, `"cpu"`, `"mps"` (Apple Silicon), `"tpu"`

**Impact on model:**
- **GPU**: 10-100x faster training, enables larger batch sizes, required for practical training
- **CPU**: Much slower, only for small experiments or debugging
- **MPS**: Apple Silicon GPU acceleration (M1/M2/M3 Macs)

**Relation to other parameters:**
- `batch_size` can be larger with GPU (more memory)
- `num_workers` for data loading is less critical with GPU (GPU processes batches faster)

---

### `trainer.default_root_dir`
**Type:** String  
**Example:** `"PRUE_EFNET_B7_finetune"`

**What it does:**
- Directory where PyTorch Lightning saves:
  - Model checkpoints
  - TensorBoard logs
  - Hyperparameter files
  - Training metrics

**Impact on model:**
- **No direct impact** - organizational only
- Helps organize experiments by model name/configuration
- Checkpoints saved here can be used for resuming training or inference

**File structure created:**
```
default_root_dir/
├── checkpoints/
│   ├── epoch=50-val_loss=0.123.ckpt
│   └── last.ckpt
├── hparams.yaml
└── events.out.tfevents.* (TensorBoard logs)
```

---

### `trainer.devices`
**Type:** List of integers  
**Example:** `[0]` or `[0, 1]` for multi-GPU

**What it does:**
- Specifies which GPU device(s) to use for training
- `[0]`: Use first GPU
- `[0, 1]`: Use first and second GPUs (data parallel)

**Impact on model:**
- **Single GPU `[0]`**: Standard setup, model trains on one GPU
- **Multi-GPU `[0, 1, ...]`**: Data parallelism - batches split across GPUs, faster training
- Model parameters are synchronized across GPUs in multi-GPU setup

**Relation to other parameters:**
- Effective batch size = `batch_size × num_gpus`
- With 2 GPUs and `batch_size=8`, effective batch size = 16

---

### `trainer.callbacks.ModelCheckpoint`
**Type:** Callback configuration  
**Example:**
```yaml
callbacks:
  - class_path: lightning.pytorch.callbacks.ModelCheckpoint
    init_args:
      monitor: val/loss
      mode: min
      save_top_k: 1
      save_last: true
      filename: "{epoch}-{val/loss:.2f}"
```

**What it does:**
- Saves model checkpoints during training based on specified criteria

**Parameters:**
- **`monitor`**: Metric to track (`val/loss`, `val/iou`, `train/loss`)
- **`mode`**: `"min"` (save when metric decreases) or `"max"` (save when metric increases)
- **`save_top_k`**: Number of best checkpoints to keep (e.g., `1` = keep only the best)
- **`save_last`**: Whether to always save the last epoch's checkpoint
- **`filename`**: Template for checkpoint names (e.g., `"{epoch}-{val/loss:.2f}"`)

**Impact on model:**
- **No direct impact on training** - affects checkpoint management
- Best checkpoint (lowest `val/loss`) is saved for inference
- `save_last: true` allows resuming from the most recent state

**Why it matters:**
- Prevents losing the best model if training continues and validation loss increases
- Enables model comparison and selection
- Allows training resumption after interruption

---

## Model Architecture Parameters

### `model.class_path`
**Type:** String (Python class path)  
**Example:** `"ftw_tools.training.trainers.CustomSemanticSegmentationTask"`

**What it does:**
- Specifies the PyTorch Lightning task class that wraps the model
- This class handles:
  - Model initialization
  - Training/validation/test loops
  - Loss computation
  - Metric calculation
  - Optimizer/scheduler setup

**Impact on model:**
- Defines the training framework and available features
- `CustomSemanticSegmentationTask` provides semantic segmentation-specific functionality

---

### `model.init_args.model`
**Type:** String  
**Example:** `"unet"`  
**Options:** `"unet"`, `"deeplabv3+"`, `"upernet"`, `"fcn"`, `"segformer"`, `"dpt"`, `"fcsiamdiff"`, `"fcsiamconc"`, `"fcsiamavg"`

**What it does:**
- Selects the segmentation architecture

**Architecture Details:**

1. **U-Net (`"unet"`)**:
   - **Encoder-Decoder structure**: Encoder downsamples (extracts features), decoder upsamples (reconstructs segmentation)
   - **Skip connections**: Connects encoder layers to decoder layers at same resolution
   - **Why it works**: Skip connections preserve fine-grained spatial details lost during downsampling
   - **Best for**: Field boundary detection (needs precise pixel-level accuracy)

2. **DeepLabV3+ (`"deeplabv3+"`)**:
   - **Atrous (dilated) convolutions**: Expands receptive field without losing resolution
   - **ASPP (Atrous Spatial Pyramid Pooling)**: Captures multi-scale context
   - **Best for**: Large fields with varying sizes

3. **UPerNet (`"upernet"`)**:
   - **Multi-scale feature extraction**: Uses features from multiple encoder levels
   - **Pyramid pooling**: Aggregates context at different scales
   - **Best for**: Complex scenes with multiple object scales

4. **FCN (`"fcn"`)**:
   - **Simple fully convolutional network**: No encoder-decoder, just convolutions
   - **Lightweight**: Fewer parameters, faster inference
   - **Best for**: Baseline comparisons, resource-constrained deployments

**Impact on model:**
- **U-Net**: Best balance of accuracy and efficiency for field delineation
- **DeepLabV3+**: Better for large-scale context, but slower
- **FCN**: Fastest but less accurate for fine boundaries

---

### `model.init_args.backbone`
**Type:** String  
**Example:** `"efficientnet-b3"`  
**Options:** `"resnet50"`, `"efficientnet-b3"`, `"efficientnet-b5"`, `"efficientnet-b7"`, `"resnet34"`, etc.

**What it does:**
- Defines the encoder backbone that extracts features from input images
- The backbone is the "feature extractor" part of the segmentation model

**Backbone Comparison:**

1. **EfficientNet-B3/B5/B7**:
   - **Scaling**: B3 < B5 < B7 (larger = more parameters, better accuracy, slower)
   - **Efficiency**: Optimized for accuracy vs. speed tradeoff
   - **Best for**: Production models (B3 for speed, B7 for accuracy)

2. **ResNet50/34**:
   - **Classic architecture**: Well-tested, widely used
   - **Residual connections**: Helps with gradient flow in deep networks
   - **Best for**: Baseline comparisons, transfer learning

**How it works:**
- Input image (256×256×8) → Backbone extracts features at multiple scales:
  - Layer 1: 256×256 (high resolution, low-level features like edges)
  - Layer 2: 128×128 (medium resolution, shapes)
  - Layer 3: 64×64 (lower resolution, object parts)
  - Layer 4: 32×32 (low resolution, high-level semantic features)
- These multi-scale features are fed to the decoder

**Impact on model:**
- **Larger backbone (B7)**: Better feature extraction, higher accuracy, slower training/inference
- **Smaller backbone (B3)**: Faster, less accurate, good for deployment
- **Choice depends on**: Accuracy requirements vs. inference speed constraints

**Relation to other parameters:**
- `freeze_backbone: true` prevents updating backbone weights (transfer learning)
- `weights: true` initializes backbone with ImageNet pretrained weights

---

### `model.init_args.weights`
**Type:** Boolean or String  
**Example:** `false` or `true` or `"path/to/checkpoint.ckpt"`

**What it does:**
- Controls initialization of the encoder backbone weights

**Options:**
- **`false`**: Random initialization (training from scratch)
- **`true`**: ImageNet pretrained weights (transfer learning)
- **`"path/to/checkpoint.ckpt"`**: Load from a specific checkpoint

**Impact on model:**
- **ImageNet pretrained (`true`)**: 
  - Encoder already knows how to extract general image features (edges, textures, shapes)
  - Faster convergence (fewer epochs needed)
  - Better final accuracy (especially with limited data)
  - **Why it works**: Natural images (ImageNet) and satellite images share low-level features

- **Random initialization (`false`)**:
  - Model learns features from scratch
  - Requires more data and epochs
  - May be necessary if ImageNet features are too different from satellite imagery

**When to use:**
- **Use `true`**: When finetuning pretrained models (most cases)
- **Use `false`**: When training from scratch or when ImageNet features are inappropriate

**Note:** When loading from a checkpoint via `--ckpt_path`, this parameter is typically ignored (weights come from checkpoint).

---

### `model.init_args.patch_weights`
**Type:** Boolean  
**Example:** `false`

**What it does:**
- If `true`, patches the first convolutional layer weights to handle multi-temporal inputs
- Standard ImageNet weights expect 3 RGB channels
- For 8-channel input (stacked temporal windows), this duplicates/adapts RGB weights

**How it works:**
- ImageNet backbone expects 3 channels (RGB)
- FTW uses 8 channels (4 bands × 2 temporal windows)
- Patching: Copies RGB weights to handle 8 channels:
  - Channels 0-3: Window B (uses RGB weights)
  - Channels 4-7: Window A (uses RGB weights)

**Impact on model:**
- **`true`**: Allows using ImageNet pretrained weights with 8-channel input
- **`false`**: First layer initialized randomly (if `weights: true`, only layers 2+ use ImageNet weights)

**When to use:**
- **Use `true`**: When using ImageNet pretrained weights with 8-channel stacked input
- **Use `false`**: When training from scratch or using single-window (4-channel) input

---

### `model.init_args.in_channels`
**Type:** Integer  
**Example:** `8` (stacked) or `4` (single window)

**What it does:**
- Defines the number of input channels to the model
- Must match the number of channels in your input images

**Channel Configurations:**

1. **`8` channels (stacked temporal)**:
   - Channels 0-3: Window B (B, G, R, NIR)
   - Channels 4-7: Window A (B, G, R, NIR)
   - **Use with**: `temporal_options: stacked`

2. **`4` channels (single window)**:
   - Channels 0-3: Single temporal window (B, G, R, NIR)
   - **Use with**: `temporal_options: windowA` or `windowB`

3. **`6` channels (RGB stacked)**:
   - Channels 0-2: Window B (RGB)
   - Channels 3-5: Window A (RGB)
   - **Use with**: `temporal_options: rgb`

**Impact on model:**
- **Mismatch causes error**: If `in_channels=8` but data has 4 channels, model will crash
- **More channels**: More information, but larger model and slower training
- **8 channels**: Captures temporal changes (crop growth, phenology)
- **4 channels**: Faster, but misses temporal information

**Relation to other parameters:**
- Must match `data.init_args.temporal_options`:
  - `stacked` → `in_channels: 8`
  - `windowA`/`windowB` → `in_channels: 4`
  - `rgb` → `in_channels: 6`

---

### `model.init_args.num_classes`
**Type:** Integer  
**Example:** `2` or `3`

**What it does:**
- Defines the number of output classes for segmentation

**Class Configurations:**

1. **`2` classes**:
   - Class 0: Background (non-field)
   - Class 1: Field
   - **Use with**: `load_boundaries: false` (2-class masks)

2. **`3` classes**:
   - Class 0: Background
   - Class 1: Field interior
   - Class 2: Field boundary
   - **Use with**: `load_boundaries: true` (3-class masks)

**Impact on model:**
- **Output shape**: Model outputs `(batch, num_classes, height, width)` logits
- **3 classes**: More detailed segmentation, better boundary detection
- **2 classes**: Simpler, faster, but boundaries may be less precise

**Model output:**
- For each pixel, model outputs `num_classes` scores (logits)
- Highest score determines predicted class
- Example: `[0.1, 0.8, 0.1]` → Class 1 (field) predicted

**Relation to other parameters:**
- Must match your label masks:
  - 2-class masks → `num_classes: 2`, `load_boundaries: false`
  - 3-class masks → `num_classes: 3`, `load_boundaries: true`

---

### `model.init_args.num_filters`
**Type:** Integer  
**Example:** `64`

**What it does:**
- **Only used for FCN model** (`model: "fcn"`)
- Defines the number of filters in FCN's convolutional layers
- Ignored for other architectures (U-Net, DeepLabV3+, etc.)

**Impact on model:**
- **More filters**: More model capacity, better accuracy, slower training
- **Fewer filters**: Faster, less accurate
- **Typical range**: 32-128 for FCN

---

### `model.init_args.freeze_backbone`
**Type:** Boolean  
**Example:** `false`

**What it does:**
- If `true`, freezes (disables gradient updates for) the encoder backbone
- Only the decoder and segmentation head are trained

**Impact on model:**
- **`true`**: Transfer learning - encoder features fixed, only decoder learns
  - Faster training (fewer parameters to update)
  - Less memory usage
  - Good when encoder is already well-trained
  - **Risk**: May limit adaptation to satellite imagery specifics

- **`false`**: End-to-end training - all layers learn
  - Slower training, more memory
  - Better adaptation to task-specific features
  - **Recommended**: For finetuning pretrained models

**When to use:**
- **Use `true`**: When encoder is from a very similar task, or for quick experiments
- **Use `false`**: For most finetuning scenarios (allows encoder to adapt)

---

### `model.init_args.freeze_decoder`
**Type:** Boolean  
**Example:** `false`

**What it does:**
- If `true`, freezes the decoder, only training the segmentation head
- Used for "linear probing" - testing if encoder features are sufficient

**Impact on model:**
- **`true`**: Linear probing - minimal adaptation
  - Very fast training
  - Tests encoder quality
  - **Usually too restrictive** for good performance

- **`false`**: Full training (recommended)

**When to use:**
- **Rarely used** - mainly for diagnostic purposes
- **Use `false`**: For actual training

---

## Loss Function Parameters

### `model.init_args.loss`
**Type:** String  
**Example:** `"ce"` or `"logcoshdice"`  
**Options:** `"ce"`, `"jaccard"`, `"focal"`, `"dice"`, `"ce+dice"`, `"logcoshdice"`, `"logcoshdice+ce"`, `"ftnmt"`, `"tversky"`, etc.

**What it does:**
- Defines the loss function that measures prediction error
- Loss guides the optimizer to update model weights

**Loss Function Details:**

1. **Cross-Entropy (`"ce"`)**:
   - **Standard classification loss**: Measures difference between predicted class probabilities and true labels
   - **Formula**: `-log(P(correct_class))`
   - **Pros**: Simple, well-understood, works well with class weights
   - **Cons**: Can struggle with class imbalance
   - **Best for**: Balanced datasets, general use

2. **Dice Loss (`"dice"`)**:
   - **Segmentation-specific**: Measures overlap between predicted and true masks
   - **Formula**: `1 - (2×intersection / (prediction + ground_truth))`
   - **Pros**: Directly optimizes IoU (Intersection over Union), handles imbalance better
   - **Cons**: Can be unstable with very small objects
   - **Best for**: Imbalanced datasets, boundary-focused tasks

3. **Log-Cosh Dice (`"logcoshdice"`)**:
   - **Smooth version of Dice**: Uses `log(cosh(dice_loss))` for stability
   - **Pros**: More stable gradients than raw Dice, better convergence
   - **Cons**: Slightly more complex
   - **Best for**: **Recommended for field delineation** (used in PRUE models)

4. **Combined Losses (`"ce+dice"`, `"logcoshdice+ce"`)**:
   - **Hybrid**: Combines classification (CE) and segmentation (Dice) losses
   - **Pros**: Benefits of both - pixel-level accuracy + boundary precision
   - **Cons**: Requires tuning weight balance
   - **Best for**: When both pixel accuracy and boundary quality matter

5. **Focal Loss (`"focal"`)**:
   - **Addresses class imbalance**: Down-weights easy examples, focuses on hard ones
   - **Formula**: `-α(1-p)^γ log(p)` where `γ` focuses on hard examples
   - **Best for**: Highly imbalanced datasets

**Impact on model:**
- **Loss choice is critical**: Directly affects what the model optimizes
- **Dice-based losses**: Better for boundary detection (field edges)
- **CE**: Better for overall pixel accuracy
- **Combined**: Best of both worlds

**Relation to other parameters:**
- `class_weights` only work with CE-based losses (`"ce"`, `"ce+dice"`)
- `ignore_index` works with most losses (except `"jaccard"`)

---

### `model.init_args.class_weights`
**Type:** List of floats  
**Example:** `[0.05, 0.2, 0.75]` for 3 classes

**What it does:**
- Assigns different importance weights to each class in the loss function
- Used to handle class imbalance (e.g., background pixels >> boundary pixels)

**How it works:**
- **Without weights**: All classes contribute equally to loss
- **With weights**: Rare classes (e.g., boundaries) contribute more
- **Example**: `[0.05, 0.2, 0.75]` means:
  - Background (class 0): Weight 0.05 (very common, low weight)
  - Field (class 1): Weight 0.2 (common, medium weight)
  - Boundary (class 2): Weight 0.75 (rare, high weight)

**Impact on model:**
- **Critical for imbalanced data**: Without weights, model ignores rare classes
- **Too high weights**: Model over-predicts rare classes (false positives)
- **Too low weights**: Model ignores rare classes (false negatives)
- **Typical values**: Inverse of class frequency, or tuned empirically

**Relation to other parameters:**
- Only works with CE-based losses (`"ce"`, `"ce+dice"`, `"logcoshdice+ce"`)
- Must have `len(class_weights) == num_classes`

**Example calculation:**
- If background:field:boundary = 90%:8%:2% of pixels
- Inverse frequency weights: `[1/0.9, 1/0.08, 1/0.02] ≈ [1.1, 12.5, 50]`
- Normalized: `[0.05, 0.2, 0.75]` (sums to 1.0)

---

### `model.init_args.ignore_index`
**Type:** Integer or `null`  
**Example:** `3` or `null`

**What it does:**
- Specifies a class index to ignore during loss computation and metric calculation
- Pixels with this class are excluded from training

**Common usage:**
- **`3`**: "Unknown" class - pixels where label is uncertain or missing
- **`null`**: No ignored pixels (all classes contribute to loss)

**Impact on model:**
- **`ignore_index: 3`**: Model doesn't learn from "unknown" pixels
  - Useful when some pixels have unreliable labels
  - Prevents model from learning incorrect patterns
- **`null`**: All labeled pixels contribute to training

**Relation to other parameters:**
- Must match your label encoding (if you use class 3 for "unknown")
- Works with most losses except `"jaccard"`

---

## Optimization Parameters

### `model.init_args.lr` (Learning Rate)
**Type:** Float  
**Example:** `0.001` (1e-3)

**What it does:**
- Controls the step size when updating model weights during training
- **Too high**: Model overshoots optimal weights, training unstable
- **Too low**: Model learns very slowly, may get stuck in local minima

**How it works:**
- **Gradient descent**: `new_weight = old_weight - lr × gradient`
- **Large lr**: Big steps, faster but unstable
- **Small lr**: Small steps, stable but slow

**Impact on model:**
- **Critical parameter**: Directly affects convergence speed and final accuracy
- **Typical values**: 
  - `0.001` (1e-3): Standard starting point
  - `0.0001` (1e-4): For finetuning pretrained models
  - `0.01` (1e-2): For training from scratch (may be too high)

**Learning rate schedule:**
- Starts at `lr`, decreases over time via cosine annealing (controlled by `patience`)
- Early epochs: Higher lr (coarse adjustments)
- Later epochs: Lower lr (fine-tuning)

**Relation to other parameters:**
- `patience` controls how quickly lr decreases
- With `freeze_backbone: true`, can use higher lr (fewer parameters to update)

---

### `model.init_args.patience`
**Type:** Integer  
**Example:** `100`

**What it does:**
- Controls the learning rate scheduler (Cosine Annealing)
- Defines the period (in epochs) over which learning rate decreases
- **Not** early stopping patience (that's a separate callback)

**How it works:**
- **Cosine Annealing**: Learning rate follows a cosine curve from `lr` to `1e-6`
- **Formula**: `lr(t) = eta_min + (lr - eta_min) × (1 + cos(π × t / patience)) / 2`
- **`patience=100`**: LR decreases over 100 epochs
- **`patience=50`**: LR decreases faster (over 50 epochs)

**Impact on model:**
- **Larger patience**: Slower LR decay, model has more time to learn
  - Good for long training runs
  - Better for complex tasks
- **Smaller patience**: Faster LR decay, model may stop learning too early
  - Good for quick experiments
  - May underfit if too small

**Typical values:**
- **`100`**: Standard for 100-epoch training
- **`50`**: For shorter training (50 epochs)
- **`200`**: For longer training (200+ epochs)

**Relation to other parameters:**
- Should be close to `max_epochs` (LR should decay over most of training)
- If `patience >> max_epochs`, LR won't decrease much (wasteful)
- If `patience << max_epochs`, LR decreases too early (may underfit)

---

## Data Loading Parameters

### `data.class_path`
**Type:** String  
**Example:** `"ftw_tools.training.datamodules.FTWDataModule"`

**What it does:**
- Specifies the PyTorch Lightning DataModule class
- Handles dataset loading, splitting, and DataLoader creation

---

### `data.init_args.batch_size`
**Type:** Integer  
**Example:** `8` or `32`

**What it does:**
- Number of images processed together in one training step
- One batch = one forward pass + one backward pass (weight update)

**Impact on model:**
- **Larger batch size (32-64)**:
  - More stable gradients (averaged over more samples)
  - Faster training (better GPU utilization)
  - **Requires more GPU memory**
  - May generalize slightly worse (sharp minima)

- **Smaller batch size (4-16)**:
  - More noisy gradients (may help escape local minima)
  - Slower training (more iterations needed)
  - Less GPU memory
  - May generalize better (flat minima)

**Typical values:**
- **GPU with 8GB VRAM**: `batch_size: 8-16`
- **GPU with 16GB+ VRAM**: `batch_size: 32-64`
- **Multi-GPU**: Effective batch size = `batch_size × num_gpus`

**Relation to other parameters:**
- Limited by GPU memory
- With `freeze_backbone: true`, can use larger batches (fewer gradients to store)

---

### `data.init_args.num_workers`
**Type:** Integer  
**Example:** `4` or `8`

**What it does:**
- Number of parallel processes for loading data from disk
- Prevents GPU from waiting for data (data loading happens in background)

**Impact on model:**
- **More workers (8-16)**: Faster data loading, better GPU utilization
  - **Requires more CPU cores and RAM**
- **Fewer workers (0-4)**: Slower data loading, GPU may idle
  - **Less CPU/RAM usage**

**Typical values:**
- **CPU with 4 cores**: `num_workers: 2-4`
- **CPU with 8+ cores**: `num_workers: 4-8`
- **`0`**: Single-threaded (slow, but simple for debugging)

**Relation to other parameters:**
- More important with slower storage (HDD vs. SSD)
- Less critical with fast storage (NVMe SSD)

---

### `data.init_args.num_samples`
**Type:** Integer  
**Example:** `-1` (all samples) or `1000`

**What it does:**
- Limits the number of samples used per epoch
- **`-1`**: Use all available samples
- **`N`**: Use only first N samples (for quick experiments)

**Impact on model:**
- **`-1`**: Full dataset, best accuracy
- **Small N (100-1000)**: Quick experiments, faster iteration, lower accuracy

**When to use:**
- **Use `-1`**: For actual training
- **Use small N**: For debugging, quick hyperparameter tests

---

### `data.init_args.train_countries` / `val_countries` / `test_countries`
**Type:** List of strings  
**Example:** `["kenya_counties_batch"]`

**What it does:**
- Specifies which country/region datasets to use for each split
- Can use multiple countries: `["kenya_counties_batch", "france"]`

**Impact on model:**
- **Training set**: Model learns from these samples
- **Validation set**: Used to monitor training, select best checkpoint
- **Test set**: Final evaluation (not used during training)

**Best practices:**
- **Train/Val/Test split**: Should be from same distribution (same country/region)
- **Multiple countries**: Increases diversity, better generalization
- **Separate test set**: Use different region for final evaluation (tests generalization)

---

### `data.init_args.temporal_options`
**Type:** String  
**Example:** `"stacked"` or `"windowB"`  
**Options:** `"stacked"`, `"windowA"`, `"windowB"`, `"median"`, `"rgb"`, `"random_window"`

**What it does:**
- Controls how temporal (multi-date) satellite imagery is processed

**Options:**

1. **`"stacked"`**:
   - **Concatenates Window A and Window B**: 8 channels total
   - **Channels 0-3**: Window B (B, G, R, NIR)
   - **Channels 4-7**: Window A (B, G, R, NIR)
   - **Best for**: Capturing temporal changes (crop growth, phenology)
   - **Requires**: `in_channels: 8`

2. **`"windowB"`** or `"windowA"`:
   - **Single temporal window**: 4 channels
   - **Best for**: Faster training, when temporal info not critical
   - **Requires**: `in_channels: 4`

3. **`"median"`**:
   - **Pixel-wise median** of Window A and Window B
   - **Reduces noise**: Median filters out outliers
   - **Requires**: `in_channels: 4`

4. **`"rgb"`**:
   - **RGB channels only** (drops NIR): 6 channels (3 per window)
   - **Requires**: `in_channels: 6`

**Impact on model:**
- **`"stacked"`**: Most information, best accuracy, slower training
- **Single window**: Faster, but misses temporal patterns
- **Must match `in_channels`**: Mismatch causes errors

**Relation to other parameters:**
- `in_channels` must match:
  - `stacked` → `8`
  - `windowA`/`windowB`/`median` → `4`
  - `rgb` → `6`

---

### `data.init_args.brightness_aug`
**Type:** Boolean  
**Example:** `false`

**What it does:**
- Applies random brightness augmentation during training
- Randomly adjusts image brightness by factor in range [0.5, 1.5]

**Impact on model:**
- **`true`**: Increases robustness to lighting variations
  - Helps model generalize to different acquisition conditions
  - May slightly slow convergence
- **`false`**: No brightness augmentation (standard)

**When to use:**
- **Use `true`**: When training data has limited lighting diversity
- **Use `false`**: When data already has good lighting diversity, or when using `preprocess_aug`

**Note**: Mutually exclusive with `preprocess_aug` (cannot use both).

---

### `data.init_args.preprocess_aug`
**Type:** Boolean  
**Example:** `false`

**What it does:**
- Replaces fixed normalization (divide by 3000) with random per-batch divisor
- Divisor drawn uniformly from [1500, 4500] for each training batch

**Impact on model:**
- **`true`**: Increases robustness to different sensor calibrations
  - Simulates different acquisition conditions
  - Helps model generalize across sensors/dates
- **`false`**: Fixed normalization (divide by 3000)

**When to use:**
- **Use `true`**: When training on data from multiple sensors or dates
- **Use `false`**: When data is from single sensor/calibration

**Note**: Mutually exclusive with `brightness_aug`.

---

### `data.init_args.resize_aug`
**Type:** Boolean  
**Example:** `false`

**What it does:**
- Applies random resized crop augmentation
- Randomly crops and resizes images to 256×256 during training

**Impact on model:**
- **`true`**: Increases robustness to scale variations
  - Model learns to handle fields of different sizes
  - Helps generalization
- **`false`**: No scale augmentation (standard)

**When to use:**
- **Use `true`**: When fields vary significantly in size
- **Use `false`**: When fields are similar size, or for faster training

---

### `data.dict_kwargs.root`
**Type:** String  
**Example:** `"data"`

**What it does:**
- Root directory where dataset files are located
- DataModule looks for country folders under this directory

**Directory structure:**
```
data/
├── kenya_counties_batch/
│   ├── s2_images/
│   ├── label_masks/
│   └── chips_kenya_counties_batch.parquet
└── france/
    └── ...
```

---

### `data.dict_kwargs.load_boundaries`
**Type:** Boolean  
**Example:** `true` or `false`

**What it does:**
- Controls whether to load 3-class masks (with boundaries) or 2-class masks

**Impact on model:**
- **`true`**: Loads 3-class masks (background, field, boundary)
  - **Requires**: `num_classes: 3`
  - **Uses**: `semantic_3class/` mask directory
- **`false`**: Loads 2-class masks (background, field)
  - **Requires**: `num_classes: 2`
  - **Uses**: `semantic_2class/` mask directory

**Relation to other parameters:**
- Must match `num_classes`:
  - `load_boundaries: true` → `num_classes: 3`
  - `load_boundaries: false` → `num_classes: 2`

---

## Other Parameters

### `seed_everything`
**Type:** Integer  
**Example:** `42`

**What it does:**
- Sets random seed for reproducibility
- Ensures same random initialization, data shuffling, and augmentation across runs

**Impact on model:**
- **No direct impact on accuracy** - purely for reproducibility
- Same seed → same results (allows fair comparison)
- Different seed → different results (tests robustness)

**When to use:**
- **Always set**: For reproducible experiments
- **Vary seed**: When testing model robustness (train multiple times with different seeds)

---

## Summary: Key Relationships

### Critical Parameter Mismatches to Avoid:

1. **`in_channels` vs `temporal_options`**:
   - `stacked` → `in_channels: 8`
   - `windowA`/`windowB` → `in_channels: 4`

2. **`num_classes` vs `load_boundaries`**:
   - `load_boundaries: true` → `num_classes: 3`
   - `load_boundaries: false` → `num_classes: 2`

3. **`class_weights` length vs `num_classes`**:
   - `len(class_weights) == num_classes`

4. **`patience` vs `max_epochs`**:
   - `patience` should be close to `max_epochs` (LR decays over training)

### Recommended Configurations:

**For 3-class field delineation (with boundaries):**
```yaml
model:
  init_args:
    model: "unet"
    backbone: "efficientnet-b3"  # or b5, b7
    in_channels: 8
    num_classes: 3
    loss: "logcoshdice"  # or "ce+dice"
    class_weights: [0.05, 0.2, 0.75]
    ignore_index: 3
    lr: 0.001
    patience: 100
data:
  init_args:
    temporal_options: stacked
  dict_kwargs:
    load_boundaries: true
```

**For 2-class field detection (no boundaries):**
```yaml
model:
  init_args:
    model: "unet"
    backbone: "efficientnet-b3"
    in_channels: 8
    num_classes: 2
    loss: "ce"
    class_weights: [0.31, 0.68]
    lr: 0.001
    patience: 100
data:
  init_args:
    temporal_options: stacked
  dict_kwargs:
    load_boundaries: false
```

---

This guide should help you understand how each parameter affects model behavior and training dynamics. Adjust parameters based on your specific dataset, hardware constraints, and accuracy requirements.
