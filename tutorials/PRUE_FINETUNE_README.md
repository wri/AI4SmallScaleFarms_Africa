# PRUE Model Finetuning Configs

These configs are designed to finetune PRUE_EFNET models on the `kenya_counties_batch` dataset.

## Available Configs

- `prue_efnet_b3_finetune_config.yaml` - EfficientNet-B3 (smallest, fastest)
- `prue_efnet_b5_finetune_config.yaml` - EfficientNet-B5 (balanced)
- `prue_efnet_b7_finetune_config.yaml` - EfficientNet-B7 (largest, most accurate)

## Key Features

- **3-class training**: `load_boundaries: true` (important for better performance)
- **Model-specific directories**: Checkpoints saved to `PRUE_EFNET_B{3,5,7}_finetune/`
- **Same data**: All use `kenya_counties_batch` dataset
- **Optimized batch sizes**: Adjusted per model size (B3: 8, B5: 6, B7: 4)

## Usage

### 1. Finetune from PRUE pretrained checkpoint:

```bash
# Download and use PRUE_EFNET_B7 pretrained model
ftw model fit --config configs/prue_efnet_b7_finetune_config.yaml \
  --ckpt_path <path_to_downloaded_PRUE_B7.ckpt>
```
# path ~/.cache/torch/hub/checkpoints/PRUE_EFNET_B7.ckpt
Or use the model registry name (auto-downloads):
```bash
# First download the PRUE checkpoint manually, or use:
ftw model fit --config configs/prue_efnet_b7_finetune_config.yaml \
  --ckpt_path $(python -c "from pathlib import Path; import torch; print(Path(torch.hub.get_dir()) / 'checkpoints' / 'PRUE_EFNET_B7.ckpt')")
```

### 2. Train from scratch (no pretrained checkpoint):

```bash
ftw model fit --config configs/prue_efnet_b7_finetune_config.yaml
```

### 3. Test your finetuned model:

```bash
ftw model test \
  --model PRUE_EFNET_B7_finetune/lightning_logs/version_X/checkpoints/last.ckpt \
  --countries kenya_counties_batch \
  --dir data \
  --gpu -1 \
  --out PRUE_EFNET_B3_finetune_iou_0.5_result_test_on_3.csv \
  --model_predicts_3_classes \
  --test_on_3_classes
```

## Configuration Details

- **Learning rate**: 5e-4 (lower than training from scratch, good for finetuning)
- **Loss**: Jaccard (works well for segmentation)
- **Classes**: 3 (background, field, boundary)
- **Input channels**: 8 (stacked temporal windows)
- **Max epochs**: 100
- **Patience**: 50 (for learning rate scheduling)

## Notes

- Adjust `batch_size` based on your GPU memory
- Adjust `devices: [0]` to use different GPUs
- Change `accelerator: "gpu"` to `"cpu"` if no GPU available
- The `default_root_dir` creates a directory named after the model for easy organization
