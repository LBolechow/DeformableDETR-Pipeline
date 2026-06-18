# LEGO Brick Detection with Deformable DETR

Part of a broader comparative study on CNN vs. Transformer-based object detection architectures. This repo contains the Deformable DETR training pipeline, built on top of the HuggingFace `transformers` implementation of `SenseTime/deformable-detr`, with a custom data pipeline and COCO-format evaluation.

For context on the full study and dataset, see the [EfficientDet D1 repo](#).

## The Model

Deformable DETR addresses the two main weaknesses of the original DETR: extremely slow convergence (10-20x more epochs than Faster R-CNN) and poor detection of small objects caused by the quadratic cost of full self-attention on high-resolution feature maps.

The fix is a deformable attention mechanism that only samples a small set of key points around a reference point instead of attending to the entire feature map. Combined with multi-scale attention across all FPN levels simultaneously, this cuts training time significantly and improves small object precision.

In practice on this dataset: converged in 200 epochs with no early stopping trigger, and despite looking unimpressive on the validation set, it achieved the **best mAP@50-95 on the held-out test set** across all five compared architectures — at the cost of the longest inference time in the comparison (79.2ms/image).

## The Study

Five architectures compared on a custom synthetic LEGO dataset:

**CNN-based:** Faster R-CNN (ResNet-50), EfficientDet D1, YOLOv8s

**Transformer-based:** RT-DETR-L, Deformable DETR

All models trained under identical conditions: same effective batch size (16), same optimizer (AdamW), same scheduler (CosineAnnealingLR), AMP enabled, deterministic cuDNN, RTX 4060 8GB.

## Dataset

600 synthetic images generated in BrickPoint Studio 2.0, each containing all 10 brick classes. 6000 annotated instances total, annotated in CVAT.

**10 brick classes across two groups:**

Geometrically similar: `2x2`, `2x3`, `1x1`, `1x2`, `1x3`, `2x2corner`, `1x1round`, `1x1curved`

Unique shapes: `2x2roundtile`, `panel1x2x1`

Dataset split into four size variants (100 / 200 / 400 / 600 images) to measure scaling behavior.

## Results

### Validation set (best checkpoint, 600 images)

| Model | mAP@50 | mAP@50-95 | Train time |
|---|---|---|---|
| RT-DETR-L | 0.946 | **0.885** | 898s |
| YOLOv8s | 0.949 | 0.873 | 188s |
| EfficientDet D1 | 0.951 | 0.848 | 1218s |
| **Deformable DETR** | 0.923 | 0.815 | 3023s |
| Faster R-CNN | 0.891 | 0.767 | 1508s |

### Test set (600 images, harder scenes)

| Model | mAP@50 | mAP@50-95 | Inference [ms/img] |
|---|---|---|---|
| **Deformable DETR** | - | **0.566** | 79.2 |
| RT-DETR-L | 0.717 | 0.653 | 21.2 |
| YOLOv8s | 0.690 | 0.607 | **8.8** |
| EfficientDet D1 | 0.655 | 0.555 | 29.7 |
| Faster R-CNN | 0.556 | 0.446 | 33.1 |

The test set contained significantly harder scenes than validation: overlapping and adjacent bricks, varied lighting, and unknown brick types not seen during training. Deformable DETR's validation results underrepresented its actual generalization ability. The 79ms inference time, however, rules it out for real-time use.

## Key Implementation Notes

**Box format:** Dataset outputs normalized `[cx, cy, w, h]` (DETR convention), converted from COCO `[x, y, w, h]` via albumentations + manual normalization.

**Label indexing:** COCO uses 1-indexed categories internally; DETR expects 0-indexed. The dataset converts on output, the evaluator converts back when building COCO predictions.

**AMP dtype:** `bfloat16` instead of `float16` — more stable for transformer training on Ampere GPUs.

**Gradient clipping:** `max_norm=0.1`, much tighter than typical CNN pipelines, which is standard for DETR-family models.

**Effective batch size:** `batch_size=2` with `accumulation_steps=8` gives effective batch of 16, matching the other models in the study. Necessary due to Deformable DETR's high VRAM usage at 640px input.

## Project Structure

```
.
├── train.py                       # entry point
├── configs/
│   └── deformable_detr.yaml       # all hyperparameters here
└── src/
    ├── config.py                  # dataclasses loading the YAML
    ├── dataset.py                 # LegoDatasetDETR + box format conversion
    ├── evaluator.py               # COCO eval via processor.post_process_object_detection
    ├── trainer.py                 # Trainer, EarlyStopping, CheckpointManager, CsvLogger
    └── utils.py                   # seed, collate_fn
```

## Setup

```bash
pip install torch torchvision transformers pycocotools albumentations opencv-python pyyaml
```

Dataset structure expected:

```
.
├── annotations/
│   ├── instances_Train.json
│   └── instances_Validation.json
└── images/
    ├── Train/
    └── Validation/
```

## Training

```bash
python train.py
```

Resumes automatically from `detr_last_checkpoint.pth` if present. To start fresh, set `resume: false` in the config or delete the checkpoint.

Key config options in `configs/deformable_detr.yaml`:

```yaml
training:
  batch_size: 2
  accumulation_steps: 8    # effective batch = 16
  epochs: 200
  learning_rate: 2.0e-4
  grad_clip_norm: 0.1      # tight clipping standard for DETR-family
  amp_dtype: bfloat16
```

## Environment

- WSL2 / Ubuntu 22.04
- CUDA 12.8 + cuDNN (deterministic mode)
- NVIDIA RTX 4060 8GB
- PyTorch with AMP (bfloat16)
