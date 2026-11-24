# LEGO Brick Detection with Deformable DETR

Part of a master's thesis comparing CNN-based and Transformer-based object detectors for LEGO brick recognition. This repository contains a **custom training and evaluation pipeline** built around Hugging Face's `transformers` implementation of [`SenseTime/deformable-detr`](https://huggingface.co/SenseTime/deformable-detr). The model architecture itself is pretrained; the data processing, training loop, checkpointing, and COCO-format evaluation are handled by this project.

## The Model

Deformable DETR improves on the original DETR architecture by replacing dense attention across entire feature maps with **deformable attention**: a limited number of sampling points around reference locations. Multi-scale feature processing also helps the detector handle objects at different sizes. These design choices improve convergence and small-object detection compared with the original DETR.

**In this study:** Deformable DETR achieved **0.8153 validation mAP@50–95** and **0.5658 test mAP@50–95** on the 600-image dataset variant. Its inference time was **79.2 ms/image**, the slowest of the five models. It did **not** achieve the best test accuracy; **RT-DETR-L** ranked first with **0.6530 test mAP@50–95**. Deformable DETR also required substantially more training time than the alternatives.

## The Study

Five object detectors were compared using a custom synthetic LEGO dataset:

- **CNN-based:** Faster R-CNN (ResNet-50), EfficientDet D1, YOLOv8s
- **Transformer-based / hybrid:** RT-DETR-L, Deformable DETR

The experiments used the same dataset variants and evaluation metrics on an **NVIDIA RTX 4060 (8 GB VRAM)**. The setups aimed for an effective training batch size of 16, but model implementations and some training parameters differed. The results therefore compare the selected **model configurations**, rather than isolating architectural differences alone.

## Dataset

The full dataset contains **600 synthetic images** (640 × 480 px), created using **BrickLink Studio 2.0** and annotated in **CVAT**, with **6,000 annotated brick instances** across 10 classes.

**Geometrically similar classes:** `2x2`, `2x3`, `1x1`, `1x2`, `1x3`, `2x2corner`, `1x1round`, `1x1curved`

**Distinctive shapes:** `2x2roundtile`, `panel1x2x1`

Scenes varied in object placement, orientation, color, background, camera position, lighting, and simulated surface wear. Annotations were prepared in COCO format.

Four dataset variants were tested: **100 / 200 / 400 / 600 images**, each split **80% training / 10% validation / 10% test**. For the largest variant, this means **480 training, 60 validation, and 60 held-out test images**. Test images were intentionally more challenging than the training and validation scenes, including close or overlapping objects and more extreme visual conditions. Additional qualitative tests examined unfamiliar brick types.

## Results

All results below are from models trained with the **600-image dataset variant**. Training times correspond to the complete recorded training runs, including early-stopping behavior where applicable.

### Validation set (best checkpoint)

| Model | mAP@50 | mAP@50–95 | Training time |
|---|---:|---:|---:|
| RT-DETR-L | 0.9460 | **0.8845** | 5,116.63 s |
| YOLOv8s | 0.9487 | 0.8733 | **564.27 s** |
| EfficientDet D1 | **0.9511** | 0.8482 | 1,217.50 s |
| **Deformable DETR** | 0.9234 | 0.8153 | 12,944.40 s |
| Faster R-CNN | 0.8911 | 0.7668 | 1,507.90 s |

### Held-out test set (60 harder scenes)

| Model | mAP@50 | mAP@50–95 | Inference (ms/image) |
|---|---:|---:|---:|
| RT-DETR-L | **0.7170** | **0.6530** | 21.2 |
| YOLOv8s | 0.6900 | 0.6070 | **8.8** |
| **Deformable DETR** | 0.6642 | 0.5658 | 79.2 |
| EfficientDet D1 | 0.6553 | 0.5548 | 29.7 |
| Faster R-CNN | 0.5560 | 0.4456 | 33.1 |

**Key takeaways:**

- **RT-DETR-L** achieved the highest mAP@50–95 on the harder held-out test set.
- **YOLOv8s** was the fastest model and provided a strong accuracy–speed trade-off.
- **Deformable DETR** ranked third in test mAP@50–95 and was the slowest at inference. It was not a favorable choice for this application's speed requirements.
- Deformable DETR's test mAP@50–95 (**0.5658**) was substantially below its validation result (**0.8153**), illustrating the difficulty of the changed test conditions.
- These numbers apply to the tested models and configurations on this synthetic dataset; they are not universal architectural rankings.

### Convergence

For the 600-image variant, Deformable DETR achieved its best validation result around **epoch 102**, and early stopping ended the run after **152 epochs** (patience: 50). On the smallest, 100-image variant, training reached the **200-epoch limit** without triggering early stopping. These are different experiments and should not be conflated.

## Key Implementation Notes

- **Bounding boxes:** COCO `[x, y, w, h]` annotations are converted into the model's normalized `[cx, cy, w, h]` representation.
- **Class indices:** COCO category IDs are mapped to the indices expected by the model and mapped back for COCO-format evaluation.
- **Mixed precision:** Uses CUDA AMP with configurable `bfloat16` autocasting.
- **Gradient clipping:** `max_norm=0.1` to constrain gradient magnitude during optimization.
- **Effective batch size:** `batch_size=2` with `accumulation_steps=8` gives 16 effective samples per optimization step, allowing training within the 8 GB VRAM budget.
- **Optimizer and schedule:** AdamW with `CosineAnnealingLR`.
- **Checkpointing:** Saves best/last model weights plus resumable training state, including optimizer, scheduler, scaler, and early stopping progress.
- **Evaluation:** Uses Hugging Face image post-processing and COCO metrics; training metrics are logged to CSV.

## Project Structure

```text
.
├── train.py                      # entry point
├── configs/
│   └── deformable_detr.yaml      # model and training configuration
└── src/
    ├── config.py                 # YAML configuration / dataclasses
    ├── dataset.py                # LegoDatasetDETR and box conversion
    ├── evaluator.py              # predictions and COCO evaluation
    ├── trainer.py                # training, early stopping, checkpoints, CSV logs
    └── utils.py                  # random seed and collate_fn
```

## Setup

```bash
pip install torch torchvision transformers pycocotools albumentations opencv-python pyyaml
```

Expected directory structure:

```text
.
├── annotations/
│   ├── instances_Train.json
│   └── instances_Validation.json
└── images/
    ├── Train/
    └── Validation/
```

The training pipeline reads the training and validation subsets; the separate held-out test set is used for final evaluation.

## Training

```bash
python train.py
```

With `resume: true`, training resumes from `detr_last_checkpoint.pth` when that checkpoint is available. To start over, set `resume: false` in the configuration or remove the checkpoint.

Selected options from `configs/deformable_detr.yaml`:

```yaml
training:
  batch_size: 2
  accumulation_steps: 8  # effective batch = 16
  epochs: 200
  learning_rate: 2.0e-4
  weight_decay: 1.0e-4
  patience: 50
  grad_clip_norm: 0.1
  seed: 42
  resume: true
  amp_dtype: bfloat16
```

## Environment

- WSL2 / Ubuntu 22.04
- NVIDIA RTX 4060 (8 GB VRAM)
- PyTorch / CUDA with AMP (`bfloat16`)

## Other Models from the Study

The remaining architectures—**YOLOv8s, RT-DETR-L, Faster R-CNN, and EfficientDet D1**—were evaluated in separate pipelines and are not included in this repository.
