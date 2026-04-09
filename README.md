# Deformable DETR Training Pipeline

## Overview

This repository provides a full training and evaluation pipeline for object detection using the Deformable DETR architecture from the Hugging Face Transformers library. The implementation is designed for COCO-format datasets and supports efficient training through mixed precision, gradient accumulation, and checkpointing.

The pipeline is suitable for research experiments as well as production-oriented training workflows.

---

## Features

* Deformable DETR model via Hugging Face Transformers
* COCO-format dataset support
* Automatic mixed precision (AMP) with bfloat16
* Gradient accumulation for memory efficiency
* Cosine annealing learning rate scheduling
* Early stopping based on validation mAP
* Resume training from checkpoints
* COCO evaluation metrics (mAP, AR)
* CSV-based logging

---

## Project Structure

```
project_root/
│
├── annotations/
│   ├── instances_Train.json
│   └── instances_Validation.json
│
├── images/
│   ├── Train/
│   └── Validation/
│
├── detr_best.pth
├── detr_last.pth
├── detr_last_checkpoint.pth
├── training_results_detr.csv
└── train.py
```

---

## Requirements

Install required dependencies:

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install albumentations
pip install pycocotools
pip install transformers
```

---

## Dataset Format

The dataset must follow the COCO annotation format.

### Images

* Training images: `images/Train/`
* Validation images: `images/Validation/`

### Annotations

* Training annotations: `annotations/instances_Train.json`
* Validation annotations: `annotations/instances_Validation.json`

Each annotation file must contain:

* `images`
* `annotations`
* `categories`

Bounding boxes are expected in COCO format: `[x, y, width, height]`.

---

## Data Processing

Images are:

* Resized while preserving aspect ratio
* Padded to a fixed resolution
* Normalized using ImageNet statistics

Bounding boxes are converted from COCO format to normalized center format required by DETR:

```
(cx, cy, width, height) in range [0, 1]
```

---

## Configuration

All parameters are defined in the `Config` class:

* `MODEL_ID` – pretrained model identifier
* `IMG_SIZE` – input resolution
* `BATCH_SIZE` – batch size
* `ACCUMULATION_STEPS` – gradient accumulation steps
* `EPOCHS` – number of training epochs
* `LR` – learning rate
* `PATIENCE` – early stopping patience
* `DEVICE` – training device

Paths for datasets, checkpoints, and logs are also configured here.

---

## Training

Run training with:

```bash
python train.py
```

### Training Details

* Model initialized from pretrained weights (`SenseTime/deformable-detr`)
* Optimizer: AdamW
* Scheduler: CosineAnnealingLR
* Mixed precision with bfloat16 when CUDA is available
* Gradient clipping for training stability

---

## Evaluation

Evaluation is performed after each epoch using COCO metrics.

Metrics include:

* mAP (IoU 0.50:0.95)
* mAP (IoU 0.50)
* mAP (IoU 0.75)
* mAP across object scales
* Average Recall (AR)

Predictions are post-processed using the corresponding Hugging Face image processor.

---

## Checkpointing

The training process automatically saves:

* `detr_best.pth` – best-performing model (based on mAP)
* `detr_last.pth` – latest model state
* `detr_last_checkpoint.pth` – full training state

Training resumes automatically if checkpoint loading is enabled.

---

## Logging

Training progress is stored in:

```
training_results_detr.csv
```

Each entry includes:

* Epoch
* Training loss
* Validation loss
* Learning rate
* Execution time
* COCO evaluation metrics

---

## Early Stopping

Training stops when no improvement in validation mAP is observed for a number of epochs defined by `PATIENCE`.

---

## Reproducibility

Random seeds are fixed across:

* Python
* NumPy
* PyTorch (CPU and CUDA)

This ensures deterministic and reproducible results.

---

## Notes

* The implementation handles images without annotations.
* Label indices are adjusted to match DETR requirements (zero-based indexing).
* Post-processing converts predictions back to COCO format for evaluation.

---

## License

No license is included by default. Add one if distribution is intended.

---

## Acknowledgments

* Deformable DETR model via Hugging Face Transformers
* COCO evaluation tools via `pycocotools`
