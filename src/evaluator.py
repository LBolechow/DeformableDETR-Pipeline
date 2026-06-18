from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from transformers import DeformableDetrImageProcessor


@dataclass
class EvalResult:
    stats: list[float]
    val_loss: float

    @property
    def map_50_95(self) -> float:
        return self.stats[0]

    @property
    def map_50(self) -> float:
        return self.stats[1]

    def stats_as_strings(self) -> list[str]:
        return [f"{s:.4f}" for s in self.stats]


def _rescale_prediction(
    box: np.ndarray,
    img_info: dict,
    img_size: int,
) -> tuple[float, float, float, float]:
    """Convert padded model-space box back to original image coordinates."""
    scale = max(img_info["height"], img_info["width"]) / img_size
    pad_x = (img_size - int(img_info["width"] / scale)) // 2
    pad_y = (img_size - int(img_info["height"] / scale)) // 2

    x_min = max(0.0, (box[0] - pad_x) * scale)
    y_min = max(0.0, (box[1] - pad_y) * scale)
    w = (box[2] - box[0]) * scale
    h = (box[3] - box[1]) * scale
    return x_min, y_min, w, h


def _collect_predictions(
    results: list[dict],
    targets: list[dict],
    coco_gt: COCO,
    idx2cat: dict,
    img_size: int,
) -> list[dict]:
    """Convert processor output to COCO prediction format."""
    predictions = []
    for i, result in enumerate(results):
        if result["scores"].numel() == 0:
            continue

        img_id = targets[i]["img_id"].item()
        img_info = coco_gt.loadImgs(img_id)[0]

        boxes = result["boxes"].cpu().float().numpy()
        scores = result["scores"].cpu().float().numpy()
        # DETR returns 0-indexed labels; convert back to 1-indexed for idx2cat
        labels = result["labels"].cpu().float().numpy()

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, w, h = _rescale_prediction(box, img_info, img_size)
            predictions.append({
                "image_id": img_id,
                "category_id": idx2cat[int(label) + 1],
                "bbox": [float(x_min), float(y_min), float(w), float(h)],
                "score": float(score),
            })

    return predictions


def _run_coco_eval(coco_gt: COCO, predictions: list[dict]) -> list[float]:
    if not predictions:
        return [0.0] * 12

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return list(coco_eval.stats)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    processor: DeformableDetrImageProcessor,
    img_size: int,
) -> EvalResult:
    model.eval()

    coco_gt = data_loader.dataset.coco
    idx2cat = data_loader.dataset.idx2cat
    total_val_loss = 0.0
    all_predictions = []

    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        outputs = model(pixel_values=images, labels=labels)
        total_val_loss += outputs.loss.item()

        target_sizes = torch.tensor(
            [(img_size, img_size)] * len(images), device=device
        )
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0
        )
        all_predictions.extend(
            _collect_predictions(results, targets, coco_gt, idx2cat, img_size)
        )

    avg_val_loss = total_val_loss / len(data_loader)
    stats = _run_coco_eval(coco_gt, all_predictions)

    return EvalResult(stats=stats, val_loss=avg_val_loss)
