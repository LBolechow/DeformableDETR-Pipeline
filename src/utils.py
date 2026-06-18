from __future__ import annotations

import random
import time

import numpy as np
import torch


CSV_HEADERS = [
    "Epoch", "Train_Loss", "Val_Loss", "LR", "Time_Sec",
    "mAP_0.50:0.95_All", "mAP_0.50_All", "mAP_0.75_All",
    "mAP_Small", "mAP_Medium", "mAP_Large",
    "AR_1", "AR_10", "AR_100", "AR_Small", "AR_Medium", "AR_Large",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def format_time(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def collate_fn(batch: list) -> tuple[torch.Tensor, list[dict]]:
    """Stack images into a single tensor; keep targets as a list."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
