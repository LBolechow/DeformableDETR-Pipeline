import os
import cv2
import time
import csv
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoModelForObjectDetection, DeformableDetrImageProcessor

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
set_seed()


class Config:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    ANNOTATIONS_DIR = os.path.join(BASE_PATH, 'annotations')
    IMAGES_ROOT = os.path.join(BASE_PATH, 'images')

    TRAIN_IMG_DIR = os.path.join(IMAGES_ROOT, 'Train')
    VAL_IMG_DIR = os.path.join(IMAGES_ROOT, 'Validation')
    TRAIN_JSON = os.path.join(ANNOTATIONS_DIR, 'instances_Train.json')
    VAL_JSON = os.path.join(ANNOTATIONS_DIR, 'instances_Validation.json')
    
    MODEL_ID = "SenseTime/deformable-detr"
    IMG_SIZE = 640
    BATCH_SIZE = 2 
    ACCUMULATION_STEPS = 8 
    PATIENCE = 50
    EPOCHS = 200
    LR = 2e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    OUTPUT_MODEL_BEST = os.path.join(BASE_PATH, "detr_best.pth")
    OUTPUT_MODEL_LAST = os.path.join(BASE_PATH, "detr_last.pth")
    CHECKPOINT_LAST = os.path.join(BASE_PATH, "detr_last_checkpoint.pth")
    CSV_LOG_FILE = os.path.join(BASE_PATH, "training_results_detr.csv")
    
    RESUME_TRAINING = True

    CSV_HEADERS = [
        "Epoch", "Train_Loss", "Val_Loss", "LR", "Time_Sec",
        "mAP_0.50:0.95_All", "mAP_0.50_All", "mAP_0.75_All",
        "mAP_Small", "mAP_Medium", "mAP_Large",
        "AR_1", "AR_10", "AR_100", "AR_Small", "AR_Medium", "AR_Large"
    ]

def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

class LegoDatasetDETR(Dataset):
    def __init__(self, img_dir, ann_path):
        self.coco = COCO(ann_path)
        self.img_dir = img_dir
        self.ids = list(self.coco.imgs.keys())
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat2idx = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}
        self.idx2cat = {i + 1: cat_id for i, cat_id in enumerate(self.cat_ids)}
        
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=Config.IMG_SIZE),
            A.PadIfNeeded(min_height=Config.IMG_SIZE, min_width=Config.IMG_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        image = cv2.imread(os.path.join(self.img_dir, img_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 1 and h > 1:
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat2idx[ann['category_id']])

        if len(boxes) > 0:
            sample = self.transform(image=image, bboxes=boxes, labels=labels)
        else:
            sample = self.transform(image=image, bboxes=[], labels=[])
            
        image_tensor = sample['image']
        aug_boxes = sample['bboxes']
        aug_labels = sample['labels']

        detr_boxes = []
        detr_labels = []
        for b, l in zip(aug_boxes, aug_labels):
            x_min, y_min, x_max, y_max = b
            cx = ((x_min + x_max) / 2.0) / Config.IMG_SIZE
            cy = ((y_min + y_max) / 2.0) / Config.IMG_SIZE
            w = (x_max - x_min) / Config.IMG_SIZE
            h = (y_max - y_min) / Config.IMG_SIZE
            detr_boxes.append([cx, cy, w, h])
            detr_labels.append(int(l) - 1)

        target = {
            "boxes": torch.tensor(detr_boxes, dtype=torch.float32).reshape(-1, 4),
            "class_labels": torch.tensor(detr_labels, dtype=torch.long),
            "img_id": torch.tensor([img_id])
        }

        return image_tensor, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets

def evaluate(model, data_loader, device, processor):
    model.eval()
    coco_gt = data_loader.dataset.coco
    coco_pred_results = []
    total_val_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            labels = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            
            outputs = model(pixel_values=images, labels=labels)
            total_val_loss += outputs.loss.item()
            target_sizes = torch.tensor([(Config.IMG_SIZE, Config.IMG_SIZE)] * len(images), device=device)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.0)

            for i, result in enumerate(results):
                if result["scores"].numel() == 0: continue
                
                img_id = targets[i]['img_id'].item()
                img_info = coco_gt.loadImgs(img_id)[0]
                
                scale = max(img_info['height'], img_info['width']) / Config.IMG_SIZE
                pad_y = (Config.IMG_SIZE - int(img_info['height'] / scale)) // 2
                pad_x = (Config.IMG_SIZE - int(img_info['width'] / scale)) // 2
                
                boxes = result["boxes"].cpu().float().numpy()
                scores = result["scores"].cpu().float().numpy()
                detr_labels = result["labels"].cpu().float().numpy()
                
                for b, s, l in zip(boxes, scores, detr_labels):
                    x_min_orig = max(0, (b[0] - pad_x) * scale)
                    y_min_orig = max(0, (b[1] - pad_y) * scale)
                    w_orig = (b[2] - b[0]) * scale
                    h_orig = (b[3] - b[1]) * scale
                    
                    coco_pred_results.append({
                        'image_id': img_id, 
                        'category_id': data_loader.dataset.idx2cat[int(l) + 1],
                        'bbox': [float(x_min_orig), float(y_min_orig), float(w_orig), float(h_orig)], 
                        'score': float(s)
                    })
                    
    avg_val_loss = total_val_loss / len(data_loader)
    if not coco_pred_results: 
        return [0.0] * 12, avg_val_loss
    
    coco_dt = coco_gt.loadRes(coco_pred_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    
    return coco_eval.stats, avg_val_loss

def main():
    use_amp = torch.cuda.is_available()
    
    train_ds = LegoDatasetDETR(Config.TRAIN_IMG_DIR, Config.TRAIN_JSON)
    val_ds = LegoDatasetDETR(Config.VAL_IMG_DIR, Config.VAL_JSON)
    
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    try:
        processor = DeformableDetrImageProcessor.from_pretrained(Config.MODEL_ID)
    except:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(Config.MODEL_ID)

    model = AutoModelForObjectDetection.from_pretrained(
        Config.MODEL_ID,
        num_labels=len(train_ds.cat_ids),
        ignore_mismatched_sizes=True
    ).to(Config.DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = torch.amp.GradScaler(enabled=use_amp)
    
    start_epoch = 0
    best_map = 0
    epochs_without_improvement = 0
    
    if Config.RESUME_TRAINING and os.path.exists(Config.CHECKPOINT_LAST):
        ckpt = torch.load(Config.CHECKPOINT_LAST, map_location=Config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch']
        best_map = ckpt['best_map']
        epochs_without_improvement = ckpt.get('epochs_without_improvement', 0)
        print(f"--> Wznowiono od epoki {start_epoch+1} (Best mAP: {best_map:.4f}, Bez poprawy od: {epochs_without_improvement} epok)")

    if start_epoch == 0:
        with open(Config.CSV_LOG_FILE, mode='w', newline='') as f:
            csv.writer(f).writerow(Config.CSV_HEADERS)

    for epoch in range(start_epoch, Config.EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, (images, targets) in enumerate(train_loader):
            images = images.to(Config.DEVICE, non_blocking=True)
            labels = [{k: v.to(Config.DEVICE, non_blocking=True) for k, v in t.items()} for t in targets]
            
            with torch.amp.autocast(device_type="cuda", enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(pixel_values=images, labels=labels)
                loss = outputs.loss / Config.ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()
            
            if (step + 1) % Config.ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * Config.ACCUMULATION_STEPS

        avg_loss = total_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epoch_dur = time.time() - epoch_start
        
        all_stats = ["N/A"] * 12
        val_loss_str = "N/A"
        
        is_eval_epoch = True
        
        if is_eval_epoch:
            print(f"\n--- Ewaluacja (Epoka {epoch+1}) ---")
            stats, v_loss = evaluate(model, val_loader, Config.DEVICE, processor)
            val_loss_str = f"{v_loss:.4f}"
            all_stats = [f"{s:.4f}" for s in stats]
            
            current_map = stats[0]
            
            if current_map > best_map:
                best_map = current_map
                epochs_without_improvement = 0
                torch.save(model.state_dict(), Config.OUTPUT_MODEL_BEST)
                print(f"✔ Nowy rekord mAP: {best_map:.4f}! Zapisano model best.")
            else:
                epochs_without_improvement += 1
                print(f"✘ Brak poprawy (Best mAP: {best_map:.4f}). Licznik cierpliwości: {epochs_without_improvement}/{Config.PATIENCE}")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_map': best_map,
            'epochs_without_improvement': epochs_without_improvement
        }, Config.CHECKPOINT_LAST)

        torch.save(model.state_dict(), Config.OUTPUT_MODEL_LAST)

        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.4f} | Val_Loss: {val_loss_str} | mAP: {all_stats[0]} | Time: {format_time(epoch_dur)}")
        
        with open(Config.CSV_LOG_FILE, mode='a', newline='') as f:
            csv.writer(f).writerow([epoch+1, f"{avg_loss:.4f}", val_loss_str, f"{current_lr:.6f}", f"{epoch_dur:.2f}"] + all_stats)
        if epochs_without_improvement >= Config.PATIENCE:
            print(f"\n[!] EARLY STOPPING: Brak poprawy przez {Config.PATIENCE} epok. Kończenie treningu.")
            break

if __name__ == "__main__":
    main()
