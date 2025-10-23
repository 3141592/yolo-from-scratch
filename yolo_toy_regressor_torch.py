# yolo_toy_regressor_2_torch.py
# PyTorch port (with history recorder) of your minimal single-box regressor.
# - Uses SmoothL1Loss (Huber), AMP, ReduceLROnPlateau
# - EarlyStopping on val_iou (maximize)
# - Records per-epoch: lr, train_loss, val_loss, val_iou and plots them.
#
# Note: PyTorch expects NCHW; images are converted from NHWC.

import os
import sys
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from datasets import load_dataset
from skimage.measure import label, regionprops

from utils.logger_quality import log_quality

# =======================
# Global Configuration
# =======================
CONFIG = {
    # Debug / Control
    "OVERFIT_ONE": False,               # Run 1-image sanity check & exit
    "DEBUG": False,                      # Verbose logs & gradient prints

    # AMP automatic mixed precision training
    "USE_AMP": False,

    # Data & Preprocessing
    "IMAGE_SIZE": 448,                  # Resize to (IMAGE_SIZE, IMAGE_SIZE)
    "SAMPLE_SIZE": 1024,                # Total images to use
    "MAX_SAMPLES": 1424,                 # Max samples to draw from parquet, 1424 total
    "TRAIN_SPLIT": 0.8,                 # Train/validation split fraction
    "TRAIN_PATH": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet",
    "VAL_PATH":   "/home/roy/src/data/voc2012/val-00000-of-00001.parquet",

    # Dataloader
    "BATCH_SIZE": 4,
    "NUM_WORKERS": 2,
    "PIN_MEMORY": True,

    # Model Init
    # Default bias to start predictions roughly centered with a modest box
    "INIT_BIAS": [0.0, 0.0, -1.3862944, -1.3862944],
    # Overfit-one special bias (log-space for w/h handled in code below)
    "OVERFIT_BIAS_LOG": [0.0, 0.0, -1.609, -1.609],  # ~log(0.2) for w,h

    # Overfit-One harness
    "OVERFIT_STEPS": 600,
    "OVERFIT_LR": 0.04,
    "OVERFIT_WD": 0.0,

    # Training
    "TOTAL_EPOCHS": 135,
    "WARMUP": 5,
    "MAX_GRAD_NORM": 1.0,               # gradient clipping L2 norm
    "EPS": 1e-7,
    "LOSS_FUNCTION": "MSELoss",         # "YOLOBBoxLoss", "MSELoss"
    "TRANSFORM": "identity",             # "identity", "sigmoid"

    # Optimizer
    "OPTIMIZER": "AdamW",               # "AdamW", "SGD"
    "LR": 1e-3,
    "WEIGHT_DECAY": 5e-4,
    "BETAS": (0.9, 0.999),
    "MOMENTUM": 0.9,

    # Scheduler (ReduceLROnPlateau on val_loss)
    "MODE": "min",
    "LR_SCHEDULER": "ReduceLROnPlateau",
    "LR_FACTOR": 0.5,
    "LR_PATIENCE": 4,
    "LR_MIN": 1e-6,

    # Early Stopping (maximize val_iou)
    "EARLY_PATIENCE": 15,
    "EARLY_MIN_DELTA": 0.0,

    # Checkpoint
    "BEST_MODEL_PATH": "best_val_iou.pt",
}

# convenience flags
OVERFIT_ONE = CONFIG["OVERFIT_ONE"]
DEBUG = CONFIG["DEBUG"]
USE_AMP = CONFIG["USE_AMP"]

# ---------- Helpers (ported 1:1) ----------
def mask_bytes_to_boxes(mask_bytes):
    im = Image.open(BytesIO(mask_bytes))
    arr = np.array(im)
    if arr.ndim == 3:
        mask_bin = np.any(arr != 0, axis=2).astype(np.uint8)
    else:
        mask_bin = (arr != 0).astype(np.uint8)
    lab = label(mask_bin)
    props = regionprops(lab)
    boxes = []
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        boxes.append((minc, minr, maxc - 1, maxr - 1))  # x_min,y_min,x_max,y_max
    return boxes

def box_to_xywh_norm(box, H, W):
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    xc = x_min + (w - 1) / 2.0
    yc = y_min + (h - 1) / 2.0
    return np.array([xc / W, yc / H, w / W, h / H], dtype=np.float32)

def iou_xywh_np(pred, gt):
    def to_xyxy(x):
        xc, yc, w, h = x
        x_min = xc - 0.5*w
        y_min = yc - 0.5*h
        x_max = xc + 0.5*w
        y_max = yc + 0.5*h
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    p = to_xyxy(pred); g = to_xyxy(gt)
    inter_min = np.maximum(p[:2], g[:2])
    inter_max = np.minimum(p[2:], g[2:])
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_area = inter[0]*inter[1]
    ap = (p[2]-p[0])*(p[3]-p[1])
    ag = (g[2]-g[0])*(g[3]-g[1])
    union = ap + ag - inter_area
    if union <= 0: return 0.0
    return float(inter_area / union)

def draw_xywh(ax, xywh, W, H, color, label=None):
    xc, yc, w, h = xywh.tolist()
    x1 = int((xc - w/2) * W); y1 = int((yc - h/2) * H)
    x2 = int((xc + w/2) * W); y2 = int((yc + h/2) * H)
    ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                   fill=False, edgecolor=color, linewidth=2, label=label))
    ax.scatter([xc*W],[yc*H], s=24, c=color)

    # ---- debug: what are we actually drawing? ----
    if DEBUG:
        # normalized box and pixel corners
        print(
            f"[dbg/draw] {label or 'box'} "
            f"xywh_norm=({xc:.6f},{yc:.6f},{w:.6f},{h:.6f})  "
            f"corners_px=({x1},{y1})→({x2},{y2})  size_px=({x2-x1},{y2-y1})  W×H=({W},{H})"
        )
        if (x2 - x1) <= 2 or (y2 - y1) <= 2:
            print("  [warn] drawn box ≤2 px in at least one dimension (looks like a dot).")

@torch.no_grad()
def iou_xywh_torch(pred, gt):
    # pred, gt: [B,4] in [xc,yc,w,h] normalized
    p = torch.stack([
        pred[:,0] - 0.5*pred[:,2],
        pred[:,1] - 0.5*pred[:,3],
        pred[:,0] + 0.5*pred[:,2],
        pred[:,1] + 0.5*pred[:,3],
    ], dim=1)
    g = torch.stack([
        gt[:,0] - 0.5*gt[:,2],
        gt[:,1] - 0.5*gt[:,3],
        gt[:,0] + 0.5*gt[:,2],
        gt[:,1] + 0.5*gt[:,3],
    ], dim=1)
    inter_mins = torch.maximum(p[:, :2], g[:, :2])
    inter_maxs = torch.minimum(p[:, 2:], g[:, 2:])
    inter_wh   = torch.clamp(inter_maxs - inter_mins, min=0.0)
    inter_area = inter_wh[:,0] * inter_wh[:,1]
    area_p = (p[:,2]-p[:,0]) * (p[:,3]-p[:,1])
    area_g = (g[:,2]-g[:,0]) * (g[:,3]-g[:,1])
    union  = area_p + area_g - inter_area
    iou    = torch.where(union > 0, inter_area / union, torch.zeros_like(union))
    return iou.mean().item()

def get_samples_from_dataset(dataset):
    max_samples = CONFIG["MAX_SAMPLES"]
    image_size = CONFIG["IMAGE_SIZE"]
    X = []
    Y = []
    for i, ex in enumerate(dataset):
        if i >= max_samples: break
        img = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB").resize((image_size,image_size))
        mask_boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
        if not mask_boxes:
            continue
        areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in mask_boxes]
        best = mask_boxes[int(np.argmax(areas))]
        orig = Image.open(BytesIO(ex["image"]["bytes"]))
        W, H = orig.size
        Y.append(box_to_xywh_norm(best, H, W))
        X.append(np.array(img).astype(np.float32)/255.0)
    X = np.stack(X)  # (N, image_size, image_size, 3)
    Y = np.stack(Y)  # (N, 4)
    return X, Y

# ---------- Model ----------
def Conv2D(in_ch, out_ch, k, s=1):
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad),
        nn.LeakyReLU(0.1, inplace=True),
    )

class YoloRegressorModel(nn.Module):
    def __init__(self):
        super().__init__()
        pool = lambda: nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat = nn.Sequential(
            # Layer 1
            Conv2D(3, 192, 7, s=2),
            pool(),
            # Layer 2
            Conv2D(192, 256, 3),
            pool(),
            # Layer 3
            Conv2D(256, 128, 1),
            Conv2D(128, 256, 3),
            Conv2D(256, 256, 1),
            Conv2D(256, 512, 3),
            pool(),
            # Layer 4
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 256, 1),
            Conv2D(256, 512, 3),
            Conv2D(512, 512, 1),
            Conv2D(512, 1024, 3),
            pool(),
            # Layer 5
            Conv2D(1024, 512, 1),
            Conv2D(512, 1024, 3),
            Conv2D(1024, 512, 1),
            Conv2D(512, 1024, 3),
            Conv2D(1024, 1024, 3),
            # stride-2 conv
            Conv2D(1024, 1024, 3, s=2),
            # Layer 6
            Conv2D(1024, 1024, 3),
            Conv2D(1024, 1024, 3),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 448, 448)
            flat = self.feat(dummy).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4)  # [xc,yc,w,h]
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return x

# ---------- Training / Eval ----------
def yolo_v1_lr_schedule(epoch, total_epochs=CONFIG["TOTAL_EPOCHS"], warmup=CONFIG["WARMUP"]):
    if epoch < warmup:
        return 1e-3 + (1e-2 - 1e-3) * (epoch + 1) / max(1, warmup)
    elif epoch < warmup + 75:
        return 1e-2
    elif epoch < warmup + 75 + 30:
        return 1e-3
    else:
        return 1e-4

class EarlyStoppingMax:
    def __init__(self, patience=12, min_delta=0.0):
        self.best = -float("inf")
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0

    def step(self, metric):
        improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.count = 0
        else:
            self.count += 1
        return improved, self.count >= self.patience

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        preds = model(xb)
        loss = loss_fn(preds, yb)
        total_loss += loss.item() * xb.size(0)
        total_iou  += iou_xywh_torch(preds, yb) * xb.size(0)
        n += xb.size(0)
    return total_loss / n, total_iou / n

def train_one_epoch(model, loader, device, opt, scaler, loss_fn, max_norm=1.0):
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type="cuda"):
            preds = model(xb)
            if CONFIG["DEBUG"]:
                # [xc,yc,w,h]
                print(f"[xc,yc,w,h]: {preds}")

            loss = loss_fn(preds, yb)
        scaler.scale(loss).backward()
        if max_norm is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(opt)
        scaler.update()

        running += loss.item() * xb.size(0)
        n += xb.size(0)
    return running / n

def main():
    # --- Load data (same parquet paths you used) ---
    ds = load_dataset(
        "parquet",
        data_files={
            "train": CONFIG["TRAIN_PATH"],
            "val":   CONFIG["VAL_PATH"],
        }
    )
    train = ds["train"]; val = ds["val"]

    sample_list = [train[i] for i in range(16)]
    X, Y = get_samples_from_dataset(sample_list)
    idx = int(CONFIG["TRAIN_SPLIT"]*len(X))
    X_tr, Y_tr = X[:idx], Y[:idx]
    X_va, Y_va = X[idx:], Y[idx:]

    # Convert to NCHW torch tensors
    X_tr_t = torch.from_numpy(np.transpose(X_tr, (0,3,1,2))).contiguous()
    Y_tr_t = torch.from_numpy(Y_tr).contiguous()
    X_va_t = torch.from_numpy(np.transpose(X_va, (0,3,1,2))).contiguous()
    Y_va_t = torch.from_numpy(Y_va).contiguous()

    train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t),
                              batch_size=CONFIG["BATCH_SIZE"],
                              shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"],
                              pin_memory=CONFIG["PIN_MEMORY"])
    val_loader   = DataLoader(TensorDataset(X_va_t, Y_va_t),
                              batch_size=CONFIG["BATCH_SIZE"],
                              shuffle=False,
                              num_workers=CONFIG["NUM_WORKERS"],
                              pin_memory=CONFIG["PIN_MEMORY"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YoloRegressorModel().to(device)
    loss_fn = nn.MSELoss()
    opt = optim.SGD(model.parameters(),
                    lr=CONFIG["LR"],
                    momentum=CONFIG["MOMENTUM"],
                    weight_decay=CONFIG["WEIGHT_DECAY"])
    plateau = optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                   mode=CONFIG["MODE"],
                                                   factor=CONFIG["LR_FACTOR"],
                                                   patience=CONFIG["LR_PATIENCE"],
                                                   min_lr=CONFIG["LR_MIN"])
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    TOTAL_EPOCHS = CONFIG["TOTAL_EPOCHS"]
    early = EarlyStoppingMax(patience=CONFIG["EARLY_PATIENCE"], min_delta=CONFIG["EARLY_MIN_DELTA"])
    best_val_iou = -1.0
    best_path = CONFIG["BEST_MODEL_PATH"]

    # ---- Simple history recorder ----
    history = {"epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_iou": []}

    for epoch_num in range(1, TOTAL_EPOCHS + 1):
        # epoch-wise LR schedule
        lr = yolo_v1_lr_schedule(epoch_num - 1, total_epochs=TOTAL_EPOCHS, warmup=CONFIG["WARMUP"])
        for pg in opt.param_groups:
            pg["lr"] = lr

        train_loss = train_one_epoch(model, train_loader, device, opt, scaler, loss_fn, CONFIG["MAX_GRAD_NORM"])
        val_loss, val_iou = evaluate(model, val_loader, device, loss_fn)
        plateau.step(val_loss)

        print(f"Epoch {epoch_num:03d} | lr {lr:.6f} | train {train_loss:.4f} | val {val_loss:.4f} | val_iou {val_iou:.4f}")

        # record history
        history["epoch"].append(epoch_num)
        history["lr"].append(lr)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        log_quality(epoch_num, lr, train_loss, val_loss, val_iou)

        improved, stop = early.step(val_iou)
        if improved:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved new best (val_iou={val_iou:.4f}) to {best_path}")
            log_quality(epoch_num, lr, train_loss, val_loss, val_iou, best=True)

        if stop:
            print("Early stopping triggered.")
            break

    # Load best & final eval
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best weights from {best_path} (best val_iou={best_val_iou:.4f})")

    # Evaluate IoU on validation (numpy version for parity with your printout)
    model.eval()

    P, T = [], []
    with torch.no_grad():
        for i in range(len(X_va)):
            x = torch.from_numpy(X_va[i]).permute(2,0,1).unsqueeze(0).to(device)
            p = model(x)[0].detach().cpu().float().numpy()
            P.append(p); T.append(Y_va[i])
    P, T = np.stack(P), np.stack(T)   # (N,4)

    print("pred mean:", P.mean(0), "pred std:", P.std(0))
    print("tgt  mean:", T.mean(0), "tgt  std:", T.std(0))

    with torch.no_grad():
        for i in range(min(5, len(X_va))):
            # 1) Get the exact tensor you would feed the model
            # If your model expects CHW float in [0,1] and normalized, do that here.
            img_np = X_va[i]                      # (H,W,3) in [0,1]?
            H, W, _ = img_np.shape

            # build model input exactly like your val pipeline
            x_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)  # (1,3,H,W)
            # apply same normalization as training/val transforms if used

            # 2) Predict
            pred = model(x_t)[0].detach().cpu().float().numpy()  # [xc,yc,w,h] in [0,1]
            # [xc,yc,w,h]
            print(f"[xc,yc,w,h]: {i} {pred}")

            # 3) Draw
            def draw_box_xywh(ax, xywh, color):
                xc, yc, w, h = xywh
                x1 = int((xc - 0.5*w) * W); x2 = int((xc + 0.5*w) * W)
                y1 = int((yc - 0.5*h) * H); y2 = int((yc + 0.5*h) * H)
                ax.add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                           fill=False, edgecolor=color, linewidth=2))
                ax.scatter([xc*W], [yc*H], s=24)  # draw center for sanity

            gt = Y_va[i]  # must be normalized [xc,yc,w,h] for the same image

            plt.figure()
            plt.imshow((img_np*255).astype(np.uint8))  # if already in [0,1]
            draw_box_xywh(plt.gca(), gt,   'g')
            draw_box_xywh(plt.gca(), pred, 'r')
            plt.title(f"IoU: {iou_xywh_np(pred, gt):.3f}")
            plt.axis('off'); plt.show()


if __name__ == "__main__":
    main()

