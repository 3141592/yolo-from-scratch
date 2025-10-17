# yolo_toy_regressor_torch.py
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
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import torch.nn.functional as F

from datasets import load_dataset
from dataclasses import dataclass
from skimage.measure import label, regionprops

from utils.logger_quality import log_quality

# =======================
# Global Configuration
# =======================
CONFIG = {
    # Debug / Control
    "OVERFIT_ONE": False,               # Run 1-image sanity check & exit
    "DEBUG": True,                      # Verbose logs & gradient prints

    # AMP automatic mixed precision training
    "USE_AMP": False,

    # Data & Preprocessing
    "IMAGE_SIZE": 448,                  # Resize to (IMAGE_SIZE, IMAGE_SIZE)
    "MAX_SAMPLES": 128,                 # Max samples to draw from parquet
    "TRAIN_SPLIT": 0.8,                 # Train/validation split fraction
    "TRAIN_PATH": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet",
    "VAL_PATH":   "/home/roy/src/data/voc2012/val-00000-of-00001.parquet",

    # Dataloader
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 2,
    "PIN_MEMORY": True,

    # Model Init
    # Default bias to start predictions roughly centered with a modest box
    "INIT_BIAS": [0.0, 0.0, -1.3862944, -1.3862944],
    # Overfit-one special bias (log-space for w/h handled in code below)
    "OVERFIT_BIAS_LOG": [0.0, 0.0, -1.609, -1.609],  # ~log(0.2) for w,h

    # Overfit-One harness
    "OVERFIT_STEPS": 600,
    "OVERFIT_LR": 1e-2,
    "OVERFIT_WD": 0.0,

    # Training
    "TOTAL_EPOCHS": 10,
    "MAX_GRAD_NORM": 1.0,               # gradient clipping L2 norm
    "EPS": 1e-7, 

    # Optimizer
    "OPTIMIZER": "AdamW",
    "LR": 3e-3,
    "WEIGHT_DECAY": 5e-4,
    "BETAS": (0.9, 0.999),
    "MOMENTUM": 0.9,

    # Scheduler (ReduceLROnPlateau on val_loss)
    "LR_SCHEDULER": "ReduceLROnPlateau",
    "LR_FACTOR": 0.5,
    "LR_PATIENCE": 4,
    "LR_MIN": 1e-6,

    # Early Stopping (maximize val_iou)
    "EARLY_PATIENCE": 12,
    "EARLY_MIN_DELTA": 0.0,

    # Checkpoint
    "BEST_MODEL_PATH": "best_val_iou.pt",
}

# convenience flags
OVERFIT_ONE = CONFIG["OVERFIT_ONE"]
DEBUG = CONFIG["DEBUG"]
USE_AMP = CONFIG["USE_AMP"]

# ---------- Helpers (ported 1:1, with CONFIG where relevant) ----------
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

@dataclass
class EpochStats:
    raw_min: float = float('inf')
    raw_max: float = float('-inf')
    raw_sum: float = 0.0
    raw_count: int = 0
    loss_sum: float = 0.0
    n_samples: int = 0

    def update_raw(self, raw: torch.Tensor):
        r = raw.detach()
        self.raw_min = min(self.raw_min, r.min().item())
        self.raw_max = max(self.raw_max, r.max().item())
        self.raw_sum += r.mean().item()
        self.raw_count += 1

    def update_loss(self, loss: torch.Tensor, batch_size: int):
        self.loss_sum += loss.detach().item() * batch_size
        self.n_samples += batch_size

    def means(self):
        raw_mean = (self.raw_sum / max(1, self.raw_count))
        loss_mean = (self.loss_sum / max(1, self.n_samples))
        return raw_mean, loss_mean

def _iou_of(model, x, y, device):
    model.eval()
    p = model(x.to(device))
    iou = iou_xywh_torch(decode_xywh(p).float(), y.float())
    return torch.as_tensor(iou).mean().item()

def overfit_one(model, loss_fn, x0, y0, device, steps, lr, mom, wd):
    """
    Overfit a single (image, box) pair. Expect IoU -> 0.9+ quickly.
    Uses a fresh optimizer & no AMP to avoid interference.
    """
    model.train()
    # freeze all but the very last linear layer
    for p in model.parameters(): p.requires_grad = False
    last = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
    for p in last.parameters(): p.requires_grad = True
    opt = optim.SGD(last.parameters(), lr=lr, momentum=mom, weight_decay=wd)

    x0 = x0.to(device)
    y0 = y0.to(device)

    print(f"[overfit-one] start IoU={_iou_of(model, x0, y0, device):.4f}")
    for t in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        raw = model(x0)
        loss  = loss_fn(raw, y0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(last.parameters(), 0.5)
        opt.step()

        if t % 50 == 0 or t <= 5:
            iou = _iou_of(model, x0, y0, device)
            pm  = raw.detach().float().mean(0)
            print(f"[overfit-one] step {t:04d}  loss={loss.item():.5f}  IoU={iou:.4f}  "
                  f"pred≈[{pm[0]:.3f},{pm[1]:.3f},{pm[2]:.3f},{pm[3]:.3f}]")
            pp = decode_xywh(raw.detach().float())[0]
            tt = y0.detach().float()[0]
            print(
                f"[overfit-one] {t:04d}  loss={loss.item():.5f}  IoU={iou:.4f}\n"
                f"    pred=[{pp[0]:.3f},{pp[1]:.3f},{pp[2]:.3f},{pp[3]:.3f}]  "
                f"tgt=[{tt[0]:.3f},{tt[1]:.3f},{tt[2]:.3f},{tt[3]:.3f}]"
            )
    print(f"[overfit-one] final IoU={_iou_of(model, x0, y0, device):.4f}")

@torch.no_grad()
def iou_xywh_torch(pred, gt):
    """
    Compute IoU between predicted and ground-truth boxes in [x_center, y_center, w, h] format.

    Args:
        pred: (N, 4) tensor of predicted boxes, values in [0,1].
        gt:   (N, 4) tensor of ground-truth boxes, values in [0,1].
        eps:  small constant to avoid division by zero / NaNs.
    Returns:
        IoU tensor of shape (N,)
    """
    eps = 1e-7
    # Clamp widths/heights to strictly positive values
    pred = pred.clone()
    gt   = gt.clone()
    pred[:, 2:] = torch.clamp(pred[:, 2:], min=eps)
    gt[:, 2:]   = torch.clamp(gt[:, 2:],   min=eps)

    # Convert from [xc, yc, w, h] → [x1, y1, x2, y2]
    p = torch.stack([
        pred[:, 0] - 0.5 * pred[:, 2],
        pred[:, 1] - 0.5 * pred[:, 3],
        pred[:, 0] + 0.5 * pred[:, 2],
        pred[:, 1] + 0.5 * pred[:, 3],
    ], dim=1)

    g = torch.stack([
        gt[:, 0] - 0.5 * gt[:, 2],
        gt[:, 1] - 0.5 * gt[:, 3],
        gt[:, 0] + 0.5 * gt[:, 2],
        gt[:, 1] + 0.5 * gt[:, 3],
    ], dim=1)

    # Clamp coordinates to [0,1] range to prevent negatives or overflow
    p = torch.clamp(p, 0.0, 1.0)
    g = torch.clamp(g, 0.0, 1.0)

    # Intersection
    inter_mins = torch.maximum(p[:, :2], g[:, :2])
    inter_maxs = torch.minimum(p[:, 2:], g[:, 2:])
    inter_wh   = torch.clamp(inter_maxs - inter_mins, min=0.0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    # Areas (clamped to avoid negative or zero)
    area_p = torch.clamp((p[:, 2] - p[:, 0]), min=0.0) * torch.clamp((p[:, 3] - p[:, 1]), min=0.0)
    area_g = torch.clamp((g[:, 2] - g[:, 0]), min=0.0) * torch.clamp((g[:, 3] - g[:, 1]), min=0.0)

    # IoU computation with eps to avoid div-by-zero
    union_area = area_p + area_g - inter_area
    iou = inter_area / (union_area + eps)

    return iou.clamp(0.0, 1.0).reshape(-1)

def get_samples_from_dataset(dataset):
    """Returns X (N, H, W, 3 in [0,1]) and Y (N,4 normalized)."""
    max_samples = CONFIG["MAX_SAMPLES"]
    image_size = CONFIG["IMAGE_SIZE"]
    X = []
    Y = []
    for i, ex in enumerate(dataset):
        if i >= max_samples: break
        img = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB").resize((image_size, image_size))
        mask_boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
        if not mask_boxes:
            continue
        areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in mask_boxes]
        best = mask_boxes[int(np.argmax(areas))]
        orig = Image.open(BytesIO(ex["image"]["bytes"]))
        W, H = orig.size
        Y.append(box_to_xywh_norm(best, H, W))
        X.append(np.array(img).astype(np.float32)/255.0)
    X = np.stack(X)  # (N, IMAGE_SIZE, IMAGE_SIZE, 3)
    Y = np.stack(Y)  # (N, 4)
    return X, Y

# ---------- Model ----------
def Conv2D(in_ch, out_ch, k, s=1):
    pad = k // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride=s, padding=pad),
        nn.LeakyReLU(0.1, inplace=True),
    )

class YOLOToyRegressor(nn.Module):
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
            dummy = torch.zeros(1, 3, CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"])
            flat = self.feat(dummy).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(4096, 4)  # [xc,yc,w,h] (logits for x,y; logs for w,h)
        )

    def forward(self, x):
        x = self.feat(x)
        return self.fc(x)

# ---------- Training / Eval ----------
def yolo_single_box_loss(raw, targets):
    """
    raw: [...,4] = [x_logit,y_logit,w_log,h_log]
    targets: [...,4] in [x,y,w,h] normalized
    """
    eps = 1e-6
    # decode xy for loss; keep wh in log-space for stability
    x = torch.sigmoid(raw[...,0])
    y = torch.sigmoid(raw[...,1])
    w_log = raw[...,2]
    h_log = raw[...,3]
    tx, ty, tw, th = targets[...,0], targets[...,1], targets[...,2], targets[...,3]
    loss_xy = F.smooth_l1_loss(x, tx) + F.smooth_l1_loss(y, ty)
    loss_wh = F.smooth_l1_loss(w_log, torch.log(tw + eps)) + \
              F.smooth_l1_loss(h_log, torch.log(th + eps))
    return 2.0 * loss_xy + 4.0 * loss_wh

def yolo_v1_lr_schedule(epoch, total_epochs=30, warmup=3):
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

def decode_xywh(raw):
    zx, zy, zw, zh = raw.unbind(-1)
    x = torch.sigmoid(zx)
    y = torch.sigmoid(zy)
    w = torch.sigmoid(zw)
    h = torch.sigmoid(zh)
    # clamp away from exact 0/1 to avoid 0-area and weird IoU divisions
    eps = 1e-7
    w = torch.clamp(w, eps, 1.0 - eps)
    h = torch.clamp(h, eps, 1.0 - eps)
    return torch.stack([x, y, w, h], dim=-1)

def _stat(x):  # tiny helper for prints
    x = x.detach()
    return f"finite={torch.isfinite(x).all().item()} min={x.min().item():.3e} max={x.max().item():.3e} mean={x.mean().item():.3e}"

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    val_stats = EpochStats()
    
    model.eval()
    total_loss = 0.0
    ious = []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            raw = model(xb)
            loss = loss_fn(raw, yb)
            if not torch.isfinite(loss):
              print("[warn] non-finite loss; skipping step")
              opt.zero_grad(set_to_none=True)
              continue
            total_loss += loss.item() * xb.size(0)

            preds = decode_xywh(raw)
            ious.extend(iou_xywh_torch(preds.float(), yb.float()).cpu().tolist())
            val_stats.update_raw(raw)
            val_stats.update_loss(loss, xb.size(0))

    avg_loss = total_loss / len(loader.dataset)
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    return float(avg_loss), float(avg_iou)

def train_one_epoch(model, loader, device, opt, scaler, loss_fn, max_norm=None):
    # DEBUG 001
    USE_AMP = False        # make sure this is False
    model.float()          # force params to fp32
    eps = 1e-7

    stats = EpochStats()

    # inside train_one_epoch(), before the loop:
    print("[dtype] model head W:", next(model.parameters()).dtype)
    # END DEBUG 001

    model.train()
    running = 0.0
    n = 0

    # locate the 4-dim head once
    last = None
    for m in model.modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == 4:
            last = m
    assert last is not None, "Could not find final Linear(out_features=4) head."

    did_probe = False

    first_bad_step = None

    for step, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # DEBUG 001
        xb = xb.to(device).float()
        yb = yb.to(device).float()
        # END DEBUG 001

        opt.zero_grad(set_to_none=True)

        # forward + loss under autocast
        with torch.amp.autocast(device_type="cuda", enabled=USE_AMP):
            raw = model(xb)
            preds = decode_xywh(raw)
            preds = preds.clamp(eps, 1.0 - eps)
            xc, yc, w, h = preds[...,0], preds[...,1], preds[...,2], preds[...,3]

            loss = loss_fn(raw, yb)
            if not torch.isfinite(loss):
              print("[warn] non-finite loss; skipping step")
              opt.zero_grad(set_to_none=True)
              continue

        # DEBUG 001
        if DEBUG and not did_probe:
            print("[dtype] xb:", xb.dtype, "raw:", raw.dtype, "preds:", preds.dtype, "yb:", yb.dtype)
        # END DEBUG 001

        stats.update_raw(raw)
        stats.update_loss(loss, xb.size(0))

        # (optional) peek at decoded preds for the first batch
        if DEBUG and not did_probe:
            with torch.no_grad():
                preds = decode_xywh(raw)
                preds = preds.clamp(eps, 1.0 - eps)
                xc, yc, w, h = preds[...,0], preds[...,1], preds[...,2], preds[...,3]

                print(f"[dbg] preds [xc,yc,w,h] batch head: {preds[:2]}")

        # snapshot params before backward (first batch only)
        if DEBUG and not did_probe:
            w_before = last.weight.detach().clone()
            b_before = last.bias.detach().clone()

        # print only for step==1 (sanity) or when something goes non-finite
        should_print = (DEBUG and step == 1)

        non_finite = (
            not torch.isfinite(raw).all()
            or not torch.isfinite(preds).all()
            or not torch.isfinite(loss)
        )

        if non_finite:
            should_print = True
            first_bad_step = first_bad_step or step  # remember the earliest offender

        if should_print:
            print("TRIPWIRE")
            print(f"[dbg] step={step}")
            print("[dbg] xb:", _stat(xb))
            print("[dbg] yb:", _stat(yb))
            print("[dbg] raw:", _stat(raw))
            print("[dbg] preds:", _stat(preds))
            print("[dbg] loss:", loss.detach().item())

        # tripwire: skip update on bad batch (or raise to halt)
        if non_finite:
            print(f"[stop] non-finite detected at step={step} — skipping optimizer step")
            continue

        if (not torch.isfinite(raw).all()
            or not torch.isfinite(preds).all()
            or not torch.isfinite(loss)):
            print("[stop] non-finite detected — skipping this batch")
            opt.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            # --- AMP path ---
            scaler.scale(loss).backward()
            if max_norm is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
        else:
            # --- normal FP32 path ---
            loss.backward()
            if max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        # first-batch grad norms (unscaled)
        if DEBUG and not did_probe:
            gw = (last.weight.grad.detach().norm().item()
                  if last.weight.grad is not None else float("nan"))
            gb = (last.bias.grad.detach().norm().item()
                  if last.bias.grad is not None else float("nan"))
            print(f"[dbg/train] grad||W||={gw:.4e}  grad||b||={gb:.4e}")

        # did params move? (first batch only)
        if DEBUG and not did_probe:
            dW = (last.weight.detach() - w_before).norm().item()
            dB = (last.bias.detach()   - b_before).norm().item()
            print(f"[dbg/train] Δ||W||={dW:.4e}  Δ||b||={dB:.4e}  loss={loss.item():.6f}")
            did_probe = True

        running += loss.item() * xb.size(0)
        n += xb.size(0)

    raw_mean, loss_mean = stats.means()
    return loss_mean, {"raw_min": stats.raw_min, "raw_max": stats.raw_max, "raw_mean": raw_mean}

def main():
    # --- Load data (parquet paths from CONFIG) ---
    ds = load_dataset(
        "parquet",
        data_files={
            "train": CONFIG["TRAIN_PATH"],
            "val":   CONFIG["VAL_PATH"],
        }
    )
    train = ds["train"]; val = ds["val"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Sample N from train, then split
    N = min(256, len(train))  # keep your existing cap here; could also add to CONFIG if desired
    sample_list = [train[i] for i in range(N)]

    X, Y = get_samples_from_dataset(sample_list)
    idx = int(CONFIG["TRAIN_SPLIT"] * len(X))
    X_tr, Y_tr = X[:idx], Y[:idx]
    X_va, Y_va = X[idx:], Y[idx:]

    # Convert to NCHW torch tensors
    X_tr_t = torch.from_numpy(np.transpose(X_tr, (0,3,1,2))).contiguous()
    Y_tr_t = torch.from_numpy(Y_tr).contiguous()
    X_va_t = torch.from_numpy(np.transpose(X_va, (0,3,1,2))).contiguous()
    Y_va_t = torch.from_numpy(Y_va).contiguous()

    if OVERFIT_ONE:
        # pick one training example
        x0 = X_tr_t[0:1]
        y0 = Y_tr_t[0:1]
        # fresh model + init
        model = YOLOToyRegressor().to(device)
        with torch.no_grad():
            last = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
            # x,y logits ~ 0.0 => sigmoid ~ 0.5; w,h logs ~ log(0.2)
            bias = torch.tensor(CONFIG["OVERFIT_BIAS_LOG"], device=last.bias.device)
            last.bias.copy_(bias)
            torch.nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        # run the harness and exit
        overfit_one(
            model,
            yolo_single_box_loss,
            x0, y0, device,
            steps=CONFIG["OVERFIT_STEPS"],
            lr=CONFIG["OVERFIT_LR"],
            mom=CONFIG["MOMENTUM"],
            wd=CONFIG["OVERFIT_WD"],
        )
        return

    # --- Sanity: do targets have variety? Is the model predicting their mean?
    if DEBUG:
        print("Y_tr mean/std:", Y_tr_t.float().mean(0), Y_tr_t.float().std(0))
        print("Y_va mean/std:", Y_va_t.float().mean(0), Y_va_t.float().std(0))

    train_loader = DataLoader(
        TensorDataset(X_tr_t, Y_tr_t),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=CONFIG["PIN_MEMORY"],
    )
    val_loader   = DataLoader(
        TensorDataset(X_va_t, Y_va_t),
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=CONFIG["PIN_MEMORY"],
    )

    model = YOLOToyRegressor().to(device)
    with torch.no_grad():
        last = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
        # For main training, start near center with a modest box
        last.bias.copy_(torch.tensor(CONFIG["INIT_BIAS"], device=last.bias.device))

    summary(model, input_size=(3, CONFIG["IMAGE_SIZE"], CONFIG["IMAGE_SIZE"]), device=str(device))

    loss_fn = yolo_single_box_loss

    # build param groups: decay everything *except* 1D params (BN/LayerNorm/scale) and biases
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    opt = optim.SGD(
        [
            {"params": decay, "weight_decay": CONFIG["WEIGHT_DECAY"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=CONFIG["LR"],
        momentum=CONFIG["MOMENTUM"],
        nesterov=True,
    )

    plateau = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=CONFIG["LR_FACTOR"], patience=CONFIG["LR_PATIENCE"], min_lr=CONFIG["LR_MIN"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available()) if USE_AMP else None

    TOTAL_EPOCHS = CONFIG["TOTAL_EPOCHS"]
    early = EarlyStoppingMax(patience=CONFIG["EARLY_PATIENCE"], min_delta=CONFIG["EARLY_MIN_DELTA"])
    best_val_iou = -1.0
    best_path = CONFIG["BEST_MODEL_PATH"]

    # ---- Simple history recorder ----
    history = {"epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(TOTAL_EPOCHS):
        lr = opt.param_groups[0]["lr"]

        train_loss, train_stats = train_one_epoch(
            model, train_loader, device, opt, scaler, loss_fn, max_norm=CONFIG["MAX_GRAD_NORM"]
        )

        val_loss, val_iou = evaluate(model, val_loader, device, loss_fn)
        plateau.step(val_loss)

        print(
            f"Epoch {epoch:03d} | lr {lr:.6f} | "
            f"train {train_loss:.6f} "
            f"(raw min {train_stats['raw_min']:.2f}, max {train_stats['raw_max']:.2f}, mean {train_stats['raw_mean']:.2f}) | "
            f"val {val_loss:.6f} "
        )

        # record history
        history["epoch"].append(epoch + 1)
        history["lr"].append(lr)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        log_quality(epoch, lr, train_loss, val_loss, val_iou)

        improved, stop = early.step(val_iou)
        if improved:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved new best (val_iou={val_iou:.4f}) to {best_path}")
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
            raw = model(x)[0].detach().cpu()
            pred = decode_xywh(raw.unsqueeze(0))[0].cpu().float().numpy()
            P.append(pred); T.append(Y_va[i])
    P, T = np.stack(P), np.stack(T)   # (N,4)

    print("pred mean:", P.mean(0), "pred std:", P.std(0))
    print("tgt  mean:", T.mean(0), "tgt  std:", T.std(0))

    val_ds = val_loader.dataset

    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(val_ds))):
            xb, yb = val_ds[i]                 # xb: (3,H,W) in [0,1], yb: (4,)
            H, W = xb.shape[-2:]
            raw  = model(xb.unsqueeze(0).to(device))[0].cpu()
            pred = decode_xywh(raw.unsqueeze(0))[0].cpu().float()      # (4,) decoded [xc,yc,w,h]
            # [xc,yc,w,h]
            if DEBUG:
                print(f"[dbg] sample {i} raw pred [xc,yc,w,h]: {pred}")

            iou = iou_xywh_torch(pred.unsqueeze(0), yb.unsqueeze(0))

            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(xb.permute(1,2,0).cpu().numpy())  # direct show, no denorm
            draw_xywh(ax, yb,   W, H, 'g', label='GT')
            draw_xywh(ax, pred, W, H, 'r', label='Pred')
            ax.set_title(f"idx={i} | IoU: {float(torch.as_tensor(iou).mean().item()):.3f}")
            ax.axis('off')
            ax.legend(loc='upper right')
            plt.show()

    last = None
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 4:
            last = m
    print("last weight std:", last.weight.data.std().item())
    print("last bias:", last.bias.data.cpu().numpy())

    acts = {}
    def hook(_, __, out): acts['pre_act'] = out.detach().cpu()

    h = last.register_forward_hook(hook)

    with torch.no_grad():
        for k in range(3):
            xb,_ = val_loader.dataset[k]
            xb = xb.unsqueeze(0).to(device)
            _ = model(xb)
            print(f"[{k}] pre_act mean={acts['pre_act'].mean().item():.4f} "
                  f"std={acts['pre_act'].std().item():.4f}  vals={acts['pre_act'].numpy()}")
    h.remove()

if __name__ == "__main__":
    main()

