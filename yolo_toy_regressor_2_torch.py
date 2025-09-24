# yolo_toy_regressor_2_torch.py
# PyTorch port (with history recorder) of your minimal single-box regressor.
# - Uses SmoothL1Loss (Huber), AMP, ReduceLROnPlateau
# - EarlyStopping on val_iou (maximize)
# - Records per-epoch: lr, train_loss, val_loss, val_iou and plots them.
#
# Note: PyTorch expects NCHW; images are converted from NHWC.
import sys
import os
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from datasets import load_dataset
from skimage.measure import label, regionprops
from torchsummary import summary

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

import torch, torch.nn as nn

def xywh_to_xyxy(b):
    x,y,w,h = b.unbind(-1)
    x1 = x - w/2; y1 = y - h/2
    x2 = x + w/2; y2 = y + h/2
    return torch.stack([x1,y1,x2,y2], dim=-1)

def box_iou_xyxy(a, b):
    tl = torch.maximum(a[..., :2], b[..., :2])
    br = torch.minimum(a[..., 2:], b[..., 2:])
    inter = (br - tl).clamp(min=0).prod(-1)
    area_a = (a[..., 2]-a[..., 0]).clamp(min=0) * (a[..., 3]-a[..., 1]).clamp(min=0)
    area_b = (b[..., 2]-b[..., 0]).clamp(min=0) * (b[..., 3]-b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def ciou_loss(pred_xywh, gt_xywh):
    """
    pred_xywh, gt_xywh: (..., 4) with xy in [0,1], w,h > 0 (normalized).
    Returns:
      loss: 1 - CIoU
      iou: IoU
      ciou: CIoU
    """
    p = xywh_to_xyxy(pred_xywh); g = xywh_to_xyxy(gt_xywh)
    iou = box_iou_xyxy(p, g)

    # center distance
    pc = (p[..., :2] + p[..., 2:]) / 2
    gc = (g[..., :2] + g[..., 2:]) / 2
    center_dist = ((pc - gc) ** 2).sum(-1)

    # enclosing diagonal
    enc_tl = torch.minimum(p[..., :2], g[..., :2])
    enc_br = torch.maximum(p[..., 2:], g[..., 2:])
    c2 = ((enc_br - enc_tl) ** 2).sum(-1) + 1e-9

    # aspect term
    pw = (p[..., 2]-p[..., 0]).clamp(min=1e-9)
    ph = (p[..., 3]-p[..., 1]).clamp(min=1e-9)
    gw = (g[..., 2]-g[..., 0]).clamp(min=1e-9)
    gh = (g[..., 3]-g[..., 1]).clamp(min=1e-9)
    v = (4/torch.pi**2) * (torch.atan(gw/gh) - torch.atan(pw/ph))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-9)

    ciou = iou - (center_dist / c2) - alpha * v
    return (1 - ciou).clamp(min=0), iou, ciou

def decode_head(t):
    # t: (..., 4) unconstrained
    tx, ty, tw, th = t.unbind(-1)
    xy = torch.sigmoid(torch.stack([tx, ty], dim=-1))         # → [0,1]
    wh = torch.exp(torch.stack([tw, th], dim=-1))             # > 0
    # Optional: scale to a max fraction of the image to avoid gigantic boxes early on
    wh = wh.clamp(max=1.0)
    xywh = torch.cat([xy, wh], dim=-1)
    # Final clamp to image-normalized bounds:
    xywh = torch.stack([
        xywh[...,0].clamp(0,1),
        xywh[...,1].clamp(0,1),
        xywh[...,2].clamp(1e-6,1.0),
        xywh[...,3].clamp(1e-6,1.0),
    ], dim=-1)
    return xywh

def get_samples_from_dataset(dataset, max_samples=500):
    X = []
    Y = []
    for i, ex in enumerate(dataset):
        if i >= max_samples: break
        img = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB").resize((448,448))
        mask_boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
        if not mask_boxes:
            continue
        areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in mask_boxes]
        best = mask_boxes[int(np.argmax(areas))]
        orig = Image.open(BytesIO(ex["image"]["bytes"]))
        W, H = orig.size
        Y.append(box_to_xywh_norm(best, H, W))
        X.append(np.array(img).astype(np.float32)/255.0)
    X = np.stack(X)  # (N, 448, 448, 3)
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
            dummy = torch.zeros(1, 3, 448, 448)
            flat = self.feat(dummy).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            #nn.Linear(flat, 4096),
            #nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4)  # [xc,yc,w,h]
        )

    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return x

# ---------- Training / Eval ----------
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

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            preds = model(xb)
            loss, iou_mean = loss_fn(preds, yb)   # unpack tuple

            total_loss += loss.item() * xb.size(0)
            total_iou  += iou_mean.item() * xb.size(0)
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
            #loss = loss_fn(preds, yb)
            loss, iou_mean = loss_fn(preds, yb)   # ⟵ unpack the tuple
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
            "train": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet",
            "val":   "/home/roy/src/data/voc2012/val-00000-of-00001.parquet",
        }
    )
    train = ds["train"]; val = ds["val"]

    sample_list = [train[i] for i in range(500)]
    X, Y = get_samples_from_dataset(sample_list, max_samples=500)
    idx = int(0.8*len(X))
    X_tr, Y_tr = X[:idx], Y[:idx]
    X_va, Y_va = X[idx:], Y[idx:]

    # Convert to NCHW torch tensors
    X_tr_t = torch.from_numpy(np.transpose(X_tr, (0,3,1,2))).contiguous()
    Y_tr_t = torch.from_numpy(Y_tr).contiguous()
    X_va_t = torch.from_numpy(np.transpose(X_va, (0,3,1,2))).contiguous()
    Y_va_t = torch.from_numpy(Y_va).contiguous()

    train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, Y_va_t), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOToyRegressor().to(device)
    summary(model, input_size=(3, 448, 448), device=str(device))

    #loss_fn = nn.SmoothL1Loss(reduction="mean")
    # New: use CIoU as the primary loss
    def loss_fn(pred_t, gt_xywh):
        pred_xywh = decode_head(pred_t)
        loss_ciou, iou, _ = ciou_loss(pred_xywh, gt_xywh)
        loss = loss_ciou.mean()
        iou_mean = iou.mean()
        return loss, iou_mean

    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    TOTAL_EPOCHS = 30
    early = EarlyStoppingMax(patience=12, min_delta=0.0)
    best_val_iou = -1.0
    best_path = "best_val_iou.pt"

    # ---- Simple history recorder ----
    history = {"epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(TOTAL_EPOCHS):
        # epoch-wise LR schedule
        lr = yolo_v1_lr_schedule(epoch, total_epochs=TOTAL_EPOCHS, warmup=3)
        for pg in opt.param_groups:
            pg["lr"] = lr

        train_loss = train_one_epoch(model, train_loader, device, opt, scaler, loss_fn)
        val_loss, val_iou = evaluate(model, val_loader, device, loss_fn)
        plateau.step(val_loss)

        print(f"Epoch {epoch+1:03d} | lr {lr:.6f} | train {train_loss:.4f} | val {val_loss:.4f} | val_iou {val_iou:.4f}")

        # record history
        history["epoch"].append(epoch + 1)
        history["lr"].append(lr)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

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
    with torch.no_grad():
        preds = []
        for xb, _ in val_loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy()
            preds.append(p)
    preds = np.concatenate(preds, axis=0)
    ious = [iou_xywh_np(p, g) for p, g in zip(preds, Y_va)]
    print("val IoU mean:", float(np.mean(ious)))
    print("val IoU median:", float(np.median(ious)))

    # ---- Plot history (loss & IoU) ----
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.plot(history["epoch"], history["val_iou"], label="val_iou")
    plt.xlabel("epoch"); plt.legend(); plt.title("Loss & IoU (PyTorch)"); plt.show()

    # visualize a few predictions
    for i in range(min(5, len(X_va))):
        img = (X_va[i]*255).astype(np.uint8)
        pred = preds[i]; gt = Y_va[i]
        plt.figure()
        plt.imshow(img)
        H, W, _ = img.shape
        def draw_box(xywh, color):
            xc,yc,w,h = xywh
            x_min = int((xc - 0.5*w)*W); x_max = int((xc+0.5*w)*W)
            y_min = int((yc - 0.5*h)*H); y_max = int((yc+0.5*h)*H)
            plt.gca().add_patch(plt.Rectangle((x_min,y_min), x_max-x_min, y_max-y_min,
                                              fill=False, edgecolor=color, linewidth=2))
        draw_box(gt,  'g')
        draw_box(pred,'r')
        plt.title(f"IoU: {iou_xywh_np(pred,gt):.3f}")
        plt.axis('off'); plt.show()

if __name__ == "__main__":
    main()

