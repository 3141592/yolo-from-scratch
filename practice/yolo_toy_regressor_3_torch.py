# yolo_toy_regressor_2_torch_fixed.py
# Changes from reorganized version:
# 1) Decode predictions before NumPy IoU summary and visualization (aligns boxes).
# 2) Remove CosineAnnealingLR; keep manual YOLO-style LR schedule only (no scheduler.step()).
# 3) Guard autocast for CPU safety (enabled only if CUDA is available).
# 4) Ensure PRIOR_W/PRIOR_H are set after estimating priors from train_loader.
# Everything else kept as-is.

import os
import math
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

from datasets import load_dataset
from skimage.measure import label, regionprops

# ------------------------------
# Global flags / constants
# ------------------------------
DEBUG = True

# ------------------------------
# Image/Geometry helpers
# ------------------------------
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

# ------------------------------
# IoU utilities (numpy & torch)
# ------------------------------
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

# ------------------------------
# CIoU (loss + helpers)
# ------------------------------
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
    p = xywh_to_xyxy(pred_xywh); g = xywh_to_xyxy(gt_xywh)
    iou = box_iou_xyxy(p, g)
    pc = (p[..., :2] + p[..., 2:]) / 2
    gc = (g[..., :2] + g[..., 2:]) / 2
    center_dist = ((pc - gc) ** 2).sum(-1)
    enc_tl = torch.minimum(p[..., :2], g[..., :2])
    enc_br = torch.maximum(p[..., 2:], g[..., 2:])
    c2 = ((enc_br - enc_tl) ** 2).sum(-1) + 1e-9
    pw = (p[..., 2]-p[..., 0]).clamp(min=1e-9)
    ph = (p[..., 3]-p[..., 1]).clamp(min=1e-9)
    gw = (g[..., 2]-g[..., 0]).clamp(min=1e-9)
    gh = (g[..., 3]-g[..., 1]).clamp(min=1e-9)
    v = (4/torch.pi**2) * (torch.atan(gw/gh) - torch.atan(pw/ph))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-9)
    ciou = iou - (center_dist / c2) - alpha * v
    return (1 - ciou).clamp(min=0), iou, ciou

# ------------------------------
# Data utilities
# ------------------------------
def get_samples_from_dataset(dataset, max_samples=500):
    X = []
    Y = []
    for i, ex in enumerate(dataset):
        if i >= max_samples: break
        img = Image.open(BytesIO(ex["image"]["bytes"]))\
              .convert("RGB").resize((448,448))
        mask_boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
        if not mask_boxes:
            continue
        areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in mask_boxes]
        best = mask_boxes[int(np.argmax(areas))]
        orig = Image.open(BytesIO(ex["image"]["bytes"]))
        W, H = orig.size
        Y.append(box_to_xywh_norm(best, H, W))
        X.append(np.array(img).astype(np.float32)/255.0)
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

# ------------------------------
# Model definition
# ------------------------------
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
            Conv2D(3, 192, 7, s=2),
            pool(),
            Conv2D(192, 256, 3),
            pool(),
            Conv2D(256, 128, 1),
            Conv2D(128, 256, 3),
            Conv2D(256, 256, 1),
            Conv2D(256, 512, 3),
            pool(),
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
            nn.MaxPool2d(2,2),
            Conv2D(1024, 512, 1),
            Conv2D(512, 1024, 3),
            Conv2D(1024, 512, 1),
            Conv2D(512, 1024, 3),
            Conv2D(1024, 1024, 3),
            Conv2D(1024, 1024, 3, s=2),
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
            nn.Dropout(0.5),
            nn.Linear(4096, 4)
        )
    def forward(self, x):
        x = self.feat(x)
        x = self.fc(x)
        return x

# ------------------------------
# LR schedule, early stopping, grads
# ------------------------------
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

def global_grad_norm(model):
    g2 = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g2 += p.grad.detach().pow(2).sum().item()
    return (g2 ** 0.5)

# ------------------------------
# Priors / head init / decode
# ------------------------------
def logit(p):
    return math.log(p/(1-p))

def init_head_biases(model, head_attr="fc", prior_xy=(0.5,0.5), prior_wh=(0.25,0.35)):
    head = getattr(model, head_attr)
    last = None
    for m in reversed(list(head.modules())):
        if isinstance(m, nn.Linear):
            last = m; break
    assert last is not None, "Couldn't find final Linear layer in head"
    nn.init.xavier_uniform_(last.weight)
    with torch.no_grad():
        last.bias.zero_()
        last.bias[0] = logit(prior_xy[0])
        last.bias[1] = logit(prior_xy[1])
        last.bias[2] = logit(prior_wh[0])
        last.bias[3] = logit(prior_wh[1])

def estimate_size_priors(loader):
    ws, hs = [], []
    for _, y in loader:
        ws.append(y[...,2].reshape(-1))
        hs.append(y[...,3].reshape(-1))
    w = torch.cat(ws).median().item()
    h = torch.cat(hs).median().item()
    eps = 1e-3
    return (min(max(w,eps),1-eps), min(max(h,eps),1-eps))

PRIOR_W, PRIOR_H = 0.25, 0.35

def decode_head(t):
    tx, ty, tw, th = t.unbind(-1)
    xy = torch.sigmoid(torch.stack([tx,ty], dim=-1))
    wh = torch.exp(torch.stack([tw,th], dim=-1)) * torch.tensor([PRIOR_W, PRIOR_H], device=t.device)
    wh = wh.clamp(min=1e-4, max=1.0)
    return torch.cat([xy, wh], dim=-1)

# ------------------------------
# Train / Eval loops
# ------------------------------
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
            loss, iou_mean = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)
            total_iou  += iou_mean.item() * xb.size(0)
            n += xb.size(0)
    return total_loss / n, total_iou / n

def train_one_epoch(model, loader, device, opt, scaler, loss_fn, max_norm=1.0, head_attr="fc"):
    model.train()
    if DEBUG:
        opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        miss = [n for n,p in model.named_parameters() if p.requires_grad and id(p) not in opt_ids]
        assert not miss, f"Params missing from optimizer: {miss[:5]}"
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                assert m.training, "BatchNorm stuck in eval() during training"
    running_loss = 0.0
    running_iou  = 0.0
    n_seen = 0
    last_linear = None
    if DEBUG:
        head = getattr(model, head_attr, model)
        for m in reversed(list(head.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m; break
    use_cuda = torch.cuda.is_available()
    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        bs = xb.size(0)
        opt.zero_grad(set_to_none=True)
        if DEBUG and step < 2:
            print(f"[dbg] xb mean/std: {xb.mean().item():.4f}/{xb.std().item():.4f} min/max: {xb.min().item():.3f}/{xb.max().item():.3f}")
            with torch.no_grad():
                t_probe = model(xb)
                print("[dbg] pre-train logits std:", [float(s) for s in t_probe.std(dim=0)])
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
            pred_t = model(xb)
            loss, iou_mean = loss_fn(pred_t, yb)
        assert loss.requires_grad, "Loss is detached; check loss_fn."
        scaler.scale(loss).backward()
        if DEBUG and step < 2:
            gnorm = global_grad_norm(model)
            print(f"[dbg] global grad norm: {gnorm:.4f}")
            if last_linear is not None and last_linear.weight.grad is not None:
                print(f"[dbg] last-linear grad mean/std: {last_linear.weight.grad.mean().item():.4e}/{last_linear.weight.grad.std().item():.4e}")
            else:
                print("[dbg] last-linear grad missing (None)")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(opt)
        scaler.update()
        if DEBUG and step < 2:
            with torch.no_grad():
                t_after = model(xb)
                print("[dbg] post-step logits std:", [float(s) for s in t_after.std(dim=0)])
        running_loss += loss.detach().item() * bs
        running_iou  += iou_mean.detach().item() * bs
        n_seen += bs
    return running_loss / max(n_seen, 1), running_iou / max(n_seen, 1)

# ------------------------------
# Main
# ------------------------------
def main():
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

    X_tr_t = torch.from_numpy(np.transpose(X_tr, (0,3,1,2))).contiguous()
    Y_tr_t = torch.from_numpy(Y_tr).contiguous()
    X_va_t = torch.from_numpy(np.transpose(X_va, (0,3,1,2))).contiguous()
    Y_va_t = torch.from_numpy(Y_va).contiguous()

    train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, Y_va_t), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLOToyRegressor().to(device)
    summary(model, input_size=(3, 448, 448), device=str(device))

    def loss_fn(pred_t, gt_xywh):
        pred_xywh = decode_head(pred_t)
        loss_ciou, iou, _ = ciou_loss(pred_xywh, gt_xywh)
        return loss_ciou.mean(), iou.mean()

    # ---- Estimate priors and initialize head biases ----
    w_med, h_med = estimate_size_priors(train_loader)
    init_head_biases(model, head_attr="fc", prior_xy=(0.5,0.5), prior_wh=(w_med, h_med))
    global PRIOR_W, PRIOR_H
    PRIOR_W, PRIOR_H = w_med, h_med

    opt = optim.SGD(model.parameters(), lr=2e-2, momentum=0.9, weight_decay=5e-4)
    TOTAL_EPOCHS = 30
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    early = EarlyStoppingMax(patience=12, min_delta=0.0)

    best_val_iou = -1.0
    best_path = "best_val_iou.pt"
    history = {"epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(TOTAL_EPOCHS):
        lr = yolo_v1_lr_schedule(epoch, total_epochs=TOTAL_EPOCHS, warmup=3)
        for pg in opt.param_groups:
            pg["lr"] = lr

        train_loss, train_iou = train_one_epoch(model, train_loader, device, opt, scaler, loss_fn, max_norm=10)
        val_loss, val_iou = evaluate(model, val_loader, device, loss_fn)

        print(f"Epoch {epoch+1:03d} | lr {lr:.6f} | train {train_loss:.4f} | val {val_loss:.4f} | val_iou {val_iou:.4f}")

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

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best weights from {best_path} (best val_iou={best_val_iou:.4f})")

    # ---- Decode before NumPy IoU summary ----
    model.eval()
    decoded_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            t = model(xb)
            xywh = decode_head(t).cpu().numpy()
            decoded_preds.append(xywh)
    preds_decoded = np.concatenate(decoded_preds, axis=0)

    ious = [iou_xywh_np(p, g) for p, g in zip(preds_decoded, Y_va)]
    print("val IoU mean (decoded):", float(np.mean(ious)))
    print("val IoU median (decoded):", float(np.median(ious)))

    # ---- Visualization with decoded predictions ----
    for i in range(min(5, len(X_va))):
        img = (X_va[i]*255).astype(np.uint8)
        pred = preds_decoded[i]; gt = Y_va[i]
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
        plt.title(f"IoU (decoded): {iou_xywh_np(pred,gt):.3f}")
        plt.axis('off'); plt.show()

if __name__ == "__main__":
    main()

