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
import matplotlib.patches as patches

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import torch.nn.functional as F

from datasets import load_dataset
from skimage.measure import label, regionprops

# --- debug toggles ---
OVERFIT_ONE = True   # set True to run the 1-image sanity check and exit
DEBUG = True

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

@torch.no_grad()
def _iou_of(model, x, y, device):
    model.eval()
    p = model(x.to(device))
    iou = iou_xywh_torch(decode_xywh(p).float(), y.float())
    return torch.as_tensor(iou).mean().item()    

def overfit_one(model, loss_fn, x0, y0, device, steps=1000, lr=1e-3, wd=0.0):
    """
    Overfit a single (image, box) pair. Expect IoU -> 0.9+ quickly.
    Uses a fresh optimizer & no AMP to avoid interference.
    """
    model.train()
    # freeze all but the very last linear layer
    for p in model.parameters(): p.requires_grad = False
    last = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
    for p in last.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(last.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.99))

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
    union_area = area_p + area_g - inter_area
    iou = (inter_area / union_area).clamp(0.0, 1.0)
    return iou.reshape(-1)   # ensure it's always a 1D tensor

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
            #nn.Dropout(0.5),
            nn.Linear(4096, 4)  # [xc,yc,w,h]
        )

    def forward(self, x):
        x = self.feat(x)
        # return raw logits: [x_logit, y_logit, w_log, h_log]
        return self.fc(x)

# ---------- Training / Eval ----------
def yolo_single_box_loss(raw, targets):
    """
    raw: [...,4] = [x_logit,y_logit,w_log,h_log]
    targets: [...,4] in [x,y,w,h] normalized
    """
    eps = 1e-6
    # decode xy for loss; keep wh in log-space for loss stability
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
    # raw: [...,4] -> normalized xy and positive wh
    x_logit, y_logit, w_log, h_log = raw[...,0], raw[...,1], raw[...,2], raw[...,3]
    x = torch.sigmoid(x_logit)
    y = torch.sigmoid(y_logit)
    w = torch.exp(w_log).clamp_min(1e-6)  # positive
    h = torch.exp(h_log).clamp_min(1e-6)
    return torch.stack([x, y, w, h], dim=-1)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        preds_raw = model(xb)
        loss = loss_fn(preds_raw, yb)
        preds = decode_xywh(preds_raw)
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
        with torch.amp.autocast(device_type="cuda", enabled=False):
            preds_raw = model(xb)
            if DEBUG:
                # [xc,yc,w,h]
                preds = decode_xywh(preds_raw)
                print(f"[xc,yc,w,h]: {preds}>>>>>>>>>>>>>>>>>>>>>>>>>>>")

            loss = loss_fn(preds_raw, yb)
        scaler.scale(loss).backward()

        # debug grads
        with torch.no_grad():
            total_norm = torch.sqrt(sum(p.grad.detach().float().pow(2).sum() for p in model.parameters() if p.grad is not None))
            head = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
            gmean = head.weight.grad.detach().float().mean().item() if head.weight.grad is not None else float('nan')
            gstd  = head.weight.grad.detach().float().std().item()  if head.weight.grad is not None else float('nan')
        if DEBUG: print(f"[dbg] grad_norm={total_norm.item():.2e} head_grad mean/std={gmean:.2e}/{gstd:.2e}")

        if max_norm is not None:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Clip gradients (L2 norm capped at 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #sample_list = [train[i] for i in range(128)]
    N = min(256, len(train))
    sample_list = [train[i] for i in range(N)]

    X, Y = get_samples_from_dataset(sample_list, max_samples=128)
    idx = int(0.8*len(X))
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
        # fresh model + init (reuse your existing init code)
        model = YOLOToyRegressor().to(device)
        with torch.no_grad():
            last = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
            torch.nn.init.normal_(last.weight, mean=0.0, std=1e-3)
            # x,y logits ~ 0.0 => sigmoid ~ 0.5; w,h logs ~ log(0.2)
            last.bias.copy_(torch.tensor([0.0, 0.0, -1.609, -1.609], device=last.bias.device))
        # run the harness and exit
        overfit_one(model, yolo_single_box_loss, x0, y0, device, steps=600, lr=1e-2, wd=0.0)
        return

    # --- Sanity: do targets have variety? Is the model predicting their mean?
    if DEBUG:
        print("Y_tr mean/std:", Y_tr_t.float().mean(0), Y_tr_t.float().std(0))
        print("Y_va mean/std:", Y_va_t.float().mean(0), Y_va_t.float().std(0))

    train_loader = DataLoader(TensorDataset(X_tr_t, Y_tr_t), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_va_t, Y_va_t), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = YOLOToyRegressor().to(device)
    with torch.no_grad():
        last = [m for m in model.modules() if isinstance(m, nn.Linear) and m.out_features == 4][0]
        # Start near center with a modest box (20% of width/height)
        last.bias.copy_(torch.tensor([0.5, 0.5, 0.2, 0.2], device=last.bias.device))
    summary(model, input_size=(3, 448, 448), device=str(device))

    loss_fn = yolo_single_box_loss

    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4, betas=(0.9, 0.999))

    plateau = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    TOTAL_EPOCHS = 5
    early = EarlyStoppingMax(patience=12, min_delta=0.0)
    best_val_iou = -1.0
    best_path = "best_val_iou.pt"

    # ---- Simple history recorder ----
    history = {"epoch": [], "lr": [], "train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(TOTAL_EPOCHS):
        # epoch-wise LR schedule
        #lr = yolo_v1_lr_schedule(epoch, total_epochs=TOTAL_EPOCHS, warmup=3)
        #for pg in opt.param_groups:
        #    pg["lr"] = lr
        lr = opt.param_groups[0]["lr"]

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

    P, T = [], []
    with torch.no_grad():
        for i in range(len(X_va)):
            x = torch.from_numpy(X_va[i]).permute(2,0,1).unsqueeze(0).to(device)
            p = model(x)[0].detach().cpu().float().numpy()
            P.append(p); T.append(Y_va[i])
    P, T = np.stack(P), np.stack(T)   # (N,4)

    print("pred mean:", P.mean(0), "pred std:", P.std(0))
    print("tgt  mean:", T.mean(0), "tgt  std:", T.std(0))

    val_ds = val_loader.dataset

    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(val_ds))):
            xb, yb = val_ds[i]                 # xb: (3,H,W) in [0,1], yb: (4,)
            H, W = xb.shape[-2:]
            pred = model(xb.unsqueeze(0).to(device))[0].cpu().float()  # (4,)
            # [xc,yc,w,h]
            print(f"[xc,yc,w,h]: {i} {pred}")

            #iou = iou_xywh_torch(pred, yb)
            iou = iou_xywh_torch(pred.unsqueeze(0), yb.unsqueeze(0))

            fig, ax = plt.subplots(figsize=(6,6))
            ax.imshow(xb.permute(1,2,0).cpu().numpy())  # direct show, no denorm
            draw_xywh(ax, yb,   W, H, 'g', label='GT')
            draw_xywh(ax, pred, W, H, 'r', label='Pred')
            ax.set_title(f"idx={i} | IoU: {iou:.3f}")
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

