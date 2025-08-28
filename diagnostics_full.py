# diagnostics_full.py
import os
from io import BytesIO
from datasets import load_dataset
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ---------------- USER CONFIG ----------------
PARQUET_TRAIN = "/home/roy/src/data/voc2012/train-00000-of-00001.parquet"
PARQUET_VAL   = "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"
RESIZED_SIZE = (224, 224)   # (W, H) - change if you trained at a different size
N_train = 400               # how many training samples to load (can reduce)
N_val = 200                 # how many val samples to load
WEIGHTS_FILE = "best_model.h5"  # model weights file (if present)
# ---------------------------------------------

def box_to_xywh_norm(box, H, W):
    x_min, y_min, x_max, y_max = box
    w_px = (x_max - x_min + 1)
    h_px = (y_max - y_min + 1)
    xc_px = x_min + (w_px - 1) / 2.0
    yc_px = y_min + (h_px - 1) / 2.0
    return np.array([xc_px / float(W), yc_px / float(H), w_px / float(W), h_px / float(H)], dtype=np.float32)

def mask_bytes_to_boxes(mask_bytes):
    im = Image.open(BytesIO(mask_bytes)).convert("RGB")
    arr = np.array(im)
    if arr.ndim == 3:
        mask_bin = np.any(arr != 0, axis=2).astype(np.uint8)
    else:
        mask_bin = (arr != 0).astype(np.uint8)
    try:
        from skimage.measure import label, regionprops
        lab = label(mask_bin)
        props = regionprops(lab)
        boxes = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            boxes.append((minc, minr, maxc - 1, maxr - 1))
        return boxes
    except Exception:
        ys, xs = np.nonzero(mask_bin)
        if len(xs) == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]

def prepare_arrays(parquet_train, parquet_val, N_train=400, N_val=200, resized=RESIZED_SIZE):
    ds = load_dataset("parquet", data_files={"train": parquet_train, "val": parquet_val})
    def build_from_split(split, N):
        X_raw_list = []
        X_pre_list = []
        Y_list = []
        for i, ex in enumerate(split):
            if i >= N: break
            # load original and boxes
            im_orig = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB")
            orig_W, orig_H = im_orig.size
            boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
            if not boxes:
                continue
            areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in boxes]
            best = boxes[int(np.argmax(areas))]
            # resize image
            im_resized = im_orig.resize(resized, resample=Image.BILINEAR)
            W_res, H_res = resized
            # scale box from original -> resized pixel coords
            scale_x = W_res / float(orig_W)
            scale_y = H_res / float(orig_H)
            x_min, y_min, x_max, y_max = best
            x_min_r = int(round(x_min * scale_x))
            x_max_r = int(round(x_max * scale_x))
            y_min_r = int(round(y_min * scale_y))
            y_max_r = int(round(y_max * scale_y))
            # normalize using resized dims
            y_xywh = box_to_xywh_norm((x_min_r, y_min_r, x_max_r, y_max_r), H_res, W_res)
            arr = np.array(im_resized).astype(np.float32)  # 0..255
            X_raw_list.append(arr / 255.0)
            X_pre_list.append(preprocess_input(arr.copy()))
            Y_list.append(y_xywh)
        if not X_raw_list:
            return (np.empty((0, H_res, W_res, 3)), np.empty((0, H_res, W_res, 3)), np.empty((0,4),dtype=np.float32))
        return np.stack(X_raw_list), np.stack(X_pre_list), np.stack(Y_list)
    X_tr, X_tr_pre, Y_tr = build_from_split(ds["train"], N_train)
    X_va, X_va_pre, Y_va = build_from_split(ds["val"], N_val)
    return X_tr, X_tr_pre, Y_tr, X_va, X_va_pre, Y_va

def make_model_pretrained_head(input_shape=(224,224,3), freeze_base=True):
    base = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    base.trainable = not freeze_base
    x = base.output
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation='sigmoid')(x)
    return models.Model(base.input, out), base

def xywh_norm_to_xyxy_pixels(xywh, H, W):
    xc, yc, w, h = [float(x) for x in xywh]
    xc_px = xc * W; yc_px = yc * H
    w_px = w * W; h_px = h * H
    x_min = xc_px - 0.5 * w_px
    y_min = yc_px - 0.5 * h_px
    x_max = xc_px + 0.5 * w_px
    y_max = yc_px + 0.5 * h_px
    return [x_min, y_min, x_max, y_max]

def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1-ax0) * max(0.0, ay1-ay0)
    area_b = max(0.0, bx1-bx0) * max(0.0, by1-by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ---- prepare arrays ----
print("Preparing arrays (this may take a bit)...")
X_tr, X_tr_pre, Y_tr, X_va, X_va_pre, Y_va = prepare_arrays(PARQUET_TRAIN, PARQUET_VAL, N_train=N_train, N_val=N_val, resized=RESIZED_SIZE)
print("Prepared arrays shapes:")
print(" X_tr:", X_tr.shape, " X_tr_pre:", X_tr_pre.shape, " Y_tr:", Y_tr.shape)
print(" X_va:", X_va.shape, " X_va_pre:", X_va_pre.shape, " Y_va:", Y_va.shape)

# ---- build model and load weights if possible ----
model, base = make_model_pretrained_head(input_shape=(RESIZED_SIZE[1], RESIZED_SIZE[0], 3), freeze_base=True)
if os.path.exists(WEIGHTS_FILE):
    try:
        model.load_weights(WEIGHTS_FILE)
        print("Loaded weights from", WEIGHTS_FILE)
    except Exception as e:
        print("Failed to load weights:", e)
else:
    print("Weights file not found:", WEIGHTS_FILE, "- predictions will come from current model parameters (possibly untrained).")

# ---- run diagnostics ----
def run_quick_diag(n_show=10, N_stats=200):
    n_val = min(len(X_va_pre), n_show)
    print("\nFirst", n_val, "validation samples (y_true vs y_pred):")
    ious = []
    for i in range(n_val):
        x = np.expand_dims(X_va_pre[i], 0)
        pred = model.predict(x)[0]
        true = Y_va[i]
        gt_px = xywh_norm_to_xyxy_pixels(true, RESIZED_SIZE[1], RESIZED_SIZE[0])
        pred_px = xywh_norm_to_xyxy_pixels(pred, RESIZED_SIZE[1], RESIZED_SIZE[0])
        iou = iou_xyxy(gt_px, pred_px)
        ious.append(iou)
        print(f"idx={i:3d}  y_true={np.round(true,4)}  y_pred={np.round(pred,4)}  IoU={iou:.3f}")
    N = min(len(X_va_pre), N_stats)
    preds = model.predict(X_va_pre[:N])
    iou_list = [iou_xyxy(xywh_norm_to_xyxy_pixels(t, RESIZED_SIZE[1], RESIZED_SIZE[0]),
                         xywh_norm_to_xyxy_pixels(p, RESIZED_SIZE[1], RESIZED_SIZE[0]))
                for p,t in zip(preds, Y_va[:N])]
    iou_arr = np.array(iou_list)
    print("\nValidation IoU summary (first", N, "examples):")
    print(" mean IoU:", float(iou_arr.mean()), " std:", float(iou_arr.std()))
    print(" percentiles (25,50,75):", np.percentile(iou_arr, [25,50,75]).tolist())
    print(" fraction IoU>0.5:", float((iou_arr > 0.5).mean()), " >0.3:", float((iou_arr>0.3).mean()))
    if X_tr_pre.shape[0] > 0:
        preds_tr = model.predict(X_tr_pre[:N])
        iou_tr = [iou_xyxy(xywh_norm_to_xyxy_pixels(t, RESIZED_SIZE[1], RESIZED_SIZE[0]),
                           xywh_norm_to_xyxy_pixels(p, RESIZED_SIZE[1], RESIZED_SIZE[0]))
                  for p,t in zip(preds_tr, Y_tr[:N])]
        iou_tr = np.array(iou_tr)
        print("\nTrain IoU summary (first", N, "examples): mean:", float(iou_tr.mean()), " std:", float(iou_tr.std()),
              " fraction >0.5:", float((iou_tr>0.5).mean()))
    return iou_arr

run_quick_diag(n_show=10, N_stats=200)
print("\nDone.")

