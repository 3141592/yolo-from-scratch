# debug_1.py -- self-contained debug script for YOLO toy regressor
import os
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models

# ---------------------
# User config: edit if needed
# ---------------------
PARQUET_TRAIN = "/home/roy/src/data/voc2012/train-00000-of-00001.parquet"
PARQUET_VAL   = "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"
dataset_split = "val"    # "train" or "val"
sample_idx = 0           # which example to inspect in that split
model_weights_path_candidates = ["best_model.h5", "model.h5", "checkpoint.h5"]
# The input size used during training (set to the size you used)
RESIZED_SIZE = (224, 224)  # (W,H)
# ---------------------

def make_model_pretrained_head(input_shape=(224,224,3), freeze_base=True):
    base = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    base.trainable = not freeze_base
    x = base.output
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(4, activation='sigmoid')(x)
    return models.Model(base.input, out)

def xywh_norm_to_xyxy_pixels(xywh, H, W):
    xc, yc, w, h = [float(x) for x in xywh]
    x_min = (xc - 0.5*w) * W
    y_min = (yc - 0.5*h) * H
    x_max = (xc + 0.5*w) * W
    y_max = (yc + 0.5*h) * H
    return [x_min, y_min, x_max, y_max]

# Load dataset
print("Loading parquet dataset...")
ds = load_dataset("parquet", data_files={"train": PARQUET_TRAIN, "val": PARQUET_VAL})
split = ds[dataset_split]
print(f"len({dataset_split}) = {len(split)}")

if sample_idx < 0 or sample_idx >= len(split):
    raise SystemExit(f"sample_idx {sample_idx} out of range for split {dataset_split} (len={len(split)})")

sample = split[sample_idx]
print("Sample loaded. image/path keys:", sample["image"].keys(), " mask keys:", sample["mask"].keys())

# Build the model and try to load weights if present
print("Building model architecture...")
model = make_model_pretrained_head(input_shape=(RESIZED_SIZE[1], RESIZED_SIZE[0], 3), freeze_base=True)

weights_loaded = False
for fname in model_weights_path_candidates:
    if os.path.exists(fname):
        try:
            model.load_weights(fname)
            print("Loaded weights from:", fname)
            weights_loaded = True
            break
        except Exception as e:
            print("Found weight file but failed to load:", fname, "error:", e)

if not weights_loaded:
    print("No pretrained weights found among candidates:", model_weights_path_candidates)
    print("Predictions will use randomly initialized (or partially initialized) model weights.")
    print("If you trained a model in another script, save it as one of the candidate filenames (e.g. 'best_model.h5') and re-run this script to inspect real predictions.")

# Prepare the image (resized) and the preprocessed version used for prediction
orig_im = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
orig_W, orig_H = orig_im.size
print("Original image size (W,H):", orig_W, orig_H)

resized_im = orig_im.resize(RESIZED_SIZE, resample=Image.BILINEAR)
X_raw = np.array(resized_im).astype(np.float32)  # 0..255
# Build both canonical arrays:
#  - X_va (raw normalized 0..1) for drawing
#  - X_va_pre (preprocessed for MobileNetV2) for model.predict
X_va = (X_raw / 255.0).astype(np.float32)
X_va_pre = preprocess_input(X_raw.copy())  # preprocess_input expects 0..255

# Construct ground truth box for this sample.
# We try to extract boxes from the mask: choose the largest connected component
def mask_bytes_to_boxes(mask_bytes):
    im = Image.open(BytesIO(mask_bytes)).convert("RGB")
    arr = np.array(im)
    if arr.ndim == 3:
        mask_bin = np.any(arr != 0, axis=2).astype(np.uint8)
    else:
        mask_bin = (arr != 0).astype(np.uint8)
    # simple connected components via numpy (avoid heavy deps)
    # We'll use skimage.label if available for better behavior
    try:
        from skimage.measure import label, regionprops
        lab = label(mask_bin)
        props = regionprops(lab)
        boxes = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            boxes.append((minc, minr, maxc - 1, maxr - 1))  # x_min,y_min,x_max,y_max
        return boxes
    except Exception:
        # fallback: bounding box of all non-zero pixels
        ys, xs = np.nonzero(mask_bin)
        if len(xs) == 0:
            return []
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        return [(x_min, y_min, x_max, y_max)]

boxes = mask_bytes_to_boxes(sample["mask"]["bytes"])
print("Found boxes from mask (x_min,y_min,x_max,y_max):", boxes)
if not boxes:
    raise SystemExit("No boxes found in the mask; cannot compute y_true for this sample.")

# pick the largest box
areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in boxes]
best_idx = int(np.argmax(areas))
best_box = boxes[best_idx]
x_min, y_min, x_max, y_max = best_box
# compute normalized (xc, yc, w, h) using ORIGINAL image size (consistent with encoding)
w_px = x_max - x_min + 1
h_px = y_max - y_min + 1
xc = x_min + (w_px - 1) / 2.0
yc = y_min + (h_px - 1) / 2.0
y_true_orig = np.array([xc / orig_W, yc / orig_H, w_px / orig_W, h_px / orig_H], dtype=np.float32)
print("y_true (normalized, relative to ORIGINAL image):", y_true_orig)

# If your training pipeline normalized relative to original image, we will use that.
# But model input is the resized image. We must convert y_true to resized-normalized coordinates for drawing:
#y_true_resized = np.array([ (y_true_orig[0] * orig_W) / RESIZED_SIZE[0],
#                            (y_true_orig[1] * orig_H) / RESIZED_SIZE[1],
#                            (y_true_orig[2] * orig_W) / RESIZED_SIZE[0],
#                            (y_true_orig[3] * orig_H) / RESIZED_SIZE[1] ], dtype=np.float32)

# --- Replace the prior y_true_resized block with this corrected version ---

# compute normalized GT relative to original (you already have y_true_orig)
# now convert via pixel-safe steps

# 1) original pixel coords (center and width/height)
xc_px = y_true_orig[0] * orig_W
yc_px = y_true_orig[1] * orig_H
w_px  = y_true_orig[2] * orig_W
h_px  = y_true_orig[3] * orig_H

# 2) scaled pixel coords on the resized image
scale_x = RESIZED_SIZE[0] / orig_W   # RESIZED_W / orig_W
scale_y = RESIZED_SIZE[1] / orig_H   # RESIZED_H / orig_H
xc_px_resized = xc_px * scale_x
yc_px_resized = yc_px * scale_y
w_px_resized  = w_px  * scale_x
h_px_resized  = h_px  * scale_y

# 3) normalized on resized image
y_true_resized = np.array([
    xc_px_resized / RESIZED_SIZE[0],
    yc_px_resized / RESIZED_SIZE[1],
    w_px_resized  / RESIZED_SIZE[0],
    h_px_resized  / RESIZED_SIZE[1]
], dtype=np.float32)

print("Corrected y_true as normalized values for resized input (xc,yc,w,h):", y_true_resized)


# note: because we used normalized fractions, a simpler mapping is:
# y_true_resized = y_true_orig.copy()  # if training normalized with respect to resized dims instead

print("Resized input size:", RESIZED_SIZE, "(W,H)")
print("y_true as normalized values for resized input (xc,yc,w,h):", y_true_resized)

# Predict using the preprocessed image
X_input = np.expand_dims(X_va_pre, axis=0)  # shape (1,H,W,3)
y_pred = model.predict(X_input)[0]
print("y_pred (normalized xc,yc,w,h) from model:", y_pred)

# Convert both to pixel boxes in the resized image space for visualization
gt_resized_px = xywh_norm_to_xyxy_pixels(y_true_resized, RESIZED_SIZE[1], RESIZED_SIZE[0])
pred_resized_px = xywh_norm_to_xyxy_pixels(y_pred, RESIZED_SIZE[1], RESIZED_SIZE[0])
print("GT box in RESIZED pixels (x_min,y_min,x_max,y_max):", gt_resized_px)
print("PRED box in RESIZED pixels (x_min,y_min,x_max,y_max):", pred_resized_px)

# Convert to ORIGINAL image coords (useful if you want to overlay on orig_img)
gt_orig_px = xywh_norm_to_xyxy_pixels(y_true_orig, orig_H, orig_W)
pred_orig_px = xywh_norm_to_xyxy_pixels(y_pred, orig_H, orig_W)  # note: if model predicted relative to resized, this rescales
print("GT box in ORIGINAL pixels:", gt_orig_px)
print("PRED box in ORIGINAL pixels (rescaled):", pred_orig_px)

# Compute IoU (resized coords)
def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1-ax0) * max(0.0, ay1-ay0)
    area_b = max(0.0, bx1-bx0) * max(0.0, by1-by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

iou_resized = iou_xyxy(gt_resized_px, pred_resized_px)
print("IoU (resized coords):", iou_resized)

# Draw overlay on resized image (X_va)
img_display = (X_va * 255.0).astype(np.uint8)
plt.figure(figsize=(6,6)); plt.imshow(img_display); ax = plt.gca()
# draw GT (green)
x0,y0,x1,y1 = gt_resized_px
ax.add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='green', linewidth=2))
# draw pred (red)
x0,y0,x1,y1 = pred_resized_px
ax.add_patch(plt.Rectangle((x0,y0), x1-x0, y1-y0, fill=False, edgecolor='red', linewidth=2))
plt.title(f"IoU: {iou_resized:.3f}")
plt.axis('off')
outname = "debug_pred_vs_gt.png"
plt.savefig(outname, bbox_inches='tight', dpi=150)
plt.show()
print("Saved debug overlay to", outname)

