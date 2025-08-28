# yolo_toy_regressor.py — minimal single-box regressor
from datasets import load_dataset
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

# --- helpers (adapt to your pipeline) ---
def mask_bytes_to_boxes(mask_bytes):
    im = Image.open(BytesIO(mask_bytes))
    arr = np.array(im)
    # if RGB -> map colors -> class mask; or if already single-channel, use it
    # For this toy, we assume you already converted mask_class (0..20,255): easiest path:
    # Here, we will do a quick fallback: if RGB, treat non-black pixels as object
    if arr.ndim == 3:
        # object pixels = anything not black (0,0,0)
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
    #w = x_max - x_min + 1
    #h = y_max - y_min + 1
    #xc = x_min + (w - 1) / 2.0
    #yc = y_min + (h - 1) / 2.0
    #return np.array([xc / W, yc / H, w / W, h / H], dtype=np.float32)
    x_min_scaled = x_min / W
    x_max_scaled = x_max / W
    y_min_scaled = y_min / H
    y_max_scaled = y_max / H

    xc = (x_max_scaled + x_min_scaled) / 2
    yc = (y_max_scaled + y_min_scaled) / 2

    w = x_max_scaled - x_min_scaled
    h = y_max_scaled - y_min_scaled
    return np.array([xc, yc, w, h], dtype=np.float32)

def iou_xywh(pred, gt):
    # pred & gt = [xc, yc, w, h] normalized
    def to_xyxy(x):
        xc, yc, w, h = x
        x_min = xc - 0.5*w
        y_min = yc - 0.5*h
        x_max = xc + 0.5*w
        y_max = yc + 0.5*h
        return np.array([x_min, y_min, x_max, y_max])
    p = to_xyxy(pred); g = to_xyxy(gt)
    inter_min = np.maximum(p[:2], g[:2])
    inter_max = np.minimum(p[2:], g[2:])
    inter = np.maximum(inter_max - inter_min, 0.0)
    inter_area = inter[0]*inter[1]
    ap = (p[2]-p[0])*(p[3]-p[1])
    ag = (g[2]-g[0])*(g[3]-g[1])
    union = ap + ag - inter_area
    if union <= 0: return 0.0
    return inter_area / union

# --- small dataset loader (replace with your parquet loader) ---
def get_samples_from_dataset(dataset, max_samples=500):
    X = []  # images resized to 224x224
    Y = []  # box targets [xc,yc,w,h] normalized
    for i, ex in enumerate(dataset):
        if i >= max_samples: break
        img = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB").resize((224,224))
        mask_boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
        if not mask_boxes:
            continue
        # choose largest box (area)
        areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in mask_boxes]
        best = mask_boxes[np.argmax(areas)]
        # original image size needed for normalization:
        orig = Image.open(BytesIO(ex["image"]["bytes"]))
        W, H = orig.size
        Y.append(box_to_xywh_norm(best, H, W))
        X.append(np.array(img).astype(np.float32)/255.0)
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y

# --- MobileNetV2 ---
def make_model_pretrained(input_shape=(224,224,3)):
    base = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    # Optionally freeze some layers
    for layer in base.layers[:-20]:
      layer.trainable = False
    x = base.output
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(4, activation='sigmoid')(x) # normalized [xc,yc,w,h]
    return models.Model(base.input, out)

# ----------------- paste this block in place of your existing main/training block -----------------
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

if __name__ == "__main__":
    # Load data from parquet files
    ds = load_dataset("parquet", data_files={
        "train": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet",
        "val":   "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"
    })
    train = ds["train"]; val = ds["val"]

    # Prepare data (same as before)
    sample_list = [train[i] for i in range(200)]
    X, Y = get_samples_from_dataset(sample_list, max_samples=200)
    idx = int(0.8*len(X))
    X_tr, Y_tr = X[:idx], Y[:idx]
    X_va, Y_va = X[idx:], Y[idx:]

    # Prepare MobileNet-style preprocessed arrays (preprocess_input expects 0..255)
    # X_tr currently in 0..1 — convert back to 0..255 then preprocess
    X_tr_pre = preprocess_input((X_tr * 255.0).astype(np.float32))
    X_va_pre = preprocess_input((X_va * 255.0).astype(np.float32))

    # --- build model & base (return base so we can freeze/unfreeze) ---
    def make_model_pretrained_head(input_shape=(224,224,3), freeze_base=True):
        base = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
        base.trainable = not freeze_base
        x = base.output
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        out = layers.Dense(4, activation='sigmoid')(x)  # normalized [xc,yc,w,h]
        return models.Model(base.input, out), base

    model, base = make_model_pretrained_head(input_shape=(224,224,3), freeze_base=True)

    # Val IoU callback (computes mean IoU on X_va_pre at epoch end)
    class ValIoUCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_val, Y_val, batch_size=16):
            super().__init__()
            self.X_val = X_val
            self.Y_val = Y_val
            self.batch_size = batch_size
        def on_epoch_end(self, epoch, logs=None):
            preds = self.model.predict(self.X_val, batch_size=self.batch_size)
            ious = [iou_xywh(p, g) for p, g in zip(preds, self.Y_val)]
            miou = float(np.mean(ious))
            print(f" -> val_mIoU: {miou:.4f}")
            if logs is not None:
                logs['val_mIoU'] = miou

    # Save best model by val_mIoU
    chk = ModelCheckpoint("best_model.h5", monitor="val_mIoU", mode="max", save_best_only=True, verbose=1)
    miou_cb = ValIoUCallback(X_va_pre, Y_va, batch_size=16)

    # ---------------------------------------------------------------------
    # Stage 1: Train HEAD only (base frozen) - safe initial training
    # ---------------------------------------------------------------------
    for layer in base.layers:
        layer.trainable = False
    # compile & train head
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    model.fit(X_tr_pre, Y_tr, validation_data=(X_va_pre, Y_va),
              epochs=8, batch_size=16, callbacks=[miou_cb, chk])

    # After this completes you should have best_model.h5 (if val_mIoU improved).
    # ---------------------------------------------------------------------
    # Optional Stage 2: Fine-tune last N layers of the backbone (uncomment to run)
    # ---------------------------------------------------------------------
    N = 15  # number of last layers to unfreeze; adjust as needed
    for layer in base.layers[-N:]:
        layer.trainable = True
    # keep BatchNorm layers frozen for stability on small batches
    for l in base.layers:
        if isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False
    # recompile with small LR for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-9, momentum=0.9), loss='mse')
    model.fit(X_tr_pre, Y_tr, validation_data=(X_va_pre, Y_va),
              epochs=8, batch_size=16, callbacks=[miou_cb, chk])

    # ---------------------------------------------------------------------
    # Evaluation & visualization using the best saved model (if present)
    # ---------------------------------------------------------------------
    # Try to load best weights if available
    try:
        model.load_weights("best_model.h5")
        print("Loaded best_model.h5 for evaluation")
    except Exception:
        print("No best_model.h5 found; evaluating current model parameters")

    preds = model.predict(X_va_pre)
    ious = [iou_xywh(p, g) for p, g in zip(preds, Y_va)]
    print("val IoU mean:", np.mean(ious))

    # visualize some results (same as your code but using preprocessed model input for predict)
    for i in range(min(5, len(X_va))):
        img = (X_va[i]*255).astype(np.uint8)  # original visual image (0..255)
        pred = preds[i]; gt = Y_va[i]
        plt.figure(figsize=(4,4)); plt.imshow(img)
        H_img, W_img, _ = img.shape
        def draw_box(xywh, color):
            xc, yc, w, h = xywh
            x_min = int((xc - 0.5*w)*W_img); x_max = int((xc+0.5*w)*W_img)
            y_min = int((yc - 0.5*h)*H_img); y_max = int((yc+0.5*h)*H_img)
            # clamp to image for neatness
            x_min = max(0, x_min); y_min = max(0, y_min)
            x_max = min(W_img-1, x_max); y_max = min(H_img-1, y_max)
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                              fill=False, edgecolor=color, linewidth=2))
        draw_box(gt, 'g')
        draw_box(pred, 'r')
        plt.title(f"IoU: {iou_xywh(pred,gt):.3f}")
        plt.axis('off')
        plt.show()
# ----------------------------------------------------------------------------------------------------

