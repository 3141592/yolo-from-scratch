# yolo_toy_regressor.py — minimal single-box regressor
from datasets import load_dataset
import numpy as np
import sys
from PIL import Image
from io import BytesIO
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Quiet loggin
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # 0=all, 1=hide INFO, 2=hide WARNING, 3=hide ERROR (not recommended)

# Manage gpu mem use
import tensorflow as tf, os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    tf.config.experimental.set_memory_growth(g, True)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

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
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    xc = x_min + (w - 1) / 2.0
    yc = y_min + (h - 1) / 2.0
    return np.array([xc / W, yc / H, w / W, h / H], dtype=np.float32)

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
    X = []  # images resized to 448x448
    Y = []  # box targets [xc,yc,w,h] normalized
    for i, ex in enumerate(dataset):
        if i >= max_samples: break
        img = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB").resize((448,448))
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

# --- custom IoU metric ---
def iou_metric(y_true, y_pred):
    # y_true, y_pred: (batch, 4) in [xc, yc, w, h] normalized
    # convert to xyxy
    xcycwh_to_xyxy = lambda t: tf.stack([
        t[...,0] - 0.5*t[...,2],  # x_min
        t[...,1] - 0.5*t[...,3],  # y_min
        t[...,0] + 0.5*t[...,2],  # x_max
        t[...,1] + 0.5*t[...,3],  # y_max
    ], axis=-1)
    p = xcycwh_to_xyxy(y_pred)
    g = xcycwh_to_xyxy(y_true)
    inter_mins = tf.maximum(p[...,:2], g[...,:2])
    inter_maxs = tf.minimum(p[...,2:], g[...,2:])
    inter_wh   = tf.maximum(inter_maxs - inter_mins, 0.0)
    inter_area = inter_wh[...,0] * inter_wh[...,1]
    pred_area  = (p[...,2]-p[...,0]) * (p[...,3]-p[...,1])
    gt_area    = (g[...,2]-g[...,0]) * (g[...,3]-g[...,1])
    union      = pred_area + gt_area - inter_area
    iou        = tf.where(union > 0, inter_area / union, tf.zeros_like(union))
    return tf.reduce_mean(iou)


# --- tiny model ---
def make_model(input_shape=(448,448,3)):
    # Input Layer
    inp = layers.Input(shape=input_shape)
    # Layer 1
    x = layers.Conv2D(192,7, strides=(2,2),padding='same')(inp)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    # Layer 2
    x = layers.Conv2D(256,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    # Layer 3
    x = layers.Conv2D(128,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    # Layer 4
    x = layers.Conv2D(256,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(256,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(512,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)

    # Layer 5
    x = layers.Conv2D(512,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(512,1,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(1024,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024,3,strides=(2, 2),padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Layer 6
    x = layers.Conv2D(1024,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(1024,3,padding='same')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)


    # Layer 7
    #x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    # Layer 8 (DropOut)
    x = layers.Dropout(0.5)(x)

    # Layer 9
    out = layers.Dense(4, activation=None, dtype='float32')(x)
    return models.Model(inp, out)

# --- train & eval (example) ---
if __name__ == "__main__":
    # Load data from parquet files
    ds = load_dataset("parquet", data_files={"train": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet", "val": "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"})
    train = ds["train"]; val = ds["val"]

    # Prepare data
    sample_list = [train[i] for i in range(200)]
    X, Y = get_samples_from_dataset(sample_list, max_samples=200)
    # split
    idx = int(0.8*len(X))
    X_tr, Y_tr = X[:idx], Y[:idx]
    X_va, Y_va = X[idx:], Y[idx:]

    model = make_model()

    # --- replace your compile() line and add this above model.fit() ---
    import tensorflow as tf
    try:
        import tensorflow_addons as tfa
        # Decoupled weight decay (closer to “weight decay 5e-4” in the paper)
        optimizer = tfa.optimizers.SGDW(
            learning_rate=1e-3,  # will be overridden by the scheduler each epoch
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=False,
        )
    except Exception:
        # Fallback: standard SGD with L2 via 'weight_decay' if available in your TF/Keras;
        # if not, you can add kernel_regularizer=... on layers instead.
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)

    # YOLOv1-style LR schedule (warmup -> 1e-2 -> 1e-3 -> 1e-4), totalling ~135 epochs.
    TOTAL_EPOCHS = 30
    WARMUP_EPOCHS = 3  # small warmup from 1e-3 up to 1e-2

    def yolo_v1_lr_schedule(epoch):
        if epoch < WARMUP_EPOCHS:
            # linear warmup 1e-3 -> 1e-2
            return 1e-3 + (1e-2 - 1e-3) * (epoch + 1) / max(1, WARMUP_EPOCHS)
        elif epoch < WARMUP_EPOCHS + 75:
            return 1e-2
        elif epoch < WARMUP_EPOCHS + 75 + 30:
            return 1e-3
        else:
            return 1e-4

    class ValIoUCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_val, Y_val, every=5, name='val_iou'):
            super().__init__()
            self.X_val = X_val
            self.Y_val = Y_val
            self.every = every
            self.name = name

        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.every != 0:
                return
            preds = self.model.predict(self.X_val, verbose=0)
            ious = [iou_xywh(p, g) for p, g in zip(preds, self.Y_val)]
            val_iou = float(np.mean(ious))
            # add to logs so it shows in History & can be monitored
            logs = logs or {}
            logs[self.name] = val_iou
            print(f"\nEpoch {epoch+1}: {self.name}={val_iou:.4f}")

    lr_cb = tf.keras.callbacks.LearningRateScheduler(yolo_v1_lr_schedule, verbose=1)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(), metrics=[iou_metric])
    model.summary()
    #sys.exit()

    val_iou_cb = ValIoUCallback(X_va, Y_va, every=1, name='val_iou')
    cbs = [
      val_iou_cb,
      ModelCheckpoint("best_val_iou.weights.h5", monitor="val_iou", mode="max",
                      save_best_only=True, save_weights_only=True, verbose=1),
      ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                        min_lr=1e-6, verbose=1),
      EarlyStopping(monitor="val_iou", mode="max", patience=12,
                    restore_best_weights=True, verbose=1),
    ]  

    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=TOTAL_EPOCHS,           # or whatever you're running
        batch_size=8,
        callbacks=cbs
    )

    # evaluate IoU on validation
    preds = model.predict(X_va)
    ious = [iou_xywh(p, g) for p,g in zip(preds, Y_va)]
    print("val IoU mean:", np.mean(ious))

    # visualize some results
    for i in range(min(5, len(X_va))):
        img = (X_va[i]*255).astype(np.uint8)
        pred = preds[i]; gt = Y_va[i]
        plt.imshow(img)
        H,W,_ = img.shape
        def draw_box(xywh, color):
            xc,yc,w,h = xywh
            x_min = int((xc - 0.5*w)*W); x_max = int((xc+0.5*w)*W)
            y_min = int((yc - 0.5*h)*H); y_max = int((yc+0.5*h)*H)
            plt.gca().add_patch(plt.Rectangle((x_min,y_min), x_max-x_min, y_max-y_min,
                                              fill=False, edgecolor=color, linewidth=2))
        draw_box(gt, 'g')
        draw_box(pred, 'r')
        plt.title(f"IoU: {iou_xywh(pred,gt):.3f}")
        plt.axis('off')
        plt.show()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['val_iou'], label='val_iou')
plt.xlabel('epoch'); plt.legend(); plt.title('Loss & IoU'); plt.show()


