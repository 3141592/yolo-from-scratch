# === build datasets and preprocessing arrays for MobileNetV2 ===
from datasets import load_dataset
from io import BytesIO
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

RESIZED_SIZE = (224, 224)   # (W, H) used for model input - change if you trained at different size

def box_to_xywh_norm(box, H, W):
    """Correct conversion: (x_min,y_min,x_max,y_max) -> [xc,yc,w,h] normalized in [0,1]
       H,W are image height and width (pixels) used for normalization.
    """
    x_min, y_min, x_max, y_max = box
    w_px = (x_max - x_min + 1)
    h_px = (y_max - y_min + 1)
    xc_px = x_min + (w_px - 1) / 2.0
    yc_px = y_min + (h_px - 1) / 2.0
    return np.array([xc_px / float(W), yc_px / float(H), w_px / float(W), h_px / float(H)], dtype=np.float32)

def mask_bytes_to_boxes(mask_bytes):
    """Return list of boxes (x_min,y_min,x_max,y_max) in ORIGINAL image pixel coords."""
    im = Image.open(BytesIO(mask_bytes)).convert("RGB")
    arr = np.array(im)
    if arr.ndim == 3:
        mask_bin = np.any(arr != 0, axis=2).astype(np.uint8)
    else:
        mask_bin = (arr != 0).astype(np.uint8)
    # use skimage if available for connected components; otherwise fallback bounding box
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

def prepare_arrays(parquet_train, parquet_val, N_train=500, N_val=200, resized=RESIZED_SIZE):
    ds = load_dataset("parquet", data_files={"train": parquet_train, "val": parquet_val})
    def build_from_split(split, N):
        X_raw_list = []
        X_pre_list = []
        Y_list = []
        for i, ex in enumerate(split):
            if i >= N: break
            # original image and size
            im_orig = Image.open(BytesIO(ex["image"]["bytes"])).convert("RGB")
            orig_W, orig_H = im_orig.size
            # largest box from mask (in original pixel coords)
            boxes = mask_bytes_to_boxes(ex["mask"]["bytes"])
            if not boxes:
                continue
            areas = [(b[2]-b[0]+1)*(b[3]-b[1]+1) for b in boxes]
            best = boxes[np.argmax(areas)]
            # resize image
            im_resized = im_orig.resize(resized, resample=Image.BILINEAR)
            W_res, H_res = resized
            # scale box from original -> resized pixel coords, then normalize by resized dims
            scale_x = W_res / float(orig_W)
            scale_y = H_res / float(orig_H)
            x_min, y_min, x_max, y_max = best
            x_min_r = int(round(x_min * scale_x))
            x_max_r = int(round(x_max * scale_x))
            y_min_r = int(round(y_min * scale_y))
            y_max_r = int(round(y_max * scale_y))
            # use box_to_xywh_norm on resized pixel coords with H_res, W_res
            y_xywh = box_to_xywh_norm((x_min_r, y_min_r, x_max_r, y_max_r), H_res, W_res)
            arr = np.array(im_resized).astype(np.float32)  # 0..255
            X_raw_list.append(arr / 255.0)                 # raw normalized to 0..1 (for visualization)
            X_pre_list.append(preprocess_input(arr.copy()))# MobileNet preprocess_input expects 0..255
            Y_list.append(y_xywh)
        if not X_raw_list:
            return (np.empty((0,)+ (resized[1],resized[0],3)), np.empty((0,)+ (resized[1],resized[0],3)),
                    np.empty((0,4),dtype=np.float32))
        return np.stack(X_raw_list), np.stack(X_pre_list), np.stack(Y_list)
    X_tr, X_tr_pre, Y_tr = build_from_split(ds["train"], N_train)
    X_va, X_va_pre, Y_va = build_from_split(ds["val"], N_val)
    return X_tr, X_tr_pre, Y_tr, X_va, X_va_pre, Y_va

# Example usage (adjust file paths / N values)
PARQUET_TRAIN = "/home/roy/src/data/voc2012/train-00000-of-00001.parquet"
PARQUET_VAL   = "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"
X_tr, X_tr_pre, Y_tr, X_va, X_va_pre, Y_va = prepare_arrays(PARQUET_TRAIN, PARQUET_VAL, N_train=400, N_val=200)
print("X_tr.shape, X_tr_pre.shape, Y_tr.shape:", X_tr.shape, X_tr_pre.shape, Y_tr.shape)
print("X_va.shape, X_va_pre.shape, Y_va.shape:", X_va.shape, X_va_pre.shape, Y_va.shape)

