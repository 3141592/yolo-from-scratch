# explore_mask_1.py
from datasets import load_dataset
from io import BytesIO
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops

# --- VOC colormap / class names (index == class id) ---
VOC_COLORMAP = [
 (0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128),
 (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0),
 (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
 (192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0),
 (0,64,128)
]
VOC_CLASS_NAMES = [
 "background","aeroplane","bicycle","bird","boat","bottle","bus","car",
 "cat","chair","cow","diningtable","dog","horse","motorbike","person",
 "pottedplant","sheep","sofa","train","tvmonitor"
]

# --- helper function (same as earlier) ---
def inspect_and_convert_mask(mask_bytes):
    im = Image.open(BytesIO(mask_bytes))
    arr = np.array(im)
    H, W = arr.shape[:2]
    meta = {"pil_mode": im.mode, "shape": arr.shape, "mapping": {}, "unique_colors": [], "unique_class_ids": []}

    if im.mode == "P":
        pal = im.getpalette()
        if pal is None:
            raise RuntimeError("Image mode 'P' but no palette found.")
        unique_pidx = np.unique(arr)
        palette_rgb = {int(p): tuple(pal[int(p)*3:int(p)*3+3]) for p in unique_pidx}
        meta["unique_colors"] = [(p, palette_rgb[int(p)]) for p in unique_pidx]

        pidx2cls = {}
        for p in unique_pidx:
            rgb = palette_rgb[int(p)]
            pidx2cls[int(p)] = VOC_COLORMAP.index(rgb) if rgb in VOC_COLORMAP else 255

        mask_class = np.full((H, W), 255, dtype=np.uint8)
        for pidx, cls in pidx2cls.items():
            if cls != 255:
                mask_class[arr == pidx] = cls
        meta["mapping"] = pidx2cls

    elif arr.ndim == 3 and arr.shape[2] == 3:
        packed = (arr[:,:,0].astype(np.uint32) << 16) | (arr[:,:,1].astype(np.uint32) << 8) | arr[:,:,2].astype(np.uint32)
        unique_packed, counts = np.unique(packed, return_counts=True)
        unique_colors = [(int((u>>16)&255), int((u>>8)&255), int(u&255)) for u in unique_packed]
        meta["unique_colors"] = list(zip(unique_colors, counts.astype(int)))

        pcol2cls = {}
        for u in unique_packed:
            rgb = (int((u>>16)&255), int((u>>8)&255), int(u&255))
            pcol2cls[int(u)] = VOC_COLORMAP.index(rgb) if rgb in VOC_COLORMAP else 255

        mask_class = np.full((H, W), 255, dtype=np.uint8)
        for u, cls in pcol2cls.items():
            if cls != 255:
                mask_class[packed == int(u)] = cls
        meta["mapping"] = {c: pcol2cls[int(c)] for c in unique_packed}

    elif arr.ndim == 2:
        vals, counts = np.unique(arr, return_counts=True)
        meta["unique_colors"] = list(zip(vals.tolist(), counts.astype(int).tolist()))
        if np.all((vals <= 20) | (vals == 255)):
            mask_class = arr.astype(np.uint8)
            mask_class[~(((mask_class <= 20) | (mask_class == 255)))] = 255
            meta["mapping"] = {int(v): int(v) for v in vals if v <= 20}
        else:
            mask_class = np.full((H, W), 255, dtype=np.uint8)
            meta["mapping"] = {}
    else:
        raise RuntimeError("Unhandled mask shape/mode: mode=%s arr.shape=%s" % (im.mode, arr.shape))

    meta["unique_class_ids"], counts = np.unique(mask_class, return_counts=True)
    meta["unique_class_ids"] = meta["unique_class_ids"].tolist()
    meta["class_counts"] = dict(zip([int(x) for x in np.unique(mask_class)], [int(x) for x in counts]))

    instances = {}
    for cls in range(0, 21):
        mask_c = (mask_class == cls)
        if not mask_c.any():
            continue
        lab = label(mask_c)
        props = regionprops(lab)
        bboxes = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            x_min, y_min, x_max, y_max = minc, minr, maxc - 1, maxr - 1
            area = p.area
            bboxes.append({"bbox": (x_min, y_min, x_max, y_max), "area": int(area)})
        instances[cls] = bboxes

    meta["instances"] = instances
    return mask_class, meta

# --- main script: load dataset, pick sample, inspect mask ---
if __name__ == "__main__":
    ds = load_dataset(
        "parquet",
        data_files={
            "train": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet",
            "val":   "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"
        }
    )
    train = ds["train"]
    val = ds["val"]
    print("len(train):", len(train))
    print("len(val):", len(val))

    # pick the sample index you want to inspect
    sample_idx = 0
    sample = train[sample_idx]   # <-- THIS ensures 'sample' is defined in this file

    print("\nsample['image'] keys:", sample['image'].keys())
    print("sample['image']['path']:", sample['image'].get('path'))
    print("sample['mask'] keys:", sample['mask'].keys())
    print("sample['mask']['path']:", sample['mask'].get('path'))

    # call the helper
    mask_class, meta = inspect_and_convert_mask(sample["mask"]["bytes"])

    print("\n=== mask inspection ===")
    print("PIL mode:", meta["pil_mode"])
    print("mask shape:", meta["shape"])
    print("unique colors (or indices) and counts (first 10):")
    for item in meta["unique_colors"][:10]:
        print("  ", item)
    print("mapping (present_color -> class_id or 255):")
    print(meta["mapping"])
    print("unique class IDs present (0..20 or 255):", meta["unique_class_ids"])
    print("per-class pixel counts:", meta["class_counts"])
    print("per-class instance bboxes (sample):")
    for cls, bboxes in meta["instances"].items():
        print(f"  {cls} {VOC_CLASS_NAMES[cls]} -> {len(bboxes)} instance(s); first bbox:", bboxes[0] if bboxes else None)

