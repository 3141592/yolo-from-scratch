from datasets import load_dataset
from io import BytesIO
from PIL import Image
import numpy as np
from skimage.measure import label, regionprops

# Load the dataset
ds = load_dataset("parquet", data_files={"train": "/home/roy/src/data/voc2012/train-00000-of-00001.parquet", "val": "/home/roy/src/data/voc2012/val-00000-of-00001.parquet"})
train = ds["train"]; val = ds["val"]
print("len(train): ", len(train))
print("len(val): ", len(val))

# Grab one example to inspect
sample = train[0]

# Image data
print("sample[\"image\"] keys:")
print(sample["image"].keys())

print("sample[\"image\"]")
print(sample["image"]["bytes"][:40])

print("sample[\"image[\"path\"]")
print(sample["image"]["path"])

# Mask data
print("sample[\"mask\"] keys:")
print(sample["mask"].keys())

print("sample[\"mask\"]")
print(sample["mask"]["bytes"][:40])

print("sample[\"mask[\"path\"]")
print(sample["mask"]["path"])

img = np.array(Image.open(BytesIO(sample["image"]["bytes"])))
print("img[:1]")
print(img[:1])

mask = np.array(Image.open(BytesIO(sample["mask"]["bytes"])))
print("mask[:1]")
print(mask[:1])

H, W = mask.shape[:2]
print("H, W", H , ", ", W)

np.unique(mask)
print("np.unique(mask): ", np.unique(mask))


present = np.unique(mask)


present = present[(present > 0) & (present < 255)]

