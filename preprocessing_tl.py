import random
import warnings
from cloud_model import CloudModel
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_path import path
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl
import torch
import json
from pandas.io.json import json_normalize
from pprint import pprint
import rasterio
import xarray
import xrspatial.multispectral as ms
import csv

# def get_xarray(filepath):
#     """Put images in xarray.DataArray format"""
#     im_arr = np.array(Image.open(filepath))
#     return xarray.DataArray(im_arr, dims=["y", "x"])


def true_color_img(tile):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    red_img = rasterio.open(tile['assets']['B04']['href'])
        # red_img = rd_img.read(1)
    #print("RED IMG: ")
    # print(red_img)
    red = xarray.DataArray(red_img.read(1), dims=["y", "x"])
    green_img = rasterio.open(tile['assets']['B03']['href'])
    green = xarray.DataArray(green_img.read(1), dims=["y", "x"])
    blue_img = rasterio.open(tile['assets']['B02']['href'])
    blue = xarray.DataArray(blue_img.read(1), dims=["y", "x"])

    return ms.true_color(r=red, g=green, b=blue)

def display_random_chip(random_number):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    with open(TRAIN_FEATURES / train_metadata['links'][random_number]['href']) as json_data:
        tile = json.load(json_data)
    ax[0].imshow(true_color_img(tile))
    ax[0].set_title(f"Chip {tile['id']}")
    with open(TRAIN_LABELS / "collection.json") as json_data:
        labels_metadata = json.load(json_data)
    with open(TRAIN_LABELS / labels_metadata['links'][random_number]['href']) as json_data:
        label_im = json.load(json_data)
    label_img = rasterio.open(label_im['assets']['labels']['href'])
    img_array = label_img.read(1)
    ax[1].imshow(img_array)
    ax[1].set_title(f"Chip {label_im['id']} label")

    plt.tight_layout()
    plt.show()

DATA_DIR = Path.cwd().parent.resolve() / "C:/Users/RACL/PycharmProjects/MLforEO"
TRAIN_FEATURES = DATA_DIR / "ref_cloud_cover_detection_challenge_v1" / "train_source"
TRAIN_LABELS = DATA_DIR / "ref_cloud_cover_detection_challenge_v1" / "train_labels"
assert TRAIN_FEATURES.exists()
submission_dir = DATA_DIR / "codeexecution"
BANDS = ["B02", "B03", "B04"]#, "B08"]

# with open(TRAIN_FEATURES / "collection.json") as json_data:
#      train_metadata = json.load(json_data)

train_metadata = pd.read_csv(DATA_DIR / "train_meta_small.csv")
# print(train_meta.head())



#with open(TRAIN_FEATURES / train_metadata['links'][238]['href']) as json_data:
#            tile = json.load(json_data)
#with rasterio.open(tile['assets']['B04']['href']) as img:
#    chip_metadata = img.meta
#    img_array = img.read(1)

## Plot one band of image ##
#plt.imshow(img_array)
#plt.title(f"B04 band for chip id {tile['id']}")
#plt.show()

## Plot true color image using all three bands ##
#fig, ax = plt.subplots(figsize=(5, 5))
#im = true_color_img(tile)
#plt.imshow(im)
#plt.title(f"All bands for chip id {tile['id']}")
#plt.show()

## Plot true colour image next to cloud label for the same image ##
#display_random_chip(1852)

##Plot datetime of chips
#train_meta["datetime"] = pd.to_datetime(train_meta["datetime"])
#train_meta["year"] = train_meta.datetime.dt.year
#train_datetime_counts = train_meta.groupby("year")[["chip_id"]].nunique().sort_index().rename(
#   columns={"chip_id": "chip_count"}
#)
#train_datetime_counts.head(4).plot(kind="bar", color="lightblue")
#plt.xticks(rotation=90)
#plt.xlabel("Year")
#plt.ylabel("Number of Chips")
#plt.title("Number of Train Chips by Year")
#plt.show()

## create sets to feed the algorithm
random.seed(9)  # set a seed for reproducibility

# put 1/3 of chips into the validation set
chip_ids = train_metadata.chip_id.unique().tolist()
val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))

val_mask = train_metadata.chip_id.isin(val_chip_ids)
val = train_metadata[val_mask].copy().reset_index(drop=True)
train = train_metadata[~val_mask].copy().reset_index(drop=True)

print(val.shape, train.shape)

# separate features from labels
feature_cols = ["chip_id"] + [f"{band}_path" for band in BANDS]

val_x = val[feature_cols].copy()
val_y = val[["chip_id", "label_path"]].copy()
print(val_x.shape, val_y.shape)

train_x = train[feature_cols].copy()
train_y = train[["chip_id", "label_path"]].copy()
print(train_x.shape, train_y.shape)

cloud_model = CloudModel(
   bands=BANDS,
   x_train=train_x,
   y_train=train_y,
   x_val=val_x,
   y_val=val_y,
)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="iou_epoch", mode="max", verbose=True
)
early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
    monitor="iou_epoch",
    patience=(cloud_model.patience),
    mode="max",
    verbose=True,
)

trainer = pl.Trainer(
    gpus=None,
    fast_dev_run=False,
    callbacks=[checkpoint_callback, early_stopping_callback],
)
if __name__ == '__main__':
    trainer.fit(model=cloud_model)

# save the model
#submission_assets_dir = submission_dir / "assets"
#submission_assets_dir.mkdir(parents=True, exist_ok=True)

#model_weight_path = submission_assets_dir / "cloud_model.pt"
#torch.save(cloud_model.state_dict(), model_weight_path)