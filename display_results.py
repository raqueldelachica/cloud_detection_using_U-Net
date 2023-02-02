import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio
import torch
import xarray
import xrspatial.multispectral as ms
import numpy as np
import tifffile as tiff
import tensorflow as tf
from imageio.core import image_as_uint
from skimage.util import img_as_ubyte, img_as_uint


def true_color_img(random_number, metadata):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    red_img = rasterio.open(metadata.get('B04_path')[random_number])
    red = xarray.DataArray(red_img.read(1), dims=["y", "x"])
    green_img = rasterio.open(metadata.get('B03_path')[random_number])
    green = xarray.DataArray(green_img.read(1), dims=["y", "x"])
    blue_img = rasterio.open(metadata.get('B02_path')[random_number])
    blue = xarray.DataArray(blue_img.read(1), dims=["y", "x"])

    return ms.true_color(r=red, g=green, b=blue)

def display_random_chip(random_number, metadata):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(true_color_img(random_number, metadata))
    ax[0].set_title(f"Chip {metadata.get('chip_id')[random_number]}")
    label_img = rasterio.open(metadata.get('label_path')[random_number])
    img_array = label_img.read(1)
    ax[1].imshow(img_array)
    # ax[1].set_title(f"Chip {test_meta.get('chip_id')[random_number]})
    plt.tight_layout()
    plt.show()

def rgb(chip_id):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    #red_img = rasterio.open('./test_images/{0}/B02.tif'.format(chip_id))
    #red = xarray.DataArray(red_img.read(1), dims=["y", "x"])
    #green_img = rasterio.open('./test_images/{0}/B03.tif'.format(chip_id))
    #green = xarray.DataArray(green_img.read(1), dims=["y", "x"])
    #blue_img = rasterio.open('./test_images/{0}/B04.tif'.format(chip_id))
    #blue = xarray.DataArray(blue_img.read(1), dims=["y", "x"])
    #color_img = ms.true_color(r=red, g=green, b=blue)
    #infra_img = rasterio.open('./test_images/{0}/B08.tif'.format(chip_id))
    #infra = xarray.DataArray(infra_img.read(1), dims=["y", "x"])
    #infra = np.array(img_as_uint(infra))
    ##color_img = np.zeros((512, 512, 4), dtype=np.uint8)
    ##color_img[:, :, 0] = red
    ##color_img[:, :, 1] = green
    ##color_img[:, :, 2] = blue
    #color_img[:, :, 3] = infra
    #return color_img
    # color_img = []  # np.zeros((4, 512, 512), dtype=np.uint8)
    # red_img = rasterio.open('./images/{0}/B02.tif'.format(chip_id))
    with rasterio.open('./test_images/{0}/B02.tif'.format(chip_id)) as ro:
        red_img = ro.read(1)
        # red_img = np.array(red_img, dtype='uint8')
        # red_img = xarray.DataArray(ro.read(1), dims=["y", "x"])
        # red_img = (red_img / 2 ** 8).astype(np.uint8)
        # red_img = np.array(img_as_ubyte(red_img))
        # red_img = np.array(red_img, dtype='float32')
        # red_img_f = (red_img).astype(np.float32)
        # red_img_temp = red_img_f - red_img_f.min()
        # red_img_temp = red_img_temp / red_img_temp.max()
        # red_img_temp = red_img_temp * 255
        # red_img_f = red_img_temp.astype(np.uint8)
        # color_img.append(red_img_f)
        red = xarray.DataArray(ro.read(1), dims=["y", "x"])
    # green_img = rasterio.open('./images/{0}/B03.tif'.format(chip_id))
    with rasterio.open('./test_images/{0}/B03.tif'.format(chip_id)) as ro:
        green_img = ro.read(1)
        # green_img = (green_img / 2 ** 8).astype(np.uint8)
        # green_img = np.array(img_as_ubyte(green_img))
        # green_img = green_img.astype(np.uint8)
        # green_img_f = (green_img).astype(np.float32)
        # green_img_temp = green_img_f - green_img_f.min()
        # green_img_temp = green_img_temp / green_img_temp.max()
        # green_img_temp = green_img_temp * 255
        # green_img_f = green_img_temp.astype(np.uint8)
        # color_img.append(green_img_f)
        green = xarray.DataArray(ro.read(1), dims=["y", "x"])
    # blue_img = rasterio.open('./images/{0}/B04.tif'.format(chip_id))
    with rasterio.open('./test_images/{0}/B04.tif'.format(chip_id)) as ro:
        blue_img = ro.read(1)
        # blue_img = (blue_img / 2 ** 8).astype(np.uint8)
        # blue_img = np.array(img_as_ubyte(blue_img))
        # blue_img = blue_img.astype(np.uint8)
        # blue_img_f = (blue_img).astype(np.float32)
        # blue_img_temp = blue_img_f - blue_img_f.min()
        # blue_img_temp = blue_img_temp / blue_img_temp.max()
        # blue_img_temp = blue_img_temp * 255
        # blue_img_f = blue_img_temp.astype(np.uint8)
        # color_img.append(blue_img_f)
        blue = xarray.DataArray(ro.read(1), dims=["y", "x"])
    multi_img = ms.true_color(r=red, g=green, b=blue)
    print("MULTI_IMG: ", multi_img)
    # infra_img = rasterio.open('./images/{0}/B08.tif'.format(chip_id))
    # infra = xarray.DataArray(infra_img.read(1), dims=["y", "x"])
    with rasterio.open('./test_images/{0}/B08.tif'.format(chip_id)) as ro:
        nr_img = ro.read(1)
        nr = xarray.DataArray(ro.read(1), dims=["y", "x"])
        # nr_img = (nr_img / 2 ** 8).astype(np.uint8)
        # nr_img = np.array(img_as_ubyte(nr_img))
        # nr_img = nr_img.astype(np.uint8)
        # nr_img = (nr_img).astype(np.float32)
        # nr_img_temp = 1 / (1 + np.exp(10 * (0.125 - nr_img)))
        # nr_img_temp = nr_img - nr_img.min()
        # print("NR IMG MAX before: ", nr_img.max())
        # nr_img_temp = nr_img_temp / nr_img_temp.max()
        # nr_img_temp = nr_img_temp * 255
        # nr_img_temp = 1 / (1 + np.exp(10 * (0.125 - nr_img_temp)))
        # nr_img = nr_img_temp.astype(np.uint8)
        # color_img.append(nr_img)
    print("NR IMG: ", nr_img)
    nir_multi = ms.true_color(r=nr, g=green, b=blue)
    # print("NR IMG MAX after: ", nr_img.max())
    # infra = np.array(img_as_ubyte(infra))
    # color_img = np.zeros((512, 512, 4), dtype=np.uint8)
    # color_img[:, :, 0] = red
    # color_img[:, :, 1] = green
    # color_img[:, :, 2] = blue
    # color_img[:, :, 3] = infra
    # color_img = np.stack(color_img,axis=-1)
    # color_img = np.stack(color_img, axis=-1)
    print("NR_IMG after ms: ", nir_multi[:, :, 0])
    # color_img[:, :, 3] = color_img[:, :, 3] / color_img[:, :, 3].max()
    # color_img[:, :, 3] = color_img[:, :, 3] * 255
    multi_img[:, :, 3] = nir_multi[:, :, 0]
    # color_img = color_img / color_img.max() * 255
    # color_img = color_img * 255
    return multi_img

DATA_DIR = Path.cwd().parent.resolve() / "C:/Users/RACL/PycharmProjects/MLforEO"
#test_meta = pd.read_csv(DATA_DIR / "test_metadata.csv")
#display_random_chip(44, test_meta)
ASSETS_DIRECTORY = DATA_DIR / "codeexecution" / "assets"
PREDICTIONS_DIRECTORY = DATA_DIR / "codeexecution" / "predictions"
# model = torch.load( ASSETS_DIRECTORY / "cloud_model.pt")
model = tf.keras.models.load_model('models/cloud_model_rgb.h5')
train_meta = pd.read_csv(DATA_DIR / "test_meta_small.csv")

chipids=train_meta['chip_id']
#for test_img_number in [91, 92, 93, 94, 95, 96, 97, 98, 99]:
fig, ax = plt.subplots(2, 3, figsize=(12, 18))
#test_img_number = 13
test_img_number = 40
#test_img_number = 48 # resultado precioso, máscara demasiado grande
#test_img_number = 28 # this one is good to see how it differentiates between cloud and white
test_img_chip = train_meta.get('chip_id')[test_img_number]
#test_img = rgb(test_img_chip)
test_img = true_color_img(test_img_number, train_meta)
img_rgb = true_color_img(test_img_number, train_meta)
mask = tiff.imread('./test_images/{0}/{1}.tif'.format(test_img_chip, test_img_chip[-4:]))
rgb_lm_prediction = tiff.imread('./codeexecution/predictions_rgb/{}.tif'.format(test_img_chip))
prediction = tiff.imread('./codeexecution/predictions/{}.tif'.format(test_img_chip))
test_img_input = np.expand_dims(test_img, 0)
keras_p = model.predict(test_img_input)
keras_prediction = (keras_p > 0.5).astype(np.uint8)

ax[0, 0].imshow(img_rgb)
#ax[i, 0].imshow(test_img)
ax[0, 0].set_title("RGB Image: " + test_img_chip[-4:], fontsize=16)
ax[0, 1].imshow(mask)
ax[0, 1].set_title("Ground truth", fontsize=16)
ax[0, 2].remove()
ax[1, 1].imshow(rgb_lm_prediction)
ax[1, 1].set_title("RGB U-NET with TL Prediction", fontsize=16)
ax[1, 2].imshow(np.squeeze(prediction))
ax[1, 2].set_title("RGB+NIR U-NET with TL Prediction", fontsize=16)
ax[1, 0].imshow(np.squeeze(keras_prediction))
ax[1, 0].set_title("RGB Raw U-NET Prediction", fontsize=16)

plt.show()

#train_meta = pd.read_csv(DATA_DIR / "train_metadata.csv")
#display_random_chip(43, train_meta)