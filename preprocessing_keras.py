from keras_model import unet_model_architecture #, jacard, jacard_loss
import os
import glob
import tifffile as tiff

import numpy as np
from matplotlib import pyplot as plt
import random

import pandas as pd
from pathlib import Path
import tensorflow as tf
import torch
import rasterio
import xarray
import xrspatial.multispectral as ms
from skimage.util import img_as_ubyte, img_as_uint
from imageio.core import image_as_uint
from rasterio.enums import Resampling
from cloud_dataset import CloudDataset
from skimage.io import imread, imshow
from skimage.transform import resize

#%% Import train and mask dataset

DATA_DIR = Path.cwd().parent.resolve() / "C:/Users/RACL/PycharmProjects/MLforEO"
submission_dir = DATA_DIR / "codeexecution_keras"
BANDS = ["B02", "B03", "B04", "B08"]
train_meta = pd.read_csv(DATA_DIR / "train_meta_small.csv")

#def normalize(img):
#    min = img.min()
#    max = img.max()
#    x = 2.0 * (img - min) / (max - min) - 1.0
#    return x
#train_x = dict()
#train_y = dict()
#val_x = dict()
#val_y = dict()


def combine_bands_img(chip_id):
    """Given the path to the directory of Sentinel-2 chip feature images,
    plots the true color image"""
    #color_img = []  # np.zeros((4, 512, 512), dtype=np.uint8)
    # red_img = rasterio.open('./images/{0}/B02.tif'.format(chip_id))
    with rasterio.open('./images/{0}/B02.tif'.format(chip_id)) as ro:
        red_img = ro.read(1)
        # red_img = np.array(red_img, dtype='uint8')
        # red_img = xarray.DataArray(ro.read(1), dims=["y", "x"])
        # red_img = (red_img / 2 ** 8).astype(np.uint8)
        # red_img = np.array(img_as_ubyte(red_img))
        # red_img = np.array(red_img, dtype='float32')
        #red_img_f = (red_img).astype(np.float32)
        #red_img_temp = red_img_f - red_img_f.min()
        #red_img_temp = red_img_temp / red_img_temp.max()
        #red_img_temp = red_img_temp * 255
        #red_img_f = red_img_temp.astype(np.uint8)
        #color_img.append(red_img_f)
        red = xarray.DataArray(ro.read(1), dims=["y", "x"])
    # green_img = rasterio.open('./images/{0}/B03.tif'.format(chip_id))
    with rasterio.open('./images/{0}/B03.tif'.format(chip_id)) as ro:
        green_img = ro.read(1)
        # green_img = (green_img / 2 ** 8).astype(np.uint8)
        # green_img = np.array(img_as_ubyte(green_img))
        # green_img = green_img.astype(np.uint8)
        #green_img_f = (green_img).astype(np.float32)
        #green_img_temp = green_img_f - green_img_f.min()
        #green_img_temp = green_img_temp / green_img_temp.max()
        #green_img_temp = green_img_temp * 255
        #green_img_f = green_img_temp.astype(np.uint8)
        #color_img.append(green_img_f)
        green = xarray.DataArray(ro.read(1), dims=["y", "x"])
    # blue_img = rasterio.open('./images/{0}/B04.tif'.format(chip_id))
    with rasterio.open('./images/{0}/B04.tif'.format(chip_id)) as ro:
        blue_img = ro.read(1)
        # blue_img = (blue_img / 2 ** 8).astype(np.uint8)
        # blue_img = np.array(img_as_ubyte(blue_img))
        # blue_img = blue_img.astype(np.uint8)
        #blue_img_f = (blue_img).astype(np.float32)
        #blue_img_temp = blue_img_f - blue_img_f.min()
        #blue_img_temp = blue_img_temp / blue_img_temp.max()
        #blue_img_temp = blue_img_temp * 255
        #blue_img_f = blue_img_temp.astype(np.uint8)
        #color_img.append(blue_img_f)
        blue = xarray.DataArray(ro.read(1), dims=["y", "x"])
    multi_img = ms.true_color(r=red, g=green, b=blue)
    print("MULTI_IMG: ", multi_img)
    # infra_img = rasterio.open('./images/{0}/B08.tif'.format(chip_id))
    # infra = xarray.DataArray(infra_img.read(1), dims=["y", "x"])
    with rasterio.open('./images/{0}/B08.tif'.format(chip_id)) as ro:
        nr_img = ro.read(1)
        nr = xarray.DataArray(ro.read(1), dims=["y", "x"])
        # nr_img = (nr_img / 2 ** 8).astype(np.uint8)
        # nr_img = np.array(img_as_ubyte(nr_img))
        # nr_img = nr_img.astype(np.uint8)
        #nr_img = (nr_img).astype(np.float32)
        #nr_img_temp = 1 / (1 + np.exp(10 * (0.125 - nr_img)))
        #nr_img_temp = nr_img - nr_img.min()
        #print("NR IMG MAX before: ", nr_img.max())
        #nr_img_temp = nr_img_temp / nr_img_temp.max()
        #nr_img_temp = nr_img_temp * 255
        #nr_img_temp = 1 / (1 + np.exp(10 * (0.125 - nr_img_temp)))
        #nr_img = nr_img_temp.astype(np.uint8)
        #color_img.append(nr_img)
    print("NR IMG: ", nr_img)
    nir_multi = ms.true_color(r=nr, g=green, b=blue)
    #print("NR IMG MAX after: ", nr_img.max())
    # infra = np.array(img_as_ubyte(infra))
    # color_img = np.zeros((512, 512, 4), dtype=np.uint8)
    # color_img[:, :, 0] = red
    # color_img[:, :, 1] = green
    # color_img[:, :, 2] = blue
    # color_img[:, :, 3] = infra
    # color_img = np.stack(color_img,axis=-1)
    #color_img = np.stack(color_img, axis=-1)
    print("NR_IMG after ms: ", nir_multi[:, :, 0])
    #color_img[:, :, 3] = color_img[:, :, 3] / color_img[:, :, 3].max()
    #color_img[:, :, 3] = color_img[:, :, 3] * 255
    multi_img[:, :, 3] = nir_multi[:, :, 0]
    # color_img = color_img / color_img.max() * 255
    # color_img = color_img * 255
    return multi_img

def rgb(chip_id):
    red_img = rasterio.open('./images/{0}/B02.tif'.format(chip_id))
    red = xarray.DataArray(red_img.read(1), dims=["y", "x"])
    green_img = rasterio.open('./images/{0}/B03.tif'.format(chip_id))
    green = xarray.DataArray(green_img.read(1), dims=["y", "x"])
    blue_img = rasterio.open('./images/{0}/B04.tif'.format(chip_id))
    blue = xarray.DataArray(blue_img.read(1), dims=["y", "x"])
    color_img = ms.true_color(r=red, g=green, b=blue)
    return color_img


#random.seed(9)  # set a seed for reproducibility

# put 1/3 of chips into the validation set
#chip_ids = train_meta.chip_id.unique().tolist()
#val_chip_ids = random.sample(chip_ids, round(len(chip_ids) * 0.33))
#val_mask = train_meta.chip_id.isin(val_chip_ids)
#val = train_meta[val_mask].copy().reset_index(drop=True)
#train = train_meta[~val_mask].copy().reset_index(drop=True)

#print(val.shape, train.shape)

# separate features from labels
#feature_cols = ["chip_id"] + [f"{band}_path" for band in BANDS]

#val_x = val[feature_cols].copy()
#val_y = val[["chip_id", "label_path"]].copy()

#train_x = train[feature_cols].copy()
#train_y = train[["chip_id", "label_path"]].copy()

#train_dataset = CloudDataset(
#            x_paths=train_x,
#            bands=BANDS,
#            y_paths=train_y,
#            transforms=None,
#        )
#val_dataset = CloudDataset(
#            x_paths=val_x,
#            bands=BANDS,
#            y_paths=val_y,
#            transforms=None,
#        )

##img_height = 256
##img_width = 256
#img_channels = 4
#print(val_dataset.__getitem__(0))

#train_dataset_chips = dict()
#train_dataset_labels = dict()
#val_dataset_chips = dict()
#val_dataset_labels = dict()
#for i in range(train_dataset.__len__()):
#    train_dataset_chips[i] = train_dataset.__getitem__(i)['chip']
#    train_dataset_labels[i] = train_dataset.__getitem__(i)['label']

#for i in range(val_dataset.__len__()):
#    val_dataset_chips[i] = val_dataset.__getitem__(i)['chip']
#    val_dataset_labels[i] = val_dataset.__getitem__(i)['label']

#print(train_dataset_chips[0])

#print(img_height, img_width, img_channels)

chipids=train_meta['chip_id']
print('Reading images')

IMG_HEIGHT = 512
IMG_WIDTH = 512
N_CHANNELS = 4

val_chip_ids = random.sample(sorted(chipids), round(len(chipids) * 0.33))
train_batch_len = len(chipids)-len(val_chip_ids)
train_x = np.zeros((train_batch_len, IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), dtype=np.uint8)
train_y = np.zeros((train_batch_len, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
val_x = np.zeros((len(val_chip_ids), IMG_HEIGHT, IMG_WIDTH, N_CHANNELS), dtype=np.uint8)
val_y = np.zeros((len(val_chip_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
i_t = 0
i_v = 0
i = 0

for img_id in chipids:
    img_m = combine_bands_img(img_id)
    #img_m = rgb(img_id)
    mask = tiff.imread('./images/{0}/{1}.tif'.format(img_id, img_id[-4:]))
    mask = mask[..., np.newaxis]
    if img_id in val_chip_ids:
        val_x[i_v, :, :, :] = img_m
        val_y[i_v, :, :, :] = mask
        i_v += 1
    else:
        train_x[i_t, :, :, :] = img_m
        train_y[i_t, :, :, :] = mask
        i_t += 1
    # val_mask = train_meta.chip_id.isin(val_chip_ids)
    # val = train_meta[val_mask].copy().reset_index(drop=True)
    # train = train_meta[~val_mask].copy().reset_index(drop=True)
    #print(mask)
    #print(mask)
    #mask = mask.transpose([1, 2, 0]) / 255
    #train_x[i] = img_m
    #train_y[i] = mask

    #train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
    #train_x[i] = img_m[:train_xsz, :, :]
    #train_y[i] = mask[:train_xsz, :]
    #val_x[i] = img_m[train_xsz:, :, :]
    #val_y[i] = mask[train_xsz:, :]
    print(img_id + ' read')
    i+=1
print('Images were read')



#metrics = ['accuracy', jacard]


def get_model():
    return unet_model_architecture(n_classes=1, height=None, width=None, channels=4)


model = get_model()
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
#model.summary()
#print("TRAIN_X: ", train_x)
#print("TRAIN_Y: ", train_y)
submission_assets_dir = submission_dir / "assets"
submission_assets_dir.mkdir(parents=True, exist_ok=True)

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], False)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='./codeexecution_keras'),
        tf.keras.callbacks.ModelCheckpoint('cloud_model.h5', verbose=1, save_best_only=True)]

print("train_x: ", train_x.shape)
print("train_y: ", train_y.shape)
print("val_x: ", val_x.shape)
print("val_y: ", val_y.shape)

history = model.fit(train_x, train_y,
                    batch_size=train_batch_len,
                    validation_split=0.1,
                    verbose=1,
                    epochs=15,
                    validation_data=(val_x, val_y),
                    shuffle=True,
                    callbacks=callbacks)

# %%
"""7"""
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, 'y', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()

#acc = history.history['jacard']
#val_acc = history.history['val_jacard']

#plt.plot(epochs, acc, 'y', label='Training Jaccard')
#plt.plot(epochs, val_acc, 'r', label='Validation Jaccard')
#plt.title('Training and validation Jacard')
#plt.xlabel('Epochs')
#plt.ylabel('Jaccard')
#plt.legend()
#plt.show()
# %%
"""8"""
#y_pred = model.predict(val_x)
#y_pred_argmax = np.argmax(y_pred, axis=3)
#val_y_argmax = np.argmax(val_y, axis=3)
#test_jacard = jacard(val_y, y_pred)
#print(test_jacard)
# %%
"""9"""
fig, ax = plt.subplots(1, 3, figsize=(12, 18))
#for i in range(0, 2):
test_img_number = random.randint(0, len(val_x)-1)
test_img = val_x[test_img_number]
#print("TEST_IMG: ", test_img)
ground_truth = val_y[test_img_number]
print("GROUND_TRUTH: ", ground_truth)
#ground_truth= ground_truth.reshape((128, 512))
test_img_input = np.expand_dims(test_img, 0)
print("Input image for prediction: ", test_img_input, test_img_input.shape)
prediction = model.predict(test_img_input)
print("PREDICTION: ", prediction)
#prediction = prediction.reshape(512, 512)
#print("PREDICTION: ", prediction)
#predicted_img = np.argmax(prediction, axis=3)[, :, :]
predicted_img = np.zeros((512, 512), dtype=np.bool_)
for n in range(prediction.shape[1]):
    for m in range(prediction.shape[2]):
        if prediction[0][n][m] > 0.5:
            predicted_img[n][m] = 1
        else:
            predicted_img[n][m] = 0
print("PREDICTED_IMG: ", predicted_img)

#ax[i, 0].imshow(true_color_img(test_img_number, train_meta))
ax[0].imshow(test_img)
ax[0].set_title("RGB Image", fontsize=16)
ax[1].imshow(ground_truth)
ax[1].set_title("Ground Truth", fontsize=16)
ax[2].imshow(predicted_img)
ax[2].set_title("Prediction", fontsize=16)
#i += i

plt.show()

submission_assets_dir = submission_dir / "assets"
submission_assets_dir.mkdir(parents=True, exist_ok=True)

model_weight_path = submission_assets_dir / "cloud_model.h5"
model.save(model_weight_path)

