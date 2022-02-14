import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Get the data from Google Drive

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/Urban Waste Localisation and Classification/dataset-original.zip', 'r') as zip_ref:
    zip_ref.extractall('sample_data/trashnet-dataset')

import pathlib
data_dir = "/content/sample_data/trashnet-dataset/dataset-original"
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

cardboard = list(data_dir.glob('cardboard/*'))
glass = list(data_dir.glob('glass/*'))
metal = list(data_dir.glob('metal/*'))
paper = list(data_dir.glob('paper/*'))
plastic = list(data_dir.glob('plastic/*'))
trash = list(data_dir.glob('trash/*'))

classes = [cardboard, glass, metal, paper, plastic, trash]

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(255,255),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(255,255),
    batch_size=32
)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10,10))
for images, labels in train_ds.take(2):
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Testing processing stuff

import skimage.color as io
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

im_size = 255

# Step 1 - Make image gray
gray_im = io.rgb2gray(images[10])
# plt.imshow(im, cmap = 'gray')

# Step 2 - Normalise the image
norm_im = (gray_im - np.min(gray_im)) / (np.max(gray_im) - np.min(gray_im))
plt.imshow(norm_im)

im = norm_im.reshape(-1, im_size, im_size, 1)

# Step 3 - Data augmentation
tester = expand_dims(images[10].numpy(), 0)

# Shifting
datagen = ImageDataGenerator(width_shift_range=[-100,100])
it = datagen.flow(tester, batch_size=1)
fig, im = plt.subplots(nrows=1, ncols=3, figsize=(15,15))
for i in range(3):
  image = next(it)[0].astype('uint8')
  im[i].imshow(image)

# Brightness changes
datagen = ImageDataGenerator(brightness_range=[0.5,2.0])
it = datagen.flow(tester, batch_size=1)
fig, im = plt.subplots(nrows=1, ncols=3, figsize=(15,15))
for i in range(3):
  image = next(it)[0].astype('uint8')
  im[i].imshow(image)

# Rotation
datagen = ImageDataGenerator(rotation_range=30, fill_mode='nearest')
it = datagen.flow(tester, batch_size=1)
fig, im = plt.subplots(nrows=1, ncols=3, figsize=(15,15))
for i in range(3):
  image = next(it)[0].astype('uint8')
  im[i].imshow(image)

# Step 4 - Standardise the data
datagen = ImageDataGenerator(featurewise_center =True,
      featurewise_std_normalization = True)

