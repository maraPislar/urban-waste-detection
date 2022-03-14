#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:25:31 2022

@author: mara
"""

# imports
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

import pathlib

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('Image Classification Started')

data_dir = "/home/mara/Documents/Year3/project/big_dataset"
data_dir = pathlib.Path(data_dir)

# split data folder in train, val and test images
# import splitfolders
# splitfolders.ratio(data_dir, output="trashnet_split", seed=1337, ratio=(.8, 0.1, 0.1)) 

data_dir = "/home/mara/Documents/Year3/project/urban-waste-detection/data/small_dataset"
results_dir = "/home/mara/Documents/Year3/project/urban-waste-detection/results/"

datagen = ImageDataGenerator(
    featurewise_center=True,
    rescale=1/255,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_dir = data_dir + "/train/"
val_dir = data_dir + "/val/"
test_dir = data_dir + "/test/"

image_size = (255, 255)

train_ds = datagen.flow_from_directory(
    directory=train_dir,
    target_size=image_size,
    class_mode="categorical",
    shuffle=True,
    batch_size=32,
    seed=42
)

val_ds = datagen.flow_from_directory(
    directory=val_dir,
    target_size=image_size,
    class_mode="categorical",
    shuffle=True,
    batch_size=32,
    seed=42
)

test_ds = datagen.flow_from_directory(
    directory=test_dir,
    target_size=image_size,
    class_mode="categorical",
    shuffle=True,
    batch_size=6,
    seed=42
)

from tensorflow.keras import layers

ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150,150,3), classes=6)

for ls in ResNet50_model.layers:
    ls.trainable=True

opt = tf.keras.optimizers.SGD(lr=0.01,momentum=0.7)
resnet50_x = layers.Flatten()(ResNet50_model.output)
resnet50_x = layers.Dense(256,activation='relu')(resnet50_x)
resnet50_x = layers.Dense(6,activation='softmax')(resnet50_x)
resnet50_x_final_model = tf.keras.Model(inputs=ResNet50_model.input, outputs=resnet50_x)
resnet50_x_final_model.compile(loss = 'categorical_crossentropy', optimizer= opt, metrics=['acc'])

number_of_epochs = 5
resnet_filepath = 'resnet50'+'-saved-model-{epoch:02d}-val_acc-{val_acc:.2f}.hdf5'
resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(resnet_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, min_lr=0.000002)
callbacklist = [resnet_checkpoint,resnet_early_stopping,reduce_lr]
resnet50_history = resnet50_x_final_model.fit(train_ds, epochs = number_of_epochs ,validation_data = val_ds,callbacks=callbacklist,verbose=1)


import matplotlib.pyplot as plt

acc = resnet50_history.history['accuracy']
val_acc = resnet50_history.history['val_accuracy']

loss = resnet50_history.history['loss']
val_loss = resnet50_history.history['val_loss']

epochs_range = range(20)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


'''
feature_extractor = ResNet50(weights="imagenet", 
                             input_shape=(image_size[0], image_size[1], 3),
                             include_top=False)

feature_extractor.trainable=False

input_ = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
x = feature_extractor(input_, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_ = tf.keras.layers.Dense(1, activation='softmax')(x)
model = tf.keras.Model(input_, output_)

model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
# training

logger.info('----------------------------------------------------')
logger.info("--------------TRAINING MODEL-----------------------")
logger.info('----------------------------------------------------')

model.fit(train_ds, epochs=5, validation_data=val_ds)

# testing

logger.info('----------------------------------------------------')
logger.info("--------------TESTING MODEL-----------------------")
logger.info('----------------------------------------------------')

STEP_SIZE_TEST=test_ds.n//test_ds.batch_size
test_ds.reset()
pred=model.predict_generator(test_ds, steps=STEP_SIZE_TEST, verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_ds.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_ds.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv(results_dir + "big_results.csv",index=False)
'''