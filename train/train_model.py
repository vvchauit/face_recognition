import tensorflow as tf
import keras
from keras.metrics import CategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from models.feature_extraction_model.inceptionresnetv2 import get_train_model
from keras.models import Model
from train.arcface_metrics import ArcFace
from keras.layers import Input
import numpy as np
import math


EPOCHS = 200
BATCH_SIZE = 256
DATA_DIR = ''
RESULTS_DIR = 'train/results'

print('INFO: TRAIN MODEL')

print('INFO: Dataset pre_processing')
print('Please wait...')

num_class = len(os.listdir(DATA_DIR))

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode = 'categorical',
    image_size=(160, 160),
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset = "training",
    validation_split = 0.2,
    seed = 23,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode = 'categorical',
    image_size=(160, 160),
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset = "validation",
    validation_split = 0.2,
    seed = 23,
)

train_ds = train_ds.prefetch(buffer_size=BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=BATCH_SIZE)

# data_augmentation = keras.layers.RandomZoom(0.2)
# augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
normalization_layer = keras.layers.Rescaling(1./255)
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

print('INFO: Create model')
model = get_train_model(num_class)

print('INFO: Compile model')
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=[CategoricalAccuracy()])

cp1 = keras.callbacks.ModelCheckpoint(os.path.join(RESULTS_DIR, "check_point_.h5"), monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')

callbacks = [
    cp1,
]

model.summary()

history = model.fit(
    normalized_train_ds,
    epochs = EPOCHS,
    verbose = 1,
    callbacks = callbacks,
    validation_data = normalized_val_ds,
    shuffle=True,
)

model.save(os.path.join(RESULTS_DIR, "final_.h5"))