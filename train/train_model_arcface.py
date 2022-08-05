import tensorflow as tf
import keras
import os
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from models.feature_extraction_model.inceptionresnetv2 import get_train_model
from keras.models import Model, load_model
from keras.utils import Sequence
from train.arcface_metrics import ArcFace
from keras.layers import Input
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 200
BATCH_SIZE = 256
DATA_DIR = ''
RESULTS_DIR = 'train/results'

print('INFO: TRAIN MODEL')

print('INFO: Dataset pre_processing')
print('Please wait...')

num_class = len(os.listdir(DATA_DIR))

x_train = []
labels = []
train_classes = []
for i, fld_name in enumerate(os.listdir(DATA_DIR)):
    fld_path = os.path.join(DATA_DIR, fld_name)
    for f_name in os.listdir(fld_path):
        f_path = os.path.join(fld_path, f_name)
        x_train.append(cv2.resize(cv2.imread(f_path), (160, 160))/255.0)
        labels.append(i)
labels = np.array(labels)
y_train = tf.keras.utils.to_categorical(labels, num_classes=num_class, dtype='int32')
del labels
x_train = np.array(x_train, dtype='float32')

class DataGenerator(Sequence):
    def __init__(self, x_train, y_train, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_train = x_train
        self.y_train = y_train
        self.img_indexes = np.arange(len(self.x_train))
        self.on_epoch_end()
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.img_indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temps = [self.img_indexes[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temps)
        return ([X, Y], Y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temps):
        return self.x_train[list_IDs_temps], self.y_train[list_IDs_temps]

print('INFO: Create model')
model = get_train_model(num_class)


inputs = Input(shape=(160, 160, 3))
labels = Input(shape=(num_class,))

base_model = get_train_model(num_class)
extract_model = Model(base_model.inputs, base_model.layers[-2].output)
x = extract_model(inputs)
output = ArcFace(n_classes=num_class)([x, labels])
model = Model([inputs, labels], output)

print('INFO: Compile model')
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['acc'])

cp = keras.callbacks.ModelCheckpoint(os.path.join(RESULTS_DIR, "check_point_lfw.h5"), monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'),

callbacks = [
    cp,
]

model.summary()

train_generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE)
history = model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_generator),
    epochs = EPOCHS,
    callbacks = callbacks,
)

model.save(os.path.join(RESULTS_DIR, "final_lfw.h5"))