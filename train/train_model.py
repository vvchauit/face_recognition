import tensorflow as tf
import keras
import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.layers import Input, BatchNormalization, Dropout, Flatten, Dense
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EPOCHS = 200
BATCH_SIZE = 1
DATA_DIR = r'D:\sw\face_rec\data\celeb_vn\celeb_vn_processed'
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
        x_train.append(f_path)
        labels.append(i)
labels = np.array(labels)
y_train = tf.keras.utils.to_categorical(labels, num_classes=num_class, dtype='int32')
del labels

class DataGenerator(Sequence):
    def __init__(self, x_train, y_train, batch_size=32, shuffle=True, is_path=False, image_shape=(160,160, 3)):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x_train = x_train
        self.y_train = y_train
        self.is_path = is_path
        self.image_shape = image_shape
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
        return (X, Y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.x_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_img(self, img_path):
        return cv2.resize(cv2.imread(img_path), (self.image_shape[0], self.image_shape[1]))

    def __data_generation(self, list_IDs_temps):
        if self.is_path:
            X = []
            for ID in list_IDs_temps:
                x = self.__load_img(self.x_train[ID])
                X.append((x/255).astype('float32'))
            X = np.array(X, dtype='float32')
            return X, self.y_train[list_IDs_temps]
        return self.x_train[list_IDs_temps], self.y_train[list_IDs_temps]

print('INFO: Create model')
input_shape = (160, 160, 3)
input = Input(shape=input_shape)
label = Input(shape=(num_class,))
base_model = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
base_model.trainable = False
classifier = base_model(input)
classifier = BatchNormalization(momentum=0.9, epsilon=2e-5)(classifier)
classifier = Dropout(0.4)(classifier)
classifier = Flatten()(classifier)
classifier = Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(classifier)
embedding = BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(classifier)
output = Dense(num_class, activation='softmax', name='predictions')(embedding)
model = Model(input, output)

print('INFO: Compile model')
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['acc'])

cp = keras.callbacks.ModelCheckpoint(os.path.join(RESULTS_DIR, "cp_inceptionResnetV2_softmax_head.h5"), monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min'),

callbacks = [
    cp,
]

model.summary()

train_generator = DataGenerator(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, is_path=True)
history = model.fit_generator(
    train_generator,
    steps_per_epoch = len(train_generator),
    epochs = EPOCHS,
    callbacks = callbacks,
)

model.save(os.path.join(RESULTS_DIR, "final_inceptionResnetV2_softmax_head.h5"))