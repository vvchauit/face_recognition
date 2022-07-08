import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.metrics import CategoricalAccuracy
from keras.preprocessing.image import ImageDataGenerator
import os
import sys
sys.path.append(r'D:\sw\face_rec\face_reg_new_ui')
from models.feature_extraction_model.inceptionresnetv2 import get_train_model


EPOCHS = 200
BATCH_SIZE = 128
DATA_DIR = 'dataset/combine'
RESULTS_DIR = 'train/results'

print('INFO: TRAIN MODEL')

print('INFO: Dataset pre_processing')
print('Please wait...')
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode = 'categorical',
    image_size=(299,299),
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset = "training",
    validation_split = 0.2,
    seed = 23,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    label_mode = 'categorical',
    image_size=(299,299),
    batch_size=BATCH_SIZE,
    shuffle=True,
    subset = "validation",
    validation_split = 0.2,
    seed = 23,
)

num_class = len(os.listdir(DATA_DIR))

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

# initial_learning_rate = 0.001

# def lr_exp_decay(epoch, lr):
#     k = 0.1
#     return initial_learning_rate * math.exp(-k*epoch)

callbacks = [
    # keras.callbacks.LearningRateScheduler(lr_exp_decay, verbose=1),
    keras.callbacks.ModelCheckpoint(os.path.join(RESULTS_DIR, "check_point.h5"), monitor='categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='max'),
]

model.summary()

history = model.fit(
    normalized_train_ds,
    epochs = EPOCHS,
    callbacks = callbacks,
    validation_data = normalized_val_ds,
)

model.save(os.path.join(RESULTS_DIR, "final.h5"))

epochs = [i for i in range(1, len(history.history['loss'])+1)]

plt.figure(1)
plt.plot(epochs, history.history['categorical_accuracy'], color='blue', label="training_accuracy")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig(os.path.join(RESULTS_DIR, "acc.png"), bbox_inches='tight')
plt.show()
plt.figure(2)
plt.plot(epochs, history.history['loss'], color='red', label="training_loss")
plt.legend(loc='best')
plt.title('training')
plt.xlabel('epoch')
plt.savefig(os.path.join(RESULTS_DIR, "loss.png"), bbox_inches='tight')
plt.show()