# references:
#   https://google.github.io/mediapipe/solutions/models.html


import tensorflow as tf
import gdown
import os
import time
import cv2
import numpy as np


# kiểm tra và load model landmark
if not os.path.exists('storage/model/landmark_model'):
    os.makedirs('storage/model/landmark_model')
landmark_model_path = r'storage/model/landmark_model/face_landmark.tflite'
landmark_model_url = 'https://drive.google.com/u/1/uc?id=1mtuMCbn2RjkMdzx94lD88jmWT9tniu-P&export=download'
if os.path.isfile(landmark_model_path) != True:
		print("face_landmark.tflite will be downloaded...")
		gdown.download(url=landmark_model_url, output=landmark_model_path, quiet=False)
interpreter = tf.lite.Interpreter(model_path=landmark_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
landmark_index = output_details[0]['index']
score_index = output_details[1]['index']

height, width = input_shape[1:3]


# summary: xác định tọa độ landmark
# params:
# 	init
# 		pixels: ảnh (array)
# 	return
# 		landmark: 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#       score: khả năng xuất hiện của khuôn mặt trong ảnh
def get_landmark(pixels):
    image = pixels
    # chuẩn hóa ảnh ngõ vào
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    factor = min(height/pixels.shape[0],width/pixels.shape[1])
    interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=interpolation)
    # image_array = ((image_resized-127.5)/127.5).astype('float32')
    image_array = (image_resized/255.0).astype('float32')
    input_data = np.expand_dims(image_array, axis=0)
    # xác định tọa độ landmark
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    landmark = interpreter.get_tensor(landmark_index)[0].reshape(468, 3) / 192
    score = interpreter.get_tensor(score_index)[0]
    return landmark, score