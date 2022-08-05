# from models.feature_extraction_model.inceptionresnetv2 import get_model

# # tạo model
# model = get_model('models/feature_extraction_model/inceptionresnetv2_512_weights_.h5')

from keras.models import load_model
model = load_model(r'models\feature_extraction_model\new.h5')

# xác định kích thước ngõ vào model
input_shape = model.layers[0].input_shape
if type(input_shape) == list:
	input_shape = input_shape[0][1:3]
else:
	input_shape = input_shape[1:3]
target_size = input_shape


import cv2
import numpy as np


# summary: chuẩn hóa ảnh ngõ vào model
# params:
# 	init
# 		face_pixels: ảnh (array)
# 	return
# 		face_pixels: ảnh đã chuẩn hóa
def img_normalize(face_pixels):
    # thêm pad nếu ảnh không phải hình vuông và resize về kích thước ngõ vào của model (160,160)
	face_pixels = face_pixels.astype('float64')
	if face_pixels.shape[0] == 0 or face_pixels.shape[1] == 0:
		raise ValueError("Detected face shape is ", face_pixels.shape,". Consider to set enforce_detection argument to False.")
	if face_pixels.shape[0] > 0 and face_pixels.shape[1] > 0:
		factor_0 = target_size[0] / face_pixels.shape[0]
		factor_1 = target_size[1] / face_pixels.shape[1]
		factor = min(factor_0, factor_1)
		dsize = (int(face_pixels.shape[1] * factor), int(face_pixels.shape[0] * factor))
		interpolation = cv2.INTER_CUBIC if factor > 1 else cv2.INTER_AREA
		face_pixels = cv2.resize(face_pixels, dsize, interpolation=interpolation)
		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - face_pixels.shape[0]
		diff_1 = target_size[1] - face_pixels.shape[1]
		face_pixels = np.pad(face_pixels, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
	if face_pixels.shape[0:2] != target_size:
		interpolation = cv2.INTER_CUBIC if max(face_pixels.shape[0], face_pixels.shape[1]) <= target_size[0] else cv2.INTER_AREA
		face_pixels = cv2.resize(face_pixels, target_size, interpolation=interpolation)
	# chuẩn hóa ảnh (chuyển thành ảnh blob)
	face_pixels = face_pixels / 255.0
	return face_pixels


from PIL import Image


# summary: chuyển ảnh về chuẩn RGB
# params:
# 	init
# 		image: ảnh (array)
# 	return
# 		ảnh RGB (aray)
def cvt2RGB(image):
    img = image.copy()
    img = Image.fromarray(image)
    img = img.convert('RGB')
    return np.array(img)


# summary: trích xuất đặc trưng từ ảnh
# params:
# 	init
# 		face_pixels: ảnh (array)
# 	return
# 		embedding: đặc trưng trích xuất từ ảnh
def feature_extraction(face_pixels):
	# chuyển ảnh về chuẩn RGB
	image = cvt2RGB(face_pixels)
	# chuẩn hóa ảnh đầu vào
	image = img_normalize(image)
	# chuyển shape ảnh từ (160,160,3) thành (1,160,160,3)
	samples = np.expand_dims(image,axis=0)
	# trích xuất đặc trưng ảnh
	yhat = model.predict(samples)
	# yhat /= np.linalg.norm(yhat, axis=1, keepdims=True)
	embedding = yhat[0]
	return embedding