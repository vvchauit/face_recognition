# references: None


import numpy as np
import cv2


# chia mặt thành các bộ phận khác nhau để nhận diên khi đeo khẩu tranh
# đối với các ảnh là mask (có đeo khẩu trang), ta cần xác định vùng ảnh chứa thông tin khuôn mặt không bị che (mắt, trán) để giữ lại 


HALF_LOW_LEFT = [148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 227, 116, 117, 119, 47, 174, 196, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11 ,12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
HALF_LOW_RIGHT = [377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 447, 345, 346, 277, 399, 419, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11 ,12, 13, 14, 15, 16, 17, 18, 200, 199, 175, 152]
HALF_UP_LEFT = [109, 67, 103, 54, 21, 162, 127, 234, 227, 116, 117, 119, 47, 174, 196, 197, 6, 168, 8, 9, 151, 10]
HALF_UP_RIGHT = [338, 297, 332, 284, 251, 389, 356, 454, 447, 345, 346, 277, 399, 419, 197, 6, 168, 8, 9, 151, 10]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACE_HALF_RIGHT = [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0, 164, 2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10]
FACE_HALF_LEFT = [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 175, 199, 200, 18, 17, 16, 15, 14, 13, 12, 11, 0, 164, 2, 94, 19, 1, 4, 5, 195, 197, 6, 168, 8, 9, 151, 10] 
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]


# summary: chia mặt thành các bộ phận (che phần đeo khẩu trang)
# params:
#   init
#       pixels: ảnh đầu vào (array)
#       landmark: landmark 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#       face_loc: tọa độ mặt (x, y, w, h)
#   return
#       face_parts = [base_img, hide_low_part]
def face_divider(pixels, landmark, face_loc=None):
    if face_loc is None:
        (x,y,w,h) = (0,0,pixels.shape[1],pixels.shape[1])
    else:
        (x,y,w,h) = face_loc
    # pixels=cv2.rectangle(pixels,(x,y), (x+w,y+h),(255, 0, 0), 2)
    # cv2.imshow('x',pixels)
    face_parts = []

    # base image
    base_img = pixels.copy()
    face_parts.append(base_img[y:y+h, x:x+w])

    # delete half low face + low face part
    # loại bỏ phần nửa mặt dưới (phần đeo khẩu trang)
    hide_low_part = pixels.copy()
    low_face_part = pixels.copy()
    stencil = np.zeros(hide_low_part.shape).astype(hide_low_part.dtype)
    points = []
    for i in HALF_LOW_LEFT:
        points.append([landmark[i][0],landmark[i][1]])
    points = np.asarray(points)
    cv2.fillPoly(hide_low_part, [points], [0,0,0])
    cv2.fillPoly(stencil, [points], [255, 255, 255])
    points = []
    for i in HALF_LOW_RIGHT:
        points.append([landmark[i][0],landmark[i][1]])
    points = np.asarray(points)
    cv2.fillPoly(hide_low_part, [points], [0,0,0])
    cv2.fillPoly(stencil, [points], [255, 255, 255])
    face_parts.append(hide_low_part[y:y+h, x:x+w])
    low_face_part = cv2.bitwise_and(low_face_part, stencil)
    face_parts.append(low_face_part[y:y+h, x:x+w])    

    return face_parts