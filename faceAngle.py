# references: 
#   https://www.geeksforgeeks.org/determine-the-face-tilt-using-opencv-python/
#   paper: https://edusj.mosuljournals.com/article_169071_92fb8de5c14f5011621ad74429f9ba1b.pdf


import numpy as np
import cv2


# tìm các góc quay của khuôn mặt, tham khảo link: https://docs.microsoft.com/en-us/azure/cognitive-services/face/images/headpose.1.jpg


LIPS = [61, 146, 146, 91, 91, 181, 181, 84, 84, 17, 17, 314, 314, 405, 405, 321, 321, 375, 375, 291, 61, 185, 185, 40, 40, 39, 39, 37, 37, 0, 0, 267, 267, 269, 269, 270, 270, 409, 409, 291, 78, 95, 95, 88, 88, 178, 178, 87, 87, 14, 14, 317, 317, 402, 402, 318, 318, 324, 324, 308, 78, 191, 191, 80, 80, 81, 81, 82, 82, 13, 13, 312, 312, 311, 311, 310, 310, 415, 415, 308]
LEFT_EYE = [33, 7, 7, 163, 163, 144, 144, 145, 145, 153, 153, 154, 154, 155, 155, 133,33, 246, 246, 161, 161, 160, 160, 159, 159, 158, 158, 157, 157, 173, 173, 133]
LEFT_EYEBROW = [46, 53, 53, 52, 52, 65, 65, 55, 70, 63, 63, 105, 105, 66, 66, 107]
LEFT_IRIS = [474, 475, 475, 476, 476, 477, 477, 474]
RIGHT_EYE = [263, 249, 249, 390, 390, 373, 373, 374, 374, 380, 380, 381, 381, 382, 382, 362, 263, 466, 466, 388, 388, 387, 387, 386, 386, 385, 385, 384, 384, 398, 398, 362]
RIGHT_EYEBROW = [276, 283, 283, 282, 282, 295, 295, 285, 300, 293, 293, 334, 334, 296, 296, 336]
RIGHT_IRIS = [469, 470, 470, 471, 471, 472, 472, 469]
FACE_OVAL = [10, 338, 338, 297, 297, 332, 332, 284, 284, 251, 251, 389, 389, 356, 356, 454, 454, 323, 323, 361, 361, 288, 288, 397, 397, 365, 365, 379, 379, 378, 378, 400, 400, 377, 377, 152, 152, 148, 148, 176, 176, 149, 149, 150, 150, 136, 136, 172, 172, 58, 58, 132, 132, 93, 93, 234, 234, 127, 127, 162, 162, 21, 21, 54, 54, 103, 103, 67, 67, 109, 109, 10]
NOSE_CENTER_POINT = 5
LEFT_POINT = 234
RIGHT_POINT =  454


# summary: function tính khoảng cách euclidean của 2 điểm với công thức sqrt((x2-x1)^2+(y2-y1)^2)
# params:
#   init
#       point1: (x1, y1) điểm đầu
#       point2: (x2, y2) điểm cuối
#   return
#       sqrt((x2-x1)^2+(y2-y1)^2)
def euclidean_distance(point1, point2):
    return np.sqrt(pow((point2[0]-point1[0]),2)+pow((point2[1]-point1[1]),2))


# summary: tính góc nghiêng (roll angle) của khuôn mặt
# params:
#   init
#       landmark: landmark 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#   return
#       angle: góc nghiêng (độ)
def roll_angle(landmark):
    # lấy tọa độ (x, y) của các điểm landmark của mắt trái
    left_eye_points = []
    for i in list(dict.fromkeys(LEFT_EYE)):
        left_eye_points.append([landmark[i][0],landmark[i][1]])
    left_eye_points = np.asarray(left_eye_points)
    x_left = [p[0] for p in left_eye_points]
    y_left = [p[1] for p in left_eye_points]
    #  tìm tọa độ điểm giữa của mắt trái
    center_left = (int(sum(x_left) / len(left_eye_points)), int(sum(y_left) / len(left_eye_points)))
    # lấy tọa độ (x, y) của các điểm landmark của mắt phải
    right_eye_points = []
    for i in list(dict.fromkeys(RIGHT_EYE)):
        right_eye_points.append([landmark[i][0],landmark[i][1]])
    right_eye_points = np.asarray(right_eye_points)
    x_right = [p[0] for p in right_eye_points]
    y_right = [p[1] for p in right_eye_points]
    # tìm tọa độ điểm giữa mắt phải
    center_right = (int(sum(x_right) / len(right_eye_points)), int(sum(y_right) / len(right_eye_points)))
    # tìm góc lệch giữa đoạn thẳng nối 2 điểm giữa 2 mắt và trục ox
    delta_x = center_left[0] - center_right[0]
    delta_y = center_left[1] - center_right[1]
    angle = np.arctan(delta_y / delta_x)
    angle = (angle * 180) / np.pi
    return angle
    

# summary: tính góc gập (pitch angle) của khuôn mặt
# params:
#   init
#       landmark: landmark 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#   return
#       angle: góc gập (độ)
def pitch_angle(landmark):
    # lấy tọa độ (x, y) của điểm landmark giữa mũi
    nose_center_point = np.asarray((landmark[NOSE_CENTER_POINT][0],landmark[NOSE_CENTER_POINT][1]))
    # lấy tọa độ (x, y) của điểm landmark ngoài cùng bên trái
    left_point = np.asarray((landmark[LEFT_POINT][0],landmark[LEFT_POINT][1]))
    # lấy tọa độ (x, y) của điểm landmark ngoài cùng bên phải
    right_point = np.asarray((landmark[RIGHT_POINT][0],landmark[RIGHT_POINT][1]))
    # tính khoảng cách euclidean của điểm landmark giữa mũi và ngoài cùng bên trái
    left_2_center_dist = euclidean_distance(left_point,nose_center_point)
    # tính khoảng cách euclidean của điểm landmark giữa mũi và ngoài cùng bên phải
    right_2_center_dist = euclidean_distance(right_point,nose_center_point)
    # shortest distance between nose_center_point and line left_point right_point
    # khoảng cách ngắn nhất giữ điểm landmark giữa mũi và đường thẳng nối 2 điểm ngoài cùng
    shortest_dist = np.cross(right_point-left_point, left_point-nose_center_point)/np.linalg.norm(right_point-left_point)
    # tính góc giữa đoạn thẳng nối 2 điểm giữa mũi và ngoài cùng bên trái so với trục ox
    angle1 = np.arcsin(shortest_dist/left_2_center_dist)
    # tính góc giữa đoạn thẳng nối 2 điểm giữa mũi và ngoài cùng bên phải so với trục ox
    angle2 = np.arcsin(shortest_dist/right_2_center_dist)
    # góc nâng hạ bằng trung bình của 2 góc đã tính
    angle = (angle1+angle2)/2
    # chuyển hệ radian sang độ
    angle = (angle * 180) / np.pi
    return angle


# summary: tính góc quay trái phải (yawn angle) của khuôn mặt
# params:
#   init
#       landmark: landmark 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#   return
#       angle: góc quay trái phải (độ)
def yawn_angle(landmark):
    # lấy tọa độ (x, y) của các điểm landmark lông mày trái
    left_eyebrow_points = []
    for i in list(dict.fromkeys(LEFT_EYEBROW)):
        left_eyebrow_points.append([landmark[i][0],landmark[i][1]])
    left_eyebrow_points = np.asarray(left_eyebrow_points)
    x_left = [p[0] for p in left_eyebrow_points]
    y_left = [p[1] for p in left_eyebrow_points]
    # tìm điểm ngoài cùng và trong cùng (theo trục ox) của lông mày trái
    min_x_left_index = np.argmin(x_left)
    max_x_left_index = np.argmax(x_left)
    min_left = left_eyebrow_points[min_x_left_index]
    max_left = left_eyebrow_points[max_x_left_index]
    # tính độ dài lông mày trái trong ảnh
    left_dist = euclidean_distance(max_left,min_left)
    # lấy tọa độ (x, y) của các điểm landmark lông mày phải
    right_eyebrow_points = []
    for i in list(dict.fromkeys(RIGHT_EYEBROW)):
        right_eyebrow_points.append([landmark[i][0],landmark[i][1]])
    right_eyebrow_points = np.asarray(right_eyebrow_points)
    x_right = [p[0] for p in right_eyebrow_points]
    y_right = [p[1] for p in right_eyebrow_points]
    # tìm điểm ngoài cùng và trong cùng (theo trục ox) của lông mày phải
    min_x_right_index = np.argmin(x_right)
    max_x_right_index = np.argmax(x_right)
    min_right = right_eyebrow_points[min_x_right_index]
    max_right = right_eyebrow_points[max_x_right_index]
    # tính độ dài lông mày trái trong phải
    right_dist = euclidean_distance(max_right,min_right)
    # tính góc quay trái phải trong ảnh bằng tương quan tỷ lệ giữa độ dài 2 lông mày
    if left_dist == right_dist:
        angle = 0
        direction = 'straight'
    elif left_dist < right_dist:
        angle = np.arcsin(1 - left_dist/right_dist)
        direction = 'right'
    elif left_dist > right_dist:
        angle = np.arcsin(1 - right_dist/left_dist)
        direction = 'left'
    angle = (angle * 180) / np.pi
    if direction == 'left':
        angle = -angle
    return angle


# summary: trả về các góc của khuôn mặt
# params:
#   init
#       landmark: landmark 468 3D landmarks flattened into a 1D tensor: (x1, y1, z1), (x2, y2, z2), ...
#   return
#       roll_angle_: góc nghiêng (độ)
#       pitch_angle_: góc nghiêng (độ)
#       yawn_angle_: góc nghiêng (độ)
def get_face_angle(landmark):
    roll_angle_ = roll_angle(landmark)
    pitch_angle_ = pitch_angle(landmark)
    yawn_angle_ = yawn_angle(landmark)
    # print(roll_angle_, pitch_angle_, yawn_angle_, yawn_direction)
    return roll_angle_, pitch_angle_, yawn_angle_



