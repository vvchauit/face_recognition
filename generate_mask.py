import os
import cv2
from utils.landmarkDetection import get_landmark
from utils.faceDivider import face_divider
import numpy as np

def add_pad(img):
    # read image
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width = int(1.25*old_image_width)
    new_image_height = int(1.25*old_image_height)
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img
    return result

def remove_pad(img):
    result = img.copy()
    h, w, c = img.shape
    pad_w = int(0.25*w)
    pad_h = int(0.25*h)
    result = result[int(pad_h/2):int(pad_h/2)+int(h-pad_h), int(pad_w/2):int(pad_w/2)+int(w-pad_w)]
    return result

def remove_face_mask_region(aligned_face):
    new_landmark, _ = get_landmark(aligned_face)
    new_landmark_ = []
    for point in new_landmark:
        point_x = int(point[0]*aligned_face.shape[1])
        point_y = int(point[1]*aligned_face.shape[0])
        point_z = int(point[2]*aligned_face.shape[1])
        new_landmark_.append((point_x,point_y,point_z))
    face_parts = face_divider(aligned_face, new_landmark_)
    return face_parts[1]

lfw_dir = 'dataset/lfw/lfw'
save_dir = 'dataset/lfw/mask'

for fld_name in os.listdir(lfw_dir):
    fld_path = os.path.join(lfw_dir, fld_name)
    if os.path.isdir(fld_path):
        save_fld_path = os.path.join(save_dir, fld_name)
        os.makedirs(save_fld_path)
        for f_name in os.listdir(fld_path):
            if f_name.lower().endswith('jpg'):
                img_path = os.path.join(fld_path, f_name)
                save_img_path = os.path.join(save_fld_path, f_name)
                img = cv2.imread(img_path)
                pad_img = add_pad(img)
                masked_img = remove_face_mask_region(pad_img)
                save_img = remove_pad(masked_img)
                cv2.imwrite(save_img_path, save_img)

