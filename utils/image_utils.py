import cv2
import numpy as np


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def add_pad(image, pad_percent):
    img = image.copy()
    h, w = img.shape[:2]
    pad_x = int(w*pad_percent/2)
    pad_y = int(h*pad_percent/2)
    new_img = np.zeros([h+pad_y*2,w+pad_x*2,3],dtype=np.uint8)
    nh, nw = new_img.shape[:2]
    if nh + nw > h + w:
        new_img[pad_y:pad_y+h, pad_x:pad_x+w] = img
    else:
        pad_x = - pad_x
        pad_y = - pad_y
        new_img = img[pad_y:pad_y+nh, pad_x:pad_x+nw]
    return new_img

def get_expanded_loc(loc, pad_percent):
    (x, y, w, h) = loc
    pad_x = int(w*pad_percent/2)
    pad_y = int(h*pad_percent/2)
    nx = x - pad_x
    ny = y - pad_y
    nw = w + 2*pad_x
    nh = h + 2*pad_y
    return (nx,ny,nw,nh)

def find_nearest_box(box_list, target_box):
    val = []
    for box in box_list:
        lst = [abs(box_i - target_box_i) for box_i, target_box_i in zip(box, target_box)]
        val.append(sum(lst) / len(lst))
    if val == []:
        return target_box
    nearest_box_idx = np.argmin(val)
    return box_list[nearest_box_idx]
