import cv2
import numpy as np
from utils.landmarkDetection import get_landmark
from utils.faceAngle import get_face_angle
from utils.faceDetection import face_detector
from utils.faceDivider import face_divider


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def find_nearest_box(box_list, target_box):
    val = []
    for box in box_list:
        lst = [abs(box_i - target_box_i) for box_i, target_box_i in zip(box, target_box)]
        val.append(sum(lst) / len(lst))
    if val == []:
        return target_box
    nearest_box_idx = np.argmin(val)
    return box_list[nearest_box_idx]


def get_aligned_face(frame, face_loc, face_loc_margin):
    (x,y,w,h) = face_loc_margin
    face = frame.copy()[y:y+h, x:x+w]
    landmark, score = get_landmark(face)
    landmark_ = []
    for point in landmark:
        point_x = int(x+point[0]*face.shape[1])
        point_y = int(y+point[1]*face.shape[0])
        point_z = int(y+point[2]*face.shape[1])
        landmark_.append((point_x,point_y,point_z))
    face_angle = get_face_angle(landmark_)
    rotate_frame = rotate_image(frame.copy(),face_angle[0])
    new_face_locs = face_detector(rotate_frame.copy())
    new_face_loc = find_nearest_box(new_face_locs[1], face_loc_margin)
    (nx,ny,nw,nh) = new_face_loc
    new_face = rotate_frame.copy()[ny:ny+nh, nx:nx+nw]
    return new_face
    

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
