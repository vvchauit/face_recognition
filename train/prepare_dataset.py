import os
from utils.faceDetection import face_detector
import cv2
import shutil
from .train_utils import get_aligned_face, remove_face_mask_region


def aligned_dataset(in_dir, out_dir='dataset/aligned'):
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for fld_name in os.listdir(in_dir):
        fld_path = os.path.join(in_dir, fld_name)
        save_fld_path = os.path.join(out_dir, fld_name)
        os.mkdir(save_fld_path)
        for file_name in os.listdir(fld_path):
            file_path = os.path.join(fld_path, file_name)
            if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
                img = cv2.imread(file_path)
                face_locs, face_locs_margin = face_detector(img.copy())
                if len(face_locs) != 1:
                    continue
                aligned_face = get_aligned_face(img.copy(), face_locs[0], face_locs_margin[0])
                save_path = os.path.join(save_fld_path, 'aligned'+file_name)
                cv2.imwrite(save_path, aligned_face)

def masked_dataset(in_dir='dataset/aligned', out_dir='dataset/masked'):
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for fld_name in os.listdir(in_dir):
        fld_path = os.path.join(in_dir, fld_name)
        save_fld_path = os.path.join(out_dir, fld_name)
        os.mkdir(save_fld_path)
        for file_name in os.listdir(fld_path):
            file_path = os.path.join(fld_path, file_name)
            if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
                aligned_face = cv2.imread(file_path)
                masked = remove_face_mask_region(aligned_face.copy())
                save_path = os.path.join(save_fld_path, 'masked'+file_name)
                cv2.imwrite(save_path, masked)


def combine_dataset(dataset1_dir, dataset2_dir, out_dir='dataset/combine'):
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    for in_dir in [dataset1_dir, dataset2_dir]:
        for fld_name in os.listdir(in_dir):
            fld_path = os.path.join(in_dir, fld_name)
            save_fld_path = os.path.join(out_dir, fld_name)
            try:
                os.mkdir(save_fld_path)
            except:
                pass
            for file_name in os.listdir(fld_path):
                file_path = os.path.join(fld_path, file_name)
                if file_path.lower().endswith(('jpg', 'jpeg', 'png')):
                    save_path = os.path.join(save_fld_path, file_name)
                    shutil.copyfile(file_path, save_path)