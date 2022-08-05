import sys
sys.path.append(r'D:\sw-work\face_recognition\face_reg_new_ui')
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from utils.faceDetection import face_detector
import numpy as np
from typing import Tuple
from utils.featureExtraction import feature_extraction
import time
import os
from utils.landmarkDetection import get_landmark
from utils.faceAngle import get_face_angle
from ui.setting import *
import functools
import random
from utils.faceDivider import face_divider
import datetime, calendar
from utils.maskDetection import mask_detector
from utils.face_geometry import get_metric_landmarks, PCF, canonical_metric_landmarks, procrustes_landmark_basis
from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
import getpass
import sqlite3
import json
import xlsxwriter
from utils.classification import predict_cosine


dataset_path = 'database/database.db'


def darkstyle(root):     
    style = ttk.Style(root)
    root.tk.call('source', 'ui/new_theme/forest-dark.tcl')
    style.theme_use('forest-dark')
    style.configure('Treeview',background=COLOR[0],fieldbackground=COLOR[0],foreground=COLOR[4])
    return style


# class main ui
class MainUI(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        style = darkstyle(self)
        self.resizable(0,0)
        self.iconphoto(False, ImageTk.PhotoImage(file='ui/images/facerecog.png'))
        self.title("Face Recognizer")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.win_w = self.winfo_screenwidth()
        self.win_h = self.winfo_screenheight()
        self.geometry('{}x{}'.format(int(0.75*self.win_w),int(0.75*self.win_h)))
        self.con = sqlite3.connect(dataset_path)
        self.cur = self.con.cursor()
        self.cur.execute('''CREATE TABLE IF NOT EXISTS EMPLOYESS (
            EMP_ID          INTEGER NOT NULL    PRIMARY KEY AUTOINCREMENT,
            EMP_NAME        TEXT    NOT NULL,
            EMP_DEPT_ID     INT     NOT NULL
            ); '''
        )
        self.cur.execute('''CREATE TABLE IF NOT EXISTS EMP_EMBS (
            FACE        TEXT        NOT NULL,
            EMB         TEXT        NOT NULL,
            MASKED_EMB  TEXT        NOT NULL,
            EMP_ID      INTERGER    NOT NULL,
            FOREIGN KEY (EMP_ID)    REFERENCES EMPLOYESS (EMP_ID)
            ); '''
        )
        self.cur.execute('''CREATE TABLE IF NOT EXISTS EMP_ATT (
            STATUS      TEXT        NOT NULL,
            CREATED     TEXT        NOT NULL,
            EMP_ID      INTERGER    NOT NULL,
            FOREIGN KEY (EMP_ID)    REFERENCES EMPLOYESS (EMP_ID)
            ); '''
        )
        self.is_mask_recog = IS_MASK_RECOG
        self.register_mode = tk.StringVar()
        self.register_mode.set('Liveness')
        self.register_mode.trace("w", self.register_mode_changed)
        self.used_users = []
        self.used_ids = []
        self.used_timestamps = []
        self.in_ids = []
        self.out_ids = []
        self.last_check = datetime.datetime.utcnow().strftime('%d-%m-%Y')
        # top frame
        self.container_top_init()
        # bottom frame
        self.container_bottom_init()
        # setting frame 
        self.container_setting_init()
        # camera center frame
        self.container_center_init()
        # option frame
        self.container_option_init()
    
    def register_mode_changed(self, *args):
        if self.register_mode.get() == 'Liveness':
            self.center_frames['RegistrationPage'].enable_loop = True
            self.center_frames['RegistrationPage'].loop()

    def new_day_reset(self):
        now = datetime.datetime.utcnow().strftime('%d-%m-%Y')
        if self.last_check != now:
            self.last_check = now
            self.in_ids = []
            self.out_ids = []

    def container_top_init(self):
        self.container_top = ttk.Frame(self,height=int(self.win_h*0.125))
        self.container_top.pack_propagate(0)
        self.container_top.pack(side=TOP,fill=X)
        self.lb_list = []
        self.face_recog_icon = ImageTk.PhotoImage(Image.open('ui/images/face-id.png').resize((64,64),Image.LANCZOS))
        self.face_regis_icon = ImageTk.PhotoImage(Image.open('ui/images/face-recognition.png').resize((64,64),Image.LANCZOS))
        self.info_icon = ImageTk.PhotoImage(Image.open('ui/images/personal-information.png').resize((64,64),Image.LANCZOS))
        self.view_icon = ImageTk.PhotoImage(Image.open('ui/images/view.png').resize((64,64),Image.LANCZOS))
        self.lb_list.append(tk.Label(self.container_top,text='Recognition'))
        self.lb_list[0]["compound"] = BOTTOM
        self.lb_list[0]["image"]=self.face_recog_icon
        self.lb_list[0].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[0].bind("<Button-1>",self.recognition_clicked)
        self.lb_list.append(tk.Label(self.container_top,text='Registration'))
        self.lb_list[1]["compound"] = BOTTOM
        self.lb_list[1]["image"]=self.face_regis_icon
        self.lb_list[1].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[1].bind("<Button-1>",self.registration_clicked)
        self.lb_list.append(tk.Label(self.container_top,text='View'))
        self.lb_list[2]["compound"] = BOTTOM
        self.lb_list[2]["image"]=self.info_icon
        # self.lb_list[2].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[2].bind("<Button-1>",self.view_clicked)
        self.lb_list.append(tk.Label(self.container_top,text='Information'))
        self.lb_list[3]["compound"] = BOTTOM
        self.lb_list[3]["image"]=self.view_icon
        self.lb_list[3].pack(side=LEFT,fill=BOTH,expand=True)
        self.lb_list[3].bind("<Button-1>",self.setting_clicked)
        for lb in self.lb_list:
            lb.configure(font=NORMAL_FONT,anchor=CENTER,fg=BLACK,highlightbackground=BLUE_GRAY[6],highlightthickness=1)
        self.lb_clicked(0)

    def lb_clicked(self, index):
        for i,lb in enumerate(self.lb_list):
            if i == index:
                lb.configure(borderwidth=1, relief="ridge",bg=TEAL[8])
            else:
                lb.configure(borderwidth=1, relief="flat",bg=TEAL[3])

    def recognition_clicked(self, event):
        self.lb_clicked(0)
        self.show_left_frame('LeftFrame1')
        self.show_right_frame('RightFrame1')
        self.show_center_frame('WebCam')

    def registration_clicked(self, event):
        self.lb_clicked(1)
        self.show_left_frame('LeftFrame2')
        self.show_right_frame('RightFrame2')
        self.show_center_frame('RegistrationPage')

    def view_clicked(self, event):
        self.lb_clicked(2)
        self.show_left_frame('LeftFrame3')
        self.show_right_frame('RightFrame3')
        self.show_center_frame('ViewPage')

    def setting_clicked(self, event):
        self.lb_clicked(3)
        self.show_left_frame('LeftFrame4')
        self.show_right_frame('RightFrame4')
        self.show_center_frame('InfoPage')

    def container_setting_init(self):
        self.container_setting = ttk.Frame(self,width=int(self.win_w*0.125),height=int(self.win_h*0.5))
        self.container_setting.pack_propagate(0)
        self.container_setting.pack(side=LEFT)
        self.left_frames = {}
        for F in (LeftFrame1,LeftFrame2,LeftFrame3,LeftFrame4):
            page_name = F.__name__
            left_frame = F(self.container_setting,self)
            self.left_frames[page_name] = left_frame
            left_frame.configure(bg=COLOR[0])
        self.last_left_frame = left_frame
        self.show_left_frame('LeftFrame1')

    def show_left_frame(self, page_name):
        self.last_left_frame.pack_forget()
        frame = self.left_frames[page_name]
        self.last_left_frame = frame
        frame.pack(fill=BOTH,expand=True)
        frame.tkraise()
    
    def container_center_init(self):
        self.container_center = ttk.Frame(self,width=int(self.win_w*0.5),height=int(self.win_h*0.5))
        self.container_center.pack_propagate(0)
        self.container_center.pack(side=LEFT)
        self.center_frames = {}
        for F in (WebCam,RegistrationPage,ViewPage,InfoPage):
            page_name = F.__name__
            center_frame = F(self.container_center,self)
            self.center_frames[page_name] = center_frame
            center_frame.configure(style='Card',padding=(5,6,7,8))
        self.last_center_frame = center_frame
        self.show_center_frame('WebCam')

    def show_center_frame(self, page_name):
        frame = self.center_frames[page_name]
        if self.last_center_frame != frame:
            try:
                self.last_center_frame.enable_loop = False
            except Exception as e:
                pass
            self.last_center_frame.pack_forget()
            try:
                frame.enable_loop = True
                frame.loop()
            except Exception as e:
                pass
            self.last_center_frame = frame
            frame.pack(fill=BOTH,expand=True)
            frame.tkraise()
        try:
            frame.default()
        except Exception as e:
                pass

    def container_option_init(self):
        self.container_option = ttk.Frame(self,width=int(self.win_w*0.125),height=int(self.win_h*0.5))
        self.container_option.pack_propagate(0)
        self.container_option.pack(side=LEFT)
        self.right_frames = {}
        for F in (RightFrame1,RightFrame2,RightFrame3,RightFrame4):
            page_name = F.__name__
            right_frame = F(self.container_option,self)
            self.right_frames[page_name] = right_frame
            right_frame.configure(bg=COLOR[0])
        self.last_right_frame = right_frame
        self.show_right_frame('RightFrame1')

    def show_right_frame(self, page_name):
        self.last_right_frame.pack_forget()
        frame = self.right_frames[page_name]
        self.last_right_frame = frame
        frame.pack(fill=BOTH,expand=True)
        frame.tkraise()

    def container_bottom_init(self):
        self.container_bottom = tk.Frame(self,height=int(self.win_h*0.125))
        self.container_bottom.configure(bg=TEAL[8],highlightbackground=BLUE_GRAY[6],highlightthickness=1)
        self.container_bottom.pack_propagate(0)
        self.container_bottom.pack(side=BOTTOM,fill=X)
        label = tk.Label(self.container_bottom,font=NORMAL_FONT,bg=BLUE_GRAY[6],fg=COLOR[4],anchor=CENTER)
        label.configure(text=tthd)
        label.pack(fill=BOTH,expand=True)


# class webcam
class WebCam(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.bg_layer = tk.Canvas(self)
        self.bg_layer.pack(anchor=CENTER)
        self.video_source = 0
        # self.video_source = r'C:\Users\trong\Downloads\1.mp4'
        self.vid = cv2.VideoCapture(self.video_source)
        if self.vid is None or not self.vid.isOpened():
            raise ValueError("Unable to open this camera. Select another video source", self.video_source)
        self.enable_loop = False
        self.loop()

    def loop(self):
        if self.enable_loop:
            is_true, frame = self.get_frame()
            if is_true:
                # bbox_layer = self.get_bbox_layer()
                # combine_layer = roi(frame,bbox_layer)
                combine_layer = self.get_bbox_layer()
                self.bg_layer.configure(width=frame.shape[1], height=frame.shape[0])
                self.bg_layer_photo = ImageTk.PhotoImage(image = Image.fromarray(combine_layer))
                self.bg_layer.create_image(frame.shape[1]//2,frame.shape[0]//2,image=self.bg_layer_photo)
            self.after(15, self.loop)

    def get_bbox_layer(self):
        is_true, frame = self.get_frame()
        if is_true:
            faces_loc_list, face_loc_margin_list = face_detector(frame)
            blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
            if faces_loc_list and face_loc_margin_list:
                self.cf_ids = []
                bbox_layer = blank_image.copy()
                for i,(x,y,w,h) in enumerate(face_loc_margin_list):
                    bbox_layer = draw_bbox(bbox_layer,(x,y,w,h), (0,255,0), 2, 10)
                    face_parts, face_angle, layer = get_face(frame,faces_loc_list[i] ,(x,y,w,h))
                    self.master.is_mask_recog = mask_detector(face_parts[0])[0]
                    id_, label, extender, face = self.classifier(face_parts, self.master.is_mask_recog)
                    info = label + ' ' + extender
                    if label != 'Unknown':
                        info += ' %'
                    text_size = 24
                    if (y-text_size>=10):
                        left_corner = (x,y-text_size)
                    else:
                        left_corner = (x,y+h)
                    bbox_layer = cv2_img_add_text(bbox_layer, info, left_corner, (0,255,0))
                    bbox_layer = roi(frame, bbox_layer)
                    # bbox_layer = self.check_in_layer(bbox_layer, id_, label, self.master.is_mask_recog)
                    # if face is not None:
                    #     bbox_layer = self.add_face(bbox_layer, face)
            else:
                bbox_layer = frame
        return bbox_layer

    def check_in_layer(self, layer, id_, label, is_masked):
        h, w, c = layer.shape
        blank_image = np.zeros((h,w,c), np.uint8)  
        ret_layer = layer.copy()
        if id_ is not None:
            blank_image[h-h//3:,:] = (100,100,100)
            ret_layer = roi(ret_layer, blank_image) 
            info = 'SUCCESSFULLY'
            left_corner = (w//2+w//15,h-h//3)
            ret_layer = cv2_img_add_text(ret_layer, info, left_corner, (0,255,0), 30)
            val = 'Face'
            if is_masked:
                val = 'Masked Face'
            info = 'Name: {}\nID: {}\nCheck: {}'.format(label, id_, val)
            left_corner = (w//2+w//15,h-h//3+40)
            ret_layer = cv2_img_add_text(ret_layer, info, left_corner, (255,255,255))
        else:
            blank_image[h-h//3:,:] = (100,100,100)
            ret_layer = roi(ret_layer, blank_image) 
            info = 'PLEASE TRY AGAIN'
            left_corner = (w//2-len(info)*8,h-h//6-16)
            ret_layer = cv2_img_add_text(ret_layer, info, left_corner, (0,0,255), 30)
        return ret_layer

    def add_face(self, img, face):
        h, w, c = img.shape
        if h//3 >= 100:
            y = h-h//3
        else:
            y = h - 100
        if w//2-2//15 >= 100:
            x = w//2-w//15-100
        else:
            x = 0
        img[y:y+100, x:x+100] = face
        return img

    def get_frame(self):
        self.master.new_day_reset()
        if self.vid.isOpened():
            is_true, frame = self.vid.read()
            # frame = cv2.imread('C:/Users/trong/Downloads/trump2.jpg')
            frame = resize_frame(self.master, frame)
            if is_true:
                return (is_true, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (is_true, None)
        else:
            return (is_true, None)

    # function để nhận diện khuôn mặt
    # sử dụng algorithm cosine similarity
    def classifier(self, face_parts, is_mask_recog=False):
        # check dataset
        if len(self.master.cur.execute('''SELECT * FROM EMP_EMBS''').fetchall()) == 0:
            return None, 'Unknown', '', None
        max_prob = 0.0
        probability_list = []
        # nếu có mang khẩu trang thì dùng feature từ ảnh đã loại bỏ vùng đeo khẩu trang
        if is_mask_recog:
            row = 'MASKED_EMB'
            audit_feature = feature_extraction(face_parts[1])
        else:
            row = 'EMB'
            audit_feature = feature_extraction(face_parts[0])
        query = 'SELECT ' + row + ' FROM EMP_EMBS'
        feature_db = []
        for emb in self.master.cur.execute(query).fetchall():
            feature = json2array(emb[0], np.float32)
            feature_db.append(feature)
        max_index, max_prob = predict_cosine(audit_feature, feature_db)
        if max_prob >= THRESHOLD:
            id_ = self.master.cur.execute('''SELECT EMP_ID FROM EMP_EMBS''').fetchall()[max_index][0]
            label = self.master.cur.execute('''SELECT EMP_NAME FROM EMPLOYESS WHERE EMP_ID = ?''', ([id_])).fetchall()[0][0]
            face = cv2.resize(json2array(self.master.cur.execute('''SELECT FACE FROM EMP_EMBS WHERE EMP_ID = ?''', ([id_])).fetchall()[0][0]), (100,100))
            ts = time.time()
            t = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y %H:%M:%S')
            # date = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d')
            # time_ = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            cico_module(self.master, id_, t)
            self.master.used_users.append(label)
            self.master.used_ids.append(id_)
            self.master.used_timestamps.append(t)
            self.cf_ids.append(id_)
            try:
                self.master.right_frames['RightFrame1'].update()
            except Exception as e:
                pass
            extender = '{:.2f}'.format(max_prob*100)
        else:
            id_ = None
            label = 'Unknown'
            extender = ''
            face = None
        return id_, label, extender, face

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


def resize_frame(master, frame):
    if frame.shape[1] > master.win_w*0.5 or frame.shape[0] > master.win_h*0.5:
        scale_x = (master.win_w*0.5)/frame.shape[1]
        scale_y = (master.win_h*0.5)/frame.shape[0]
        scale = min(scale_x, scale_y)
        interpolation = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        frame = cv2.resize(frame, (int(frame.shape[1]*scale),int(frame.shape[0]*scale)), interpolation=interpolation)
    return frame


def roi(lower, upper):
        alpha_u = upper / 255.0
        alpha_l = 1.0 - alpha_u
        if upper.shape == lower.shape:
            return (alpha_u * upper[:, :] + alpha_l * lower[:, :]).astype('uint8')
        else:
            return lower


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, length=10):
    (x,y,w,h) = bbox
    image = cv2.line(image, (x,y),(x+length,y),color,thickness)
    image = cv2.line(image, (x,y),(x,y+length),color,thickness)
    image = cv2.line(image, (x+w-length,y), (x+w,y),color,thickness)
    image = cv2.line(image, (x+w,y),(x+w,y+length),color,thickness)
    image = cv2.line(image, (x,y+h),(x+length,y+h),color,thickness)
    image = cv2.line(image, (x,y+h),(x,y+h-length),color,thickness)
    image = cv2.line(image, (x+w,y+h),(x+w-length,y+h),color,thickness)
    image = cv2.line(image, (x+w,y+h),(x+w,y+h-length),color,thickness)
    return image


def cv2_img_add_text(img, text, left_corner: Tuple[int, int], text_rgb_color=(255, 0, 0), text_size=24, font=FONT, **option):
    pil_img = img
    if isinstance(pil_img, np.ndarray):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_text = ImageFont.truetype(font=font, size=text_size, encoding=option.get('encoding', 'utf-8'))
    draw.text(left_corner, text, text_rgb_color, font=font_text)
    cv2_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    if option.get('replace'):
        img[:] = cv2_img[:]
        return None
    return cv2_img

def user_create(master, emp_name, emp_dept_id=0):
    master.cur.execute('''INSERT INTO EMPLOYESS (EMP_NAME, EMP_DEPT_ID) VALUES (?, ?)''', (emp_name, emp_dept_id))
    master.con.commit()
    emp_id = master.cur.lastrowid
    master.right_frames['RightFrame2'].user_list_frame.reload_user_list()
    master.right_frames['RightFrame3'].reload_user_list()
    return str(emp_id)

def append_dataset(master, face, emb, masked_emb, emp_id):
    face_json = json.dumps(face.tolist())
    emb_json = json.dumps(emb.tolist())
    masked_emb_json = json.dumps(masked_emb.tolist())
    master.cur.execute('''INSERT INTO EMP_EMBS (EMP_ID, FACE, EMB, MASKED_EMB) VALUES (?, ?, ?, ?)''', (emp_id, face_json, emb_json, masked_emb_json))
    master.con.commit()
    master.right_frames['RightFrame2'].user_list_frame.reload_user_list()
    master.right_frames['RightFrame3'].reload_user_list()

def empty_user(master, emp_id):
    flag = False
    for obj in master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall():
        if emp_id == obj[0] and flag == False:
            flag = True
    if flag == True:
        master.cur.execute('''DELETE FROM EMP_EMBS WHERE EMP_ID=?''', ([emp_id]))
    master.con.commit()

def user_remove(master, emp_id):
    flag = False
    for obj in master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall():
        if emp_id == obj[0] and flag == False:
            flag = True
    if flag == True:
        empty_user(master, emp_id)
        empty_cico(master, emp_id)
        master.cur.execute('''DELETE FROM EMPLOYESS WHERE EMP_ID=?''', ([emp_id]))
        master.con.commit()
    master.right_frames['RightFrame2'].user_list_frame.reload_user_list()
    master.right_frames['RightFrame3'].reload_user_list()

def empty_cico(master, emp_id):
    flag = False
    for obj in master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall():
        if emp_id == obj[0] and flag == False:
            flag = True
    if flag == True:
        master.cur.execute('''DELETE FROM EMP_ATT WHERE EMP_ID=?''', ([emp_id]))
    master.con.commit()

def cico_module(master, emp_id, created):
    if len(master.cur.execute('''SELECT * FROM EMP_ATT WHERE CREATED LIKE ? AND EMP_ID = ? AND STATUS = ?''', (master.last_check + '%', emp_id, 'check-in')).fetchall()) == 0:
        master.cur.execute('''INSERT INTO EMP_ATT (EMP_ID, STATUS, CREATED) VALUES (?, ?, ?)''', (emp_id, 'check-in', created))
    else:
        if len(master.cur.execute('''SELECT * FROM EMP_ATT WHERE CREATED LIKE ? AND EMP_ID = ? AND STATUS = ?''', (master.last_check + '%', emp_id, 'check-out')).fetchall()) == 0:
            master.cur.execute('''INSERT INTO EMP_ATT (EMP_ID, STATUS, CREATED) VALUES (?, ?, ?)''', (emp_id, 'check-out', created))
        else:
            master.cur.execute('''UPDATE EMP_ATT SET CREATED = ? WHERE CREATED LIKE ? AND EMP_ID = ? AND STATUS = ?''', (created, master.last_check + '%', emp_id, 'check-out'))
    master.con.commit()


def export_cico(master, by='month', sel='this', out_path='cico/cico.xlsx'):
    workbook = xlsxwriter.Workbook(out_path)
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    merge_format = workbook.add_format({'align': 'center', 'bold': True})
    today = datetime.datetime.utcnow().strftime('%d-%m-%Y')
    d, m, y = today.split('-') 
    if by == 'month':
        if sel in ['last', 'this']:
            if sel == 'last':
                m = int(m) - 1
                if m <= 0: 
                    m = 12
                    y = int(y) - 1
            d = int(d)
            m = int(m)
            y = int(y)
            num_days = calendar.monthrange(y, m)[1]
            days = [datetime.date(y, m, d) for d in range(1, num_days+1)]
            for i, lb_id in enumerate(master.cur.execute('''SELECT DISTINCT EMP_ID FROM EMPLOYESS''').fetchall()):
                val = (i*(num_days+5+3)+1)
                worksheet.merge_range('A{}:D{}'.format(val, val), 'CHECK-IN CHECK-OUT', merge_format)
                lb_id = lb_id[0]
                label = master.cur.execute('''SELECT EMP_NAME FROM EMPLOYESS WHERE EMP_ID = ?''', ([lb_id])).fetchall()[0][0]
                worksheet.merge_range('A{}:D{}'.format(val+1, val+1), 'Label id: {}     Label: {}'.format(lb_id, label), bold)
                worksheet.merge_range('A{}:D{}'.format(val+2, val+2), 'Details'.format(lb_id, label))
                worksheet.write('A{}'.format(val+3), 'Day', bold)
                worksheet.write('B{}'.format(val+3), 'Check-in', bold)
                worksheet.write('C{}'.format(val+3), 'Check-out', bold)
                worksheet.write('D{}'.format(val+3), 'Note', bold)
                for j, day in enumerate(days):
                    day_str = day.strftime('%d-%m-%Y')
                    ci = master.cur.execute('''SELECT CREATED FROM EMP_ATT WHERE EMP_ID = ? AND STATUS = ? AND CREATED LIKE ?''', ([lb_id], 'check-in',day_str+'%')).fetchall()
                    if len(ci) == 1:
                        ci = ci[0][0].split(' ')[1]
                    else:
                        ci = ''
                    co = master.cur.execute('''SELECT CREATED FROM EMP_ATT WHERE EMP_ID = ? AND STATUS = ? AND CREATED LIKE ?''', ([lb_id], 'check-out', day_str+'%')).fetchall()
                    if len(co) == 1:
                        co = co[0][0].split(' ')[1]
                    else:
                        co = ''
                    worksheet.write('A{}'.format(val+4+j), day_str)
                    worksheet.write('B{}'.format(val+4+j), ci)
                    worksheet.write('C{}'.format(val+4+j), co)
                    worksheet.write('D{}'.format(val+4+j), '')
    workbook.close()


# class registration page
class RegistrationPage(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.webcam_frame = master.center_frames['WebCam']
        self.frames = []
        for i in range(4):
            self.frames.append(ttk.Frame(self))
        # info_frame
        info_lb = ttk.Label(self.frames[0],font=NORMAL_FONT)
        info_lb.pack(fill=BOTH,expand=True)
        info_lb['text']='Please click Add new user to create new user\nor click to user name to change user data'
        self.frames[0].pack(expand=True)
        # add_user_frame
        ttk.Label(self.frames[1],text='Add new user',font=BOLD_FONT,anchor=CENTER).pack(side=TOP,fill=BOTH,pady=20)
        user_name_frame = ttk.Frame(self.frames[1])
        user_name_frame.pack(side=TOP,fill=BOTH,expand=True)
        ttk.Label(user_name_frame,text='User name',font=BOLD_FONT).pack(side=LEFT,fill=BOTH,pady=20,padx=15)
        self.user_name_var = tk.StringVar()
        user_name_entry = ttk.Entry(user_name_frame, textvariable=self.user_name_var)
        user_name_entry.pack(side=LEFT,fill=BOTH,pady=20)
        button_frame = ttk.Frame(self.frames[1])
        button_frame.pack(side=TOP,fill=BOTH,expand=True)
        self.ok_btn = ttk.Button(button_frame,text='Ok',command=self.ok_clicked)
        self.ok_btn.pack(side=LEFT,fill=BOTH,pady=20)
        self.cancel_btn = ttk.Button(button_frame,text='Cancel',command=self.cancel_clicked)
        self.cancel_btn.pack(side=RIGHT,fill=BOTH,pady=20)
        # camera_frame
        self.bg_layer = tk.Canvas(self.frames[2])
        self.bg_layer.pack(anchor=CENTER)
        # add_img_frame
        self.upload_icon = ImageTk.PhotoImage(Image.open('ui/images/upload.png'))
        self.user_lb = ttk.Label(self.frames[3],font=BOLD_FONT)
        self.user_lb.pack(side=TOP,expand=True,anchor='s')
        self.browse_lb = tk.Label(self.frames[3],text='Or browse for directory',bg='Green',fg=WHITE,font=BOLD_FONT)
        self.browse_lb.bind('<Button-1>', self.browse_files)
        self.browse_lb.pack(side=TOP,expand=True,anchor='n')
        # 
        self.process_popup = ProcessPopup(self)
        self.new_user_faces = []
        self.face_parts = []
        self.labels = []
        self.pitchs = ['Center','Up','Down']
        self.yawns = ['Straight','Left','Right']
        for pitch in self.pitchs:
            for yawn in self.yawns:
                self.labels.append(pitch+'_'+yawn)
        for i in range(9):
            self.new_user_faces.append(None)
            self.face_parts.append(None)
        self.is_changed = False
        self.is_detected = False
        self.enable_loop = False
        self.enable_get_face = False
        self.quick_done = False
        self.choose_frame(0)

    def choose_frame(self, index):
        for i, frame in enumerate(self.frames):
            if i == index:
                frame.pack(expand=True)
                if index == 3:
                    self.master.left_frames['LeftFrame2'].chosen_lb(2)
                else:
                    self.master.left_frames['LeftFrame2'].chosen_lb(index)
            else:
                frame.pack_forget()     

    def add_new_user_clicked(self):
        self.user_name_var.set('')
        self.choose_frame(1)

    def choose_user_clicked(self, id_, label):
        self.username = label
        self.id = id_
        if self.master.register_mode.get() == 'Liveness':
            self.master.right_frames['RightFrame2'].user_list_frame.pack_forget()
            self.master.right_frames['RightFrame2'].register_status_frame.pack(fill=BOTH,expand=True)
            self.master.left_frames['LeftFrame2'].done_btn.pack(side=BOTTOM,fill=X,ipady=10)
            self.choose_frame(2)
            for i in range(9):
                self.new_user_faces[i] = None
                self.face_parts[i] = None
            self.enable_get_face = True
            self.t_start = time.process_time()
            self.t1_start = time.process_time()
        elif self.master.register_mode.get() == 'Photo':
            self.user_lb['text'] = 'Name: {} (id: {})'.format(self.username, self.id)
            self.choose_frame(3)
    
    def browse_files(self, event):
        file_path_list  = askopenfilenames(initialdir='C:/Users/%s/Desktop'%getpass.getuser(),filetypes=[('Image Files','*jpeg;*jpg;*png'),("All files","*.*")])
        images = []
        for file_path in file_path_list:
            if file_path.lower().endswith(('.png','.jpg','.jpeg')):
                images.append(resize_frame(self.master, cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)))
        self.process_popup.show_popup()
        if images:
            for image in images:
                first_flag = True
                faces_loc_list, faces_loc_margin_list = face_detector(image)
                if faces_loc_list and faces_loc_margin_list:
                    if first_flag:
                        empty_user(self.master, self.id)
                        first_flag == False
                    face_parts, face_angle, layer = get_face(image,faces_loc_list[0] ,faces_loc_margin_list[0])
                    feature_masked = []
                    for i,face_part in enumerate(face_parts):
                        feature_masked.append(feature_extraction(face_part))
                    append_dataset(self.master, face_parts[0], feature_masked[0], feature_masked[1], self.id)
            self.process_popup.hide_popup()
            self.default()
    
    def ok_clicked(self):
        self.username = self.user_name_var.get()
        if self.username == '':
            messagebox.showwarning('Warning','User name cannot be empty!')
            self.user_name_var.set('')
        elif self.username == 'None':
            messagebox.showwarning('Warning','User name cannot be None!')
            self.user_name_var.set('')
        else:
            self.id = user_create(self.master, self.username)
            self.choose_user_clicked(self.id, self.username)
            
    def cancel_clicked(self):
        self.choose_frame(0)
        self.username = ''
        for i in range(9):
            self.new_user_faces[i] = None
            self.face_parts[i] = None
        self.enable_get_face = False        

    def default(self):
        self.master.left_frames['LeftFrame2'].chosen_lb(0)
        self.username = ''
        self.user_name_var.set('')
        self.choose_frame(0)
        self.master.right_frames['RightFrame2'].register_status_frame.pack_forget()
        self.master.right_frames['RightFrame2'].user_list_frame.pack(fill=BOTH,expand=True)
        self.master.left_frames['LeftFrame2'].done_btn.pack_forget()
        for i in range(9):
            self.new_user_faces[i] = None
            self.face_parts[i] = None
            self.master.right_frames['RightFrame2'].register_status_frame.status[i].configure(text='...')
        self.is_changed = False
        self.is_detected = False
        self.enable_get_face = False
        self.quick_done = False
    
    def loop(self):
        if self.master.register_mode.get() == 'Photo':
            self.enable_loop = False
        if self.enable_loop:
            is_true, frame = self.webcam_frame.get_frame()
            if is_true:
                if self.enable_get_face:
                    bbox_layer, bbox_frame, bbox_location = self.get_bbox_layer(frame)
                    combine_layer = roi(frame,bbox_layer)
                    faces_loc_list, faces_loc_margin_list = face_detector(bbox_frame)
                    if faces_loc_list and faces_loc_margin_list:
                        self.is_detected = True
                        face_parts, face_angle, layer = get_face(bbox_frame,faces_loc_list[0] ,faces_loc_margin_list[0], True, True)
                        (x,y,w,h) = bbox_location
                        croped_combine_layer = roi(combine_layer[y:y+h,x:x+w],layer)
                        combine_layer[y:y+h,x:x+w] = croped_combine_layer
                        for i,label in enumerate(self.labels):
                            if self.check_face_angle(face_angle) == label:
                                if self.new_user_faces[i] is None and not mask_detector(face_parts[0])[0]:
                                    self.new_user_faces[i] = face_parts[0]
                                    self.face_parts[i] = face_parts
                                    self.is_changed = True
                    else:
                        self.is_detected = False
                    if self.is_detected:
                        self.t_start = time.process_time()
                    self.timer = time.process_time() - self.t_start
                    if self.is_changed:
                            self.is_changed = False
                            self.t1_start = time.process_time()
                    if self.timer >= INSTRUCTOR_TIME:
                        instructor_layer = self.instructor_layer(frame, 9, (255,255,0), 30)
                        combine_layer = roi(combine_layer,instructor_layer)
                    else:
                        self.timer1 = time.process_time() - self.t1_start
                        if self.timer1 >= INSTRUCTOR_TIME:
                            instructor_index = [i for i,f in enumerate(self.new_user_faces) if  f is None]
                            instructor_layer = self.instructor_layer(frame, instructor_index[0], (255,255,0), 30)
                            combine_layer = roi(combine_layer,instructor_layer)
                    ct = 0
                    for i,new_user_face in enumerate(self.new_user_faces):
                        if new_user_face is None:
                            self.master.right_frames['RightFrame2'].register_status_frame.status[i].configure(text='...')
                        else:
                            ct += 1
                            self.master.right_frames['RightFrame2'].register_status_frame.status[i].configure(text='ok')
                    progress = ct/9*100
                    pgbar_layer = self.progress_bar_layer(frame, progress)
                    combine_layer = roi(combine_layer,pgbar_layer)
                    self.bg_layer.configure(width=frame.shape[1], height=frame.shape[0])
                    self.bg_layer_photo = ImageTk.PhotoImage(image = Image.fromarray(combine_layer))
                    self.bg_layer.create_image(frame.shape[1]//2,frame.shape[0]//2,image=self.bg_layer_photo)
                    if ct == 9 or self.quick_done:
                        try:
                            if ct > 0:
                                self.process_popup.show_popup()
                                self.quick_done = False
                                empty_user(self.master, self.id)
                                for i,new_user_face in enumerate(self.new_user_faces):
                                    feature_masked = []
                                    if new_user_face is not None:
                                        for face_part in self.face_parts[i]:
                                            feature_masked.append(feature_extraction(face_part))
                                        append_dataset(self.master, new_user_face, feature_masked[0], feature_masked[1], self.id)
                                self.process_popup.hide_popup()
                        except Exception as e:
                            print(e)
                            messagebox.showerror(title='Register Error', message='Some error during process.',)
                        self.default()
            self.after(15, self.loop)

    def get_bbox_layer(self, frame, bbox_size = (R*2,R*2)):
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        center_x = frame.shape[1]/2
        center_y = frame.shape[0]/2
        w,h = bbox_size
        x = int(center_x - w/2)
        y = int(center_y - h/2)
        if bbox_size[0] >= frame.shape[0] or bbox_size[1] >= frame.shape[1] or bbox_size < (150,150):
            return blank_image, frame.copy(), (0,0,frame.shape[1],frame.shape[0])
        bbox_layer = cv2.rectangle(blank_image, (x,y), (x+w,y+h), (0,0,0), 2)
        bbox_frame = frame.copy()[y:y+h,x:x+w]
        # return bbox_layer, frame, (0,0,frame.shape[1],frame.shape[0])
        return bbox_layer, bbox_frame, (x,y,w,h)

    def check_face_angle(self, face_angle):
        pitch = ''
        yawn = ''
        if -5.0 <= face_angle[1] <=15.0:
            pitch = self.pitchs[0]
        elif face_angle[1] > 15.0:
            pitch = self.pitchs[1]
        elif face_angle[1] < -15.0:
            pitch = self.pitchs[2]
        else:
            if face_angle[1] > 0:
                pitch = 'Slightly'+self.pitchs[1]
            else:
                pitch = 'Slightly'+self.pitchs[2]
        if -10.0 <= face_angle[2] <= 10.0:
            yawn = self.yawns[0]
        elif face_angle[2] > 20.0:
            yawn = self.yawns[1]
        elif face_angle[2] < -20.0:
            yawn = self.yawns[2]
        else:
            if face_angle[1] > 0:
                yawn = 'Slightly'+self.yawns[1]
            else:
                yawn = 'Slightly'+self.yawns[2]
        return pitch+'_'+yawn
    
    def progress_bar_layer(self, frame, progress, r=R, lenght=L):
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        return_layer = blank_image.copy()
        center_x = round(frame.shape[1]/2)
        center_y = round(frame.shape[0]/2)
        center = (center_x, center_y)
        b_circle_arc_points = calculate_circle_arc_points(center, r, N)
        m_circle_arc_points = calculate_circle_arc_points(center, round(r-lenght/2), N)
        s_circle_arc_points = calculate_circle_arc_points(center, r-lenght, N)
        c_point = round(progress*N/100)
        for i in range(0,N):
            if i < c_point:
                cv2.line(return_layer, (s_circle_arc_points[i]), (b_circle_arc_points[i]), (100, 160, 110), 2)
            else:
                cv2.line(return_layer, (s_circle_arc_points[i]), (m_circle_arc_points[i]), (200, 200, 200), 1)
        return return_layer

    def instructor_layer(self, frame, instructor_index, color=(0,0,255), text_size=24):
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
        return_layer = blank_image.copy()
        font = ImageFont.truetype(FONT, text_size)
        text = INSTRUCTOR[instructor_index]
        text_width = 0
        text_height = text_size
        for c in text:
            text_width += font.getsize(c)[0]
        text_x = (frame.shape[1] - text_width) // 2
        text_y = (frame.shape[0] - text_height) // 2
        return_layer = cv2_img_add_text(return_layer, text, (text_x,text_y), color, text_size)
        return return_layer


def face_axis_layer(frame, landmarks_,draw_line=False, draw_arc=True):
    frame_height, frame_width, channels = frame.shape
    points_idx = [33,263,61,291,199]
    points_idx = points_idx + [key for (key,val) in procrustes_landmark_basis]
    points_idx = list(set(points_idx))
    points_idx.sort()
    focal_length = frame_width
    center = (frame_width/2, frame_height/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeff = np.zeros((4, 1))
    pcf = PCF(near=1,far=10000,frame_height=frame_height,frame_width=frame_width,fy=camera_matrix[1,1])
    blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
    return_layer = blank_image.copy()
    landmarks = np.array([(lm[0],lm[1],lm[2]) for lm in landmarks_])
    landmarks = landmarks / np.array([frame_width, frame_height, frame_width])[None,:]
    landmarks = landmarks.T
    metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)
    model_points = metric_landmarks[0:3, points_idx].T
    image_points = landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None,:]
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeff, flags=cv2.SOLVEPNP_ITERATIVE)
    # _, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeff)
    (p2, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 15.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeff)
    (p3, jacobian) = cv2.projectPoints(np.array([(0.0, 15.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeff)
    (p4, jacobian) = cv2.projectPoints(np.array([(15.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeff)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(p2[0][0][0]), int(p2[0][0][1]))
    p3 = (int(p3[0][0][0]), int(p3[0][0][1]))
    p4 = (int(p4[0][0][0]), int(p4[0][0][1]))
    # draw face axis lines
    if draw_line:
        return_layer = cv2.line(return_layer, p1, p4, (0,0,255), 2)
        return_layer = cv2.line(return_layer, p1, p3, (0,255,0), 2)
        return_layer = cv2.line(return_layer, p1, p2, (255,0,0), 2)
    # draw face axis arcs
    if draw_arc:
        c_frame = (round(frame_width/2), round(frame_height/2))
        left_point = (round(c_frame[0]-R+L/2),c_frame[1])
        right_point = (round(c_frame[0]+R-L/2),c_frame[1])
        top_point = (c_frame[0],round(c_frame[1]-R+L/2))
        bottom_point = (c_frame[0],round(c_frame[1]+R-L/2))
        nose_center_point = np.asarray((landmarks_[5][0],landmarks_[5][1]))
        sc_axis = abs(nose_center_point[1]-c_frame[1])
        axes_x = (R-L,sc_axis)
        f_axis = abs(nose_center_point[0]-c_frame[0])
        axes_y = (f_axis,R-L)
        if nose_center_point[1] < c_frame[1]:
            return_layer = cv2.ellipse(return_layer, c_frame, axes_x, 0., 180, 360, (0,0,255))
        elif nose_center_point[1] > c_frame[1]:
            return_layer = cv2.ellipse(return_layer, c_frame, axes_x, 0., 0, 180, (0,0,255))
        else:
            return_layer = cv2.line(return_layer, left_point, right_point, (0,0,255))
        if nose_center_point[0] > c_frame[0]:
            return_layer = cv2.ellipse(return_layer, c_frame, axes_y, 0., -90, 90, (0,0,255))
        elif nose_center_point[0] < c_frame[0]:
            return_layer = cv2.ellipse(return_layer, c_frame, axes_y, 0., 90, 270, (0,0,255))
        else:
            return_layer = cv2.line(return_layer, top_point, bottom_point, (0,0,255))
    return return_layer


def calculate_circle_arc_points(center_point, r, n=100):
    (c_x, c_y) = center_point
    pi = np.pi
    points = [None]*n
    for i in range(0,n):
        point_x = np.cos(2*pi/n*i-pi/2)*r
        point_y = np.sin(2*pi/n*i-pi/2)*r
        points[i] = (round(point_x+c_x), round(point_y+c_y))
    return points


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def euclidean_distance(point1, point2):
    return np.sqrt(pow((point2[0]-point1[0]),2)+pow((point2[1]-point1[1]),2))


def get_face(frame,face_location,face_location_margin,get_bbox_layer=False,get_axis_layer=False):
    (x,y,w,h) = face_location_margin
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
    # 
    new_faces_loc, new_faces_loc_margin = face_detector(rotate_frame.copy())
    new_face_loc_margin = find_nearest_box(new_faces_loc_margin, face_location_margin)
    new_face_loc = find_nearest_box(new_faces_loc, face_location)
    (nx,ny,nw,nh) = new_face_loc_margin
    new_face = rotate_frame.copy()[ny:ny+nh, nx:nx+nw]
    new_landmark, _ = get_landmark(new_face)
    new_landmark_ = []
    for point in new_landmark:
        point_x = int(x+point[0]*new_face.shape[1])
        point_y = int(y+point[1]*new_face.shape[0])
        point_z = int(y+point[2]*new_face.shape[1])
        new_landmark_.append((point_x,point_y,point_z))
    face_parts = face_divider(rotate_frame, new_landmark_, new_face_loc)
    blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)
    return_layer = blank_image.copy()
    if get_bbox_layer:
        return_layer = draw_bbox(return_layer,face_location_margin)
    if get_axis_layer:
        axis_layer = face_axis_layer(frame, landmark_)
        return_layer = roi(return_layer,axis_layer)
    return face_parts, face_angle, return_layer


# def find_nearest_box(box_list, target_box):
#     (tx, ty, tw, th) = target_box
#     # target box center point
#     tbcp = (int(tx+tw/2),int(ty+th/2))
#     dists = []
#     for box in box_list:
#         (x, y, w, h) = box
#         bcp = (int(x+w/2),int(y+h/2))
#         dists.append(euclidean_distance(tbcp, bcp))
#     if dists == []:
#         return target_box
#     nearest_box_indx = np.argmin(dists)
#     return box_list[nearest_box_indx]


def find_nearest_box(box_list, target_box):
    val = []
    for box in box_list:
        lst = [abs(box_i - target_box_i) for box_i, target_box_i in zip(box, target_box)]
        val.append(sum(lst) / len(lst))
    if val == []:
        return target_box
    nearest_box_idx = np.argmin(val)
    return box_list[nearest_box_idx]


# class view page
class ViewPage(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.frames = []
        # frame 0
        self.frames.append(ttk.Frame(self))
        # frame 0 instructor label
        instructor_lb = 'View'
        ttk.Label(self.frames[0],text=instructor_lb,font=BOLD_FONT,anchor=CENTER).pack(fill=BOTH,expand=True)
        # frame 1
        self.frames.append(ttk.Frame(self))
        self.scrollbar = ttk.Scrollbar(self.frames[1],orient=VERTICAL)
        self.canvas = tk.Canvas(self.frames[1],yscrollcommand=self.scrollbar.set,highlightthickness=0,bg=COLOR[0])
        self.canvas.pack(fill=BOTH,expand=True)
        self.scrollbar.config(command=self.canvas.yview)
        self.frame = tk.Frame(self.canvas,bg=COLOR[0])
        self.frame.bind("<Configure>", self.update_scroll_region)
        self.scrollbar.pack(side=RIGHT,fill=Y)
        self.canvas.pack(side=LEFT,fill=BOTH,expand=True)
        self.canvas.create_window((0,0),window=self.frame,anchor="nw",tags="self.frame")
        self.labels = []

        # show start frame (0)
        self.show_frame(0)

    def update_scroll_region(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

    def show_frame(self, index):
        for frame in self.frames:
            frame.pack_forget()
        self.frames[index].pack(fill=BOTH,expand=True)

    def show_user(self, id_, label):     
        for widget in self.frame.winfo_children():
            widget.destroy()
        indices = [i for i, x in enumerate(self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall()) if x == id_]
        for j, i in enumerate(indices):
            image = ImageTk.PhotoImage(Image.fromarray(json2array(self.master.cur.execute('''SELECT FACE FROM EMP_EMBS WHERE EMP_ID=?''', ([id_])).fetchall()[i][0])))
            self.labels.append(tk.Label(self.frame,image=image,))
            self.labels[j].pack(fill=X,side=TOP)
        self.show_frame(1)


class InfoPage(ttk.Frame):
    def __init__(self,container,master):
        ttk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        ttk.Label(self,text='Infomation',font=BOLD_FONT,anchor=CENTER).pack(fill=X,ipady=10)
        ttk.Label(self,text='Version: 0.1',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Date: 2022-04-14 11:00:00 AM',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Code IDE: Visual Code Studio v1.66.2',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Machine Learning Back End: Tensorflow v2.8.0',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Programming Language: Python 3.9.',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Python GUI Library: Tkinter v0.0.1',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Face Detection Model Architecture: DNN',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Face Landmark Detection Model Architecture: MobileNetV2-like with customized blocks',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)
        ttk.Label(self,text='Face Feature Extraction Model Architecture: InceptionResnet v2',font=NORMAL_FONT,anchor=W).pack(fill=X,ipady=10)


class LeftFrame1(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        tk.Label(self,text='Export cico data',font=BOLD_FONT,anchor=W,bg=COLOR[0],fg=COLOR[4]).pack()
        self.export_btn = tk.Button(self, text='Export', command=lambda:export_cico(master))
        self.export_btn.pack()


class LeftFrame2(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.lb_list = []
        self.lb_list.append(tk.Label(self,text='Tutorial'))
        self.lb_list.append(tk.Label(self,text='Step 1: Enter Username'))
        self.lb_list.append(tk.Label(self,text='Step 2: Add User Data'))
        self.lb_list.append(tk.Label(self,text='Step 3: Wait processing'))
        for lb in self.lb_list:
            lb.configure(font=NORMAL_FONT,anchor=W,bg=COLOR[0],fg=COLOR[4])
            lb.pack(side=TOP,fill=X,ipady=10)
        tk.Label(self,text='Register Mode',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=BOTH,ipady=5)
        register_modes = ['Liveness', 'Photo']
        self.drop = tk.OptionMenu(self, self.master.register_mode, *register_modes)
        self.drop.configure(font=NORMAL_FONT,bg=COLOR[0],fg=COLOR[4])
        self.drop.pack(side=TOP,fill=BOTH,ipady=10)
        self.chosen_lb(0)
        self.done_btn = tk.Label(self,text='Quick Done')
        self.done_btn.configure(font=NORMAL_FONT,anchor=CENTER,bg=COLOR[1],fg=COLOR[4])
        self.done_btn.bind('<Button-1>', self.done_click)
        self.done_btn.pack_forget()

    def drop_lock(self, check_var):
        if check_var:
            self.drop.configure(state='normal')
        else:
            self.drop.configure(state='disabled')

    def done_click(self, event):
        self.master.center_frames['RegistrationPage'].quick_done = True
    
    def chosen_lb(self, index):
        if index in [0, 1]:
            self.drop_lock(True)
        else:
            self.drop_lock(False)
        for i,lb in enumerate(self.lb_list):
            if i == index:
                lb.configure(font=BOLD_FONT,bg=COLOR[1])
            else:
                lb.configure(font=NORMAL_FONT,bg=COLOR[0])
                

class LeftFrame3(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class LeftFrame4(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class RightFrame1(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.ct = 0
        tk.Label(self,text='Timestamps recording',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=X)
        self.frame = tk.Frame(self,bg=COLOR[0])
        self.frame.pack(side=BOTTOM,fill=BOTH,expand=True)
        self.scrollbarx = ttk.Scrollbar(self.frame,orient=HORIZONTAL)
        self.scrollbary = ttk.Scrollbar(self.frame,orient=VERTICAL)
        self.treeview = ttk.Treeview(self.frame,columns=("Id", "Name", "Time"))
        self.treeview.configure(height=100,selectmode="extended",xscrollcommand=self.scrollbarx.set,yscrollcommand=self.scrollbary.set)
        self.scrollbarx.config(command=self.treeview.xview)
        self.scrollbary.config(command=self.treeview.yview)
        self.scrollbary.pack(side=RIGHT,fill=Y)
        self.scrollbarx.pack(side=BOTTOM,fill=X)
        self.treeview.heading('Id', text="Id", anchor=W)
        self.treeview.heading('Name', text="Name", anchor=W)
        self.treeview.heading('Time', text="Time", anchor=W)
        self.treeview.column('#0', stretch=NO, minwidth=0, width=0)
        self.treeview.column('#1', stretch=NO, minwidth=0, width=30)
        self.treeview.column('#2', stretch=NO, minwidth=0, width=60)
        self.treeview.column('#3', stretch=NO, minwidth=0, width=120)
        self.treeview.pack()

    def update(self):
        self.treeview.insert('', 0, value=(self.master.used_ids[-1],self.master.used_users[-1],self.master.used_timestamps[-1]))


class RightFrame2(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.user_list_frame = UserList(self,master)
        self.user_list_frame.configure(bg=COLOR[0])
        self.user_list_frame.pack(fill=BOTH,expand=True)
        self.register_status_frame = RegisterStatus(self,master)
        self.register_status_frame.configure(bg=COLOR[0])
        self.user_list_frame.reload_user_list()


class RightFrame3(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.bin_icon = ImageTk.PhotoImage(Image.open('ui/images/bin.png').resize((20,20),Image.LANCZOS))
        tk.Label(self,text='User List',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=BOTH,ipady=5)
        self.main_frame = tk.Frame(self,bg=COLOR[0])
        self.main_frame.pack(side=TOP,fill=BOTH,expand=True)
        self.scrollbar = ttk.Scrollbar(self.main_frame,orient='vertical')
        self.canvas = tk.Canvas(self.main_frame,yscrollcommand=self.scrollbar.set,highlightthickness=0,bg=COLOR[0])
        self.scrollbar.config(command=self.canvas.yview)
        self.scrollbar.pack(side=RIGHT,fill=Y)
        self.canvas.pack(side=LEFT,fill=BOTH,expand=True)
        self.frame = tk.Frame(self.canvas,bg=COLOR[0])
        self.frame.bind("<Configure>", self.update_scroll_region)
        self.canvas.create_window((0,0),window=self.frame,anchor="nw",tags="self.frame")
        # self.frame.pack(fill=BOTH,expand=True)
        self.frames = []
        self.choose_user_btns = []
        self.delete_user_btns = []
        self.reload_user_list()

    def update_scroll_region(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

    def choose_user(self, id_, label, event):
        self.master.center_frames['ViewPage'].show_user(id_, label)

    def delete_user(self, id, event):
        user_remove(self.master, id)

    def reload_user_list(self):   
        for frame in self.frames:         
            for frame in self.frames:
                for widget in frame.winfo_children():
                    widget.destroy()
        self.choose_user_btns = []
        self.delete_user_btns = []
        if self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall():
            for i,id_ in enumerate(list(dict.fromkeys(self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall()))):
                id_ = id_[0]
                self.frames.append(tk.Frame(self.frame,bg=COLOR[0]))
                self.frames[i].pack(fill=X,side=TOP)
                indexes = [j for j,x in enumerate(self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall()) if x[0] == id_]
                label = self.master.cur.execute('''SELECT EMP_NAME FROM EMPLOYESS''').fetchall()[indexes[0]][0]
                self.choose_user_btns.append(tk.Label(self.frames[i],text=label))
                self.choose_user_btns[i].configure(font=NORMAL_FONT,anchor=W,bg=COLOR[0],fg=COLOR[4])
                self.choose_user_btns[i].bind('<Button-1>', functools.partial(self.choose_user,id_,label))
                try:
                    icon_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(json2array(self.master.cur.execute('''SELECT FACE EMP_EMBS WHERE EMP_ID=?''', ([id_])).fetchall()[random.choice(indexes)][0]),(100,100))))
                    create_tool_tip(self.choose_user_btns[i],COLOR[1],COLOR[0],'{} (id:{})'.format(label,id_),icon_img)
                except:
                    create_tool_tip(self.choose_user_btns[i],COLOR[1],COLOR[0],'{} (id:{})'.format(label,id_))
                self.delete_user_btns.append(tk.Label(self.frames[i],image=self.bin_icon))
                self.delete_user_btns[i].configure(anchor=CENTER,bg=COLOR[0])
                self.delete_user_btns[i].bind('<Button-1>', functools.partial(self.delete_user,id_))
                create_tool_tip(self.delete_user_btns[i],'red',COLOR[0])
                self.delete_user_btns[i].pack(side=LEFT,ipady=5,ipadx=5)
                self.choose_user_btns[i].pack(side=RIGHT,fill=X,ipady=5,expand=True)


class RightFrame4(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master


class RegisterStatus(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        labels = []
        pitchs = ['Center','Up','Down']
        yawns = ['Straight','Left','Right']
        for pitch in pitchs:
            for yawn in yawns:
                labels.append(pitch+' '+yawn)
        tk.Label(self,text='Register Status',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=BOTH,ipady=10)
        self.left_frame = tk.Frame(self,bg=COLOR[0])
        self.left_frame.pack(side=LEFT,fill=BOTH,expand=True)
        self.right_frame = tk.Frame(self,bg=COLOR[0])
        self.right_frame.pack(side=RIGHT,fill=BOTH,expand=True)
        self.status = []
        for i,label in enumerate(labels):
            tk.Label(self.left_frame,text=label,font=NORMAL_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=BOTH,ipady=10)
            self.status.append(tk.Label(self.right_frame,text='...',font=NORMAL_FONT,bg=COLOR[0],fg=COLOR[4]))
            self.status[i].pack(side=TOP,fill=BOTH,ipady=10)        


class UserList(tk.Frame):
    def __init__(self,container,master):
        tk.Frame.__init__(self,container)
        self.container = container
        self.master = master
        self.bin_icon = ImageTk.PhotoImage(Image.open('ui/images/bin.png').resize((20,20),Image.LANCZOS))
        tk.Label(self,text='User List',font=BOLD_FONT,bg=COLOR[0],fg=COLOR[4]).pack(side=TOP,fill=BOTH,ipady=5)
        self.add_icon = ImageTk.PhotoImage(Image.open('ui/images/add-user.png').resize((30,30),Image.LANCZOS))
        self.add_new_user_lb = tk.Label(self,text='Add new user   ',font=NORMAL_FONT,bg=COLOR[0],fg=COLOR[4])
        self.add_new_user_lb["compound"] = RIGHT
        self.add_new_user_lb["image"]=self.add_icon
        create_tool_tip(self.add_new_user_lb,COLOR[1],COLOR[0])
        self.add_new_user_lb.pack(side=TOP,fill=BOTH,ipady=5)
        self.add_new_user_lb.bind('<Button-1>', self.add_new_user)
        self.main_frame = tk.Frame(self,bg=COLOR[0])
        self.main_frame.pack(side=TOP,fill=BOTH,expand=True)
        self.scrollbar = ttk.Scrollbar(self.main_frame,orient='vertical')
        self.canvas = tk.Canvas(self.main_frame,yscrollcommand=self.scrollbar.set,highlightthickness=0,bg=COLOR[0])
        self.scrollbar.config(command=self.canvas.yview)
        self.frame = tk.Frame(self.canvas,bg=COLOR[0])
        self.frame.bind("<Configure>", self.update_scroll_region)
        self.scrollbar.pack(side=RIGHT,fill=Y)
        self.canvas.pack(side=LEFT,fill=BOTH,expand=True)
        self.canvas.create_window((0,0),window=self.frame,anchor="nw",tags="self.frame")
        # self.frame.pack(fill=BOTH,expand=True)
        self.frames = []
        self.choose_user_btns = []
        self.delete_user_btns = []

    def update_scroll_region(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox('all'))

    def add_new_user(self, event):
        self.master.center_frames['RegistrationPage'].add_new_user_clicked()

    def choose_user(self, id, label, event):
        self.master.center_frames['RegistrationPage'].choose_user_clicked(id, label)

    def delete_user(self, id, event):
        user_remove(self.master, id)

    def reload_user_list(self):   
        for frame in self.frames:         
            for widget in frame.winfo_children():
                widget.destroy()
        self.choose_user_btns = []
        self.delete_user_btns = []
        if self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall():
            for i,id_ in enumerate(list(dict.fromkeys(self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall()))):
                id_ = id_[0]
                self.frames.append(tk.Frame(self.frame,bg=COLOR[0]))
                self.frames[i].pack(fill=X,side=TOP)
                indexes = [j for j,x in enumerate(self.master.cur.execute('''SELECT EMP_ID FROM EMPLOYESS''').fetchall()) if x[0] == id_]
                label = self.master.cur.execute('''SELECT EMP_NAME FROM EMPLOYESS''').fetchall()[indexes[0]][0]
                self.choose_user_btns.append(tk.Label(self.frames[i],text=label))
                self.choose_user_btns[i].configure(font=NORMAL_FONT,anchor=W,bg=COLOR[0],fg=COLOR[4])
                self.choose_user_btns[i].bind('<Button-1>', functools.partial(self.choose_user,id_,label))
                try:
                    icon_img = ImageTk.PhotoImage(Image.fromarray(cv2.resize(json2array(self.master.cur.execute('''SELECT FACE FROM EMP_EMBS WHERE EMP_ID=?''', ([id_])).fetchall()[random.choice(indexes)][0]),(100,100))))
                    create_tool_tip(self.choose_user_btns[i],COLOR[1],COLOR[0],'{} (id:{})'.format(label,id_),icon_img)
                except:
                    create_tool_tip(self.choose_user_btns[i],COLOR[1],COLOR[0],'{} (id:{})'.format(label,id_))
                self.delete_user_btns.append(tk.Label(self.frames[i],image=self.bin_icon))
                self.delete_user_btns[i].configure(anchor=CENTER,bg=COLOR[0])
                self.delete_user_btns[i].bind('<Button-1>', functools.partial(self.delete_user,id_))
                create_tool_tip(self.delete_user_btns[i],'red',COLOR[0])
                self.delete_user_btns[i].pack(side=LEFT,ipady=5,ipadx=5)
                self.choose_user_btns[i].pack(side=RIGHT,fill=X,ipady=5,expand=True)


class ProcessPopup(object):
    def __init__(self, parent):
        self.parent = parent
        self.wd = None
        
    def show_popup(self):
        self.parent.master.left_frames['LeftFrame2'].chosen_lb(3)
        if self.wd:
            return
        # self.wd = Toplevel(self.master,background=TOOLTIP_BG,relief=SOLID,borderwidth=1)
        # self.wd.wm_overrideredirect(1)
        # self.wd.eval('tk::PlaceWindow . center')
        # label = Label(self.wd,text='Please wait',justify=LEFT,background=TOOLTIP_BG,fg=TOOLTIP_FG,font=TOOLTIP_FONT)
    
    def hide_popup(self):
        wd = self.wd
        self.wd = None
        if wd:
            wd.destroy()


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
    
    def showtip(self,text=None,image=None):
        self.text = text
        self.image = image
        if self.tipwindow or not (text or image):
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 57
        y = y + cy + self.widget.winfo_rooty() +27
        self.tipwindow = tw = Toplevel(self.widget,background=TOOLTIP_BG,relief=SOLID,borderwidth=1)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        if self.text:
            label = Label(tw,text=self.text,justify=LEFT,background=TOOLTIP_BG,fg=TOOLTIP_FG,font=TOOLTIP_FONT)
            label.pack()
        if self.image:
            image = Label(tw,image=image,justify=LEFT,background=TOOLTIP_BG,relief=SOLID)
            image.pack()
    
    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def create_tool_tip(widget, color1=COLOR[0], color2=COLOR[0], text=None, image=None):
    tool_tip = ToolTip(widget)  
    def enter(event):
        widget.configure(bg=color1)
        tool_tip.showtip(text, image)    
    def leave(event):
        widget.configure(bg=color2)
        tool_tip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)


def json2array(json_, dtype=np.uint8):
    list_ = json.loads(json_)
    return np.array(list_, dtype=dtype)


if __name__ == '__main__':
    MainUI().mainloop()