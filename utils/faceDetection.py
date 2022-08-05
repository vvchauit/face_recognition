# references:
#   https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/


import cv2


# threshold để xét có phải mặt hay không
MIN_SCORE = 0.5
#  kích thước tối thiểu của mặt tìm được
MIN_FACE_SIZE = 100


prototxtPath = 'models/face_detection_model/deploy.prototxt'
weightsPath = 'models/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
# load model bằng module dnn của opencv
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


# summary: tìm tọa độ mặt trong ảnh bằng module dnn của opencv
# params:
#   init
#       pixels: ảnh đầu vào (array)
#   return
#       locs: list các tọa độ mặt
#       scores: độ tin cậy
def face_detector_caffe(pixels):
    # chuyển ảnh ngõ vào thành ảnh blob
    blob = cv2.dnn.blobFromImage(pixels, 1.0, (224, 224),(104.0, 177.0, 123.0))
    # tìm tọa độ khung bao quanh khuôn mặt trong ảnh
    faceNet.setInput(blob)
    detections = faceNet.forward()
    locs = []
    scores = []
    # giữ lại những ảnh có độ tin cậy cao hơn MIN_SCORE
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > MIN_SCORE:
            # tọa độ các điểm của bbox [0,1]
            box = detections[0, 0, i, 3:7]
            (xmin, ymin, xmax, ymax) = box
            locs.append((xmin, ymin, xmax, ymax))
            scores.append(confidence)
    return locs, scores


# summary: tìm tọa độ mặt trong ảnh
# params:
#   init
#       pixels: ảnh đầu vào (array)
#   return
#       faces_location: list tọa độ mặt (shape là hình vuông)
#       faces_location_margin: list tọa độ mặt đã thêm lề
def face_detector(pixels):
    image = pixels
    # lấy kích thước của ảnh gốc
    base_width, base_height = pixels.shape[1], pixels.shape[0]
    # xác định các bbox
    bboxes, scores=face_detector_caffe(image)
    faces_location = []
    faces_location_margin = []
    for box in bboxes:
        # chuyển tọa độ các điểm thành pixel
        xmin = int(max(1,(box[0] * base_width)))
        ymin = int(max(1,(box[1] * base_height)))
        xmax = int(min(base_width,(box[2] * base_width)))
        ymax = int(min(base_height,(box[3] * base_height)))
        # điều chỉnh tọa độ bbox sao cho box thành hình vuông
        bb_width = xmax-xmin
        bb_height = ymax-ymin
        offset_x = 0
        offset_y = 0
        if bb_width > bb_height:
            offset_y = int((bb_width - bb_height)/2)
            bb_height = bb_width
        elif bb_width < bb_height:
            offset_x = int((bb_height - bb_width)/2)
            bb_width = bb_height
        face_location = (xmin-offset_x, ymin-offset_y, bb_width, bb_height)
        # thêm lề cho bbox (25%)
        # note: thêm lề do yêu cầu của landmark model cần lề 25%
        margin_x = int(bb_width*0.25)
        margin_y = int(bb_height*0.25)
        offset_x = int(offset_x+margin_x/2)
        offset_y = int(offset_y+margin_y/2)
        bb_width += margin_x
        bb_height += margin_y
        # loại bỏ các box có kích thước nhỏ hơn MIN_FACE_SIZE
        if (bb_height>=MIN_FACE_SIZE) and (bb_width>=MIN_FACE_SIZE) and ((ymin-offset_y)>=0) and ((ymin-offset_y+bb_height)<=base_height) and ((xmin-offset_x)>=0) and ((xmin-offset_x+bb_width)<=base_width):
            faces_location.append(face_location)
            faces_location_margin.append((xmin-offset_x,ymin-offset_y,bb_width,bb_height))
    return faces_location, faces_location_margin