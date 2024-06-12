import os
import cv2
img_path = 'C:/code/yolo/yolov7/data/wider_person/images/val/'
label_path = 'C:/code/yolo/yolov7/data/wider_person/labels/val/'
# img_path = 'C:/code/yolo/yolov7/data/voc_person/images/2007/'
# label_path = 'C:/code/yolo/yolov7/data/voc_person/labels/2007/'

f = os.listdir(img_path)
def paint(label_file, img_file):
    img = cv2.imread(img_file)
    img_h, img_w, _ = img.shape
    with open(label_file, 'r') as f:
        obj_lines = [l.strip() for l in f.readlines()]
    for obj_line in obj_lines:
        cls, cx, cy, nw, nh = [float(item) for item in obj_line.split(' ')]
        color = (0, 0, 255) if cls == 0.0 else (0, 255, 0)
        x_min = int((cx - (nw / 2.0)) * img_w)
        y_min = int((cy - (nh / 2.0)) * img_h)
        x_max = int((cx + (nw / 2.0)) * img_w)
        y_max = int((cy + (nh / 2.0)) * img_h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.imshow('Ima', img)
    cv2.waitKey(0)
for i in f:
    label_path_name = label_path + i.replace('jpg','txt').replace('png','txt')
    img_path_name = img_path + i
    print(label_path_name)
    print(img_path_name)
    paint(label_path_name,img_path_name)