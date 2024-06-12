import xml.etree.ElementTree as ET
import os
import shutil
from os import getcwd
space = 40*70
xml_path = "C:/code/yolo/yolov7/data/caltech_person/caltech_voc/Annotations/"
jpg_path = "C:/code/yolo/yolov7/data/caltech_person/images/raw/"
# save_xml_path = "data/CaltechPedestrian/caltech_voc/Annotations_filtered"
save_jpg_path = "C:/code/yolo/yolov7/data/caltech_person/images/train/"
save_txt_path = 'C:/code/yolo/yolov7/data/caltech_person/labels/train/'

def filter_xml():
    for file_name in os.listdir(xml_path):
        file_path = os.path.join(xml_path, file_name)
        # print(file_path)
        in_file = open(file_path)
        tree = ET.parse(in_file)  # ET是一个xml文件解析库，ET.parse（）打开xml文件。parse--"解析"
        root = tree.getroot()  # 获取根节点

        name = file_name.replace('.xml' ,'')
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        list = []

        write_flag = False

        for obj in root.findall('object'):  # 找到根节点下所有“object”节点
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            bb = convert((w, h), (x1, x2, y1, y2))
            if (x2-x1)*(y2-y1) >= space:
                write_flag = True
            list.append(bb)
            

        if write_flag:
            out_file = open(
                "%s%s.txt" % (save_txt_path, name), "w"
            )
            for item in list:
                out_file.write(
                    '0' + " " + " ".join([str(a) for a in item]) + "\n"
            )
            out_file.close()
            # tree.write(os.path.join(save_xml_path, file_name))
            # print("save xml file: {}".format(os.path.join(save_xml_path, file_name)))
            # jpg_name = file_name.replace('xml', 'jpg')
            # old_path = os.path.join(jpg_path, jpg_name)
            # new_path = os.path.join(save_jpg_path, jpg_name)
            # shutil.copyfile(old_path, new_path)
            # print("save jpg file: {}".format(new_path))


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

filter_xml()