import os
import re
from PIL import Image

sets=['train']
#需要填写变量image_path、annotations_path、full_path
image_path = "C:/code/yolo/yolov7/data/inria_person/images/2008/"                          # 图片存放路径，路径固定
annotations_path = "C:/code/yolo/yolov7/data/inria_person/Test/annotations/" #文件夹目录                                          # INRIA标签存放路径
annotations= os.listdir(annotations_path) #得到文件夹下的所有文件名称

# 获取文件夹下所有图片的图片名
def get_name(file_dir):
   list_file=[]
   for root, dirs, files in os.walk(file_dir):
      for file in files:
         # splitext()将路径拆分为文件名+扩展名，例如os.path.splitext(“E:/lena.jpg”)将得到”E:/lena“+".jpg"
         if os.path.splitext(file)[1] == '.png':
            list_file.append(os.path.join(root, file))
   return list_file

# 在labels目录下创建每个图片的标签txt文档
def text_create(name,bnd):
   full_path = "C:/code/yolo/yolov7/data/inria_person/labels/2008/%s.txt"%(name)
   size = get_size(name + '.png')
   file = open(full_path, 'w')
   for item in bnd:
      convert_size = convert(size, item)
      file.write('0 ' + str(convert_size[0]) + ' ' + str(convert_size[1]) + ' ' + str(convert_size[2]) + ' ' + str(convert_size[3]))
      file.write('\n')

# 获取要查询的图片的w,h
def get_size(image_id):
   im = Image.open('C:/code/yolo/yolov7/data/inria_person/images/2008/%s'%(image_id))       # 源图片存放路径
   size = im.size
   w = size[0]
   h = size[1]
   return (w,h)

# 将Tagphoto的x,y,w,h格式转换成yolo的X,Y,W,H
def convert(size, box):
   dw = 1./size[0]
   dh = 1./size[1]
   x = (box[0] + box[2])/2.0
   y = (box[1] + box[3])/2.0
   w = box[2] - box[0]
   h = box[3] - box[1]
   x = x*dw
   w = w*dw
   y = y*dh
   h = h*dh
   return (x,y,w,h)

# 将处理的图片路径放入一个txt文件夹中
for image_set in sets:
   if not os.path.exists('C:/code/yolo/yolov7/data/inria_person/Train/labels'):
      os.makedirs('C:/code/yolo/yolov7/data/inria_person/Train/labels')                # 生成的yolo标签存放路径，路径固定
   image_names = get_name(image_path)
   list_file = open('C:/code/yolo/yolov7/data/inria_person/Train/%s.txt'%(image_set), 'w')
   for image_name in image_names:
      list_file.write('%s\n'%(image_name))
   list_file.close()


for file in annotations: #遍历文件夹
   str_name = file.replace('.txt', '')
   if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
      with open(annotations_path+''+file) as f : #打开文件
         list = []
         iter_f = iter(f); #创建迭代器
         for line in iter_f: #遍历文件，一行行遍历，读取文本
            str_XY = "(Xmax, Ymax)"
            if str_XY in line:
               strlist = line.split(str_XY)
               strlist1 = "".join(strlist[1:])    # 把list转为str
               strlist1 = strlist1.replace(':', '')
               strlist1 = strlist1.replace('-', '')
               strlist1 = strlist1.replace('(', '')
               strlist1 = strlist1.replace(')', '')
               strlist1 = strlist1.replace(',', '')
               b = strlist1.split()
               bnd = (float(b[0]) ,float(b[1]) ,float(b[2]) ,float(b[3]))
               list.append(bnd)
               text_create(str_name, list)
            else:
               continue

# images= os.listdir(image_path) #得到文件夹下的所有文件名称
# for file in images: #遍历文件夹
#    # print(file)
#    if ".png" in file:
#       continue
#    str_name = file.replace('.jpg', '')
#    list = []
#    text_create(str_name, list)