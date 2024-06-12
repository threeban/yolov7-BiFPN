from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random


# 1、定义数据集路径
cocoRoot = "C:/code/yolo/COCO/"
dataType = "val2017"
annFile = os.path.join(cocoRoot, f"annotations/instances_{dataType}.json")
print(f"Annotation file: {annFile}")

# 2、为实例注释初始化COCO的API
coco = COCO(annFile)


# 3、采用不同函数获取对应数据或类别
ids = coco.getCatIds("person")[0]  # 采用getCatIds函数获取"person"类别对应的ID
print(f'"person" 对应的序号: {ids}')
id = coco.getCatIds(["dog"])[0]  # 获取某一类的所有图片，比如获取包含dog的所有图片
imgIds = coco.catToImgs[id]
print(f"包含dog的图片共有：{len(imgIds)}张, 分别是：", imgIds)


cats = coco.loadCats(1)  # 采用loadCats函数获取序号对应的类别名称
print(f'"1" 对应的类别名称: {cats}')

imgIds = coco.getImgIds(
    catIds=[1]
)  # 采用getImgIds函数获取满足特定条件的图片（交集），获取包含person的所有图片
print(f"包含person的图片共有：{len(imgIds)}张")


# 4、将图片进行可视化
imgId = imgIds[10]
imgInfo = coco.loadImgs(imgId)[0]
print(f"图像{imgId}的信息如下：\n{imgInfo}")

imPath = os.path.join(cocoRoot, "images", dataType, imgInfo["file_name"])
im = cv2.imread(imPath)
plt.axis("off")
plt.imshow(im)
plt.show()


plt.imshow(im)
plt.axis("off")
annIds = coco.getAnnIds(imgIds=imgInfo["id"])  # 获取该图像对应的anns的Id
anns = coco.loadAnns(annIds)
print(f'图像{imgInfo["id"]}包含{len(anns)}个ann对象，分别是:\n{annIds}')


coco.showAnns(anns)
print(f"ann{annIds[3]}对应的mask如下：")
mask = coco.annToMask(anns[3])
plt.imshow(mask)
plt.axis("off")
