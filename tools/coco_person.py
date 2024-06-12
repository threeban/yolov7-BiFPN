from pycocotools.coco import COCO
from tqdm import tqdm
import shutil
import os


# 将ROI的坐标转换为yolo需要的坐标
# size是图片的w和h
# box里保存的是ROI的坐标（x，y的最大值和最小值）
# 返回值为ROI中心点相对于图片大小的比例坐标，和ROI的w、h相对于图片大小的比例
def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


# 获取所需要的类名和id
# path为类名和id的对应关系列表的地址（标注文件中可能有很多类，我们只加载该path指向文件中的类）
# 返回值是一个字典，键名是类名，键值是id
# def get_classes_and_index(path):
#     D = {}
#     f = open(path)
#     for line in f:
#         temp = line.rstrip().split(",", 2)
#         print("temp[0]:" + temp[0] + "\n")
#         print("temp[1]:" + temp[1] + "\n")
#         D[temp[1]] = temp[0]
#     return D


if __name__ == "__main__":
    dst_dir = "C:/code/yolo/coco_person"
    dataDir = "C:/code/yolo/COCO"  # COCO数据集所在的路径
    dataType = "val2017"  # 要转换的COCO数据集的子集名
    images_dir = os.path.join(dst_dir, "images", dataType)
    labels_dir = os.path.join(dst_dir, "labels", dataType)
    annFile = "%s/annotations/instances_%s.json" % (
        dataDir,
        dataType,
    )  # COCO数据集的标注文件路径
    # classes = get_classes_and_index(
    #     "C:/code/yolo/coco_person/coco_list.txt"
    # )

    images_dir = os.path.join(images_dir, dataType)
    labels_dir = os.path.join(labels_dir, dataType)
    # 创建YOLO这俩文件夹images_dir labels_dir
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    coco = COCO(annFile)  # 加载解析标注文件
    list_file = open("%s/%s.txt" % (dst_dir, dataType), "w")  # YOLO数据集训练验证txt

    imgIds = coco.getImgIds()  # 获取标注文件中所有图片的COCO Img ID
    catIds = coco.getCatIds()  # 获取标注文件总所有的物体类别的COCO Cat ID

    for imgId in tqdm(imgIds):
        objCount = 0  # 一个标志位，用来判断该img是否包含我们需要的标注
        Img = coco.loadImgs(imgId)[0]  # 加载图片信息
        filename = Img["file_name"]  # 获取图片名
        width = Img["width"]  # 获取图片尺寸
        height = Img["height"]  # 获取图片尺寸
        # print("imgId :%s" % imgId)
        # print("Img :%s" % Img)
        # print("filename :%s, width :%s ,height :%s" % (filename, width, height))
        annIds = coco.getAnnIds(
            imgIds=imgId, catIds=catIds, iscrowd=None
        )  # 获取该图片对应的所有COCO物体类别标注ID
        # print("annIds :%s" % annIds)
        for annId in annIds:
            anns = coco.loadAnns(annId)[0]  # 加载标注信息
            catId = anns["category_id"]  # 获取该标注对应的物体类别的COCO Cat ID
            cat = coco.loadCats(catId)[0]["name"]  # 获取该COCO Cat ID对应的物体种类名
            # print 'anns :%s' % anns
            # print 'catId :%s , cat :%s' % (catId,cat)

            # 如果该类名在我们需要的物体种类列表中，将标注文件转换为YOLO需要的格式
            classes = {"person": "0"}
            if cat in classes:
                objCount = objCount + 1
                out_file = open(
                    os.path.join(labels_dir, str(imgId).rjust(12,'0') + ".txt"), "w"
                )
                cls_id = classes[cat]  # 获取该类物体在yolo训练中的id
                box = anns["bbox"]
                size = [width, height]
                bb = convert(size, box)
                out_file.write(
                    str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n"
                )
                out_file.close()

        if objCount > 0:
            # list_file.write('data/images/%s\n' % filename) # 相对路径
            list_file.write(
                os.path.join(dst_dir, "images/%s/%s\n" % (dataType, filename))
            )  # 绝对路径
            src_img = os.path.join(dataDir, "%s/%s" % (dataType, filename))
            dst_img = os.path.join(images_dir, filename)
            shutil.copy(src_img, dst_img)

    list_file.close()
