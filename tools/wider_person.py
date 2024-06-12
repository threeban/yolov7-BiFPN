import os
from PIL import Image
import shutil


# coding=utf-8
def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset


def convert(size, box0, box1, box2, box3):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box0 + box2) / 2 * dw
    y = (box1 + box3) / 2 * dh
    w = (box2 - box0) * dw
    h = (box3 - box1) * dh
    return (x, y, w, h)


if __name__ == '__main__':

    outpath_txt = 'data/wider_person/labels/val'   
    # 注意：这里F:\\WiderPerson是你存储文件侧地方，可以替换，后面的不要动
    outpath_jpg = 'data/wider_person/image/val'
    # 注意：这里F:\\WiderPerson是你存储文件侧地方，可以替换，后面的不要动
    # os.makedirs(outpath_txt)
    # os.makedirs(outpath_jpg)

    path = 'data/wider_person/val.txt'
    with open(path, 'r') as f:
        img_ids = [x for x in f.read().splitlines()]
    
    for img_id in img_ids:  # '000040'
        img_path = 'data/wider_person/Images/' + img_id + '.jpg'

        with Image.open(img_path) as Img:
            img_size = Img.size

        ans = ''

        label_path = img_path.replace('Images', 'Annotations') + '.txt'
        
        outpath = outpath_txt + '/' + img_id + '.txt'

        with open(label_path, encoding=check_charset(label_path)) as file:
            line = file.readline()
            count = int(line.split('\n')[0])  # 里面行人个数
            line = file.readline()
            while line:
                cls = int(line.split(' ')[0])
                # if cls == 1
                if cls == 1  or cls == 3:
                    xmin = float(line.split(' ')[1])
                    ymin = float(line.split(' ')[2])
                    xmax = float(line.split(' ')[3])
                    ymax = float(line.split(' ')[4].split('\n')[0])
                    # print(img_size[0], img_size[1], xmin, ymin, xmax, ymax)
                    bb = convert(img_size, xmin, ymin, xmax, ymax)
                    ans = ans + '0' + ' ' + ' '.join(str(a) for a in bb) + '\n'
                line = file.readline()
        with open(outpath, 'w') as outfile:
            outfile.write(ans)
        # 想保留原文件用copy
        # shutil.copy(img_path, outpath_jpg + '/' + img_id + '.jpg')
        # 直接移动用这个
        # shutil.move(img_path, outpath_jpg + '/' + img_id + '.jpg')