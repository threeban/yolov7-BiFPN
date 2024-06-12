import os
from tqdm import tqdm
import cv2
from skimage import io
#import os
path = r'C:\code\yolo\yolov7\data\inria_person\images\2000/' #path后面记得加 /
fileList = os.listdir(path)
for i in tqdm(fileList):
    image = io.imread(path+i)  # image = io.imread(os.path.join(path, i))
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imencode('.png',image)[1].tofile(path+i)