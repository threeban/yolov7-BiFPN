# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())

# import tensorflow as tf
# print(tf.__version__)


# import os
# from tqdm import tqdm

# path = "C:/code/yolo/coco_person/labels/train2017"

# os.chdir(path)
# for file in  tqdm(os.listdir(path)):
#     os.rename(file,file.split('.')[0].rjust(12,'0')+'.txt')
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())