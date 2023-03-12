import os
import numpy as np
import PIL
from PIL import Image
import cv2

path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\image_file\temp" #需要转化的文件夹路径，jpg和png都能一起批量转化（8转24）
for root, dirs, files in os.walk(path):
    for name in files:
        print("files:",os.path.join(root,name))
        filename = os.path.join(root,name)
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img_shape = img.shape
        imgs = np.zeros(shape=(img_shape[0], img_shape[1], 3), dtype=np.float32)
        imgs[:, :, 0] = img[:, :]
        imgs[:, :, 1] = img[:, :]
        imgs[:, :, 2] = img[:, :]
        cv2.imwrite(filename, imgs)