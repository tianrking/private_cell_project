import os
import shutil
import cv2
path1 = r'D:\ww\lwd\data_only\Data\CTdata\label-aug'
files = os.listdir(path1)
import numpy as np
from PIL import Image
import numpy as np
from PIL import Image
list111=[]

for file in files:
    #file = 'Patient01_62_mask.png'
    way1 = os.path.join(path1, file)  # label  512*512
    # way2 = os.path.join(save, file)  # label  512*512
    # way2 = os.path.join(path1, c[i+1])
    _dir = r"D:\ww\lwd\data_only\Data\CTdata"
    _dir_label = _dir + "\\" + "label-aug" + "\\" + file
    _dir_image = _dir + "\\" + "image-aug" + "\\" + file
    img1 = Image.open(way1)
    a = np.array(img1)
    #print(np.unique(a))
    # print(np.unique(a))
    a = np.unique(a)
    lennnn=len(a)
    if len(a)==1:
        shutil.copy(_dir_label, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/1/label-aug")
        shutil.copy(_dir_image, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/1/image-aug")
        print("1")
    elif len(a)==2:
        shutil.copy(_dir_label, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/2/label-aug")
        shutil.copy(_dir_image, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/2/image-aug")
        print("2")
    elif len(a)==3:
        shutil.copy(_dir_label, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/3/label-aug")
        shutil.copy(_dir_image, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/3/image-aug")
        print("3")
    elif len(a)==4:
        shutil.copy(_dir_label, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/4/label-aug")
        shutil.copy(_dir_image, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/4/image-aug")
        print("4")
    elif len(a) == 5:
        shutil.copy(_dir_label, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/5/label-aug")
        shutil.copy(_dir_image, r"D:\ww\lwd\data_only\Data\CTdata\CLASSIFY/5/image-aug")
        print("5")