import os
import cv2
path1 = r'D:\ww\lwd\data_only\Data\CTdata\class\label'
path2 = r'G:\daima\UNET-ZOO-master\train\text'
save = r'G:\codes\unet-nested-multiple-classification-master\data\labels'

files = os.listdir(path1)
import numpy as np
from PIL import Image
list111=[]

for file in files:
    #file = 'Patient01_62_mask.png'
    way1 = os.path.join(path1, file)  # label  512*512
    # way2 = os.path.join(save, file)  # label  512*512
    # way2 = os.path.join(path1, c[i+1])

    img1 = Image.open(way1)
    a = np.array(img1)
    #print(np.unique(a))
    # print(np.unique(a))
    a = np.unique(a)
    a = a.tolist()
    list111.append(a)
    # if 85 in np.unique(a):
    #     print(np.unique(a))
    # a = np.where(a == 64, 1, a)
    # a = np.where(a == 191,2, a)
    # a = np.where(a == 127, 3, a)
    # a = np.where(a == 255, 4, a)



    # else:
    #     print(a)
    #     list111.append(a)
    #     print(list111)
    #img = Image.fromarray(a.astype('uint8')).convert('L')
    #img.save(way2)
b_list = []
for i in list111:
    if i not in b_list:
        b_list.append(i)
print(b_list)
