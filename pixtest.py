import cv2
import os
import numpy as np

def color_dict(labelFolder):
    colorDict = []
    #  获取文件夹内的文件名
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        #  如果是灰度，转成RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        #  为了提取唯一值，将RGB转成一个数
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        # print(unique)
        #  将第i个像素矩阵的唯一值添加到colorDict中
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        #  对目前i个像素矩阵里的唯一值再取唯一值
        # print(colorDict)
        colorDict = sorted(set(colorDict))
        # print(colorDict)
        #  若唯一值数目等于总类数(包括背景)ClassNum，停止遍历剩余的图像
        # if(len(colorDict) == classNum):
        #     break
    #  存储颜色的RGB字典，用于预测时的渲染结果
    # print(colorDict)
    colorDict_RGB = []
    for k in range(len(colorDict)):
        #  对没有达到九位数字的结果进行左边补零(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        #  前3位R,中3位G,后3位B
        color_RGB = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        # print(color_RGB)
        colorDict_RGB.append(color_RGB)
    # print(colorDict_RGB)
    #  转为numpy格式
    colorDict_RGB = np.array(colorDict_RGB)
    print(colorDict_RGB)
    #  存储颜色的GRAY字典，用于预处理时的onehot编码
    # print(colorDict_RGB.shape)
    colorDict_GRAY = colorDict_RGB.reshape((colorDict_RGB.shape[0], 1 ,colorDict_RGB.shape[1])).astype(np.uint8)
    # print(colorDict_GRAY)
    # print(colorDict_GRAY.shape)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    # print(colorDict_GRAY)
    # print(colorDict_GRAY.shape)
    return colorDict_RGB, colorDict_GRAY

path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe_heren\label_all"
color_dict1 = color_dict(path)
# filename = readTif(path)
# print(color_dict1)