import os
import cv2
import numpy as np
'''
读取文件夹下图片并保存到指定路径从   
'''
readpathname =  r"D:\ww\lwd\data_only\Data\CTdata\label"
outpathname =  r"D:\ww\lwd\data_only\Data\CTdata\label"
def read_path(file_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        print(filename)

        img = cv2.imread(file_pathname+'/'+filename)
        #print(img.shape)

        ####change to gray
        #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）不需要这种操作只需注释掉即可
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        # for i in range(500):
        #     for j in range(500):
        #         # if 0 < img[i, j][0] < 110 and 0 < img[i, j][1] < 110 and img[i,j][2]>120:#根据三通道阈值填充图像(找红色像素点)
        #         sc = int(img[i, j][1]) - int(img[i, j][2])
        #         if  0 < img[i,j][0] < 130 and 110 < img[i,j][1] and 0 < img[i,j][2] <130 and sc > 20:#根据三通道阈值填充图像(找绿色像素点)
        #         # sc = int(img[i, j][0]) - int(img[i, j][2])
        #         # if 0 < img[i, j][1] < 120 and 110 < img[i, j][0] and 0 < img[i, j][2] < 120 and sc > 15:#根据三通道阈值填充图像(找蓝色像素点)
        #             img[i, j] = (255, 255, 255)
        #         else:
        #             img[i, j] = (0, 0, 0)
		#
        # # 轮廓转换为实心掩码图
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为单通道二值图
        # ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
        # # cv2.imshow('img', thresh)
        # # cv2.waitKey()
        # for i in range(img.shape[0]):
        #     rightpoint = 0
        #     leftpoint = img.shape[1]
        #     for j in range(img.shape[1]):
        #         if thresh[i][j] == 255:
        #             if j < leftpoint:
        #                 leftpoint = j
        #             if j > rightpoint:
        #                 rightpoint = j
        #     for s in range(leftpoint, rightpoint + 1):
        #         thresh[i][s] = 255
        #     kernel = np.ones((3, 3), np.uint8)
        #     thresh = cv2.dilate(thresh, kernel, iterations=9)
        #     thresh = cv2.erode(thresh, kernel, iterations=9)
        #     # thresh = cv2.erode(thresh, kernel, iterations=1)
        #     # thresh = cv2.dilate(thresh, kernel, iterations=1)

        # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#单通道转换回三通道
        # image_np = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)#单通道转换回三通道
        #####save figure
        cv2.imwrite(outpathname+"/"+filename, img)

#注意*处如果包含家目录（home）不能写成~符号代替
#读取的目录
read_path(readpathname)  # vscode里面读取图片文件夹的正确方式，pycharm里不知道。。。。
#print(os.getcwd())
