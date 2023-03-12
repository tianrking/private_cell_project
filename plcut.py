import os
import cv2

'''
读取文件夹下图片并保存到指定路径  
'''
readpathname = r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\1"
outpathname = r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\2"
def read_path(file_pathname, save_pathname):
    #遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        # print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        ####change to gray
      #（下面第一行是将RGB转成单通道灰度图，第二步是将单通道灰度图转成3通道灰度图）不需要这种操作只需注释掉即可
        # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # image_np=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        for i in range(500):
            for j in range(500):
                # 找绿色
                # sc1 = int(img[i, j][1]) - int(img[i, j][0]) # G-B
                # sc2 = int(img[i, j][1]) - int(img[i, j][2]) # G-R
                # if img[i, j][0] >= 0 and img[i, j][1] >= 90 and img[i, j][2] > 0 and sc1 > 30 and sc2 > 30:

                # 找黄色
                sc1 = int(img[i, j][1]) - int(img[i, j][0]) # G-B
                sc2 = int(img[i, j][2]) - int(img[i, j][0]) # R-B
                if img[i, j][0] >= 0 and img[i, j][1] >= 90 and img[i, j][2] >= 90  and sc1 > 30 and sc2 > 30:

                    img[i, j] = (255, 255, 255)
                else:
                    img[i, j] = (0, 0, 0)

        # 轮廓转换为实心掩码图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为单通道二值图
        ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
        # cv2.imshow('img', thresh)
        # cv2.waitKey()
        for i in range(img.shape[0]):
            rightpoint = 0
            leftpoint = img.shape[1]
            for j in range(img.shape[1]):
                if thresh[i][j] == 255:
                    if j < leftpoint:
                        leftpoint = j
                    if j > rightpoint:
                        rightpoint = j
            for s in range(leftpoint, rightpoint + 1):
                thresh[i][s] = 255
            # kernel = np.ones((3, 3), np.uint8)
            # thresh = cv2.dilate(thresh, kernel, iterations=1)
            # thresh = cv2.erode(thresh, kernel, iterations=1)
            # thresh = cv2.erode(thresh, kernel, iterations=1)
            # thresh = cv2.dilate(thresh, kernel, iterations=1)
        out= cv2.medianBlur(thresh, ksize=9)
            # out2 = cv2.dilate(out, kernel, iterations=1)

        image_np = cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)#单通道转换回三通道
        ret1, image_out = cv2.threshold(image_np, 6, 255, cv2.THRESH_BINARY)
        #####save figure
        cv2.imwrite(save_pathname+"/"+filename[:-4] + ".png", image_out) #只能是png
        # cv2.imwrite(outpathname+"/"+filename, img)
#注意*处如果包含家目录（home）不能写成~符号代替
#读取的目录
read_path(readpathname, outpathname)  # vscode里面读取图片文件夹的正确方式，pycharm里不知道。。。。
#print(os.getcwd())
