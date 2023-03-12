import cv2
import os
import numpy as np
#
file_root = r'F:\ww\project\dataset_formal\final_data\E0011/'  # 当前文件夹下的所有图片
file_list = os.listdir(file_root)
save_out = r"F:\ww\project\dataset_formal\gray_final_data\E0011/" # 保存图片的文件夹名称
a=0
for img_name in file_list:
    img_path = file_root + img_name
    img = cv2.imread(img_path)
# img = cv2.imread(r"F:\ww\project\dataset_formal\final_data\test\D2021.01.20_S00120_I0939_D_WELL01_RUN212.png")  # 获取BGR图像，第二个参数默认值为1，可不填
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b=0
    for i in range(0, 500):  # 选取图像的10-99行
        for j in range(0, 500):  # 选取图像的80-99列
            # if gray[i, j] <= 10 and gray[i, j] > 0:
            #     gray[i, j] = 0
            # if gray[i, j] <= 58 and gray[i, j] >= 26:
            #     gray[i, j] = 38
            # if gray[i, j] <= 25 and gray[i, j] > 10:
            #     gray[i, j] = 15
            # if gray[i, j] <= 125 and gray[i, j] >= 98:
            #     gray[i, j] = 113
            # if gray[i, j] <= 91 and gray[i, j] >= 64:
            #     gray[i, j] = 76
            # if gray[i,j]!=0 and gray[i,j]!=38 and gray[i,j]!=15 and gray[i,j]!=113 and gray[i,j]!=76:
            #     b=b+1

            if img[i, j][0] <= 20 and img[i, j][1] <= 20 and img[i, j][2] <= 20:  # 黑色
                img[i, j] = 0
            if img[i, j][0] <= 20 and img[i, j][1] <= 20 and img[i, j][2] >= 200:  # 红色
                img[i, j] = 25
            if img[i, j][1] >= 200 and img[i, j][2] >= 200:  # 黄色
                img[i, j] = 50
            if img[i, j][0] < 10 and img[i, j][1] == 255 and img[i, j][2] < 10:  # 绿色
                img[i, j] = 96
            if img[i, j][0] == 255 and img[i, j][1] < 10 and img[i, j][2] < 10:  # 蓝色
                img[i, j] = 133
            if img[i, j][0] != 133 and img[i, j][0] != 96 and img[i, j][0] != 50 and img[i, j][0] != 25 and img[i, j][0] != 0:
                a=a+1
                print(img[i, j])
    # print(a)
    out_name = img_name.split('.')[0]
    save_path = save_out + img_name
    cv2.imwrite(save_path, img)
# save_path = r"F:\ww\project\dataset_formal\final_data\test\1.png"
# cv2.imwrite(save_path, img)
# print(a)
