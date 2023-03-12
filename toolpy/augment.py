import random
import cv2
import os

filepath1 = r"D:\ww\lwd\data_only\Data\CTdata\val\image-all" # 原文件夹1
filepath2 = r"D:\ww\lwd\data_only\Data\CTdata\val\label-all"	# 原文件夹2
filepath3 = r"D:\ww\lwd\data_only\Data\CTdata\val\image-aug"	# 保存文件夹1，与原文件夹1对应
filepath4 = r"D:\ww\lwd\data_only\Data\CTdata\val\label-aug"	# 保存文件夹2，与原文件夹2对应
filename1 = os.listdir(filepath1)
filename2 = os.listdir(filepath2)

for i in range(len(filename1)):
	image = cv2.imread(filepath1 + '/' + filename1[i])
	label = cv2.imread(filepath2 + '/' + filename2[i])
	flipCode = random.choice([-1,0,1])
	image_result = cv2.flip(image, flipCode)
	label_result = cv2.flip(label, flipCode)
	cv2.imwrite(filepath3 + "/" + filename1[i][:-4] + "-aug" + ".png", image_result)
	cv2.imwrite(filepath4 +
				"/" + filename1[i][:-4] + "-aug" + ".png", label_result)