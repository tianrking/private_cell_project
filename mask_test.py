import cv2
import os
import numpy as np


begin_path = r"F:\ww\lwd\data_only\Data\yuanhe\test\predict_shai-2300-2-4-5-150"#原图文件夹
mask_path = r"F:\ww\lwd\data_only\Data\yuanhe\test\label_shai"#掩码图文件夹
save_path = r"F:\ww\lwd\data_only\Data\yuanhe\test\label_goule"#保存路径

def union_image_mask(image_path, mask_path):
	# 读取原图
	imagefilename = os.listdir(image_path)
	maskfilename = os.listdir(mask_path)
	# print(filename)
	for i in range(len(imagefilename)):
		img = cv2.imread(image_path + '/' + imagefilename[i])
		# print(file_name)
		mask = cv2.imread(mask_path + '/' + maskfilename[i])
		gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		# ret, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
		# cv2.imshow("img1", binary)
		# cv2.waitKey(0)
		contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# print(contours)
		# cv2.imshow("img", img)
		# cv2.waitKey(0)
		cv2.drawContours(img, contours, -1, (0, 255, 255), 1)
		# cv2.imshow("img", img)
		# cv2.waitKey(0)

		cv2.imwrite(save_path + '\\' + imagefilename[i], img)


	# cv2.waitKey(0)
union_image_mask(begin_path,mask_path)
