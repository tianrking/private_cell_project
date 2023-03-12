import os
import cv2
import shutil


xibao_label_path = r"F:\ww\lwd\data_only\Data\xibao\train\label_3"
zhiyun_label_path = r"F:\ww\lwd\data_only\Data\zhiyun\train_zhiyun\label_shai"
yuanhe_label_path = r"F:\ww\lwd\data_only\Data\yuanhe\train_yuanhe\label_all_shai"

xibao_zhiyun_image_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun\image"
xibao_yuanhe_image_path = r"F:\ww\lwd\data_only\Data\xibao_yuanhe\image"
xibao_zhiyun_yuanhe_image_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\image"
zhiyun_yuanhe_image_path = r"F:\ww\lwd\data_only\Data\zhiyun_yuanhe\image"

xibao_zhiyun_label_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun\label"
xibao_yuanhe_label_path = r"F:\ww\lwd\data_only\Data\xibao_yuanhe\label"
xibao_zhiyun_yuanhe_label_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\label"
zhiyun_yuanhe_label_path = r"F:\ww\lwd\data_only\Data\zhiyun_yuanhe\label"

image_path = r"F:\ww\lwd\data_only\Data\image_rgb\all"
# for xibao_name in os.listdir(xibao_path):
# 	# print(xibao_name)
# 	for yuanhe_name in os.listdir(yuanhe_path):
# 		if xibao_name == yuanhe_name:
# 			img_xibao_yuanhe = cv2.imread(xibao_path + '/' + xibao_name)
# 			img_yuanhe = cv2.imread(yuanhe_path + '/' + yuanhe_name)
# 			for i in range(img_xibao_yuanhe.shape[0]):
# 				for j in range(img_xibao_yuanhe.shape[1]):
# 					if all(img_yuanhe[i, j] == (255, 255, 255)):
# 						img_xibao_yuanhe[i, j] = (128, 128, 128)
# 			cv2.imwrite(xibao_yuanhe_path + "/" + xibao_name, img_xibao_yuanhe)
#
# 		for zhiyun_name in os.listdir(zhiyun_path):
# 			if xibao_name == zhiyun_name and xibao_name == yuanhe_name:
# 				img_xibao_zhiyun_yuanhe = cv2.imread(xibao_path+'/'+xibao_name)
# 				img_zhiyun = cv2.imread(zhiyun_path + '/' + zhiyun_name)
# 				for i in range(img_xibao_zhiyun.shape[0]):
# 					for j in range(img_xibao_zhiyun.shape[1]):
# 						if all(img_zhiyun[i, j] == (255, 255, 255)):
# 							img_xibao_zhiyun[i, j] = (128, 128, 128)
# 				cv2.imwrite(xibao_zhiyun_yuanhe_path+"/"+xibao_name, img)
#
# 			if xibao_name == yuanhe_name:
# 				img = cv2.imread(xibao_path + '/' + xibao_name)
# 				cv2.imwrite(xibao_yuanhe_path + "/" + xibao_name, img)

xibao_name = os.listdir(xibao_label_path)
zhiyun_name = os.listdir(zhiyun_label_path)
yuanhe_name = os.listdir(yuanhe_label_path)

for name in xibao_name:
	if name in zhiyun_name:
		img_result = cv2.imread(xibao_label_path + '/' + name)
		img_zhiyun = cv2.imread(zhiyun_label_path + '/' + name)
		for i in range(img_result.shape[0]):
			for j in range(img_result.shape[1]):
				if all(img_zhiyun[i, j] == (255, 255, 255)):
					img_result[i, j] = (192, 192, 192)
		cv2.imwrite(xibao_zhiyun_label_path + "/" + name, img_result)
		shutil.copy(image_path + "/" + name[:-3] + 'JPG', xibao_zhiyun_image_path)
		if name in yuanhe_name:
			img_yuanhe = cv2.imread(yuanhe_label_path + '/' + name)
			for i in range(img_result.shape[0]):
				for j in range(img_result.shape[1]):
					if all(img_zhiyun[i, j] == (255, 255, 255)):
						img_zhiyun[i, j] = (192, 192, 192)
					if all(img_yuanhe[i, j] == (255, 255, 255)):
						img_zhiyun[i, j] = (128, 128, 128)
						img_result[i, j] = (128, 128, 128)
			cv2.imwrite(zhiyun_yuanhe_label_path + "/" + name, img_zhiyun)
			cv2.imwrite(xibao_zhiyun_yuanhe_label_path + "/" + name, img_result)
			shutil.copy(image_path + "/" + name[:-3] + 'JPG', zhiyun_yuanhe_image_path)
			shutil.copy(image_path + "/" + name[:-3] + 'JPG', xibao_zhiyun_yuanhe_image_path)
	if name in yuanhe_name:
		img_result = cv2.imread(xibao_label_path + '/' + name)
		img_yuanhe = cv2.imread(yuanhe_label_path + '/' + name)
		for i in range(img_result.shape[0]):
			for j in range(img_result.shape[1]):
				if all(img_yuanhe[i, j] == (255, 255, 255)):
					img_result[i, j] = (128, 128, 128)
		cv2.imwrite(xibao_yuanhe_label_path + "/" + name, img_result)
		shutil.copy(image_path + "/" + name[:-3] + 'JPG', xibao_yuanhe_image_path)

# for name in zhiyun_name:
# 	if name in yuanhe_name:
# 		img_result = cv2.imread(zhiyun_path + '/' + name)
# 		img_yuanhe = cv2.imread(yuanhe_path + '/' + name)
# 		for i in range(img_result.shape[0]):
# 			for j in range(img_result.shape[1]):
# 				if all(img_result[i, j] == (255, 255, 255)):
# 					img_result[i, j] = (192, 192, 192)
# 				if all(img_yuanhe[i, j] == (255, 255, 255)):
# 					img_result[i, j] = (128, 128, 128)

