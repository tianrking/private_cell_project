import cv2
import  os
from osgeo import gdal


def readTif(fileName): # 可用os的文件操作代替
	dataset = gdal.Open(fileName)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
	return GdalImg_data
test_iamge_path = r"F:\ww\project\dataset_formal\mask_purple\dataset\all\label"
result_path = r"F:\ww\project\dataset_formal\mask_purple\dataset\all\label_2zhi"
imageList = os.listdir(test_iamge_path)
for i in range(len(imageList)):
	img = readTif(test_iamge_path + "\\" + imageList[i])
	if (len(img.shape) == 2):
		# print(img.shape)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	else:
		# print(img.shape)
		img = img.swapaxes(1, 0)
		img = img.swapaxes(1, 2)
		# print(img.shape)
		# print(i)
	rel, img_result = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
	# print(img_result.shape)
	cv2.imwrite(result_path + "\\" + imageList[i][:-4] + ".png", img_result) # 只能是png才行