import  os
from osgeo import gdal


def readTif(fileName):# 文件打开程序，可用os替代
	dataset = gdal.Open(fileName)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
	return GdalImg_data
test_iamge_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe_heren\label_all"
imageList = os.listdir(test_iamge_path)
rgb_list = []
gray_list = []
k = 0
for i in range(len(imageList)):
	img = readTif(test_iamge_path + "\\" + imageList[i])
	if (len(img.shape) == 2):
		gray_list.append(imageList[i])
		# print(img.shape)
	else:
		rgb_list.append(imageList[i])
		k = k+1
print(k)