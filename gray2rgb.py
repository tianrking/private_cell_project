import cv2
import  os
from osgeo import gdal

def readTif(fileName):# 可用os的文件操作代替
	dataset = gdal.Open(fileName)
	width = dataset.RasterXSize
	height = dataset.RasterYSize
	GdalImg_data = dataset.ReadAsArray(0, 0, width, height)
	return GdalImg_data
test_iamge_path = r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\image_rgb\E0018"
result_path = r"F:\ww\lwd\Unet_RSimage_Multi-band_Multi-class-master\Data\image_rgb\E0018"
imageList = os.listdir(test_iamge_path)
for i in range(len(imageList)):
	img = readTif(test_iamge_path + "\\" + imageList[i])
	img_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.imwrite(result_path + "\\" + imageList[i][:-4] + ".JPG", img_result)