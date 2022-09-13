import cv2
import  os
import numpy as np

pix0 = 0
pix128 = 0
pix192 = 0
pix255 = 0
test_iamge_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\validation\label-aug"
imageList = os.listdir(test_iamge_path)
for i in range(len(imageList)):
	ImagePath = test_iamge_path + "/" + imageList[i]
	# print(ImagePath)
	img = cv2.imread(ImagePath).astype(np.uint32)
	#  如果是灰度，转成RGB
	# if (len(img.shape) == 2):
# 	img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
	for r in range(500):
		for c in range(500):
			# print(img[r,c])
			if all(img[r,c] == (0, 0, 0)):
				pix0 = pix0 +1
			elif all(img[r,c] == (128,128,128)):
				pix128 = pix128 +1
			elif all(img[r,c] == (192,192,192)):
				pix192 = pix192 +1
			elif all(img[r,c] == (255,255,255)):
				pix255 = pix255 +1
print(pix0/len(imageList),pix128/len(imageList),pix192/len(imageList),pix255/len(imageList))