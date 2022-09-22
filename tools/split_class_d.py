import cv2
from PIL import Image
import os
import numpy as np

class FIND_LOCATION:
	x_min = 0
	x_max = 0
	y_min = 0
	y_max = 0

	x_list = []
	y_list = []

	np_im = 0

	ROI = 0

	def __init__(self, dir):
		im = Image.open(dir)
		self.np_im = np.asarray(im)

	def read_img(self, dir):
		im = Image.open(dir)
		self.np_im = np.asarray(im)
		return self.np_im

	def Print(self):
		print("%s %s %s %s\n" % (self.x_max, self.x_min, self.y_min, self.y_max))

	def find_white(self):

		for x in range(0, 500):
			for y in range(0, 500):
				if (self.np_im[x][y].sum() == 765):
					self.x_list.append(x)
					self.y_list.append(y)

		self.x_min = min(self.x_list)
		self.x_max = max(self.x_list)
		self.y_min = min(self.y_list)
		self.y_max = max(self.y_list)

	def get_location(self):
		return self.y_min, self.x_min, self.y_max, self.x_max

	def clip_img(self, clip_origin_location, save_location):
		gg = cv2.imread(clip_origin_location)
		self.ROI = gg[self.x_min:self.x_max, self.y_min:self.y_max]
		cv2.imwrite(save_location, self.ROI)

		# cv2.imwrite(save_location, self.ROI)
	def reset(self):
		self.x_min = 0
		self.x_max = 0
		self.y_min = 0
		self.y_max = 0

		self.x_list = []
		self.y_list = []

		self.np_im = 0

		self.ROI = 0


## train_label_path 白色最小外接矩形
train_label_path = r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\label"

## train_label_path 根据白色最小外接矩形裁剪原图路径
train_image_path = r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\image"

validation_label_path = r"F:\ww\lwd\data_only\Data\yuanhe_heren\validation\label-aug"
validation_image_path = r"F:\ww\lwd\data_only\Data\yuanhe_heren\validation\image-aug"


## 裁剪图路径
output_dir_label = r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\label-clip"
output_dir_image = r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\image-clip"
output_dir_validation_label_path = r"F:\ww\lwd\data_only\Data\yuanhe_heren\validation\label-aug-clip"
output_dir_validation_image_path = r"F:\ww\lwd\data_only\Data\yuanhe_heren\validation\image-aug-clip"

dic_name = os.walk(r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\label")

# dic_name = os.walk(r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\label_file\E0001")

# kk = FIND_LOCATION(r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\label\D2020.09.07_S00021_I0939_D_WELL08_RUN098.png")
# kk.find_white()
# kk.clip_img(r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\image\D2020.09.07_S00021_I0939_D_WELL08_RUN098.JPG",
# 			r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\image-clip\D2020.09.07_S00021_I0939_D_WELL08_RUN098.png")
# kk.clip_img(r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\label\D2020.09.07_S00021_I0939_D_WELL08_RUN098.png",
# 			r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\label-clip\D2020.09.07_S00021_I0939_D_WELL08_RUN098.png")
 # D2020.09.07_S00021_I0939_D_WELL08_RUN098

g ={}

# dir_tt = r"F:\ww\lwd\data_only\Data\tt"
time_test = 0
for path, dir_list, file_list in dic_name:
	for file_name in file_list:

		# i_label_dir = r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\label\D2020.09.07_S00021_I0939_D_WELL07_RUN093.png"
		# o_label_clip_dir = r"F:\ww\lwd\data_only\Data\yuanhe_heren\train\label-clip\D2020.09.07_S00021_I0939_D_WELL07_RUN093.png"

		i_image_dir = path.split("label")[0] + "image" + "\\" + file_name.split(".png")[0] + ".png"
		i_label_dir = path + "\\" + file_name

		o_image_clip_dir = path.split("label")[0]+"image-clip"+"\\"+file_name
		o_label_clip_dir = path.split("label")[0] + "label-clip" + "\\" + file_name

		g[time_test] = FIND_LOCATION(i_label_dir)
		# g[time_test] = FIND_LOCATION(
		# 	path+"\\"+file_name)
		g[time_test].find_white()
		# kk.Print()

		# i_image_dir = path.split("label")[0]+"image"+"\\"+file_name.split(".png")[0]+".JPG"
		# i_label_dir = path+"\\"+file_name
		#
		# # o_image_clip_dir = path.split("label")[0]+"image-clip"+"\\"+file_name
		# o_label_clip_dir = path.split("label")[0]+"label-clip"+"\\"+file_name


		# o_val_image_dir = path+"\\"+file_name
		# o_val_label_dir = path.split("label")[0]+"label-clip"+"\\"+file_name
		#
		# g[time_test].clip_img(
		# 	i_image_dir,
		# 	o_image_clip_dir)
		g[time_test].clip_img(
			i_label_dir,
			o_label_clip_dir)

		# print(i_image_dir + "\n" + o_image_clip_dir )
		print(i_label_dir + "\n" + o_label_clip_dir)

		# print(path+"\\"+file_name+" \n"+path.split("label")[0]+"label-clip"+"\\"+file_name)
		#
		# print(path.split("label")[0]+"image"+"\\"+file_name.split(".png")[0]+".JPG")
		# print(path.split("label")[0] + "image-clip" + "\\" + file_name)
		# kk.reset()
		# kk.clip_img(path+"\\"+file_name,dir_tt+"\\"+file_name)


		print(time_test)
		# print(path+"\\"+file_name,dir_tt+"\\"+file_name)
		time_test = time_test + 1
		if time_test==8:
			break
