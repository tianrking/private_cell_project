'''
本程序用作在某一个图上修改部分像素值
且添加像素的来源为同样大小的另外一张图片
针对原图-标签的数据集编写
'''
import os
import cv2
import shutil

# 标签图地址
file_initial_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun\label_all" # 需要添加像素原图文件夹
file_add_path = r"F:\ww\lwd\data_only\Data\heren\label_all" # 像素值来源文件夹
file_result_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_heren\label_all" # 结果保存文件夹

# 原图地址
image_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun\image_all" # 上述标签对应的原图地址
save_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_heren\image_all" # 原图重新保存地址

# 生成文件名列表
file_initial_name = os.listdir(file_initial_path)
file_add_name = os.listdir(file_add_path)
save_name = os.listdir(save_path)
k = 0
for name in file_initial_name:
	# if (name[:-4] + ".JPG") not in save_name:
	if name in file_add_name:
		img_initial = cv2.imread(file_initial_path + '/' + name)
		img_add = cv2.imread(file_add_path + '/' + name)
		for i in range(img_add.shape[0]):
			for j in range(img_add.shape[1]):
				if all(img_add[i, j] == (255, 255, 255)):
					img_initial[i, j] = (64, 64, 64)
		cv2.imwrite(file_result_path + "/" + name[:-4] + ".png", img_initial)
		shutil.copy(image_path + "/" + name[:-4] + ".JPG", save_path + "/" + name[:-4] + ".JPG")
		print(k)
		k=k+1
