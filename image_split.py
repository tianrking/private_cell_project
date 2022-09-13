import os
import shutil
import numpy as np
'''
自动划分训练集、验证集、测试集
需要自己创建文件夹
'''

all_image_path = r"F:\ww\lwd\data_only\Data\zhiyun_heren\image_all" # 全部原图所在文件夹
all_label_path = r"F:\ww\lwd\data_only\Data\zhiyun_heren\label_all" # 全部标签图所在文件夹
train_path = r"F:\ww\lwd\data_only\Data\zhiyun_heren\train" # train保存上级文件夹
test_path = r"F:\ww\lwd\data_only\Data\zhiyun_heren\test" # test保存上级文件夹
validation_path = r"F:\ww\lwd\data_only\Data\zhiyun_heren\validation" # val保存上级文件夹


all_image_name_list = os.listdir(all_image_path) # 文件名做成的列表，带后缀
all_label_name_list =os.listdir(all_label_path)
cur_state = np.random.get_state()
np.random.shuffle(all_image_name_list)
np.random.set_state(cur_state)
np.random.shuffle(all_label_name_list)
num = len(all_image_name_list)
# print(num)
for index in range(num):
	if index < num * 0.8:
		shutil.copy(all_image_path + "/" + all_image_name_list[index],
					train_path + "/image")
		shutil.copy(all_label_path + "/" + all_label_name_list[index],
					train_path + "/label")
	elif index < num * 0.9:
		shutil.copy(all_image_path + "/" + all_image_name_list[index],
					validation_path + "/image")
		shutil.copy(all_label_path + "/" + all_label_name_list[index],
					validation_path + "/label")
	else :
		shutil.copy(all_image_path + "/" + all_image_name_list[index],
					test_path + "/image")
		shutil.copy(all_label_path + "/" + all_label_name_list[index],
					test_path + "/label")