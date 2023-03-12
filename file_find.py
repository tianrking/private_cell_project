import os
import shutil
'''
将总数据集中的文件按名字不同分开放在不同的文件夹中
需要有一个已经分好的文件夹做标准
'''
# 图名写进文档
# filename = os.listdir(r"F:\ww\lwd\data_only\Data\image_rgb")
# for i in range(1,19):
# 	name = os.listdir(r"F:\ww\lwd\data_only\Data\image_rgb"+"/"+filename[i])
# 	writechar = name[0]
# 	print(writechar[0:-7])
# 	f=open(r"F:\ww\lwd\data_only\Data\image_rgb\name.txt",'a')
# 	f.write(writechar[0:-7]+"\n")
# 	f.close()
##########################
all_image_path = r"F:\ww\lwd\data_only\Data\zhiyun_yuanhe_heren\image_all" # 全部原图所在文件夹
all_label_path = r"F:\ww\lwd\data_only\Data\zhiyun_yuanhe_heren\label_all" # 全部标签图所在文件夹
save_image_path = r"F:\ww\lwd\data_only\Data\zhiyun_yuanhe_heren\image_file" # 原图分类保存上级文件夹
save_label_path = r"F:\ww\lwd\data_only\Data\zhiyun_yuanhe_heren\label_file" # 标签图分类保存上级文件夹
txt_path = r"F:\ww\lwd\data_only\Data\image_rgb\name.txt"

all_image_name_list = os.listdir(all_image_path) # 文件名做成的列表，带后缀
all_label_name_list =os.listdir(all_label_path)

txt_file = open(r"F:\ww\lwd\data_only\Data\image_rgb\name.txt","r")
txt_list = txt_file.read().split("\n") # 以换行符为分界读取txt文件中的内容形成列表
txt_file.close()

file_name = os.listdir(r"F:\ww\lwd\data_only\Data\image_rgb")
for i in range(1,19): # 生成文件夹
	if not os.path.exists(save_image_path + "/" + file_name[i]):
		os.mkdir(save_image_path + "/" + file_name[i])
	if not os.path.exists(save_label_path + "/" + file_name[i]):
		os.mkdir(save_label_path + "/" + file_name[i])
# print(txt_list[1])
for name1 in all_image_name_list:
	for name2_index in range(len(txt_list)):
		if name1[0:-7] == txt_list[name2_index]:
			shutil.copy(all_image_path + "/" + name1, save_image_path + "/" + file_name[name2_index+1])
			shutil.copy(all_label_path + "/" + name1[0:-3] + "png", save_label_path + "/" + file_name[name2_index + 1])