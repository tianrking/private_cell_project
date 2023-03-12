import os
import shutil

readpathname = r"F:\ww\lwd\data_only\class_Data\xibao_only\fen_zu_wen_jian\E0001" # 起始文件夹名字，结尾为数字
outpathname = r"F:\ww\lwd\data_only\class_Data\xibao_only\ji_he" # 目标文件夹

def read_path(file_pathname, save_pathname):
	#遍历该目录下的所有图片文件
	for filename in os.listdir(file_pathname):
		filepath = file_pathname + "\\" + filename
		# print(filepath)
		shutil.copy(filepath, save_pathname)


for i in range(18):
	if i+1 < 10:
		newpath = readpathname.replace(readpathname[-1], str(i+1))
	if i+1 >= 10:
		newpath = readpathname.replace(readpathname[-2:], str(i+1))
	print(newpath)
	read_path(newpath, outpathname)