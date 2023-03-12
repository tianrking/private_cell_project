import os

file1_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\label_all"
file2_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe_heren\label_all"

file1_name = os.listdir(file1_path)
file2_name = os.listdir(file2_path)

file1_name_new = []
file2_name_new = []

for i in range(len(file2_name)):
	file1_name_new.append(file1_name[i][:-4])
	file2_name_new.append(file2_name[i][:-4])
# print(file1_name_new)
k=0
for i in range(len(file1_name_new)):
	if file1_name_new[i] != file2_name_new[i]:
		k = k+1
print(k)
		# print(file1_name_new[i], file2_name_new[i])