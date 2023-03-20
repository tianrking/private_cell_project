from sklearn.model_selection import train_test_split

import os
import shutil

import yaml
import sys
import getopt

root_dir = None
argv = sys.argv[1:]
opts = []


try:
    opts, args = getopt.getopt(argv, "d:")
except:
    print("Error")

for opt, arg in opts:
    if opt in ['-d']:
        root_dir = arg

import os
import shutil
def save_images_to_dir(image_files, dst_dir):
    """
    将图像文件保存到指定目录中，目录结构为dst_dir/image.jpg
    """
    os.makedirs(dst_dir, exist_ok=True)
    for file_path in image_files:
        file_name = os.path.basename(file_path)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copyfile(file_path, dst_path)

# root_dir = "/root/DDD/heren-yuanhe-zhiyun/"
# 原始图片数据路径
data_dir = 'image_roi_all'
data_dir = root_dir + data_dir
# 标签数据图片路径
label_dir = 'label_all'
label_dir = root_dir + label_dir
# 划分比例
test_size = 0.2
# 训练集和测试集保存路径

train_image_dir = root_dir + 'train/' + 'image_dir'
train_label_dir = root_dir + 'train/' + 'label_dir'
test_image_dir = root_dir + 'test/' + 'image_dir'
test_label_dir = root_dir + 'test/' + 'label_dir'

# 获取所有原始图片文件的路径
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
# 获取所有标签图片文件的路径
label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

# 使用train_test_split函数将数据集划分为训练集和测试集
train_image_files, test_image_files, train_label_files, test_label_files = train_test_split(image_files, label_files, test_size=test_size, random_state=42)

# 将训练集和测试集保存到指定路径
save_images_to_dir(train_image_files, train_image_dir)
save_images_to_dir(test_image_files, test_image_dir)
save_images_to_dir(train_label_files, train_label_dir)
save_images_to_dir(test_label_files, test_label_dir)
