from sklearn.model_selection import train_test_split

import os
import shutil

import yaml
import sys
import getopt

config = None
argv = sys.argv[1:]
opts = []


try:
    opts, args = getopt.getopt(argv, "d:")
except:
    print("Error")

for opt, arg in opts:
    if opt in ['-d']:
        config = arg


class train_test_split_class:

    def __init__(self) -> None:
        pass

    def set_path(self, train_image_files_path, test_image_files_path) -> None:
        self.train_dir = train_image_files_path
        self.test_dir = test_image_files_path

    @classmethod
    def _set_path(cls, path) -> bool:
        pass


def save_images_to_dir(image_files, labels, dst_dir):
    """
    将图像文件保存到指定目录中，目录结构为dst_dir/label/image.jpg
    """
    os.makedirs(dst_dir, exist_ok=True)
    for i, file_path in enumerate(image_files):
        label = labels[i]
        file_name = os.path.basename(file_path)
        dst_subdir = os.path.join(dst_dir, str(label))
        os.makedirs(dst_subdir, exist_ok=True)
        dst_path = os.path.join(dst_subdir, file_name)
        shutil.copyfile(file_path, dst_path)

# 假设您的数据集在'/path/to/data'

# data_dir = '/root/DDD/heren/label_all'
# data_dir = r'E:\w0x7ce_td\A\heren-yuanhe\label_all'


# 获取所有图像文件的路径
image_files = [os.path.join(data_dir, f)
               for f in os.listdir(data_dir) if f.endswith('.png')]

# 定义标签，这里类别0
labels = [0] * len(image_files)

# 使用train_test_split函数将数据集划分为训练集和测试集
train_image_files, test_image_files, train_labels, test_labels = train_test_split(
    image_files, labels, test_size=0.2, random_state=42)

save_images_to_dir(train_image_files, train_labels, 'train')
save_images_to_dir(test_image_files, test_labels, 'test')
