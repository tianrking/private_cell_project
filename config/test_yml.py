# pip install pyyaml

# coding:utf-8

import yaml
import sys
import getopt

config = None
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "c:u:")
except:
    print("Error")

for opt, arg in opts:
    if opt in ['-c']:
        config = arg

# 打开配置文件
# f = open(r"E:\w0x7ce_td\O\config\unet.yaml", encoding="utf-8")
f = open(config, encoding="utf-8")
config = yaml.load(f, Loader=yaml.FullLoader)
# print(config)

# #  训练图像标签路径
# train_image_path = config['base_config']['train_image_path']
# # #  训练数据标签路径
# train_label_path = config['base_config']['train_label_path']
# # #  验证数据图像路径
# validation_image_path = config['base_config']['validation_image_path']
# # #  验证数据标签路径
# validation_label_path = config['base_config']['validation_label_path']


class D_path:

    # self.config = 0

    def __init__(self, config) -> None:
        self.config = config
        self.set_path(config)

    def set_path(self, config) -> None:
        self.train_image_path = self.config['base_config']['train_image_path']
        self.train_label_path = self.config['base_config']['train_label_path']
        self.validation_image_path = self.config['base_config']['validation_image_path']
        self.validation_label_path = self.config['base_config']['validation_label_path']

    @classmethod
    def Print_config(cls, config) -> None:
        pass


class D_parameter:

    def __init__(self, config) -> None:
        self.config = config
        self.set_parameter(self.config)

    def set_parameter(self, config) -> None:
        self.batch_size = config['model_config']['batch_size']
        self.classNum = config['model_config']['class_Num']
        self.input_size = config['model_config']['input_size']
        self.epochs = config['model_config']['epochs']
        self.learning_rate = config['model_config']['learning_rate']

        self.premodel_path = config['model_config']['premodel_path']

        self.model_path = config['model_config']['model_path']


aa = D_path(config)
print(aa.train_image_path)
print(aa.train_label_path)
