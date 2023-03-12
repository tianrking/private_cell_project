# pip install pyyaml

# coding:utf-8

import yaml

# 打开配置文件
f = open(r"E:\w0x7ce_td\O\config\unet.yaml", encoding="utf-8")
config = yaml.load(f, Loader=yaml.FullLoader)
print(config)
