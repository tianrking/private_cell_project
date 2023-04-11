import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from fcn_test import *
from seg_unet import unet
from deeplabv3plus_model import Deeplabv3
from dataProcess import testGenerator, saveResult, color_dict

import tensorflow as tf
# import keras
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 0.6 sometimes works better for folks
# tf.keras.backend.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))


#  训练模型保存地址
model_path = r"E:\w0x7ce_td\1_2_3_4_train_test_0.8\seg_fcn\output_weight\heren-yuanhe-zhiyun-xibao.h5"

#  测试数据路径
test_iamge_path = r"E:\w0x7ce_td\A\0.8\heren-yuanhe-zhiyun-xibao\test\image_dir"  
#  测试数据标签路径
test_label_path = r"E:\w0x7ce_td\A\0.8\heren-yuanhe-zhiyun-xibao\test\label_dir"
#  结果保存路径
save_path = r"E:\w0x7ce_td\A\predict\seg_fcn"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 5
#  模型输入图像大小
xx = 512
input_size = (xx,xx, 3)
#  生成图像大小
# output_size = (200, 200)
output_size = (xx,xx)

colorDict_RGB, colorDict_GRAY = color_dict(test_label_path, classNum)
# print(colorDict_RGB)
# model = unet(model_path)
# model = Deeplabv3(model_path)
model =  FCN_8S(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)
# print(results)
#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_RGB, output_size)
