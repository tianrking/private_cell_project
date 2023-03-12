import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from seg_unet import unet
from dataProcess import testGenerator, saveResult, color_dict

import tensorflow as tf
# import keras
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 0.6 sometimes works better for folks
# tf.keras.backend.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))


#  训练模型保存地址
model_path = r"D:\ww\lwd\data_only\Data\yuanhe_heren\model/temp_clip/weights.44-0.0074.hdf5"
#model_path = r"D:\ww\lwd\data_only\Data\yuanhe_heren\model\temp_clip\weights.63-0.0196.hdf5"
#  测试数据路径
test_iamge_path = r"D:\ww\lwd\data_only\Data\yuanhe_heren\test\image-clip-resize"   #单词写错image
#  结果保存路径
save_path = r"D:\ww\lwd/data_only/Data/yuanhe_heren/test/temp-clip"
#  测试数据数目
test_num = len(os.listdir(test_iamge_path))
#  类的数目(包括背景)
classNum = 3
#  模型输入图像大小
input_size = (512, 512, 3)
#  生成图像大小
output_size = (200, 200)
#  测试数据标签路径
test_label_path = r"D:\ww\lwd\data_only\Data\yuanhe_heren\test\label-clip-resize"
colorDict_RGB, colorDict_GRAY = color_dict(test_label_path, classNum)
# print(colorDict_RGB)
model = unet(model_path)

testGene = testGenerator(test_iamge_path, input_size)

#  预测值的Numpy数组
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)
# print(results)
#  保存结果
saveResult(test_iamge_path, save_path, results, colorDict_RGB, output_size)
