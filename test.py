import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Unet_model import unet
from deeplabv3plus_model import Deeplabv3
from SegNet_model import segnet
from attention_unet import at_unet
from FCN_model import fcn_8
from loos_unet import loss_unet
from double_unet import double_unet
from dataProcess import testGenerator, saveResult, color_dict
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)



#  训练模型保存地址
model_path = r"F:\ww\lwd\multi_model\model_weight\at_unet\loss_unet-2\best_weights.59-3111.62966.hdf5"
filepath = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\image_file"
savepath = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\multi_predict\loss_unet"
filename = os.listdir(filepath)
filename1 = ["E0015","E0016","E0017","E0018"]
for name in filename1:
    #  测试数据路径
    test_iamge_path = filepath + '/' + name
    #  结果保存路径
    save_path = savepath + "/" + name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    #  测试数据数目
    test_num = len(os.listdir(test_iamge_path))
    #  类的数目(包括背景)
    classNum = 4
    #  模型输入图像大小
    input_size = (512, 512, 3)
    #  生成图像大小
    output_size = (500, 500)
    #  训练数据标签路径
    train_label_path =  r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\label"
    colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)
    # print(colorDict_RGB)
    model = loss_unet(pretrained_weights=model_path)

    testGene = testGenerator(test_iamge_path, input_size)

    #  预测值的Numpy数组
    results = model.predict_generator(testGene,
                                      test_num,
                                      verbose = 1)
    # print(results)
    #  保存结果
    saveResult(test_iamge_path, save_path, results, colorDict_GRAY, output_size)
