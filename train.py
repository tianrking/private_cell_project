import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Unet_model import unet
from fcn_test import FCN_8S
from deeplabv3plus_model import Deeplabv3 # accuracy
from SegNet_model import segnet #accuracy
from FCN_model import fcn_8 #accuracy
from attention_unet import at_unet
from loos_unet import loss_unet
from double_unet import  double_unet
from dataProcess import trainGenerator, color_dict
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
'''
数据集相关参数
'''
#  训练数据图像路径
train_image_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\image-aug"
    # r"F:\ww\unet\keras-segmentation-master\data\dataset1\images_prepped_train"
#  训练数据标签路径
train_label_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\train\label-aug"
#  验证数据图像路径
validation_image_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\validation\image-aug"
#r"F:\ww\unet\keras-segmentation-master\data\dataset1\images_prepped_test"
#  验证数据标签路径
validation_label_path = r"F:\ww\lwd\data_only\Data\xibao_zhiyun_yuanhe\validation\label-aug"



'''
模型相关参数
'''
#  批大小
batch_size = 4
#  类的数目(包括背景)
classNum = 4
#  模型输入图像大小
input_size = (512, 512, 3)
#  训练模型的迭代总轮数
epochs = 120
#  初始学习率
learning_rate = 1e-4

premodel_path = None

#  训练模型保存地址
model_path = r"F:\w0x7ce_storage\deeplabv3\cell_yuanhe_heren\weights.{epoch:02d}-{val_loss:.4f}.hdf5" #test deeplabv3
# model_path = r"E:\lwd\model\diceandce\weights.{epoch:02d}-{val_loss:.5f}.hdf5"
    # "Model\\unet_model.hdf5"
    # \weights.{epoch:02d}-{val_loss:.5f}.hdf5

#  训练数据数目
train_num = len(os.listdir(train_image_path))
#  验证数据数目
validation_num = len(os.listdir(validation_image_path))
#  训练集每个epoch有多少个batch_size
steps_per_epoch = train_num / batch_size
#  验证集每个epoch有多少个batch_size
validation_steps = validation_num / batch_size
#  标签的颜色字典,用于onehot编码
colorDict_RGB, colorDict_GRAY = color_dict(train_label_path, classNum)


#  得到一个生成器，以batch_size的速率生成训练数据
train_Generator = trainGenerator(batch_size,
                                 train_image_path, 
                                 train_label_path,
                                 classNum ,
                                 colorDict_RGB,
                                 input_size)

#  得到一个生成器，以batch_size的速率生成验证数据
validation_data = trainGenerator(batch_size,
                                 validation_image_path,
                                 validation_label_path,
                                 classNum,
                                 colorDict_RGB,
                                 input_size)

#  定义模型
model = Deeplabv3(pretrained_weights=models_dir)
# model = loss_unet()
# model = seg_hrnet(pretrained_weights = premodel_path,
#                  input_size = input_size,
#                  classNum = classNum,
#                  learning_rate = learning_rate)



#  打印模型结构
model.summary()
#  回调函数
#  val_loss连续10轮没有下降则停止训练
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 50)
#  当3个epoch过去而val_loss不下降时，学习率减半
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)

model_checkpoint = ModelCheckpoint(model_path,
                                   monitor = 'loss',
                                   verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                   save_best_only = True,)
                                   # period=5)

#  获取当前时间
start_time = datetime.datetime.now()

#  模型训练
history = model.fit_generator(train_Generator,
                    steps_per_epoch = steps_per_epoch,
                    epochs = epochs,
                    callbacks = [early_stopping, reduce_lr, model_checkpoint],
                    validation_data = validation_data,
                    validation_steps = validation_steps)

#  训练总时间
end_time = datetime.datetime.now()
log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
time = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
print(log_time)
with open('TrainTime_%s.txt'%time,'w') as f:
    f.write(log_time)
    
#  保存并绘制loss,acc
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
book = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = book.add_sheet('test', cell_overwrite_ok=True)
for i in range(len(acc)):
    sheet.write(i, 0, str(acc[i]))
    sheet.write(i, 1, str(val_acc[i]))
    sheet.write(i, 2, str(loss[i]))
    sheet.write(i, 3, str(val_loss[i]))
book.save(r'AccAndLoss_%s.xls'%time)
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("accuracy_%s.png"%time, dpi = 300)
plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("loss_%s.png"%time, dpi = 300)
plt.show()
