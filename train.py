#READ CONFIG TRAIN.yml
################################
DEBUG = 0
################################

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import os
# os.environ["TF_KERAS"] = '1'

import tensorflow as tf

# tf.compat.v1.ConfigProto() #这是tensorflow2.0+版本的写法，这个方法的作用就是设置运行tensorflow代码的时候的一些配置，例如如何分配显存，是否打印日志等;所以它的参数都是　配置名称＝True/False(默认为False) 这种形式
# gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)# 限制GPU资源的使用，此处allow_growth=True是动态分配显存，需要多少，申请多少，不是一成不变、而是一直变化
# sess = tf.compat.v1.Session(config=config)  # 　让这些配置生效

import os
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
# 选择编号为0的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"#我的笔记本只有一块GPU，编号是0，所以这里调用编号为0的GPU

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from seg_unet import *
from fcn_test import *
from deeplabv3plus_model import *

# from Model.seg_unet import unet
# from loss_unet import loss_unet
from dataProcess import trainGenerator, color_dict
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import datetime
import xlwt
import tensorflow
# from tensorflow.python.client import device_lib
# #
# # print(device_lib.list_local_devices())
# # print(tensorflow.test.is_gpu_available())

################################
# Read Configuration
################################

import yaml
import sys
import getopt

class D_path:
        
    def __init__(self,config) -> None:
        self.config = config
        self.set_path(config)
            
    def set_path(self,config) -> None:
        self.train_image_path = self.config['base_config']['train_image_path']
        self.train_label_path = self.config['base_config']['train_label_path']
        self.validation_image_path = self.config['base_config']['validation_image_path']
        self.validation_label_path = self.config['base_config']['validation_label_path']
        
    @classmethod
    def Print_config(cls) -> bool:
        
        pass
        
        
class D_parameter:
    
    def __init__(self,config) -> None:
        self.config = config
        self.set_parameter(self.config)
        
    def set_parameter(self,config) -> None:
        self.batch_size = config['model_config']['batch_size']
        self.classNum = config['model_config']['class_Num']
        self.input_size = config['model_config']['input_size']
        self.epochs = config['model_config']['epochs']
        self.learning_rate = config['model_config']['learning_rate']
        
        self.premodel_path = config['model_config']['premodel_path']
        
        self.model_path = config['model_config']['model_path']
    
    @classmethod
    def Print_config(cls,config) -> bool:
        pass

config = None
read_config_flag = False
data_dir = None
data_config = None
argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "c:u:")
except:
    print("Error")

for opt, arg in opts:
    if opt in ['-c']:
        config = arg
        read_config_flag = True
        
if read_config_flag is True:
    f = open(config, encoding="utf-8")
    config = yaml.load(f, Loader=yaml.FullLoader)
    data_dir = D_path(config)
    data_config = D_parameter(config)

################################


'''
数据集相关参数
'''

data_list = ["heren","heren-yuanhe","heren-yuanhe-zhiyun","heren-yuanhe-zhiyun-xibao"]

model_list = []

premodel_path = None

for data_type in data_list:
#  训练数据图像路径

    # train_image_path = r"E:\w0x7ce_td\A\0.8\%s\image_roi_all"%(data_type)
    # # print(train_image_path)
    #     # r"F:\ww\unet\keras-segmentation-master\data\dataset1\images_prepped_train"
    # #  训练数据标签路径
    # train_label_path = r"E:\w0x7ce_td\A\0.8\%s\label_all"%(data_type)
    # #  验证数据图像路径
    # validation_image_path = r"E:\w0x7ce_td\A\0.8\%s\image_roi_all"%(data_type)
    # #r"F:\ww\unet\keras-segmentation-master\data\dataset1\images_prepped_test"
    # #  验证数据标签路径
    # validation_label_path = r"E:\w0x7ce_td\A\0.8\%s\label_all"%(data_type)
    
    train_image_path = r"E:\w0x7ce_td\A\0.8\%s\train\image_dir"%(data_type)
    train_label_path = r"E:\w0x7ce_td\A\0.8\%s\train\label_dir"%(data_type)
    validation_image_path = r"E:\w0x7ce_td\A\0.8\%s\test\image_dir"%(data_type)
    validation_label_path = r"E:\w0x7ce_td\A\0.8\%s\test\label_dir"%(data_type)


    '''
    模型相关参数
    '''
    #  批大小c
    batch_size = 4
    #  类的数目(包括背景)
    classNum = 5
    #  模型输入图像大小
    input_size = (512, 512, 3)
    #  训练模型的迭代总轮数
    if DEBUG == 1 :
        epochs = 1
    else :
        epochs = 100
    #  初始学习率
    learning_rate = 1e-5
    #  预训练模型地址,没有为none
    #xibao最佳模型路径F:\ww\lwd\data_only\Data\xibao\model\model3000-4-4-5-120\weights.107-0.05957.hdf5
    # premodel_path =  r"E:\ww\CTsegthor\model\model-1out-stage4\weights.78-0.00929.hdf5"
        #r"F:\ww\lwd\data_only\Data\xibao_yuanhe\model\4-4-4-120-xibao-1\weights.19-0.00613.hdf5"
        # r"E:\ww\CTsegthor\model\all\weights.12-0.01424.hdf5"比较好的一次unet可以作为premodel
        # None
    # premodel_path = None

    #  训练模型保存地址

    model_path = r"E:\w0x7ce_td\O\output_weight\%s-weights.{epoch:02d}-{val_loss:.5f}.hdf5"%(data_type)
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

    if premodel_path is None:

        # model = unet(   input_size = input_size,
        #                 classNum = classNum,
        #                 learning_rate = learning_rate)
        # model = FCN_8S(   input_size = input_size,
        #                 classNum = classNum,
        #                 learning_rate = learning_rate)
        
        model = Deeplabv3(   input_size = input_size,
                classNum = classNum,
                learning_rate = learning_rate)

    else:
        # model = unet(pretrained_weights = premodel_path,
        #                 input_size = input_size,
        #                 classNum = classNum,
        #                 learning_rate = learning_rate)
        # model = FCN_8S(pretrained_weights = premodel_path,
        #                 input_size = input_size,
        #                 classNum = classNum,
        #                 learning_rate = learning_rate)
        model = Deeplabv3(pretrained_weights = premodel_path,
                        input_size = input_size,
                        classNum = classNum,
                        learning_rate = learning_rate)

    #  打印模型结构
    model.summary()
    #  回调函数
    #  val_loss连续10轮没有下降则停止训练
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
    #  当3个epoch过去而val_loss不下降时，学习率减半
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)

    model_checkpoint = ModelCheckpoint(model_path,
                                    monitor = 'loss',
                                    verbose = 1,# 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                    save_best_only = True)
                                    # period=3)
                                    # period=5#每5次保存模型文件)

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
        
    model_path = r"E:\w0x7ce_td\O\output_weight\%s.h5"%(data_type)
    model_list.append(model_path)
    
    try:
        model.save(model_path)
    except:
        print("save error")
        
    
    premodel_path = model_path
    print(premodel_path)
    
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
    # plt.show()
    
    # break
