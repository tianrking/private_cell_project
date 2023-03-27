from tensorflow.keras.layers import add
from tensorflow.python.keras.models import Model
import tensorflow
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.optimizers import Adam
tensorflow.compat.v1.disable_eager_execution()

def FCN_8S(pretrained_weights=None ,input_size = (512, 512, 3), classNum = 5 ,learning_rate=1e-5):
    inputs = Input(input_size)
###编码器部分
    # 0 height 1 weight
    nClasses =classNum
    conv1 = Conv2D(filters=32, input_shape=(input_size[0], input_size[1], 1),
                   kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv1')(inputs)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block1_conv2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv1')(pool1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block2_conv2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv1')(pool2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv2')(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block3_conv3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
    score_pool3 = Conv2D(filters=nClasses, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool3')(pool3)#此行代码为后面的跳层连接做准备

    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv1')(pool3)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv2')(conv4)
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block4_conv3')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
    score_pool4 = Conv2D(filters=nClasses, kernel_size=(3, 3), padding='same',
                         activation='relu', name='score_pool4')(pool4)#此行代码为后面的跳层连接做准备

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv1')(pool4)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv2')(conv5)
    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu',
                   name='block5_conv3')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(conv5)
###1×1卷积部分，加入了Dropout层以免过拟合
    fc6 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu',
                 name='fc6')(pool5)
    fc6 = Dropout(0.3, name='dropout_1')(fc6)

    fc7 = Conv2D(filters=1024, kernel_size=(1, 1), padding='same', activation='relu',
                 name='fc7')(fc6)
    fc7 = Dropout(0.3, name='dropour_2')(fc7)
###下面的代码为跳层连接结构
    score_fr = Conv2D(filters=nClasses, kernel_size=(1, 1), padding='same',
                      activation='relu', name='score_fr')(fc7)

    score2 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score2")(score_fr)

    add1 = add(inputs=[score2, score_pool4], name="add_1")

    score4 = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score4")(add1)

    add2 = add(inputs=[score4, score_pool3], name="add_2")

    UpSample = Conv2DTranspose(filters=nClasses, kernel_size=(8, 8), strides=(8, 8),
                               padding="valid", activation=None,
                               name="UpSample")(add2)

    outputs = Conv2D(nClasses, 1, activation='softmax')(UpSample)
    #因softmax的特性，跳层连接部分的卷积层都有nClasses个卷积核，以保证softmax的运行
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
