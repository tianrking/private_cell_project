from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.layers import Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, concatenate, \
    UpSampling2D
from keras.layers import merge
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()


# def unet(pretrained_weights = None, input_size = (256, 256, 4), classNum = 2, learning_rate = 1e-5):

def weighted_categorical_crossentropy(Y_pred, Y_gt, weights=(0.013, 1, 0.167, 0.277)):
    """
    (0.008, 1, 0.107, 0.162)
    weighted_categorical_crossentropy between an output and a target
    loss=-weight*y*log(y')
    :param Y_pred:A tensor resulting from a softmax
    :param Y_gt:A tensor of the same shape as `output`
    :param weights:numpy array of shape (C,) where C is the number of classes
    :return:categorical_crossentropy loss
    Usage:
    weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    """
    weights = np.array(weights)
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.compat.v1.reduce_sum(Y_pred, axis=- 1, keep_dims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = - Y_gt * tf.compat.v1.log(output)
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weights * loss)
    return loss


def categorical_dice(Y_pred, Y_gt, weight_loss=(0.013, 1, 0.167, 0.277)):
    """
    multi label dice loss with weighted
    WDL=1-2*(sum(w*sum(r&p))/sum((w*sum(r+p)))),w=array of shape (C,)
    :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_pred is softmax result
    :param Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                       self.numclass],Y_gt is one hot result
    :param weight_loss: numpy array of shape (C,) where C is the number of classes
    :return:
    """
    weight_loss = np.array(weight_loss)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_mean(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss = -tf.reduce_mean(weight_loss * gen_dice_coef)
    return loss

def categorical_focal_loss(Y_pred, Y_gt, gamma=2, alpha=(1, 1, 1, 1)):
    """
     Categorical focal_loss between an output and a target
    :param Y_pred: A tensor of the same shape as `y_pred`
    :param Y_gt: A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param alpha: Sample category weight,which is shape (C,) where C is the number of classes
    :param gamma: Difficult sample weight
    :return:
    """
    weight_loss = np.array(alpha)
    epsilon = 1.e-5
    # Scale predictions so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # Clip the prediction value to prevent NaN's and Inf's
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    # Calculate Cross Entropy
    cross_entropy = -Y_gt * tf.math.log(output)
    # Calculate Focal Loss
    loss = tf.pow(1 - output, gamma) * cross_entropy
    loss = tf.reduce_sum(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss = tf.reduce_mean(weight_loss * loss)
    return loss


def categorical_dicePcrossentroy(Y_pred, Y_gt, weight=(0.013, 1, 0.167, 0.277), lamda=0.5):
    """
    hybrid loss function from dice loss and crossentroy
    loss=Ldice+lamda*Lfocalloss
    :param Y_pred:A tensor resulting from a softmax(-1,z,h,w,numclass)
    :param Y_gt: A tensor of the same shape as `y_pred`
    :param gamma:Difficult sample weight
    :param alpha:Sample category weight,which is shape (C,) where C is the number of classes
    :param lamda:trade-off between dice loss and focal loss,can set 0.1,0.5,1
    :return:diceplusfocalloss
    """
    weight_loss = np.array(weight)
    smooth = 1.e-5
    smooth_tf = tf.constant(smooth, tf.float32)
    Y_pred = tf.cast(Y_pred, tf.float32)
    Y_gt = tf.cast(Y_gt, tf.float32)
    # Compute gen dice coef:
    numerator = Y_gt * Y_pred
    numerator = tf.reduce_sum(numerator, axis=(1, 2, 3))
    denominator = Y_gt + Y_pred
    denominator = tf.reduce_sum(denominator, axis=(1, 2, 3))
    gen_dice_coef = tf.reduce_sum(2. * (numerator + smooth_tf) / (denominator + smooth_tf), axis=0)
    loss1 = tf.reduce_mean(weight_loss * gen_dice_coef)
    epsilon = 1.e-5
    # scale preds so that the class probas of each sample sum to 1
    output = Y_pred / tf.reduce_sum(Y_pred, axis=- 1, keepdims=True)
    # manual computation of crossentropy
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    loss = -Y_gt * tf.math.log(output)
    loss = tf.reduce_mean(loss, axis=(1, 2, 3))
    loss = tf.reduce_mean(loss, axis=0)
    loss2 = tf.reduce_mean(weight_loss * loss)
    total_loss = (1 - lamda) * (1 - loss1) + lamda * loss2
    return total_loss

def loss_unet(pretrained_weights=None, input_size=(512, 512, 3), classNum=4, learning_rate=1e-4):
    inputs = Input(input_size)
    #  2D卷积层
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs))
    conv1 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1))
    #  对于空间数据的最大池化
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1))
    conv2 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2))
    conv3 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3))
    conv4 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4))
    #  Dropout正规化，防止过拟合
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4))
    conv5 = BatchNormalization()(
        Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5))
    drop5 = Dropout(0.5)(conv5)
    #  上采样之后再进行卷积，相当于转置卷积操作
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))

    try:
        merge6 = concatenate([drop4, up6], axis=3)
    except:
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6))
    conv6 = BatchNormalization()(
        Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6))

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    try:
        merge7 = concatenate([conv3, up7], axis=3)
    except:
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7))
    conv7 = BatchNormalization()(
        Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7))

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    try:
        merge8 = concatenate([conv2, up8], axis=3)
    except:
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8))
    conv8 = BatchNormalization()(
        Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8))

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    try:
        merge9 = concatenate([conv1, up9], axis=3)
    except:
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9))
    conv9 = BatchNormalization()(
        Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9))
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(classNum, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    #  用于配置训练模型（优化器、目标函数、模型评估标准）
    model.compile(optimizer=Adam(lr=learning_rate), loss=categorical_dicePcrossentroy, metrics=['accuracy'])

    #  如果有预训练的权重
    if (pretrained_weights):
        model=load_model(pretrained_weights,custom_objects={'categorical_dicePcrossentroy': categorical_dicePcrossentroy})

    return model

