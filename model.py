import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from keras.models import Sequential
from keras.layers import MaxPool3D, UpSampling3D
from keras.layers.core import Activation, Lambda, Dense, Flatten
from keras.layers.convolutional import Conv3D, Deconv3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
#from keras.optimizers import adam
from constants import *
from visualize import *

alpha = 0
#opt = adam(lr=0.005)

def build_model():
        kernel_size=(4, 4, 4)
        strides=(2, 2, 2)

        model = Sequential()
        model.add(Conv3D(32, kernel_size=(4, 4, 4), strides=strides, padding='same', activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv3D(64, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv3D(128, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv3D(256, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(Conv3D(512, kernel_size=(2, 2, 2), strides=1, padding='same', activation='relu'))
        model.add(Deconv3D(512, kernel_size=(2, 2, 2), strides=1, padding='same', activation='relu'))
        model.add(Deconv3D(256, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(Deconv3D(128, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Deconv3D(64, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Deconv3D(32, kernel_size=(2, 2, 2), strides=strides, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Deconv3D(1, kernel_size=(4, 4, 4), strides=1, padding='same', activation='sigmoid'))

        model.compile(loss='mse', metrics=[keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.MeanSquaredError()], optimizer='adam')
        model.summary()

        return model


def build_model_2():
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(2, 2, 2), padding='same', activation='relu', input_shape=input_shape))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(64, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(128, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(256, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(512, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(MaxPool3D(pool_size=(2, 2, 2)))

        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Deconv3D(512, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Deconv3D(256, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Deconv3D(128, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Deconv3D(64, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(UpSampling3D(size=(2, 2, 2)))
        model.add(Deconv3D(32, kernel_size=(2, 2, 2), padding='same', activation='relu'))
#        model.add(Deconv3D(1, kernel_size=(2, 2, 2), padding='same', activation='relu'))
        model.add(Dense(1))
#        model.add(Lambda(lambda x: scale(x)))

        model.compile(loss='mse', metrics=[keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.MeanSquaredError()], optimizer='adam')
        model.summary()

        return model

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
