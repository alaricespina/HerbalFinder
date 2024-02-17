import keras

import keras.backend as K
import tensorflow as tf
from keras.datasets import cifar10
 
from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import cv2 
import numpy as np 
from keras.datasets import cifar10 
from keras import backend as K 
from keras.utils import np_utils

import math 
from keras.optimizers import SGD 
from keras.callbacks import LearningRateScheduler

num_classes = 10

def load_cifar10_data(img_rows, img_cols):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:,:,:,:]])
    X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)
    
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')

    # preprocess data
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    return X_train, Y_train, X_valid, Y_valid
 
def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def MRV_Inception(Input_Shape = (64, 64, 3), CONV_CONSTANT = 64, NUM_CLASSES = 10, **kwargs):
    # Input_Shape = (64, 64, 3), CONV_CONSTANT = 64, REDUC_CONSTANT = 16, POOL_CONSTANT = 32
    REDUC_CONSTANT = kwargs["REDUC_CONSTANT"]
    POOL_CONSTANT = kwargs["POOL_CONSTANT"]


    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)   
    

    input_layer = Input(Input_Shape)

    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2')(x)
    x = Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1')(x)
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1')(x)
    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT,
                        filters_3x3_reduce=REDUC_CONSTANT * 2,
                        filters_3x3=CONV_CONSTANT * 2,
                        filters_5x5_reduce=REDUC_CONSTANT,
                        filters_5x5=CONV_CONSTANT // 2,
                        filters_pool_proj=POOL_CONSTANT,
                        name='inception_3a')

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 2,
                        filters_3x3_reduce=REDUC_CONSTANT * 8,
                        filters_3x3=CONV_CONSTANT * 3,
                        filters_5x5_reduce=REDUC_CONSTANT * 2,
                        filters_5x5=CONV_CONSTANT * 3 // 2,
                        filters_pool_proj=POOL_CONSTANT * 2,
                        name='inception_3b')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 3,
                        filters_3x3_reduce=REDUC_CONSTANT * 6,
                        filters_3x3=CONV_CONSTANT * 13 // 4,
                        filters_5x5_reduce=REDUC_CONSTANT * 2,
                        filters_5x5=CONV_CONSTANT * 3 // 4,
                        filters_pool_proj=POOL_CONSTANT * 2,
                        name='inception_4a')


    # x1 = AveragePooling2D((4, 4), strides=3)(x)
    # x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    # x1 = Flatten()(x1)
    # x1 = Dense(1024, activation='relu')(x1)
    # x1 = Dropout(0.7)(x1)
    # x1 = Dense(10, activation='softmax', name='auxilliary_output_1')(x1)

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 5 // 2,
                        filters_3x3_reduce=REDUC_CONSTANT * 7,
                        filters_3x3=CONV_CONSTANT * 7 // 2,
                        filters_5x5_reduce=REDUC_CONSTANT * 3 // 2,
                        filters_5x5=CONV_CONSTANT,
                        filters_pool_proj=POOL_CONSTANT * 2,
                        name='inception_4b')

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 2,
                        filters_3x3_reduce=REDUC_CONSTANT * 9,
                        filters_3x3=CONV_CONSTANT * 4,
                        filters_5x5_reduce=REDUC_CONSTANT * 3 // 2,
                        filters_5x5=CONV_CONSTANT,
                        filters_pool_proj=POOL_CONSTANT * 2,
                        name='inception_4c')

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 7 // 4,
                        filters_3x3_reduce=REDUC_CONSTANT * 9,
                        filters_3x3=CONV_CONSTANT * 9 // 2,
                        filters_5x5_reduce=REDUC_CONSTANT * 2,
                        filters_5x5=CONV_CONSTANT,
                        filters_pool_proj=POOL_CONSTANT * 2,
                        name='inception_4d')


    # x2 = AveragePooling2D((4, 4), strides=3)(x)
    # x2 = Conv2D(128, (1, 1), padding='same', activation='relu')(x2)
    # x2 = Flatten()(x2)
    # x2 = Dense(1024, activation='relu')(x2)
    # x2 = Dropout(0.7)(x2)
    # x2 = Dense(10, activation='softmax', name='auxilliary_output_2')(x2)

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 4,
                        filters_3x3_reduce=REDUC_CONSTANT * 10,
                        filters_3x3=CONV_CONSTANT * 5,
                        filters_5x5_reduce=REDUC_CONSTANT * 2,
                        filters_5x5=CONV_CONSTANT * 2,
                        filters_pool_proj=POOL_CONSTANT * 4,
                        name='inception_4e')

    x = MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(x)

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 4,
                        filters_3x3_reduce=REDUC_CONSTANT * 10,
                        filters_3x3=CONV_CONSTANT * 5,
                        filters_5x5_reduce=REDUC_CONSTANT * 2,
                        filters_5x5=CONV_CONSTANT * 2,
                        filters_pool_proj=POOL_CONSTANT * 4,
                        name='inception_5a')

    x = inception_module(x,
                        filters_1x1=CONV_CONSTANT * 6,
                        filters_3x3_reduce=REDUC_CONSTANT * 12,
                        filters_3x3=CONV_CONSTANT * 6,
                        filters_5x5_reduce=REDUC_CONSTANT * 3,
                        filters_5x5=CONV_CONSTANT * 2,
                        filters_pool_proj=POOL_CONSTANT * 4,
                        name='inception_5b')

    x = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(x)

    x = Dropout(0.4)(x)

    x = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(input_layer, x, name='inception_v1')

    return model


# epochs = 25
# initial_lrate = 0.01

# def decay(epoch, steps=100):
#     initial_lrate = 0.01
#     drop = 0.96
#     epochs_drop = 8
#     lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
#     return lrate

# sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

# lr_sc = LearningRateScheduler(decay, verbose=1)

# model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

# history = model.fit(X_train, [y_train, y_train, y_train], validation_data=(X_test, [y_test, y_test, y_test]), epochs=epochs, batch_size=256, callbacks=[lr_sc])
