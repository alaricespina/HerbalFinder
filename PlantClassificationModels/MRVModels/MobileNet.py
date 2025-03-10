import tensorflow as tf
#import all necessary layers
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model
# MobileNet block

# stem of the model


def mobilnet_block (x, filters, strides):
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# main part of the model
def MRV_MobileNet(Input_Shape = (64,64,3), CONV_CONSTANT=64, DENSE_CONSTANT = 10, NUM_CLASSES=10):
    
    input = Input(shape = Input_Shape)

    x = Conv2D(filters = CONV_CONSTANT // 2, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)


    x = mobilnet_block(x, filters = CONV_CONSTANT, strides = 1)
    x = mobilnet_block(x, filters = CONV_CONSTANT * 2, strides = 2)
    x = mobilnet_block(x, filters = CONV_CONSTANT * 2, strides = 1)
    x = mobilnet_block(x, filters = CONV_CONSTANT * 4, strides = 2)
    x = mobilnet_block(x, filters = CONV_CONSTANT * 4, strides = 1)
    x = mobilnet_block(x, filters = CONV_CONSTANT * 8, strides = 2)

    for _ in range (5):
        x = mobilnet_block(x, filters = CONV_CONSTANT * 8, strides = 1)

    x = mobilnet_block(x, filters = CONV_CONSTANT * 16, strides = 2)
    x = mobilnet_block(x, filters = CONV_CONSTANT * 16, strides = 1)

    x = AvgPool2D (pool_size = 2, strides = 1, data_format='channels_first')(x)
    output = Dense(units = DENSE_CONSTANT, activation = 'softmax')(x)
    output = Dense(units = NUM_CLASSES, activation = 'softmax')(x)
    model = Model(inputs=input, outputs=output)

    return model 

if __name__ == "__main__":
    model = MRV_MobileNet((64, 64, 3), 4 , 4*128, 11)
    model.summary()


