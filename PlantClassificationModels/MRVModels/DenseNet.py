import tensorflow as tf
from tensorflow.keras.layers import Input, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import ReLU, AvgPool2D, Flatten, Dense
from tensorflow.keras import Model

def MRV_DenseNet(Input_Shape = (64,64,3), CONV_CONSTANT=64, DENSE_CONSTANT = 10, NUM_CLASSES=10):
    
    input = Input(shape = Input_Shape)

    x = Conv2D(filters = CONV_CONSTANT // 2, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = CONV_CONSTANT // 2, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = CONV_CONSTANT, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = CONV_CONSTANT, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = CONV_CONSTANT * 2, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters = CONV_CONSTANT * 2, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Flatten()(x)
    output = Dense(units = DENSE_CONSTANT, activation = 'softmax')(x)
    output = Dense(units = NUM_CLASSES, activation = 'softmax')(x)
    model = Model(inputs=input, outputs=output)

    return model 

if __name__ == "__main__":
    model = MRV_DenseNet((64, 64, 3), 4 , 4*128, 11)
    model.summary()