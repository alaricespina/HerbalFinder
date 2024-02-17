from keras.layers import Conv2D, ReLU, concatenate
from keras.layers import Dropout, AveragePooling2D, MaxPooling2D, Input, Flatten, Dense
from keras import Model 



def fire_module(x,s1,e1,e3):
    s1x = Conv2D(s1,kernel_size = 1, padding = 'same')(x)
    s1x = ReLU()(s1x)
    e1x = Conv2D(e1,kernel_size = 1, padding = 'same')(s1x)
    e3x = Conv2D(e3,kernel_size = 3, padding = 'same')(s1x)
    x = concatenate([e1x,e3x])
    x = ReLU()(x)
    return x

def MRV_SqueezeNet(INPUT_SHAPE = (64, 64, 3), CONV_CONSTANT = 64 , NUM_CLASSES = 10):
    input = Input(INPUT_SHAPE)
    x = Conv2D(96,kernel_size=(7,7),strides=(2,2),padding='same')(input)
    x = MaxPooling2D(pool_size=(3,3), strides = (2,2))(x)
    x = fire_module(x, s1 = CONV_CONSTANT, e1 = CONV_CONSTANT * 4, e3 = CONV_CONSTANT * 4) #2
    x = fire_module(x, s1 = CONV_CONSTANT, e1 = CONV_CONSTANT * 4, e3 = CONV_CONSTANT * 4) #3
    x = fire_module(x, s1 = CONV_CONSTANT * 2, e1 = CONV_CONSTANT * 8, e3 = CONV_CONSTANT * 8) #4
    x = MaxPooling2D(pool_size=(3,3), strides = (2,2))(x)
    x = fire_module(x, s1 = CONV_CONSTANT * 2, e1 = CONV_CONSTANT * 8, e3 = CONV_CONSTANT * 8) #5
    x = fire_module(x, s1 = CONV_CONSTANT * 3, e1 = CONV_CONSTANT * 12, e3 = CONV_CONSTANT * 12) #6
    x = fire_module(x, s1 = CONV_CONSTANT * 3, e1 = CONV_CONSTANT * 12, e3 = CONV_CONSTANT * 12) #7
    x = fire_module(x, s1 = CONV_CONSTANT * 4, e1 = CONV_CONSTANT * 16, e3 = CONV_CONSTANT * 16) #8
    x = MaxPooling2D(pool_size=(3,3), strides = (2,2))(x)
    x = fire_module(x, s1 = CONV_CONSTANT * 4, e1 = CONV_CONSTANT * 16, e3 = CONV_CONSTANT * 16) #9
    x = Dropout(0.5)(x)
    x = Conv2D(NUM_CLASSES, activation="softmax", kernel_size = 1)(x)
    #output = AveragePooling2D(pool_size=(13,13))(x)
    x = AveragePooling2D(pool_size=(3,3))(x)
    output = Flatten()(x)
    


    model = Model(input, output)
    
    return model