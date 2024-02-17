from keras.layers import Permute, Reshape 
from keras.layers import Conv2D, BatchNormalization, Add, DepthwiseConv2D, Dense
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import ReLU, Input, concatenate 
from keras import Model

def channel_shuffle(x, groups):
    _, width, height, channels = x.get_shape().as_list()
    group_ch = channels // groups
    x = Reshape([width, height, group_ch, groups])(x)
    x = Permute([1, 2, 4, 3])(x)
    x = Reshape([width, height, channels])(x)
    return x

def shuffle_unit(x, groups, channels,strides):
    y = x
    x = Conv2D(channels//4, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = channel_shuffle(x, groups)
    x = DepthwiseConv2D(kernel_size = (3,3), strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    if strides == (2,2):
       channels = channels - y.shape[-1]
    x = Conv2D(channels, kernel_size = 1, strides = (1,1),padding = 'same', groups=groups)(x)
    x = BatchNormalization()(x)
    
    if strides ==(1,1):
        x = Add()([x,y])
    if strides == (2,2):  
        y = AveragePooling2D((3,3), strides = (2,2), padding = 'same')(y)
        x = concatenate([x,y])
    x = ReLU()(x)
    return x

def MRV_Shuffle_Net(nclasses, start_channels ,input_shape = (224,224,3), rep = [3, 7, 3]):
    groups = 2
    input_layer = Input (input_shape)
    x =  Conv2D (24,kernel_size=3,strides = (2,2), padding = 'same', use_bias = True)(input_layer)
    x =  BatchNormalization()(x)
    x =  ReLU()(x)
    x = MaxPooling2D (pool_size=(3,3), strides = 2, padding='same')(x)
    repetitions = rep
    for i, repetition in enumerate(repetitions):
        channels = start_channels * (2**i)
        x  = shuffle_unit(x, groups, channels,strides = (2,2))
        for i in range(repetition):
            x = shuffle_unit(x, groups, channels,strides=(1,1))
            
    x = GlobalAveragePooling2D()(x)
    output = Dense(nclasses,activation='softmax')(x)
    model = Model(input_layer, output)
    return model
