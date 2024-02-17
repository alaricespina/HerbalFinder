#import all necessary layers
from keras.layers import Input, DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization
from keras.layers import ReLU, AvgPool2D, Flatten, Dense
from keras import Model

# MobileNet block
def mobilnet_block (x, filters, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def MRV_MobileNet(input_size = (224, 224, 3), _C = 64, _D = 512):

    #stem of the model
    input = Input(shape = input_size)
    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # main part of the model
    x = mobilnet_block(x, filters = _C, strides = 1)
    x = mobilnet_block(x, filters = _C * 2, strides = 2)
    x = mobilnet_block(x, filters = _C * 2, strides = 1)
    x = mobilnet_block(x, filters = _C * 4, strides = 2)
    x = mobilnet_block(x, filters = _C * 4, strides = 1)
    x = mobilnet_block(x, filters = _C * 8, strides = 1)

    for _ in range (5):
        x = mobilnet_block(x, filters = _C * 8 , strides = 1)

    x = mobilnet_block(x, filters = _C * 16, strides = 1)
    x = mobilnet_block(x, filters = _C * 16, strides = 1)

    x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
    # x = AvgPool2D (pool_size = 7, strides = 1)(x)
    x = Flatten()(x)

    x = Dense (units = _D)(x)
    output = Dense (units = 10, activation="softmax")(x)
    model = Model(inputs=input, outputs=output)
    
    return model

#plot the model

#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_dtype=False,show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)