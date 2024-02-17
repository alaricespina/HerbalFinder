from keras.layers import BatchNormalization, Activation, Conv2D, Dropout, Concatenate, AveragePooling2D, Flatten, Dense, Input
from keras import Model

compression = 1

def denseblock(input, l, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l): 
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate > 0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp

## transition Blosck
def transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
         Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    return avg

#output layer
def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(10, activation='softmax')(flat)
    return output

def MRV_DenseNet(image_shape, dense_layers = 7, conv_filters = 8):
    l = dense_layers
    input = Input(shape=image_shape)
    First_Conv2D = Conv2D(30, (3,3), use_bias=False ,padding='same')(input)
    First_Block = denseblock(First_Conv2D, l, conv_filters, 0.5)
    First_Transition = transition(First_Block, conv_filters, 0.5)
    Last_Block = denseblock(First_Transition, l, conv_filters, 0.5)
    output = output_layer(Last_Block)
    model = Model(inputs=[input], outputs=[output])

    return model


