from keras.layers import Input, Conv2D, AveragePooling2D, Dense
from keras import Model 


# Input (32 x 32 x 1)

def MRV_LeNet(x_in, input_layer_object, CONV_CONSTANT = 6, DENSE_CONSTANT = 84, NUM_CLASSES = 10):

    x = Conv2D(CONV_CONSTANT, (5, 5), activation="tanh")(x_in)

    x = AveragePooling2D((2, 2), 2)(x)

    x = Conv2D(CONV_CONSTANT * 3, (5, 5), activation="tanh")(x)

    x = AveragePooling2D((2, 2), 2)(x)

    x = Conv2D(CONV_CONSTANT * 20, (5, 5), activation="tanh")(x)

    x = Dense(84)(x)

    output = Dense(10, activation='softmax')(x)

    model = Model(inputs = input_layer_object, outputs = output) 

    return model