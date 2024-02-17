# import necessary layers

from keras.layers import Input, Conv2D
from keras.layers import MaxPool2D, Flatten, Dense
from keras import Model



def MRV_VGG(Input_Shape = (224, 224, 3), CONV_CONSTANT = 64 , NUM_CLASSES = 10, **kwargs):
    DENSE_CONSTANT = kwargs["DENSE_CONSTANT"]
    
    # Input Layer Shape Recommended : 224, 224, 3 

    # 1st Conv Block
    input_layer = Input(shape=Input_Shape)
    x = Conv2D (filters = CONV_CONSTANT, kernel_size =3, padding ='same', activation='tanh')(input_layer)
    x = Conv2D (filters = CONV_CONSTANT, kernel_size =3, padding ='same', activation='tanh')(x)
    x = MaxPool2D(pool_size = 2, strides = 2, padding ='same')(x)

    # 2nd Conv Block

    x = Conv2D (filters = CONV_CONSTANT * 2, kernel_size =3, padding ='same', activation='tanh')(x)
    x = Conv2D (filters = CONV_CONSTANT * 2, kernel_size =3, padding ='same', activation='tanh')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 3rd Conv block  
    x = Conv2D (filters = CONV_CONSTANT * 4, kernel_size =3, padding ='same', activation='tanh')(x) 
    x = Conv2D (filters = CONV_CONSTANT * 4, kernel_size =3, padding ='same', activation='tanh')(x) 
    x = Conv2D (filters = CONV_CONSTANT * 4, kernel_size =3, padding ='same', activation='tanh')(x) 
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 4th Conv block

    x = Conv2D (filters = CONV_CONSTANT * 8, kernel_size =3, padding ='same', activation='tanh')(x)
    x = Conv2D (filters = CONV_CONSTANT * 8, kernel_size =3, padding ='same', activation='tanh')(x)
    x = Conv2D (filters = CONV_CONSTANT * 8, kernel_size =3, padding ='same', activation='tanh')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # 5th Conv block

    x = Conv2D (filters = CONV_CONSTANT * 8, kernel_size =3, padding ='same', activation='tanh')(x)
    x = Conv2D (filters = CONV_CONSTANT * 8, kernel_size =3, padding ='same', activation='tanh')(x)
    x = Conv2D (filters = CONV_CONSTANT * 8, kernel_size =3, padding ='same', activation='tanh')(x)
    x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)

    # ALL CONV WAS RELU 
    
    # Fully connected layers  
    x = Flatten()(x) 
    x = Dense(units = DENSE_CONSTANT, activation ='relu')(x) 
    x = Dense(units = DENSE_CONSTANT, activation ='relu')(x) 
    output = Dense(units = NUM_CLASSES, activation ='softmax')(x)

    # creating the model

    model = Model (inputs= input_layer, outputs =output)

    return model
