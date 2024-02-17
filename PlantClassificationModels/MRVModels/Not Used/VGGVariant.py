from keras.layers import Conv2D, BatchNormalization, AveragePooling2D, Flatten, Dense 
from keras import Model

# x_in
# x = Conv2D(CONV_CONSTANT, (5, 5), activation="tanh", padding="same")(x_in)
# x = BatchNormalization()(x)
# x = Conv2D(CONV_CONSTANT, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = AveragePooling2D((2, 2), 2)(x)
# x = Conv2D(CONV_CONSTANT * 2, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = Conv2D(CONV_CONSTANT * 2, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = AveragePooling2D((2, 2), 2)(x)
# x = Conv2D(CONV_CONSTANT * 4, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = Conv2D(CONV_CONSTANT * 4, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = AveragePooling2D((2, 2), 2)(x)
# x = Conv2D(CONV_CONSTANT * 8, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = Conv2D(CONV_CONSTANT * 8, (5, 5), activation="tanh", padding="same")(x)
# x = BatchNormalization()(x)
# x = AveragePooling2D((2, 2), 2)(x)
# x = Flatten()(x)
# x = Dense(DENSE_CONSTANT)(x)
# x = BatchNormalization()(x)
# output = Dense(10, activation='softmax')(x)

# model = Model(inputs = x_in, outputs = output) 

# model.compile(loss="categorical_crossentropy",
#               optimizer = "adam",
#               metrics = ["accuracy"])