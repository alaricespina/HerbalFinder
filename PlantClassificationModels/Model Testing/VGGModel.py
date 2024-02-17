import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings 
warnings.filterwarnings("ignore")

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential 

from BaseClassifier import BaseClassifier

class CustomVGG19Model(BaseClassifier):
    def __init__(self, model_name = "CUSTOM_VGG19MODEL"):
        super().__init__(model_name = model_name)

    def prepare_model(self):
        dense_units = 64
        dense_activation = "relu"
        base_unit = 8
        activation = "relu"
        kernel_size = (3, 3)
        padding = "same"

        self.model = Sequential([
            Conv2D(base_unit * 1, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 1, kernel_size, activation=activation, padding=padding),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(base_unit * 2, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 2, kernel_size, activation=activation, padding=padding),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(base_unit * 4, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 4, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 4, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 4, kernel_size, activation=activation, padding=padding),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            Conv2D(base_unit * 8, kernel_size, activation=activation, padding=padding),
            MaxPooling2D(pool_size=(2,2)),
            Flatten(),
            Dense(dense_units, activation=dense_activation),
            Dropout(0.5),
            Dense(dense_units, activation=dense_activation),
            Dropout(0.5),
            Dense(self.num_unique_classes, activation="softmax")
        ])

        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

    
