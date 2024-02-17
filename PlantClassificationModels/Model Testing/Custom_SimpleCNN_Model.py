from BaseClassifier import BaseClassifier 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

class CustomSimpleCNNModel(BaseClassifier):
    def __init__(self, model_name = "CUSTOM_SimpleCNNMODEL"):
        super().__init__(model_name = model_name)
        #self.checkpoint_path = "Training Checkpoints/" + self.model_name

    def prepare_model(self):
        self.model = Sequential([
            Conv2D(8, (3,3), activation="tanh"),
            Flatten(),
            Dense(self.num_unique_classes, activation="softmax")
        ])

        self.model.compile(loss = "categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
