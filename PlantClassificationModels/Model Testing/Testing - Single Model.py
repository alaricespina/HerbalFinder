import random 
import os 

import numpy as np 
from Custom_VGG_19_Model import CustomVGG19Model
from Custom_SimpleCNN_Model import CustomSimpleCNNModel 
from Custom_Inception_Model import CustomInceptionV3Model
from TensorflowPreTrainedModels import EfficientNetV2B3Model

class TestModel(EfficientNetV2B3Model):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    Model = TestModel()
    Model.prepare_data()
    Model.prepare_model()
    
    # Model.load_existing_model()
    # Model.load_history()
    
    # Inception - val_dense_3_accuracy
    # Simple CNN - val_accuracy
    # VGG - val_accuracy
    Model.fit_data(desired_size = 32, epochs = 5, ckpt_metric = "val_accuracy", ckpt_mode = "max")
    
    print(Model.model.summary())
    Model.show_history()

    # X_TEST = Model.x_test 
    # Y_TEST = Model.y_test 

    # # Random Integer Indexing
    # LL = random.randint(0, len(X_TEST) // 2) 
    # UL = LL + random.randint(2, 10)
    # print(f"Bounds Obtained: {LL} -> {UL}\n")

    # # Single Prediction
    # _, rsp = Model.predict_given_data(np.array([X_TEST[UL]]))
    # print(f"Random Single Prediction\n(PREDICTED):\n{rsp}\n(ACTUAL)\n{np.argmax(Y_TEST[UL])}\n")

    # # Group Prediction
    # _, rgp = Model.predict_given_data(np.array(X_TEST[LL:UL]))
    # print(f"Random Group Prediction\n(PREDICTED)\n{rgp}\n(ACTUAL)\n{np.argmax(Y_TEST[LL:UL], axis=1)}\n")

    # # Model Evaluation (Interface Function)
    # print(Model.eval())

