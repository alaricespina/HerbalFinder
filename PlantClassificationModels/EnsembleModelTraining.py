import pandas as pd 
import matplotlib.pyplot as plt 
import pickle 

from keras.utils import to_categorical
import os 
import numpy as np 
import cv2 

from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder

from MRVModels.Inception import MRV_Inception
from MRVModels.SqueezeNet import MRV_SqueezeNet
from MRVModels.VGG import MRV_VGG
from MRVModels.ResNet import MRV_ResNet34

from keras.models import load_model


# Load Unaugmented Data

def load_dataset(npz_dataset_path = "NPZ DATASET/Augmented_CV_Dataset_22_Steps.npz"):
    loaded_arr = np.load(npz_dataset_path)
    _X = loaded_arr["raw_X"]
    _y = loaded_arr["raw_y"]

    return _X, _y 

def transform_data(input_y_data, categorize_data=True, save=True):
    le = LabelEncoder()
    _y = le.fit_transform(input_y_data)
    
    if categorize_data:
        _y = to_categorical(_y)

    if save:
        with open("LabelEncoderData.pkl", "wb") as f:
            pickle.dump(le, f)

    return _y

def split_data(raw_x, raw_y, test_size = 0.2, validation_size = 0.1):
    X_t, X_test, y_t, y_test = train_test_split(raw_x, raw_y, test_size=test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(X_t, y_t, test_size=validation_size)

    return (X_train, X_valid, X_test, y_train, y_valid, y_test)

def build_models():
    CONV_CONSTANT = 4
    DENSE_CONSTANT = CONV_CONSTANT * 128
    NUM_CLASSES = 11

    Inception_Model = MRV_Inception((64, 64, 3), CONV_CONSTANT, NUM_CLASSES, REDUC_CONSTANT = 4, POOL_CONSTANT = 4)
    ResNet_Model = MRV_ResNet34((64, 64, 3), CONV_CONSTANT, NUM_CLASSES)
    VGG_Model = MRV_VGG((64, 64, 3), CONV_CONSTANT, NUM_CLASSES, DENSE_CONSTANT = DENSE_CONSTANT)
    SqueezeNet_Model = MRV_SqueezeNet((64, 64, 3), CONV_CONSTANT, NUM_CLASSES)

    models =  (Inception_Model, ResNet_Model, VGG_Model, SqueezeNet_Model)

    for model in models:
        model.compile(loss="categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])
              
    return models 

def train_single_model(model, X_train, y_train, X_valid, y_valid, epochs = 100):
    history = model.fit(X_train, 
                    y_train, 
                    validation_data = (X_valid, y_valid),
                    validation_split = 0.2,
                    batch_size = X_train.shape[0]//128,
                    epochs = epochs,
                    verbose = 2)
    
    return history.history

def plot_historical_data(history_dict):
    plt.figure()
    plt.plot(history_dict["accuracy"], label="accuracy")
    plt.plot(history_dict["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history_dict["loss"], label="loss")
    plt.plot(history_dict["val_loss"], label="val_loss")
    plt.legend()
    plt.show()

def save_model_and_history(model_obj, model_name, history_obj, history_name, model_folder = "KERAS MODELS", history_folder = "HISTORY OBJECTS"):
    model_obj.save(os.path.join(model_folder, model_name + ".keras"))

    with open(os.path.join(history_folder, history_name + ".pkl"), "wb") as f:
        pickle.dump(history_obj, f)

def evaluate_model(model, X_test, y_test):
    test_acc = model.evaluate(X_test, y_test)

    print("Testing Accuracy:", test_acc)

    test_predictions = model.predict(X_test)
    unique_values = np.argmax(test_predictions, axis=1)
    print("Unique Values Predicted:", unique_values, len(unique_values))

def perform_tasks(**kwargs):
    _model_obj = kwargs["model"]
    _model_name = kwargs["model_name"]
    _history_name = kwargs["history_name"]
    X_train, y_train, X_valid, y_valid, X_test, y_test = kwargs["data"]

    
    _history = train_single_model(_model_obj, X_train, y_train, X_valid, y_valid, epochs=100)
    save_model_and_history(_model_obj, _model_name, _history, _history_name)
    plot_historical_data(_history)
    evaluate_model(_model_obj, X_test, y_test)

    del _model_obj, _history


if __name__ == "__main__":
    os.system("cls")
    _X, _y = load_dataset()
    _y = transform_data(_y)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(_X, _y)



    #bundled_data = split_data(_X, _y)
    bundled_data = (X_train, y_train, X_valid, y_valid, X_test, y_test)

    #Inception_Model, ResNet_Model, VGG_Model, SqueezeNet_Model = build_models()
    models = list(build_models())
    model_names = ["MRV_Inception", "MRV_ResNet", "MRV_VGG", "MRV_SqueezeNet"]
    history_names = ["MRV_Inception_HISTORY", "MRV_ResNet_HISTORY", "MRV_VGG_HISTORY", "MRV_SqueezeNet_HISTORY"]
    
    # Inception done
    i = 3
    perform_tasks(model = models[i], 
                    model_name = model_names[i],
                    history_name = history_names[i],
                    data = bundled_data)

    # Release Model Memory
    del models[i]



        

    








