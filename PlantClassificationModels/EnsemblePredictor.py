import cv2 
import os 
import numpy as np 
import pickle
from keras.models import load_model


class EnsemblePredictor():
    # Load Models
    def __init__(self, ENABLE_ROI_TRANSFORM = False, NUM_ROI_TRANSFORMATIONS = 16):
        # VGG Model
        self.MRV_VGG_Model = load_model("PlantClassificationModels/KERAS MODELS/MRV_VGG.keras")
        
        # MRV_Inception_Model
        self.MRV_Inception_Model = load_model("PlantClassificationModels/KERAS MODELS/MRV_Inception.keras")

        # MRV_ResNet_Model
        self.MRV_ResNet_Model = load_model("PlantClassificationModels/KERAS MODELS/MRV_ResNet.keras")

        # MRV_SqueezeNet_Model
        self.MRV_SqueezeNet_Model = load_model("PlantClassificationModels/KERAS MODELS/MRV_SqueezeNet.keras")

        os.system("CLS")
        print("INFO: MODELS LOADED")

        # Region of Interest Constant Num
        self.ROI_CN = 16

        # Enable ROI Transformations
        self.ROI_TRANSFORM_BOOL = ENABLE_ROI_TRANSFORM

        # Default Image Size Dimensions
        self.IMAGE_SIZE = (64, 64)

        # Load Label Encoder
        with open("PlantClassificationModels/LabelEncoderData.pkl", "rb") as f:
            self.le = pickle.load(f)
    
    # Preprocess Image (BGR)
    def preprocess_image(self, image_name):
        _im = cv2.imread(image_name, cv2.IMREAD_ANYCOLOR)
        w, h, w_steps, h_steps = self.get_dimensions(_im)

        if self.ROI_TRANSFORM_BOOL:
            region_proposals = np.array(self.get_proposals(_im, w_steps, h_steps))
        
        else:
            region_proposals = np.array([self.resize_normalize_image(_im, self.IMAGE_SIZE)])

        return region_proposals

    # Resize Normalize Image
    def resize_normalize_image(self, image, size=(64, 64)):
        _image = np.array(image)
        _image = cv2.resize(_image, size)
        _image = cv2.normalize(_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return _image  

    # Get Dimensions of Images
    def get_dimensions(self, image):
        w, h = image.shape[:2]
        w_steps = w // self.ROI_CN
        h_steps = h // self.ROI_CN
        
        return w, h, w_steps, h_steps 

    # Get Region Proposals of Images
    def get_proposals(self, image, w_steps, h_steps):
        
        region_proposals = []

        for x in range(0, self.ROI_CN):
            for y in range(0, self.ROI_CN):
                for i in range(x + 1, self.ROI_CN):
                    for j in range(y + 1, self.ROI_CN):
                        _left = x * w_steps
                        _top = y * h_steps
                        _right = i * w_steps
                        _bottom = j * h_steps

                        cropped_image = image.copy()

                        try:
                            cropped_image = cropped_image[_top:_bottom + 1, _left:_right + 1]
                        
                            cropped_image = self.resize_normalize_image(cropped_image)
                        
                            region_proposals.append(cropped_image)
                        except:
                            pass 

        return region_proposals

    def predict_single_model(self, model, input_image, verbose = 0):
        raw_y = model.predict(input_image, verbose = verbose)
        print(raw_y)
        pred_y = np.argmax(np.array(raw_y), axis=1)
        return self.le.classes_[pred_y]
        
    # Predict Region Proposals on available models
    def predict_image(self, region_proposals):
        results = []
        for model in [
            self.MRV_Inception_Model,
            self.MRV_ResNet_Model,
            self.MRV_SqueezeNet_Model,
            self.MRV_VGG_Model
        ]:
            results.append(self.predict_single_model(model, region_proposals).tolist()[0])

        return results
        
if __name__ == "__main__":
    E = EnsemblePredictor(ENABLE_ROI_TRANSFORM=True)
    _proposals = E.preprocess_image("Segmented Herbal Leaf Images/Artocarpus Heterophyllus (Jackfruit)/1.jpg")
    _predictions = E.predict_image(_proposals)
    os.system("CLS")
    print("PREDICTIONS:")
    print(_predictions)




