import cv2 
import os 
from PIL import Image
import numpy as np 
import pickle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import rembg 

from tqdm import tqdm 

model = load_model("PlantClassificationModels/KERAS MODELS/MRV_VGG.keras")

with open("PlantClassificationModels/LabelEncoderData.pkl", "rb") as f:
    le = pickle.load(f)
    
GET_ROI_TRANSFORMATIONS = False
ROI_CW = 2

def load_image(joined_path):
    im = cv2.imread(joined_path, cv2.IMREAD_ANYCOLOR)
    return im

def get_dimensions(im):
    w, h = im.shape[:2]
    w_steps = w // ROI_CW
    h_steps = h // ROI_CW
    w_residual = w - w_steps * ROI_CW
    h_residual = h - h_steps * ROI_CW

    return w, h, w_steps, h_steps, w_residual, h_residual

def resize_normalize_image(im, size=(64, 64)):
    image = np.array(im) 
    image = cv2.resize(image, (64, 64))
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return image 
    
def transform_image(im, w_steps, h_steps):
    
    proposed_images = [] 

    for x in range(0, ROI_CW):
        for y in range(0, ROI_CW):
            for i in range(x + 1, ROI_CW):
                for j in range(y + 1, ROI_CW):
                    _left = x * w_steps
                    _top = y * h_steps
                    _right = (i) * w_steps
                    _bottom = (j) * h_steps

                    cropped_image = im.copy()
                    try:
                        cropped_image = cropped_image[_top:_bottom + 1, _left:_right + 1]
                        # print("Targets:", _top, _left, _bottom, _right)
                        # print("Cropped Image Dimensions:", cropped_image.shape)
                        cropped_image = resize_normalize_image(cropped_image)
                        
                        proposed_images.append(cropped_image)
                    except:
                        pass
                    
    proposed_images = np.array(proposed_images)
    return proposed_images

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

pil_image = Image.open("input_img.jpg")

output_path = 'output.png'

output = rembg.remove(pil_image, bgcolor=(0, 0, 0, 0))
output.save(output_path)

t_image = Image.open("output.png")
c_image = t_image.crop(t_image.getbbox())
c_image.save("Cropped.png")

t_image = Image.open("Cropped.png").convert("RGB")
t_image.show()
conv_image = t_image.copy().save("Saved.jpg")

# _a = "C:/Users/Alaric/Documents/GitHub/CPE124_E01_2Q2324_GROUP_1_SERVICE_LEARNING_REPOSITORY/PlantClassificationModels/Segmented Herbal Leaf Images/Artocarpus Heterophyllus (Jackfruit)/100.jpg"
# a = cv2.imread(_a, cv2.IMREAD_ANYCOLOR)
a = cv2.imread("Saved.jpg", cv2.IMREAD_ANYCOLOR)
a = resize_normalize_image(a)

cv2.imshow("Hatdog", a)
cv2.waitKey(0)
cv2.destroyAllWindows()
a = np.array([a])

y = model.predict(a)
pred_y = np.argmax(y, axis=1)
label = le.classes_[pred_y]
y, pred_y, label

orig_folder = "Segmented Herbal Leaf Images"

total_plant_class_tally = {}

for plant_class in os.listdir(orig_folder):
    print(plant_class)
    plant_class_predictions = {}

    for testing_image in tqdm(os.listdir(os.path.join(orig_folder, plant_class))):
        
        im = load_image(os.path.join(orig_folder, plant_class, testing_image))
        w, h, w_steps, h_steps, w_residual, h_residual = get_dimensions(im)
        
        if GET_ROI_TRANSFORMATIONS:
            proposed_images = transform_image(im, w_steps, h_steps)
        
        else:
            proposed_images = np.array([resize_normalize_image(im, (64, 64))])


        _im = proposed_images[0]
        _im = np.array(_im)

        _im = rotate_image(_im, 36)
        # cv2.imshow("hatdog", _im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        raw_y = model.predict(proposed_images)
        raw_y = np.array(raw_y)
        pred_y = np.argmax(raw_y, axis=1)
        predict_dict = {}
        for i in pred_y:
            predict_dict[i] = predict_dict.get(i, 0) + 1

        transformed_dict = {}
        for predict in predict_dict:
            transformed_dict[le.classes_[predict]] = predict_dict[predict] / len(pred_y)

        x = list(dict(sorted(transformed_dict.items(), key=lambda x:x[1], reverse=True)).items())[0]
        
        plant_class_predictions[testing_image] = x
        
        break
    print(plant_class_predictions)

    plant_class_tally = {}

    for (file_name, (item_name, confidence)) in plant_class_predictions.items():
        plant_class_tally[item_name] = plant_class_tally.get(item_name, 0) + 1

    total_plant_class_tally[plant_class] = plant_class_tally

    break


accuracies = {}

for key, value in total_plant_class_tally.items():
    total = 0 

    for _k, _v in value.items():
        total += _v

    if key in value:
        accuracies[key] = value[key] / total 
    else:
        accuracies[key] = 0

accuracies
        
total_acc = 0
for _k, _v in accuracies.items():
    total_acc += _v 

total_acc