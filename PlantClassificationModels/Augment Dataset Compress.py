import pandas as pd 
import numpy as np 
import cv2 
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os 

import matplotlib.pyplot as plt 

from tqdm import tqdm 

Original_Path = "Segmented Herbal Leaf Images"

print(os.listdir(Original_Path))

IMAGE_RESIZE = (64, 64)

x_data = []
y_data = []

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

for folder in tqdm(os.listdir(Original_Path)):
  for image in tqdm(os.listdir(os.path.join(Original_Path, folder))):
      _r = cv2.imread(os.path.join(Original_Path, folder, image), cv2.IMREAD_ANYCOLOR)
      _r = cv2.resize(_r, IMAGE_RESIZE)
      _r = cv2.normalize(_r, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

      _ref = _r
      
      # Rotate 45 degrees each 

      for x in range(0, 360+1, 22):
        _ref = rotate_image(_ref, x)

        for i in [-1, 0, 1]:            
            _ = cv2.flip(_ref, i)

            # 0
            x_data.append(_)
            y_data.append(folder)

            # 90
            _ = cv2.rotate(_, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_data.append(_)
            y_data.append(folder)

            # 180
            _ = cv2.rotate(_, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_data.append(_)
            y_data.append(folder)

            # 270
            _ = cv2.rotate(_, cv2.ROTATE_90_COUNTERCLOCKWISE)
            x_data.append(_)
            y_data.append(folder)

_X = np.array(x_data)
_y = np.array(y_data)

np.savez_compressed("Augmented_CV_Dataset_36_Steps",
                    raw_X = _X,
                    raw_y = _y)






