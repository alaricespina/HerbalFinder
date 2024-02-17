import sys 
import cv2 
import copy 
import numpy as np 
import pandas as pd 
from PIL import Image 
import os 
from sklearn.preprocessing import LabelEncoder



# Parser the data from annotation file
def get_data(input_path):
	"""Parse the data from annotation file
	Assume that input path is the folder containing 
	the folders for each class. And only training csv would be used
	
	Args:
		input_path: annotation file path

	Returns:
		all_data: list(filepath, width, height, list(bboxes))
		classes_count: dict{key:class_name, value:count_num} 
			e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
		class_mapping: dict{key:class_name, value: idx}
			e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
	"""
	found_bg = False
	all_imgs = {}
	classes_count = {}
	class_mapping = {}

	train_annotations = []

	for folder in os.listdir(input_path):		
		for file in os.listdir(os.path.join(input_path, folder)):
			if ".csv" in file and "train" in file:
				_df = pd.read_csv(os.path.join(input_path, folder, file))
				train_annotations.append(_df)

	train_df = pd.concat(train_annotations)
	
	classes = train_df["class"].unique()
	for _class in classes:
		classes_count[_class] = len(train_df[train_df["class"] == _class])
	
	#print(classes_count)

	le = LabelEncoder()
	le.fit_transform(train_df["class"])

	for idx, _class in enumerate(le.classes_):
		class_mapping[_class] = idx
	
	#print(class_mapping)

	train_df["width"] = 200
	train_df["height"] = 200
	
	bbox_data = train_df[["x0","y0","x1","y1"]]
	bbox_data = bbox_data.to_numpy()

	filepath_data = train_df["image_name"].to_numpy()
	width_data = train_df["width"].to_numpy()
	height_data = train_df["height"].to_numpy()

	print(bbox_data.shape)
	print(filepath_data.shape)
	print(width_data.shape)
	print(height_data.shape)
	print(bbox_data[0], filepath_data[0], width_data[0], height_data[0])

	all_data = [
		filepath_data.copy(), 
		width_data.copy(), 
		height_data.copy(), 
		bbox_data.copy()
		]

	return all_data, classes_count, class_mapping

	# with open(input_path,'r') as f:

	# 	print('Parsing annotation files')

	# 	for line in f:

	# 		# Print process
	# 		sys.stdout.write('\r'+'idx=' + str(i))
	# 		i += 1

	# 		line_split = line.strip().split(',')

	# 		(filename,x1,y1,x2,y2,class_name) = line_split

	# 		if class_name not in classes_count:
	# 			classes_count[class_name] = 1
	# 		else:
	# 			classes_count[class_name] += 1

	# 		if class_name not in class_mapping:
	# 			if class_name == 'bg' and found_bg == False:
	# 				print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
	# 				found_bg = True
	# 			class_mapping[class_name] = len(class_mapping)

	# 		if filename not in all_imgs:
	# 			all_imgs[filename] = {}
				
	# 			img = cv2.imread(filename)
	# 			(rows,cols) = img.shape[:2]
	# 			all_imgs[filename]['filepath'] = filename
	# 			all_imgs[filename]['width'] = cols
	# 			all_imgs[filename]['height'] = rows
	# 			all_imgs[filename]['bboxes'] = []
				
	# 		all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


	# 	all_data = []
	# 	for key in all_imgs:
	# 		all_data.append(all_imgs[key])
		
	# 	# make sure the bg class is last in the list
	# 	if found_bg:
	# 		if class_mapping['bg'] != len(class_mapping) - 1:
	# 			key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
	# 			val_to_switch = class_mapping['bg']
	# 			class_mapping['bg'] = len(class_mapping) - 1
	# 			class_mapping[key_to_switch] = val_to_switch
		
		# return all_data, classes_count, class_mapping

def get_new_img_size(width, height, img_min_side=200):
	if width <= height:
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
	else:
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side

	return resized_width, resized_height

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img

if __name__ == "__main__":
    print("Testing Utility")