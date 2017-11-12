# import region
import csv
import cv2
import numpy as np


# collect the data
def data_generator():
	csv_file = 'data\driving_log.csv'
	img_path = 'data\img\\'
	# imgs will be the featurs while steerings will be the labels
	imgs = []
	steerings = []
	with open(csv_file, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
		
			# read the angles and add a correction factor
			steering_center = float(row[3])
			correction = 0.1
			steering_center_flipped = steering_center * -1
			steering_left = steering_center + correction
			steering_right = steering_center - correction

			file_center = row[0].split('\\')[-1]
			file_left = row[1].split('\\')[-1]
			file_right = row[2].split('\\')[-1]
			
			# read the images, and flip the center camera image
			img_center = cv2.imread(img_path + file_center)
			img_center_flipped = cv2.flip(img_center,1)
			img_left = cv2.imread(img_path + file_left)
			img_right = cv2.imread(img_path + file_right)

			# append fetures with images
			imgs.append(img_center)
			imgs.append(img_center_flipped)
			imgs.append(img_left)
			imgs.append(img_right)
			
			# append labels with angles 
			steerings.append(steering_center)
			steerings.append(steering_center_flipped)
			steerings.append(steering_left)
			steerings.append(steering_right)
	return imgs, steerings

# 
X_train, y_train = data_generator()
X_train = np.array(X_train)
y_train = np.array(y_train)

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Create the Sequential model
model = Sequential()

# Normalize the data and focus on important area
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# 3 (5x5) Convolutional layers
model.add(Convolution2D(24, 5, 5, border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, border_mode='valid'))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))

# 2 (3x3) Convolutional layers
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))

model.add(Flatten())

# 3 fully connected layers
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.compile('adam', 'mse')
history = model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2)

# save the model
model.save('model.h5')