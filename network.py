import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

samples = []

dir_img = '../../dados_simulador/IMG/'

with open('../../dados_simulador/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)

	for line in reader:
		samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for line in batch_samples:

				correction = 0.2

				steering_center = float(line[3]) #steering
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				path_center = dir_img + line[0].split('/')[-1]
				path_left = dir_img + line[1].split('/')[-1]
				path_right = dir_img + line[2].split('/')[-1]

				image_center = cv2.imread(path_center)
				image_left = cv2.imread(path_left)
				image_right = cv2.imread(path_right)	
				
				images.extend([image_left])
				images.extend([image_center])
				images.extend([image_right])

				angles.extend([steering_left])
				angles.extend([steering_center])
				angles.extend([steering_right])

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

from netnvidia import NVidia

network = NVidia()
network.train(train_generator, validation_generator, len(train_samples), len(validation_samples))










