import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import pickle

samples = []

def load_data(dir_data):
	with open(dir_data + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)

		for line in reader:

			line[0] = dir_data + 'IMG/' + line[0].split('/')[-1]
			line[1] = dir_data + 'IMG/' + line[1].split('/')[-1]
			line[2] = dir_data + 'IMG/' + line[2].split('/')[-1]

			samples.append(line)

load_data('../../mini_train/') # track 1

print("samples ", len(samples))


def get_data(samples):

	images = []	
	measurements = []

	for line in samples:

		correction = 0.2

		steering_center = float(line[3]) #steering
		steering_left = steering_center + correction
		steering_right = steering_center - correction

		image_center = cv2.imread(line[0])
		image_left = cv2.imread(line[1])
		image_right = cv2.imread(line[2])	
		
		images.extend([image_left])
		images.extend([image_center])
		images.extend([image_right])

		measurements.extend([steering_left])
		measurements.extend([steering_center])
		measurements.extend([steering_right])

	X_train = np.array(images)
	y_train = np.array(measurements)
	return shuffle(X_train, y_train)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
		rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
	)

def get_data_generator(samples, batch_size=32):
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

				image_center = cv2.imread(line[0])
				image_left = cv2.imread(line[1])
				image_right = cv2.imread(line[2])
				
				images.append(image_left)
				images.append(image_center)
				images.append(image_right)

				angles.append(steering_left)
				angles.append(steering_center)
				angles.append(steering_right)

			
			X_train = np.array(images)
			y_train = np.array(angles)

			#print("shape before aug", X_train.shape)
			
			batches = 0
			for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train)):
				batches += 1

				X_train = np.concatenate((X_train, X_batch), axis=0)
				y_train = np.concatenate((y_train, y_batch), axis=0)

				#maximum 2 times the size of train data
				if batches == 2:
					break

			#print("shape after aug", X_train.shape)
			

			yield shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 32




train_generator = get_data_generator(train_samples, batch_size=batch_size)
validation_generator = get_data_generator(validation_samples, batch_size=batch_size)


from netnvidia import NVidia

network = NVidia()

n_epochs = 3

'''
for e in range(n_epochs):

	print("epoch %d" % e)

	for X_train, y_train in get_data_generator(train_samples, batch_size=256):
		batches = 0
		for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
			loss = model.fit(X_batch, Y_batch)
			#print("loss %0.4f" % loss)
			print("%d / %d" % (batches , len(X_train) / batch_size))
			batches += 1
			if batches >= len(X_train) / batch_size:
				break

'''
history_object = network.train_generator(train_generator, validation_generator, 
									len(train_samples)/batch_size, 
									len(validation_samples)/batch_size)

### save the history so we can view it at any time. 
### when we are training on GPU we can't plot 
pickle.dump(history_object.history, open('loss_history.p', 'wb'))










