import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
import pickle
from sklearn.model_selection import train_test_split

# all paths for the sample images
samples = []

# given a dir path, reads the driving_log.csv to extract the path for images.
# replaces the path images and adds the line to the "samples" vector
def load_data(dir_data):
	with open(dir_data + 'driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)

		for line in reader:

			line[0] = dir_data + 'IMG/' + line[0].split('/')[-1]
			line[1] = dir_data + 'IMG/' + line[1].split('/')[-1]
			line[2] = dir_data + 'IMG/' + line[2].split('/')[-1]

			samples.append(line)

# loads the image paths from these directories
# this was usefull because the transfer to the ec2 amazon cloud could 
# did on demand (as needed)
load_data('../../track1/') 
load_data('../../track1_1/')
load_data('../../track1_2/')
load_data('../../track1_3/')
load_data('../../track1_4/')
load_data('../../track1_5/')
load_data('../../track1_6/')
load_data('../../track2/')
load_data('../../track2_1/')

# print the number of samples
print("samples ", len(samples))

# 20% for validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# generator to provide the data for training
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

				#steering
				steering_center = float(line[3]) 
				# adds a correction for the images captured on the left and right cameras
				# this allows the car to be redirected to the center if it is leaving the runway
				steering_left = steering_center + correction
				steering_right = steering_center - correction

				# reads the paths for the three captured images
				image_left = cv2.imread(line[1])
				image_center = cv2.imread(line[0])
				image_right = cv2.imread(line[2])


				split_yuv_and_add(images, angles, image_left, steering_left)
				split_yuv_and_add(images, angles, image_center, steering_center)
				split_yuv_and_add(images, angles, image_right, steering_right)

			X_train = np.array(images)
			y_train = np.array(angles)
			
			yield shuffle(X_train, y_train)

# splits the given image in YUV to produce augmentation data
def split_yuv_and_add(images, angles, img, steering):
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)

	# before add the, reshapes the images to one channel on last dimenstion (160, 320, 1)
	images.append(y.reshape(160, 320, -1))
	images.append(u.reshape(160, 320, -1))
	images.append(v.reshape(160, 320, -1))

	# repeats the same steering angles three times (Y, U and V)
	angles.append(steering)
	angles.append(steering)
	angles.append(steering)


# compile and train the model using the generator function
batch_size = 32

train_generator = get_data_generator(train_samples, batch_size=batch_size)
validation_generator = get_data_generator(validation_samples, batch_size=batch_size)


from netnvidia import NVidia

network = NVidia()

history_object = network.train_generator(train_generator, validation_generator, 
									len(train_samples)/batch_size, 
									len(validation_samples)/batch_size)

### save the history so we can view it at any time. 
### when we are training on GPU we can't plot there
pickle.dump(history_object.history, open('loss_history.p', 'wb'))










