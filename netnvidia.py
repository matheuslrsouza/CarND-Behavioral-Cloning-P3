import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

class NVidia:

	def __init__(self):
		self.model = Sequential()

		self.model.add(Lambda(lambda x: (x / 255.0) -0.5, input_shape=(160, 320, 3)))
		self.model.add(Cropping2D(cropping=((70, 20), (0, 0))))

		self.model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
		self.model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
		self.model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
		self.model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		self.model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		
		self.model.add(Flatten())
		self.model.add(Dense(100, activation='relu'))
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dense(10, activation='relu'))

		#output
		self.model.add(Dense(1))

		self.model.summary()

	def train(self, train_generator, validation_generator, 
				samples_per_epoch, nb_val_samples):

		self.model.compile(loss='mse', optimizer='adam')

		self.model.fit_generator(
				train_generator, samples_per_epoch=samples_per_epoch, 
				validation_data=validation_generator, nb_val_samples=nb_val_samples, 
            	nb_epoch=1)


		self.model.save('model.h5')





