import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras import optimizers

class NVidia:

	def __init__(self):
		self.model = Sequential()

		self.model.add(Lambda(lambda x: (x / 255.0) -0.5, input_shape=(160, 320, 3)))
		self.model.add(Cropping2D(cropping=((70, 20), (0, 0))))

		self.model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
		self.model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
		self.model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
		#self.model.add(Dropout(0.8))

		self.model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		self.model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		
		self.model.add(Flatten())
		self.model.add(Dropout(0.6))
		self.model.add(Dense(100, activation='relu'))
		self.model.add(Dense(50, activation='relu'))
		self.model.add(Dense(10, activation='relu'))
		#self.model.add(Dropout(0.8))
		#output
		self.model.add(Dense(1))

		self.model.summary()

	def train_generator(self, train_generator, validation_generator, 
				steps_per_epoch, validation_steps):

		#adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		#adam = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
		self.model.compile(loss='mse', optimizer='adam')

		history = self.model.fit_generator(
						train_generator, steps_per_epoch=steps_per_epoch, 
						validation_data=validation_generator, validation_steps=validation_steps, 
		            	epochs=16)


		self.model.save('model.h5')

		return history

	def train(self, X_train, y_train):

		self.model.compile(loss='mse', optimizer='adam')
		self.model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

		self.model.save('model.h5')
