import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

class LeNet:

	def __init__(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		self.model = Sequential()

		self.model.add(Lambda(lambda x: (x / 255.0) -0.5, input_shape=(160, 320, 3)))
		self.model.add(Cropping2D(cropping=((70, 20), (0, 0))))

		self.model.add(Conv2D(6, (5, 5), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


		self.model.add(Conv2D(16, (5, 5), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		self.model.add(Flatten())
		self.model.add(Dense(120, activation='relu'))
		self.model.add(Dense(84, activation='relu'))

		self.model.add(Dense(1))

	def train(self):

		self.model.compile(loss='mse', optimizer='adam')
		self.model.fit(self.X_train, self.y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

		self.model.save('model.h5')
