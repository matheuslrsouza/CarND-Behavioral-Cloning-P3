import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Lambda, Dropout, Input
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.layers import AveragePooling2D, Conv2DTranspose, Reshape
from keras.layers.merge import Multiply
from keras import optimizers

import matplotlib.image as mpimg

import cv2
import numpy as np
from keras.models import load_model

import glob



inputs = Input(shape=(160, 320, 3))

model = load_model('model.h5')

conv2d_1 = model.get_layer('conv2d_1').output
avg1 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv2d_1)

conv2d_2 = model.get_layer('conv2d_2').output
avg2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv2d_2)

conv2d_3 = model.get_layer('conv2d_3').output
avg3 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv2d_3)

conv2d_4 = model.get_layer('conv2d_4').output
avg4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv2d_4)

conv2d_5 = model.get_layer('conv2d_5').output
avg5 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same', data_format=None)(conv2d_5)

y = Flatten()(avg5)
y = Dense(2*33*64, activation='relu')(y)
y = Reshape((2, 33, 64))(y)


# --------- conv4 ------

#64@4x35
conv4_deconv = Conv2DTranspose(64,
       kernel_size=(3, 3),
       strides=(1, 1), 
       use_bias=False, 
       kernel_initializer=keras.initializers.Ones()
    )(y)

#pointwise multiplication
pointwise_conv4 = Multiply()([conv4_deconv, avg4])

# ----------------------

# --------- conv3 ------

conv3_deconv = Conv2DTranspose(48,
       kernel_size=(3, 3),
       strides=(1, 1), 
       use_bias=False, 
       kernel_initializer=keras.initializers.Ones()
    )(pointwise_conv4)

#pointwise multiplication
pointwise_conv3 = Multiply()([conv3_deconv, avg3])

# --------- conv2 ------

conv2_deconv = Conv2DTranspose(36,
       kernel_size=(5, 5),
       strides=(2, 2), 
       use_bias=False, 
       kernel_initializer=keras.initializers.Ones()
    )(pointwise_conv3)

#pointwise multiplication
pointwise_conv2 = Multiply()([conv2_deconv, avg2])

# --------- conv1 ------

#kernel 6 para corrigir o valor gerado
conv1_deconv = Conv2DTranspose(24,
       kernel_size=(5, 6),
       strides=(2, 2), 
       use_bias=False, 
       kernel_initializer=keras.initializers.Ones()
    )(pointwise_conv2)

#pointwise multiplication
pointwise_conv1 = Multiply()([conv1_deconv, avg1])

input_deconv = Conv2DTranspose(1,
       kernel_size=(6, 6),
       strides=(2, 2), 
       use_bias=False, 
       kernel_initializer=keras.initializers.Ones()
    )(pointwise_conv1)

model = Model(inputs=model.input, outputs=input_deconv)


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


#images = glob.glob('./run_track1/*.jpg')
images = ['./examples/input_salient_track2_6.jpg']
total = len(images)

# Initial call to print 0% progress
printProgressBar(0, total, prefix = 'Progress:', suffix = 'Complete', length = 50)


batch_print = 100

for i, path in enumerate(images):

	image_array = mpimg.imread(path)
	yuv = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
	y, u, v = cv2.split(yuv)
	y = y.reshape(160, 320, -1)

	filename = path.split('/')[-1]

	img_predicted = model.predict(y[None, :, :, :], batch_size=1)

	img_predicted = img_predicted.reshape(img_predicted.shape[1:])


	img = ((img_predicted - img_predicted.min())*255 / (img_predicted.max() - img_predicted.min())).astype(np.uint8)
		
	
	new=[[[j if j <= 80 else 0,j if j > 80 else 0,0] for j in i] for i in img]
	dt = np.dtype('uint8')
	new=np.array(new,dtype=dt)

	new = np.lib.pad(new, ((70, 20), (0, 0), (0, 0)), 'constant', constant_values=0)

	new_image = cv2.addWeighted(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),0.8,new,1.0,0.)
	cv2.imwrite('./examples/result_'+filename, new_image)
	#cv2.imwrite('./avg_'+filename, img)

	if i%batch_print == 0:
		printProgressBar(i + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50)

printProgressBar(i + 1, total, prefix = 'Progress:', suffix = 'Complete', length = 50)




