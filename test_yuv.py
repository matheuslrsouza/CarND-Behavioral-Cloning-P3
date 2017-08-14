import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import cv2
import numpy as np

plot = False

image_array = cv2.imread('./img_tests/track2_shadows.jpg')
img2 = mpimg.imread('./img_tests/track2_noshadows.jpg')

yuv = cv2.cvtColor(image_array, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(yuv)

yuv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YUV)
y2, u2, v2 = cv2.split(yuv2)

if plot:
	plt.imshow(image_array)
	plt.show()

	plt.imshow(y)
	plt.show()

	#plt.imshow(y * 2)
	#plt.show()

	#plt.imshow(y2)
	#plt.show()


	plt.imshow(u)
	plt.show()

	#plt.imshow(u2)
	#plt.show()

	plt.imshow(v)
	plt.show()

	#plt.imshow(v2)
	#plt.show()
else:
	filename = 'track2_shadows.jpeg'
	cv2.imwrite('./examples/y_'+filename, y)
	cv2.imwrite('./examples/u_'+filename, u)
	cv2.imwrite('./examples/v_'+filename, v)
