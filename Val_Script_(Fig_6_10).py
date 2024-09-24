# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:17:22 2020

@author: Sz-PC
"""

#%% Cell 1
# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot

#%% Cell 2
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

#%% Cell 3
# plot & save source, generated and target images as a single file
def plot_images(src_img, gen_img, gen_img2, tar_img):
	images = vstack((src_img, gen_img, gen_img2, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0	
	#titles = ['Source', 'U-Net Gen', 'O-Net Gen', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 4, 1 + i)
		# turn off axis
		pyplot.axis('off')		
		# show title
		#pyplot.title(titles[i])
		# plot raw pixel data
		pyplot.imshow(images[i])
	pyplot.show()
	pyplot.close()

#%% Cell 4
# plot & save single image as a single file
def plot_single_image(src_img):
	images = src_img
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	# plot images row by row
	for i in range(len(images)):
		# turn off axis
		pyplot.axis('off')		
		# plot raw pixel data
		pyplot.imshow(images[i])
	pyplot.show()
	pyplot.close()

#%% Cell 5
# load dataset (PCM - 'pcm_256_val.npz'; DIC - 'dic_256_val.npz')
[X1, X2] = load_real_samples('dic_256_val.npz')
print('Loaded', X1.shape, X2.shape)

#%% Cell 6
# load model
model = load_model('dic_model_398344.h5')
modelo = load_model('o-dic_model_398344.h5')

#%% Cell 7
# select values for index from index file (0, 1 & 2)
index = [0];

#%% Cell 8
src_image, tar_image = X1[index], X2[index]

#%% Cell 9
# generate image from source
gen_image = model.predict(src_image)
gen_imageo = modelo.predict(src_image)

#%% Cell 10
# plot & save all four images
plot_images(src_image, gen_image, gen_imageo, tar_image)

#%% Cell 11
# plot & save one image
plot_single_image(src_image)
plot_single_image(gen_image)
plot_single_image(gen_imageo)
plot_single_image(tar_image)