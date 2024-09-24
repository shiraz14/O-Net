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
	# scale from [-1,1] to [0,1]
	src_img = (src_img + 1) / 2.0
	gen_img = (gen_img + 1) / 2.0
	gen_img2 = (gen_img2 + 1) / 2.0
	tar_img = (tar_img + 1) / 2.0
	images = vstack((src_img, gen_img, gen_img2, tar_img))	
	titles = ['Source', 'U-Net Gen', 'O-Net Gen', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 4, 1 + i)
		# turn off axis
		pyplot.axis('off')		
		# show title
		pyplot.title(titles[i])
		# plot raw pixel data
		pyplot.imshow(images[i])
	pyplot.show()
	pyplot.close()

#%% Cell 4
# load dataset
[X1, X2] = load_real_samples('dic_256_val_Fig_8.npz')
print('Loaded', X1.shape, X2.shape)

#%% Cell 5
# load model
model = load_model('dic_model_382568_(U-Net).h5')
modelo = load_model('dic_model_382568_(O-Net).h5')

#%% Cell 6
# select values for index from index file (0, 1 & 2)
index = [0];

#%% Cell 7
src_image, tar_image = X1[index], X2[index]

#%% Cell 8
# generate image from source
gen_image = model.predict(src_image)
gen_imageo = modelo.predict(src_image)

#%% Cell 9
# plot & save all three images
plot_images(src_image, gen_image, gen_imageo, tar_image)