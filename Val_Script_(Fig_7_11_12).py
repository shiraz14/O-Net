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
	X1 = data['arr_0']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5	
	return [X1]

#%% Cell 3
# plot & save source, generated and target images as a single file
def plot_images(src_img, gen_img):
	images = vstack((src_img, gen_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 2, 1 + i)
		# turn off axis
		pyplot.axis('off')		
		# show title
		pyplot.title(titles[i])
        # plot raw pixel data
		pyplot.imshow(images[i])
	pyplot.show()
	pyplot.close()

#%% Cell 4
# load dataset (DIC - 'Diatom Sample_dic_256_test.npz'; PCM - 'Diatom Sample_pcm_256_test.npz')
[X1] = load_real_samples('Diatom Sample_pcm_256_test.npz')
print('Loaded', X1.shape)

#%% Cell 5
# load model (Fig 7 & 11 - 398344; 12 - 473280)
model = load_model('o-pcm_model_473280.h5')

#%% Cell 6
# select index = 9
index = [9]

#%% Cell 7
src_image = X1[index]

#%% Cell 8
# generate image from source
gen_image = model.predict(src_image)

#%% Cell 9
# plot & save all the five images (for Fig 7, 11 & 12)
plot_images(src_image, gen_image)