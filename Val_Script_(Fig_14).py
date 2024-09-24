# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:17:22 2020

@author: Sz-PC
"""
# Cell 0
# example of loading a pix2pix model and using it for image to image translation
from keras.models import load_model
from numpy import load
from matplotlib import pyplot
import numpy as np

#%%
# Cell 1
# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1 = data['arr_0']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5	
	return [X1]

#%%
# Cell 2
# plot & save source, generated and target images as a single file
def plot_images(src_img, src_nimg, gen_img, gen_img2, tar_img):
    # create figure
    fig = pyplot.figure(figsize=(256, 256))
  
    # setting values to rows and column variables
    rows = 1
    columns = 5
    
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # showing image
    pyplot.imshow(src_img)
    pyplot.axis('off')
    pyplot.title("Source Image")
    
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 2)
    # showing image
    pyplot.imshow(src_nimg)
    pyplot.axis('off')
    pyplot.title("Source with Noise")
    
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 3)
    # showing image
    pyplot.imshow(gen_img)
    pyplot.axis('off')
    pyplot.title("U-Net Gen Image")
    
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 4)
    # showing image
    pyplot.imshow(gen_img2)
    pyplot.axis('off')
    pyplot.title("O-Net Gen Image")
    
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 5)
    # showing image
    pyplot.imshow(tar_img)
    pyplot.axis('off')
    pyplot.title("Expected Image")

#%%
# Cell 3
# plot & save source, generated and target images as separate individual files
def plot_imagea(src_img, gen_img):
    filename4 = 'Source Image (Val).png'
    filename5 = 'Generated Image (Val).png'    
    #import function
    #import numpy as np
    #save images as separate files
    x = np.squeeze(src_img)
    pyplot.axis('off')
    pyplot.imshow(x)
    pyplot.savefig(filename4)
    pyplot.close()
    y = np.squeeze(gen_img)
    pyplot.axis('off')
    pyplot.imshow(y)
    pyplot.savefig(filename5)
    pyplot.close()    
    
#%%
# Cell 4
# load dataset
[X1] = load_real_samples('Noise_pcm_256_test.npz')
print('Loaded', X1.shape)

#%%
# Cell 5
# load model
modelu = load_model('pcm_model_398344.h5')
modelo = load_model('o-pcm_model_398344.h5')

#%%
# Cell 6
# select random example
#ix = randint(0, len(X1), 1)
#ix = [1]
exp_image = X1[[0]]
src_image = X1[[1]]
n_src_image = X1[[2]]

#%%
# Cell 7
# generate image from source
ugen_image = modelu.predict(n_src_image)
ogen_image = modelo.predict(n_src_image)

#%%
# Cell 8
# squeeze all images into 3D array
exp_image = np.squeeze(exp_image)
src_image = np.squeeze(src_image)
n_src_image = np.squeeze(n_src_image)
ugen_image = np.squeeze(ugen_image)
ogen_image = np.squeeze(ogen_image)

#%%
# Cell 9
# plot & save both images
plot_images(src_image, n_src_image, ugen_image, ogen_image, exp_image)