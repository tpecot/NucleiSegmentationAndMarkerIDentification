# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

"""
Functions needed to run the notebooks
"""

"""
Import python packages
"""

import numpy as np
import tensorflow as tf

import os
from scipy import ndimage
from scipy.misc import bytescale
import threading
from threading import Thread, Lock
import h5py
import re
import datetime

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imsave
import skimage.external.tifffile as tiff

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.engine import Layer, InputSpec
from keras.utils import np_utils


"""
Helper functions
"""

def categorical_sum(y_true, y_pred):
	return K.sum(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred*0, axis=-1)))

def rate_scheduler(lr = .001, decay = 0.95):
	def output_fn(epoch):
		epoch = np.int(epoch)
		new_lr = lr * (decay ** epoch)
		return new_lr
	return output_fn

def process_image(channel_img, win_x, win_y):
	p50 = np.percentile(channel_img, 50)
	channel_img /= max(p50,0.01)
	avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
	channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size

	return channel_img

def process_image_onlyLocalAverageSubtraction(channel_img, win_x, win_y):
	avg_kernel = np.ones((2*win_x + 1, 2*win_y + 1))
	channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size

	return channel_img

def nikon_getfiles(direc_name,channel_name):
	imglist = os.listdir(direc_name)
	imgfiles = [i for i in imglist if channel_name in i]
    
	def sorted_nicely(l):
		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		return sorted(l, key = alphanum_key)

	imgfiles = sorted_nicely(imgfiles)
	return imgfiles

def get_image(file_name):
	if '.tif' in file_name:
		im = tiff.imread(file_name)
		im = bytescale(im)
		im = np.float32(im)
		#im = np.float32(tiff.imread(file_name))
	else:
		im = np.float32(imread(file_name))
	return im

def format_coord(x,y,sample_image):
	numrows, numcols = sample_image.shape
	col = int(x+0.5)
	row = int(y+0.5)
	if col>= 0 and col<numcols and row>=0 and row<numrows:
		z = sample_image[row,col]
		return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
	else:
		return 'x=%1.4f, y=1.4%f'%(x,y)

"""
Data generator for training_data
"""

def data_generator(channels, batch, pixel_x, pixel_y, labels, win_x = 30, win_y = 30):
	img_list = []
	l_list = []
	for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
		img = channels[b, x-win_x:x+win_x+1, y-win_y:y+win_y+1, :]
		img_list += [img]
		l_list += [l]

	return np.stack(tuple(img_list),axis = 0), np.array(l_list)

def data_generator_lnet(channels, batch, pixel_x, pixel_y, labels, win_x = 30, win_y = 30):
	img_list = []
	l_list = []
	for b, x, y, l in zip(batch, pixel_x, pixel_y, labels):
		img = channels[b, x-int(win_x/2):x+int(win_x/2), y-int(win_y/2):y+int(win_y/2), :]
		img_list += [img]
		l_list += [l]

	return np.stack(tuple(img_list),axis = 0), np.array(l_list)

def get_data_sample(file_name):
	training_data = np.load(file_name)
	channels = training_data["channels"]
	batch = training_data["batch"]
	labels = training_data["y"]
	pixels_x = training_data["pixels_x"]
	pixels_y = training_data["pixels_y"]
	win_x = training_data["win_x"]
	win_y = training_data["win_y"]

	total_batch_size = len(labels)
	num_test = np.int32(np.floor(total_batch_size/10))
	num_train = np.int32(total_batch_size - num_test)
	full_batch_size = np.int32(num_test + num_train)

	"""
	Split data set into training data and validation data
	"""
	arr = np.arange(len(labels))
	arr_shuff = np.random.permutation(arr)

	train_ind = arr_shuff[0:num_train]
	test_ind = arr_shuff[num_train:num_train+num_test]

	X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)
	train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}

	return train_dict, (X_test, y_test)

def get_data_sample_lnet(file_name):
	training_data = np.load(file_name)
	channels = training_data["channels"]
	batch = training_data["batch"]
	labels = training_data["y"]
	pixels_x = training_data["pixels_x"]
	pixels_y = training_data["pixels_y"]
	win_x = training_data["win_x"]
	win_y = training_data["win_y"]

	total_batch_size = len(labels)
	num_test = np.int32(np.floor(total_batch_size/10))
	num_train = np.int32(total_batch_size - num_test)
	full_batch_size = np.int32(num_test + num_train)

	"""
	Split data set into training data and validation data
	"""
	arr = np.arange(len(labels))
	arr_shuff = np.random.permutation(arr)

	train_ind = arr_shuff[0:num_train]
	test_ind = arr_shuff[num_train:num_train+num_test]

	X_test, y_test = data_generator_lnet(channels.astype("float32"), batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)
	train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}
	
	return train_dict, (X_test, y_test)

def get_data_sample_unet(file_name_training, file_name_validation, imaging_field_x = 256, imaging_field_y = 256):
	training_data = np.load(file_name_training)
	channels_training = training_data["channels"]
	labels_training = training_data["y"]
	validation_data = np.load(file_name_validation)
	channels_validation = validation_data["channels"]
	labels_validation = validation_data["y"]
    
	X_test = channels_validation[:,0:imaging_field_x,0:imaging_field_y,:]
	Y_test = labels_validation[:,0:imaging_field_x,0:imaging_field_y,:]
    
	train_dict = {"channels": channels_training, "labels": labels_training}

	return train_dict, (X_test, Y_test)


def random_sample_generator_unet(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2):

	do_augmentation = True
	cpt = 0

	n_images = len(x_init)
	arr = np.arange(n_images)
	np.random.shuffle(arr)
    
	while(True):

		# buffers for a batch of data
		x = np.zeros((batch_size, dim1, dim2, n_channels))
		y = np.zeros((batch_size, dim1, dim2, n_classes))
        
		for k in range(batch_size):

			# get random image
			if cpt == n_images:
				cpt = 0
			img_index = arr[cpt]

			# open images
			x_big = x_init[img_index]
			y_big = y_init[img_index]

			# get random crop
			start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
			start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

			patch_x = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
			patch_y = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
			patch_x = np.asarray(patch_x)
			patch_x = patch_x.astype('float32')
			patch_y = np.asarray(patch_y)
			patch_y = patch_y.astype('float32')

			if(do_augmentation):

				augmentation_mode = np.random.randint(low=0, high=3)
				if augmentation_mode == 0:
					# flip
					rand_flip = np.random.randint(low=0, high=4)
					if(rand_flip == 2):
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = np.flipud(patch_x[:,:,i])
						for i in range(patch_y.shape[2]):
							patch_y[:,:,i] = np.flipud(patch_y[:,:,i])
					if(rand_flip == 3):
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = np.fliplr(patch_x[:,:,i])
						for i in range(patch_y.shape[2]):
							patch_y[:,:,i] = np.fliplr(patch_y[:,:,i])

				if augmentation_mode == 1:                    
					# rotate
					rand_rotate = np.random.randint(low=0, high=4)
					if(rand_rotate > 0):
						for i in range(patch_x.shape[2]):                    
							patch_x[:,:,i] = np.rot90(patch_x[:,:,i], rand_rotate)
						for i in range(patch_y.shape[2]):                        
							patch_y[:,:,i] = np.rot90(patch_y[:,:,i], rand_rotate)

				if augmentation_mode == 2:                                                
					# illumination
					ifactor = 1 + np.random.uniform(-0.75, 0.75)
					patch_x = ifactor*patch_x

				if augmentation_mode == 3:
					# Shearing
					rand_shearing = np.random.randint(low=0, high=2)
					if(rand_shearing > 0):
						afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = 255.*(trf.warp(patch_x[:,:,i]/1024., afine_tf))
						for i in range(patch_y.shape[2]):
							patch_y[:,:,i] = 255.*(trf.warp(patch_y[:,:,i]/1024., afine_tf))

#				if augmentation_mode == 4:
#					# flip
#					rand_flip = np.random.randint(low=0, high=4)
#					if(rand_flip == 2):
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = np.flipud(patch_x[i,:,:])
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = np.flipud(patch_y[i,:,:])
#					if(rand_flip == 3):
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = np.fliplr(patch_x[i,:,:])
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = np.fliplr(patch_y[i,:,:])
#					# rotate
#					rand_rotate = np.random.randint(low=0, high=4)
#					if(rand_rotate > 0):
#						for i in range(patch_x.shape[0]):                    
#							patch_x[i,:,:] = np.rot90(patch_x[i,:,:], rand_rotate)
#						for i in range(patch_y.shape[0]):                        
#							patch_y[i,:,:] = np.rot90(patch_y[i,:,:], rand_rotate)
#					# illumination
#					ifactor = 1 + np.random.uniform(-0.75, 0.75)
#					patch_x *= ifactor
#					# Shearing
#					rand_shearing = np.random.randint(low=0, high=2)
#					if(rand_shearing > 0):
#						afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = 500.*(trf.warp(patch_x[i,:,:]/500., afine_tf))
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = 500.*(trf.warp(patch_y[i,:,:]/500., afine_tf))

			# save image to buffer
			x[k, :, :, :] = patch_x
#			y[i, :, :] = [np.ravel(patch_y[0,:,:]),np.ravel(patch_y[1,:,:]),np.ravel(patch_y[2,:,:])]
			y[k, :, :, :] = patch_y
			cpt += 1
            
		# return the buffer
		yield(x, y)

def random_sample_generator_unet2(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2, nb_augmentations):

	cpt = 0

	n_images = len(x_init)
	arr = np.arange(n_images)
	np.random.shuffle(arr)
    
	while(True):

		# buffers for a batch of data
		x = np.zeros((batch_size*nb_augmentations, dim1, dim2, n_channels))
		y = np.zeros((batch_size*nb_augmentations, dim1, dim2, n_classes))
        
		for k in range(batch_size):

			# get random image
			if cpt == n_images:
				cpt = 0
			img_index = arr[cpt]

			# open images
			x_big = x_init[img_index]
			y_big = y_init[img_index]

			# get random crop
			start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
			start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

			patch_x_original = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
			patch_y_original = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
			patch_x_original = np.asarray(patch_x_original)
			patch_x_original = patch_x_original.astype('float32')
			patch_y_original = np.asarray(patch_y_original)
			patch_y_original = patch_y_original.astype('float32')

			if np.random.randint(0, 10) == 10:
				x[k*nb_augmentations+cpt, :, :, :] = patch_x_original
				y[k*nb_augmentations+cpt, :, :, :] = patch_y_original
                
			else:
				for cpt in range(nb_augmentations):           
            
					patch_x = patch_x_original
					patch_y = patch_y_original
                
					if np.random.randint(0, 1) == 1:
						# flip
						rand_flip = np.random.randint(0, 1)
						if(rand_flip == 0):
							for i in range(patch_x.shape[2]):
								patch_x[:,:,i] = np.flipud(patch_x[:,:,i])
							for i in range(patch_y.shape[2]):
								patch_y[:,:,i] = np.flipud(patch_y[:,:,i])
						if(rand_flip == 1):
							for i in range(patch_x.shape[2]):
								patch_x[:,:,i] = np.fliplr(patch_x[:,:,i])
							for i in range(patch_y.shape[2]):
								patch_y[:,:,i] = np.fliplr(patch_y[:,:,i])

					if np.random.randint(0, 1) == 1:
						# rotate
#						rand_rotate = np.random.randint(low=0, high=4)
						random_degree = random.uniform(-25, 25)
						for i in range(patch_x.shape[2]):                    
							patch_x[:,:,i] = sk.transform.rotate(patch_x[:,:,i], random_degree)
						for i in range(patch_y.shape[2]):                    
							patch_y[:,:,i] = sk.transform.rotate(patch_y[:,:,i], random_degree)
#				if(rand_rotate > 0):
#					for i in range(patch_x.shape[2]):                    
#						patch_x[:,:,i] = np.rot90(patch_x[:,:,i], rand_rotate)
#					for i in range(patch_y.shape[2]):                        
#						patch_y[:,:,i] = np.rot90(patch_y[:,:,i], rand_rotate)

					if np.random.randint(0, 1) == 1:
						# illumination
						ifactor = 1 + np.random.uniform(-0.75, 0.75)
						patch_x = ifactor*patch_x

					if np.random.randint(0, 1) == 1:
						# Shearing
						rand_shearing = np.random.randint(low=0, high=2)
						if(rand_shearing > 0):
							afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
							for i in range(patch_x.shape[2]):
								patch_x[:,:,i] = 255.*(trf.warp(patch_x[:,:,i]/1024., afine_tf))
							for i in range(patch_y.shape[2]):
								patch_y[:,:,i] = 255.*(trf.warp(patch_y[:,:,i]/1024., afine_tf))

					if np.random.randint(0, 1) == 1:
						# rotate
						random_degree = random.uniform(-25, 25)
						for i in range(patch_x.shape[2]):                    
							patch_x[:,:,i] = sk.util.random_noise(patch_x[:,:,i])
						for i in range(patch_y.shape[2]):                    
							patch_y[:,:,i] = sk.util.random_noise(patch_y[:,:,i])
                
					# save image to buffer
					x[k*nb_augmentations+cpt, :, :, :] = patch_x
					y[k*nb_augmentations+cpt, :, :, :] = patch_y
                
			cpt += 1
        
		# return the buffer
		yield(x, y)
       
        
def random_sample_generator_centralPixelClassification(img, img_ind, x_coords, y_coords, y_init, batch_size, n_channels, n_classes, win_x, win_y, do_augmentation = True):

	cpt = 0

	n_images = len(img_ind)
	arr = np.arange(n_images)
	np.random.shuffle(arr)
    
	while(True):

		# buffers for a batch of data
#		x = np.zeros((batch_size, 1, dim1, dim2))
		batch_x = np.zeros(tuple([batch_size] + [2*win_x+1,2*win_y+1] + [n_channels]))
		batch_y = np.zeros(tuple([batch_size] + [n_classes]))
		# get one image at a time
		for k in range(batch_size):

			# get random image
			if cpt == n_images:
				cpt = 0
			img_index = arr[cpt]

			# open images
			patch_x = img[img_ind[img_index], (x_coords[img_index]-win_x):(x_coords[img_index]+win_x+1), (y_coords[img_index]-win_y):(y_coords[img_index]+win_y+1), :]
			patch_x = np.asarray(patch_x)
			patch_x = patch_x.astype('float32')
			current_class = np.asarray(y_init[img_index])
			current_class = current_class.astype('float32')

			if(do_augmentation):

				augmentation_mode = np.random.randint(low=0, high=3)
				if augmentation_mode == 0:
					# flip
					rand_flip = np.random.randint(low=0, high=4)
					if(rand_flip == 2):
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = np.flipud(patch_x[:,:,i])
					if(rand_flip == 3):
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = np.fliplr(patch_x[:,:,i])

				if augmentation_mode == 1:                    
					# rotate
					rand_rotate = np.random.randint(low=0, high=4)
					if(rand_rotate > 0):
						for i in range(patch_x.shape[2]):                    
							patch_x[:,:,i] = np.rot90(patch_x[:,:,i], rand_rotate)

				if augmentation_mode == 2:                                                
					# illumination
					ifactor = 1 + np.random.uniform(-0.75, 0.75)
					patch_x *= ifactor

				if augmentation_mode == 3:
					# Shearing
					rand_shearing = np.random.randint(low=0, high=2)
					if(rand_shearing > 0):
						afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = 255.*(trf.warp(patch_x[:,:,i]/1024., afine_tf))

#				if augmentation_mode == 4:
#					# flip
#					rand_flip = np.random.randint(low=0, high=4)
#					if(rand_flip == 2):
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = np.flipud(patch_x[i,:,:])
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = np.flipud(patch_y[i,:,:])
#					if(rand_flip == 3):
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = np.fliplr(patch_x[i,:,:])
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = np.fliplr(patch_y[i,:,:])
#					# rotate
#					rand_rotate = np.random.randint(low=0, high=4)
#					if(rand_rotate > 0):
#						for i in range(patch_x.shape[0]):                    
#							patch_x[i,:,:] = np.rot90(patch_x[i,:,:], rand_rotate)
#						for i in range(patch_y.shape[0]):                        
#							patch_y[i,:,:] = np.rot90(patch_y[i,:,:], rand_rotate)
#					# illumination
#					ifactor = 1 + np.random.uniform(-0.75, 0.75)
#					patch_x *= ifactor
#					# Shearing
#					rand_shearing = np.random.randint(low=0, high=2)
#					if(rand_shearing > 0):
#						afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = 500.*(trf.warp(patch_x[i,:,:]/500., afine_tf))
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = 500.*(trf.warp(patch_y[i,:,:]/500., afine_tf))

			# save image to buffer
			batch_x[k, :, :, :] = patch_x
			batch_y[k, :] = current_class
			cpt += 1

		# return the buffer
		yield(batch_x, batch_y)

        
def random_sample_generator_centralPixelClassification_lnet(img, img_ind, x_coords, y_coords, y_init, batch_size, n_channels, n_classes, win_x, win_y, do_augmentation = True):

	cpt = 0

	n_images = len(img_ind)
	arr = np.arange(n_images)
	np.random.shuffle(arr)
    
	while(True):

		# buffers for a batch of data
#		x = np.zeros((batch_size, 1, dim1, dim2))
		batch_x = np.zeros(tuple([batch_size] + [win_x,win_y] + [n_channels]))
		batch_y = np.zeros(tuple([batch_size] + [n_classes]))
		# get one image at a time
		for k in range(batch_size):

			# get random image
			if cpt == n_images:
				cpt = 0
			img_index = arr[cpt]

			# open images
			patch_x = img[img_ind[img_index], (x_coords[img_index]-int(win_x/2)):(x_coords[img_index]+int(win_x/2)), (y_coords[img_index]-int(win_y/2)):(y_coords[img_index]+int(win_y/2)), :]
			patch_x = np.asarray(patch_x)
			patch_x = patch_x.astype('float32')
			current_class = np.asarray(y_init[img_index])
			current_class = current_class.astype('float32')

			if(do_augmentation):

				augmentation_mode = np.random.randint(low=0, high=3)
				if augmentation_mode == 0:
					# flip
					rand_flip = np.random.randint(low=0, high=4)
					if(rand_flip == 2):
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = np.flipud(patch_x[:,:,i])
					if(rand_flip == 3):
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = np.fliplr(patch_x[:,:,i])

				if augmentation_mode == 1:                    
					# rotate
					rand_rotate = np.random.randint(low=0, high=4)
					if(rand_rotate > 0):
						for i in range(patch_x.shape[2]):                    
							patch_x[:,:,i] = np.rot90(patch_x[:,:,i], rand_rotate)

				if augmentation_mode == 2:                                                
					# illumination
					ifactor = 1 + np.random.uniform(-0.75, 0.75)
					patch_x *= ifactor

				if augmentation_mode == 3:
					# Shearing
					rand_shearing = np.random.randint(low=0, high=2)
					if(rand_shearing > 0):
						afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
						for i in range(patch_x.shape[2]):
							patch_x[:,:,i] = 255.*(trf.warp(patch_x[:,:,i]/1024., afine_tf))

#				if augmentation_mode == 4:
#					# flip
#					rand_flip = np.random.randint(low=0, high=4)
#					if(rand_flip == 2):
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = np.flipud(patch_x[i,:,:])
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = np.flipud(patch_y[i,:,:])
#					if(rand_flip == 3):
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = np.fliplr(patch_x[i,:,:])
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = np.fliplr(patch_y[i,:,:])
#					# rotate
#					rand_rotate = np.random.randint(low=0, high=4)
#					if(rand_rotate > 0):
#						for i in range(patch_x.shape[0]):                    
#							patch_x[i,:,:] = np.rot90(patch_x[i,:,:], rand_rotate)
#						for i in range(patch_y.shape[0]):                        
#							patch_y[i,:,:] = np.rot90(patch_y[i,:,:], rand_rotate)
#					# illumination
#					ifactor = 1 + np.random.uniform(-0.75, 0.75)
#					patch_x *= ifactor
#					# Shearing
#					rand_shearing = np.random.randint(low=0, high=2)
#					if(rand_shearing > 0):
#						afine_tf = trf.AffineTransform(shear=6.28*np.random.ranf())
#						for i in range(patch_x.shape[0]):
#							patch_x[i,:,:] = 500.*(trf.warp(patch_x[i,:,:]/500., afine_tf))
#						for i in range(patch_y.shape[0]):
#							patch_y[i,:,:] = 500.*(trf.warp(patch_y[i,:,:]/500., afine_tf))

			# save image to buffer
			batch_x[k, :, :, :] = patch_x
			batch_y[k, :] = current_class
			cpt += 1

		# return the buffer
		yield(batch_x, batch_y)
"""
Training convnets
"""
def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_channel(channel, name):
	def func(y_true, y_pred):
		y_pred_tmp = K.cast(K.equal( K.argmax(y_pred, axis=-1), channel), "float32")
		true_positives = K.sum(K.round(K.clip(y_true[:,channel,:,:] * y_pred_tmp, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred_tmp, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		possible_positives = K.sum(K.round(K.clip(y_true[:,channel,:,:], 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return 2*((precision*recall)/(precision+recall+K.epsilon()))
	func.__name__ = name
	return func

def weighted_crossentropy_3classes(weight1, weight2, weight3):

	def func(y_true, y_pred):
		class_weights = ([[[[weight1, weight2, weight3]]]])
		unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
		weights = tf.reduce_sum(class_weights * y_true, axis=-1)
		weighted_losses = weights * unweighted_losses
		return tf.reduce_mean(weighted_losses)

	return func

def train_model_sample(model = None, dataset = None,  optimizer = None, 
	expt = "", batch_size = 32, n_epoch = 100,
	direc_save = "./trained_classifiers/", 
	direc_data = "./training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

#	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_{epoch:03d}.h5")
	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + ".h5")
	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + ".npz")

	train_dict, (X_test, Y_test) = get_data_sample(training_data_file_name)

	# the data, shuffled and split between train and test sets
	print(train_dict["pixels_x"].shape[0], 'training samples')
	print(X_test.shape[0], 'test samples')

	# determine the number of channels and classes
	input_shape = model.layers[0].output_shape
	n_channels = input_shape[-1]
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# convert class vectors to binary class matrices
	train_dict["labels"] = np_utils.to_categorical(train_dict["labels"], n_classes)
	Y_test = np_utils.to_categorical(Y_test, n_classes)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

	train_generator = random_sample_generator_centralPixelClassification(train_dict["channels"], train_dict["batch"], train_dict["pixels_x"], train_dict["pixels_y"], train_dict["labels"], batch_size, n_channels, n_classes, train_dict["win_x"], train_dict["win_y"])

	# fit the model on the batches generated by datagen.flow()
	loss_history = model.fit_generator(train_generator,
						steps_per_epoch=int(len(train_dict["labels"])/batch_size),
						epochs=n_epoch,
						validation_data=(X_test,Y_test),
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only=True),
							LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history = loss_history.history)


def train_model_sample_unet(model = None, dataset_training = None,  dataset_validation = None,  optimizer = None, expt = "", 
	batch_size = 5, n_epoch = 100, imaging_field_x = 256, imaging_field_y = 256,
	direc_save = "./trained_classifiers/", 
	direc_data = "./training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)):

	training_data_file_name = os.path.join(direc_data, dataset_training + ".npz")
	validation_data_file_name = os.path.join(direc_data, dataset_validation + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

#	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset_training + "_" + expt + "_{epoch:03d}.h5")
	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset_training + "_" + expt + ".h5")
	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset_training + "_" + expt + ".npz")

	train_dict, (X_test, Y_test) = get_data_sample_unet(training_data_file_name, validation_data_file_name, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y)

	# data information (one way for the user to check if the training dataset makes sense)
	print(train_dict["channels"].shape[0], 'training images')
	print(X_test.shape[0], 'test images')

	# determine the number of channels and classes
	input_shape = model.layers[0].output_shape
	n_channels = input_shape[-1]
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# determine the weights for the weighted cross-entropy based on class distribution for training dataset
	w1 = max(np.sum(train_dict['labels'][:,:,:,0]), np.sum(train_dict['labels'][:,:,:,1]), np.sum(train_dict['labels'][:,:,:,2])) / np.sum(train_dict['labels'][:,:,:,0])
	w2 = max(np.sum(train_dict['labels'][:,:,:,0]), np.sum(train_dict['labels'][:,:,:,1]), np.sum(train_dict['labels'][:,:,:,2])) / np.sum(train_dict['labels'][:,:,:,1])
	w3 = max(np.sum(train_dict['labels'][:,:,:,0]), np.sum(train_dict['labels'][:,:,:,1]), np.sum(train_dict['labels'][:,:,:,2])) / np.sum(train_dict['labels'][:,:,:,2])

	# prepare the model compilation
	model.compile(loss = weighted_crossentropy_3classes(weight1 = w1, weight2 = w2, weight3 = w3), optimizer = optimizer, metrics = [f1])

	# prepare the generation of data
	train_generator = random_sample_generator_unet2(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_x, 1) 
    
	# fit the model
	loss_history = model.fit_generator(train_generator,
						steps_per_epoch=int(len(train_dict["labels"])/batch_size),
						epochs=n_epoch,
						validation_data=(X_test,Y_test),
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only=True),LearningRateScheduler(lr_sched)])
    
	np.savez(file_name_save_loss, loss_history = loss_history.history)


def train_model_sample_lnet(model = None, dataset = None,  optimizer = None, 
	expt = "", batch_size = 32, n_epoch = 100,
	direc_save = "./trained_classifiers/", 
	direc_data = "./training_data_npz/", 
	lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)):

	training_data_file_name = os.path.join(direc_data, dataset + ".npz")
	todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

#	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + "_{epoch:03d}.h5")
	file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + ".h5")
	file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + ".npz")

	train_dict, (X_test, Y_test) = get_data_sample_lnet(training_data_file_name)

	# the data, shuffled and split between train and test sets
	print(train_dict["pixels_x"].shape[0], 'training samples')
	print(X_test.shape[0], 'test samples')

	# determine the number of channels and classes
	input_shape = model.layers[0].output_shape
	n_channels = input_shape[-1]
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# convert class vectors to binary class matrices
	train_dict["labels"] = np_utils.to_categorical(train_dict["labels"], n_classes)
	Y_test = np_utils.to_categorical(Y_test, n_classes)

	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])

	train_generator = random_sample_generator_centralPixelClassification_lnet(train_dict["channels"], train_dict["batch"], train_dict["pixels_x"], train_dict["pixels_y"], train_dict["labels"], batch_size, n_channels, n_classes, train_dict["win_x"], train_dict["win_y"])

	# fit the model on the batches generated by datagen.flow()
	loss_history = model.fit_generator(train_generator,
						steps_per_epoch=int(len(train_dict["labels"])/batch_size),
						epochs=n_epoch,
						validation_data=(X_test,Y_test),
						callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only=True),
							LearningRateScheduler(lr_sched)])

	np.savez(file_name_save_loss, loss_history = loss_history.history)

"""
Executing convnets
"""

def get_image_sizes(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:       
		img_list_channels += [nikon_getfiles(data_location, channel)]
	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	return img_temp.shape
	
def get_images_from_directory(data_location, channel_names):
	img_list_channels = []
	for channel in channel_names:
		img_list_channels += [nikon_getfiles(data_location, channel)]

	img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

	n_channels = len(channel_names)
	all_images = []

	for stack_iteration in range(len(img_list_channels[0])):
		all_channels = np.zeros((1, img_temp.shape[0],img_temp.shape[1], n_channels), dtype = 'float32')
		for j in range(n_channels):
			channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
			all_channels[0,:,:,j] = channel_img
		all_images += [all_channels]
	
	return all_images

# Unet
def run_model_unet(img, model, imaging_field_x = 256, imaging_field_y = 256, normalizing_window_size_x = 64, normalizing_window_size_y = 64):
    
	for j in range(img.shape[-1]):
		img[0,:,:,j] = process_image(img[0,:,:,j], normalizing_window_size_x, normalizing_window_size_y)

	img = np.pad(img, pad_width = [(0,0), (5,5), (5,5), (0,0)], mode = 'reflect')
            
	n_classes = model.layers[-1].output_shape[-1]
	image_size_x = img.shape[1]
	image_size_y = img.shape[2]
	model_output = np.zeros((image_size_x-10,image_size_y-10,n_classes), dtype = np.float32)
	current_output = np.zeros((1,imaging_field_x,imaging_field_y,n_classes), dtype = np.float32)
    
	x_iterator = 0
	y_iterator = 0
    
	while x_iterator<=(image_size_x-imaging_field_x) and y_iterator<=(image_size_y-imaging_field_y):
		current_output = model.predict(img[:,x_iterator:(x_iterator+imaging_field_x),y_iterator:(y_iterator+imaging_field_y),:])
		model_output[x_iterator:(x_iterator+imaging_field_x-10),y_iterator:(y_iterator+imaging_field_y-10),:] = current_output[:,5:(imaging_field_x-5),5:(imaging_field_y-5),:]
        
		if x_iterator<(image_size_x-2*imaging_field_x):
			x_iterator += (imaging_field_x-10)
		else:
			if x_iterator == (image_size_x-imaging_field_x):
				if y_iterator < (image_size_y-2*imaging_field_y):
					y_iterator += (imaging_field_y-10)
					x_iterator = 0
				else:
					if y_iterator == (image_size_y-imaging_field_y):
						y_iterator += (imaging_field_y-10)
					else:
						y_iterator = (image_size_y-imaging_field_y)
						x_iterator = 0
			else:
				x_iterator = image_size_x-(imaging_field_x)

	return model_output

#def run_model_on_directory_unet(data_location, channel_names, output_location, model, imaging_field_x = 256, imaging_field_y = 256, normalizing_window_size_x = 64, normalizing_window_size_y = 64):
    
#	n_classes = model.layers[-1].output_shape[-1]
        
#	return processed_image_list


def run_models_on_directory_unet(data_location, channel_names, output_location, model, normalizing_window_size_x = 64, normalizing_window_size_y = 64):

	# determine the number of channels and classes as well as the imaging field dimensions
	input_shape = model.layers[0].output_shape
	n_channels = input_shape[-1]
	imaging_field_x = input_shape[1]
	imaging_field_y = input_shape[2]
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# determine the image size
	image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

	# process images
	counter = 0

	image_list = get_images_from_directory(data_location, channel_names)
	processed_image_list = []

	for img in image_list:
		print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
		processed_image = run_model_unet(img, model, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y, normalizing_window_size_x = normalizing_window_size_x, normalizing_window_size_y = normalizing_window_size_y)
		processed_image_list += [processed_image]
  
		# Save images
		for feat in range(n_classes):
			current_class = processed_image[:,:,feat]
			cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(counter) + '.tif')
			tiff.imsave(cnnout_name,current_class)
		counter += 1

#	processed_image_list = run_model_on_directory_unet(data_location, channel_names, output_location, model, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y, normalizing_window_size_x = normalizing_window_size_x, normalizing_window_size_y = normalizing_window_size_y)
#	model_output += [np.stack(processed_image_list, axis = 0)]
        
	return processed_image_list

# central pixel based networks
def run_model_pixByPix(img, model, win_x = 30, win_y = 30, std = False, split = True, process = True, bs=32, maxDim=800, normalization = 1):

	if normalization == 1:
		for j in range(img.shape[-1]):
			img[0,:,:,j] = process_image(img[0,:,:,j], win_x, win_y)
	else:
		for j in range(img.shape[-1]):
			img[0,:,:,j] = process_image_onlyLocalAverageSubtraction(img[0,:,:,j], win_x, win_y)
        
	img = np.pad(img, pad_width = [(0,0), (win_x, win_x), (win_y,win_y), (0,0)], mode = 'reflect')
	n_classes = model.layers[-1].output_shape[-1]
	image_size_x = img.shape[1]
	image_size_y = img.shape[2]
	model_output = np.zeros((image_size_x-2*win_x,image_size_y-2*win_y,n_classes), dtype = np.float32)
	x_minIterator = win_x
	x_maxIterator = min(image_size_x,maxDim)-win_x
	y_minIterator = win_y
	y_maxIterator = min(image_size_y,maxDim)-win_y
    
	while x_minIterator<(image_size_x-win_x) and y_minIterator<(image_size_y-win_y):
		test_images = []
		if normalization == 3:
			for x in range(x_minIterator, x_maxIterator):
				for y in range(y_minIterator, y_maxIterator):
					test_images.append(img[0,x-win_x:x+win_x,y-win_y:y+win_y,:])
		else:
			for x in range(x_minIterator, x_maxIterator):
				for y in range(y_minIterator, y_maxIterator):
					test_images.append(img[0,x-win_x:x+win_x+1,y-win_y:y+win_y+1,:])
               
		test_images = np.asarray(test_images)
		test_images = test_images.astype('float32')

		predictions = model.predict(test_images, verbose=1, batch_size=bs)

		cpt = 0
		for x in range(x_minIterator, x_maxIterator):
			for y in range(y_minIterator, y_maxIterator):
				model_output[x-win_x,y-win_y,:] = predictions[cpt,:]
				cpt += 1

		if x_maxIterator < image_size_x-win_x:
			x_minIterator = min(x_maxIterator,image_size_x)
			if image_size_x-x_minIterator < maxDim:
				x_maxIterator = image_size_x-win_x
			else:
				x_maxIterator = x_minIterator+maxDim-win_x
		else:       
			x_minIterator = win_x
			x_maxIterator = min(image_size_x,maxDim)-win_x
			y_minIterator = min(y_maxIterator,image_size_y)
			if image_size_y-y_minIterator < maxDim:
				y_maxIterator = image_size_y-win_y
			else:
				y_maxIterator = y_minIterator+maxDim-win_y

	return model_output

def run_model_pixByPixOnMasks(img, mask, model, win_x = 30, win_y = 30, bs=32, maxDim=800, normalization = 1):

	if normalization == 1:
		for j in range(img.shape[-1]):
			img[0,:,:,j] = process_image(img[0,:,:,j], win_x, win_y)
	else:
		for j in range(img.shape[-1]):
			img[0,:,:,j] = process_image_onlyLocalAverageSubtraction(img[0,:,:,j], win_x, win_y)
        
	img = np.pad(img, pad_width = [(0,0), (win_x, win_x),(win_y,win_y), (0,0)], mode = 'reflect')
	mask = np.pad(mask, pad_width = [(0,0), (win_x, win_x),(win_y,win_y), (0,0)], mode = 'reflect')
            
	n_classes = model.layers[-1].output_shape[-1]
	image_size_x = img.shape[1]
	image_size_y = img.shape[2]
	model_output = np.zeros((image_size_x-2*win_x,image_size_y-2*win_y,n_classes), dtype = np.float32)

	x_minIterator = win_x
	x_maxIterator = min(image_size_x,maxDim)-win_x
	y_minIterator = win_y
	y_maxIterator = min(image_size_y,maxDim)-win_y

	while x_minIterator<(image_size_x-win_x) and y_minIterator<(image_size_y-win_y):
		test_images = []
		if normalization == 3:
			for x in range(x_minIterator, x_maxIterator):
				for y in range(y_minIterator, y_maxIterator):
					if mask[0,x,y,:] > 0:
						test_images.append(img[0,x-win_x:x+win_x,y-win_y:y+win_y,:])
		else:
			for x in range(x_minIterator, x_maxIterator):
				for y in range(y_minIterator, y_maxIterator):
					if mask[0,x,y,:] > 0:
						test_images.append(img[0,x-win_x:x+win_x+1,y-win_y:y+win_y+1,:])

		test_images = np.asarray(test_images)
		test_images = test_images.astype('float32')
        
		predictions = model.predict(test_images, verbose=1, batch_size=bs)
        
		cpt = 0
		for x in range(x_minIterator, x_maxIterator):
			for y in range(y_minIterator, y_maxIterator):
				if mask[0,x,y,:] > 0:
					model_output[x-win_x,y-win_y,:] = predictions[cpt,:]
					cpt += 1

		if x_maxIterator < image_size_x-win_x:
			x_minIterator = min(x_maxIterator,image_size_x)
			if image_size_x-x_minIterator < maxDim:
				x_maxIterator = image_size_x-win_x
			else:
				x_maxIterator = x_minIterator+maxDim-win_x
		else:       
			x_minIterator = win_x
			x_maxIterator = min(image_size_x,maxDim)-win_x
			y_minIterator = min(y_maxIterator,image_size_y)
			if image_size_y-y_minIterator < maxDim:
				y_maxIterator = image_size_y-win_y
			else:
				y_maxIterator = y_minIterator+maxDim-win_y

	return model_output


def run_model_on_directory_pixByPix(data_location, channel_names, mask_names, output_location, model, win_x = 30, win_y = 30, bs=32, maxDim=800, normalization = 1):
	n_classes = model.layers[-1].output_shape[-1]
	counter = 0

	image_list = get_images_from_directory(data_location, channel_names)
	processed_image_list = []
    
	if mask_names == "None":
		for img in image_list:
			print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
			processed_image = run_model_pixByPix(img, model, win_x = win_x, win_y = win_y, bs=bs, maxDim=maxDim, normalization = normalization)
			processed_image_list += [processed_image]

			# Save images
			for feat in range(n_classes):
				current_class = processed_image[:,:,feat]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) +"_frame_"+ str(counter) + '.tif')
				tiff.imsave(cnnout_name,current_class)
			counter += 1

	else:
		mask_list = get_images_from_directory(data_location, mask_names)
		for img in image_list:
			print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
			processed_image = run_model_pixByPixOnMasks(img, mask_list[counter], model, win_x = win_x, win_y = win_y, bs=bs, maxDim=maxDim, normalization = normalization)
			processed_image_list += [processed_image]

			# Save images
			for feat in range(n_classes):
				current_class = processed_image[:,:,feat]
				cnnout_name = os.path.join(output_location, 'feature_' + str(feat) +"_frame_"+ str(counter) + '.tif')
				tiff.imsave(cnnout_name,current_class)
			counter += 1
    
	return processed_image_list


def run_models_on_directory(data_location, channel_names, mask_names, output_location, model, bs=32, maxDim=800, normalization = 1):

	# determine the number of channels and classes as well as the imaging field dimensions
	input_shape = model.layers[0].output_shape
	n_channels = input_shape[-1]
	if normalization == 3:
		imaging_field_x = int(input_shape[1]/2)
		imaging_field_y = int(input_shape[2]/2)
	else:
		imaging_field_x = int((input_shape[1]-1)/2)
		imaging_field_y = int((input_shape[2]-1)/2)
	output_shape = model.layers[-1].output_shape
	n_classes = output_shape[-1]

	# determine the image size
	image_size_x, image_size_y = get_image_sizes(data_location, channel_names)

	# process images
	cpt = 0
	model_output = []
	processed_image_list= run_model_on_directory_pixByPix(data_location, channel_names, mask_names, output_location, model, win_x = imaging_field_x, win_y = imaging_field_y, bs=bs, maxDim=maxDim, normalization = normalization)
	model_output += [np.stack(processed_image_list, axis = 0)]

	return model_output