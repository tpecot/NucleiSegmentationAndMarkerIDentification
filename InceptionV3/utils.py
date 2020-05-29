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
from skimage.transform import rescale, resize, downscale_local_mean, rotate, AffineTransform, warp
from skimage.io import imread, imsave
import skimage.external.tifffile as tiff
import matplotlib.pyplot as plt

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import copy

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

def getfiles(direc_name,channel_name):
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
    else:
        im = np.float32(imread(file_name))
    return im

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
    test_ind = arr_shuff[num_train:full_batch_size]

    X_test, y_test = data_generator(channels.astype("float32"), batch[test_ind], pixels_x[test_ind], pixels_y[test_ind], labels[test_ind], win_x = win_x, win_y = win_y)
    train_dict = {"channels": channels.astype("float32"), "batch": batch[train_ind], "pixels_x": pixels_x[train_ind], "pixels_y": pixels_y[train_ind], "labels": labels[train_ind], "win_x": win_x, "win_y": win_y}

    return train_dict, (X_test, y_test)

        

def GenerateRandomImgaugAugmentation(
        pAugmentationLevel=10,           # number of augmentations
        pEnableFlipping1=True,          # enable x flipping
        pEnableFlipping2=True,          # enable y flipping
        pEnableRotation90=True,           # enable rotation
        pEnableRotation=True,           # enable rotation
        pMaxRotationDegree=5,
        pEnableShearX=True,             # enable x shear
        pEnableShearY=True,             # enable y shear
        pMaxShearDegree=5,             # maximum shear degree
        pEnableDropOut=True,            # enable pixel dropout
        pMaxDropoutPercentage=0.01,     # maximum dropout percentage
        pEnableBlur=True,               # enable gaussian blur
        pBlurSigma=.03,                  # maximum sigma for gaussian blur
        pEnableSharpness=True,          # enable sharpness
        pSharpnessFactor=0.01,           # maximum additional sharpness
        pEnableEmboss=True,             # enable emboss
        pEmbossFactor=0.01,              # maximum emboss
        pEnableBrightness=True,         # enable brightness
        pBrightnessFactor=0.01,         # maximum +- brightness
        pEnableRandomNoise=True,        # enable random noise
        pMaxRandomNoise=0.01,           # maximum random noise strength
        pEnableInvert=True,             # enables color invert
        pEnableContrast=True,           # enable contrast change
        pContrastFactor=0.01,            # maximum +- contrast
):
    
    augmentationMap = []
    augmentationMapOutput = []


    if pEnableFlipping1:
        aug = iaa.Fliplr()
        augmentationMap.append(aug)
        
    if pEnableFlipping2:
        aug = iaa.Flipud()
        augmentationMap.append(aug)

    if pEnableRotation90:
        randomNumber = random.Random().randint(0,3)
        aug = iaa.Rot90(randomNumber)

    if pEnableRotation:
        if random.Random().randint(0, 1)==1:
            randomRotation = random.Random().random()*pMaxRotationDegree
        else:
            randomRotation = -random.Random().random()*pMaxRotationDegree
        aug = iaa.ShearX(randomRotation)
        augmentationMap.append(aug)

    if pEnableShearX:
        if random.Random().randint(0, 1)==1:
            randomShearingX = random.Random().random()*pMaxShearDegree
        else:
            randomShearingX = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearX(randomShearingX)
        augmentationMap.append(aug)

    if pEnableShearY:
        if random.Random().randint(0, 1)==1:
            randomShearingY = random.Random().random()*pMaxShearDegree
        else:
            randomShearingY = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearY(randomShearingY)
        augmentationMap.append(aug)

    if pEnableDropOut:
        randomDropOut = random.Random().random()*pMaxDropoutPercentage
        aug = iaa.Dropout(p=randomDropOut, per_channel=False)
        augmentationMap.append(aug)

    if pEnableBlur:
        randomBlur = random.Random().random()*pBlurSigma
        aug = iaa.GaussianBlur(randomBlur)
        augmentationMap.append(aug)

    if pEnableSharpness:
        randomSharpness = random.Random().random()*pSharpnessFactor
        aug = iaa.Sharpen(randomSharpness)
        augmentationMap.append(aug)

    if pEnableEmboss:
        randomEmboss = random.Random().random()*pEmbossFactor
        aug = iaa.Emboss(randomEmboss)
        augmentationMap.append(aug)

    if pEnableBrightness:
        if random.Random().randint(0, 1)==1:
            randomBrightness = 1 - random.Random().random()*pBrightnessFactor
        else:
            randomBrightness = 1 + random.Random().random()*pBrightnessFactor
        aug = iaa.Add(randomBrightness)
        augmentationMap.append(aug)

    if pEnableRandomNoise:
        if random.Random().randint(0, 1)==1:
            randomNoise = 1 - random.Random().random()*pMaxRandomNoise
        else:
            randomNoise = 1 + random.Random().random()*pMaxRandomNoise
        aug = iaa.MultiplyElementwise(randomNoise,  per_channel=True)
        augmentationMap.append(aug)
        
    if pEnableInvert:
        aug = iaa.Invert(1)
        augmentationMap.append(aug)

    if pEnableContrast:
        if random.Random().randint(0, 1)==1:
            randomContrast = 1 - random.Random().random()*pContrastFactor
        else:
            randomContrast = 1 + random.Random().random()*pContrastFactor
        aug = iaa.contrast.LinearContrast(randomContrast)
        augmentationMap.append(aug)

    arr = np.arange(len(augmentationMap))
    np.random.shuffle(arr)
    for i in range(pAugmentationLevel):
        augmentationMapOutput.append(augmentationMap[arr[i]])
    
        
    return iaa.Sequential(augmentationMapOutput)


def random_sample_generator_centralPixelClassification(img, img_ind, x_coords, y_coords, y_init, batch_size, n_channels, n_classes, win_x, win_y, nb_augmentations = 1):

    cpt = 0

    n_images = len(img_ind)
    arr = np.arange(n_images)
    np.random.shuffle(arr)

    while(True):

        # buffers for a batch of data
        batch_x = np.zeros(tuple([batch_size] + [2*win_x+1,2*win_y+1] + [n_channels]))
        batch_y = np.zeros(tuple([batch_size] + [n_classes]))
        # get one image at a time
        for k in range(batch_size):

            # get random image
            img_index = arr[cpt%len(img_ind)]

            # open images
            patch_x = img[img_ind[img_index], (x_coords[img_index]-win_x):(x_coords[img_index]+win_x+1), (y_coords[img_index]-win_y):(y_coords[img_index]+win_y+1), :]
            patch_x = np.asarray(patch_x)
            current_class = np.asarray(y_init[img_index])
            current_class = current_class.astype('float32')

            augmentationMap = GenerateRandomImgaugAugmentation()
            patch_x = augmentationMap(image=patch_x)


            # save image to buffer
            batch_x[k, :, :, :] = patch_x.astype('float32')
            batch_y[k, :] = current_class
            cpt += 1

        # return the buffer
        yield(batch_x, batch_y)
        

"""
Training convnets
"""
def train_model_sample(model = None, dataset = None,  optimizer = None, 
    expt = "", batch_size = 32, n_epoch = 100,
    direc_save = "./trained_classifiers/", 
    direc_data = "./training_data_npz/", 
    lr_sched = rate_scheduler(lr = 0.01, decay = 0.95), nb_augmentations = 1, normalization = 1):

    training_data_file_name = os.path.join(direc_data, dataset + ".npz")
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + ".h5")
    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + dataset + "_" + expt + ".npz")

    if nb_augmentations == 1:
        train_dict, (X_test, Y_test) = get_data_sample(training_data_file_name)
    else:
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

    train_generator = random_sample_generator_centralPixelClassification(train_dict["channels"], train_dict["batch"], train_dict["pixels_x"], train_dict["pixels_y"], train_dict["labels"], batch_size, n_channels, n_classes, train_dict["win_x"], train_dict["win_y"], nb_augmentations)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(train_generator, 
                                       steps_per_epoch = int(nb_augmentations*len(train_dict["labels"])/batch_size), 
                                       epochs = n_epoch, validation_data = (X_test,Y_test), 
                                       callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only=True), LearningRateScheduler(lr_sched)])

    np.savez(file_name_save_loss, loss_history = loss_history.history)

"""
Executing convnets
"""

def get_image_sizes(data_location, channel_names):
    img_list_channels = []
    for channel in channel_names:       
        img_list_channels += [getfiles(data_location, channel)]
    img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

    return img_temp.shape

def get_images_from_directory(data_location, channel_names):
    img_list_channels = []
    for channel in channel_names:
        img_list_channels += [getfiles(data_location, channel)]

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
