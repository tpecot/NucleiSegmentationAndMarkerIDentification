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
import skimage as sk
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

def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn

def process_image(channel_img, win_x, win_y):
    p50 = np.percentile(channel_img, 50)
    channel_img /= max(p50,0.01)
    avg_kernel = np.ones((win_x + 1, win_y + 1))
    channel_img -= ndimage.convolve(channel_img, avg_kernel)/avg_kernel.size
    return channel_img

def getfiles(direc_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if '.png' in i]
    
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

def get_image_dim3(file_name):
    if '.tif' in file_name:
        im = tiff.imread(file_name)
        im = bytescale(im)
        im = np.float32(im)
    else:
        im = np.float32(imread(file_name))
    if len(im.shape) < 3:
        output_im = np.zeros((im.shape[0], im.shape[1], 1))
        output_im[:, :, 0] = im
        im = output_im
    return im

"""
Data generator for training_data
"""
def get_data_sample(training_directory, validation_directory, nb_channels = 1, nb_classes = 3, imaging_field_x = 256, imaging_field_y = 256, nb_augmentations = 1, dim_norm1 = 64, dim_norm2 = 64, validationTrainingRatio = 0.1, class_to_dilate = [0,0,0], dil_radius = 1):

    channels_training = []
    labels_training = []
    channels_validation = []
    labels_validation = []
    
    imglist_training_directory = os.path.join(training_directory, 'images/')
    masklist_training_directory = os.path.join(training_directory, 'masks/')

    # adding evaluation data into validation
    if validation_directory is not None:

        imglist_validation_directory = os.path.join(validation_directory, 'images/')
        masklist_validation_directory = os.path.join(validation_directory, 'masks/')

        imageValFileList = [f for f in os.listdir(imglist_validation_directory) if os.path.isfile(os.path.join(imglist_validation_directory, f))]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imglist_validation_directory, imageFile)
            current_image = get_image_dim3(imagePath)
            channels_validation.append(current_image.astype('uint8'))
            maskPath = os.path.join(masklist_validation_directory, baseName + ".png")
            current_mask_image = get_image(maskPath)
            current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
            current_mask[: , :, (nb_classes-1)] = np.where(current_mask_image == 0, 1, 0)
            for i in range(nb_classes):
                if i > 0:
                    current_mask[: , :, i-1] = np.where(current_mask_image == i, 1, 0)
            if np.sum(class_to_dilate) > 0:
                for i in range(nb_classes):
                    if class_to_dilate[i] == 1:
                        strel = sk.morphology.disk(dil_radius)
                        current_mask[: , :, i] = sk.morphology.binary_dilation(current_mask[: , :, i], selem = strel)
                for i in range(nb_classes):
                    if class_to_dilate[i] != 1:
                        for k in range(nb_classes):
                            if class_to_dilate[k] == 1:
                                current_mask[:, :, i] -= current_mask[:, :, k]
                        current_mask[:, :, i] = current_mask[:, :, i] > 0
            labels_validation.append(current_mask.astype('int32'))
            
        imageFileList = [f for f in os.listdir(imglist_training_directory) if os.path.isfile(os.path.join(imglist_training_directory, f))]
        for imageFile in imageFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image_dim3(imagePath)
            channels_training.append(current_image.astype('uint8'))
            maskPath = os.path.join(masklist_training_directory, baseName + ".png")
            current_mask_image = get_image(maskPath)
            current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
            current_mask[: , :, (nb_classes-1)] = np.where(current_mask_image == 0, 1, 0)
            for i in range(nb_classes):
                if i > 0:
                    current_mask[: , :, i-1] = np.where(current_mask_image == i, 1, 0)
            if np.sum(class_to_dilate) > 0:
                for i in range(nb_classes):
                    if class_to_dilate[i] == 1:
                        strel = sk.morphology.disk(dil_radius)
                        current_mask[: , :, i] = sk.morphology.binary_dilation(current_mask[: , :, i], selem = strel)
                for i in range(nb_classes):
                    if class_to_dilate[i] != 1:
                        for k in range(nb_classes):
                            if class_to_dilate[k] == 1:
                                current_mask[:, :, i] -= current_mask[:, :, k]
                        current_mask[:, :, i] = current_mask[:, :, i] > 0
            labels_training.append(current_mask.astype('int32'))
            
    # splitting train data into train and validation
    else:
        imageValFileList = [f for f in os.listdir(imglist_training_directory) if os.path.isfile(os.path.join(imglist_training_directory, f))]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image_dim3(imagePath)
            maskPath = os.path.join(imglist_training_directory, baseName + ".png")
            current_mask_image = get_image(maskPath)
            current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
            current_mask[: , :, (nb_classes-1)] = np.where(current_mask_image == 0, 1, 0)
            for i in range(nb_classes):
                if i > 0:
                    current_mask[: , :, i-1] = np.where(current_mask_image == i, 1, 0)
            if np.sum(class_to_dilate) > 0:
                for i in range(nb_classes):
                    if class_to_dilate[i] == 1:
                        strel = sk.morphology.disk(dil_radius)
                        current_mask[: , :, i] = sk.morphology.binary_dilation(current_mask[: , :, i], selem = strel)
                for i in range(nb_classes):
                    if class_to_dilate[i] != 1:
                        for k in range(nb_classes):
                            if class_to_dilate[k] == 1:
                                current_mask[:, :, i] -= current_mask[:, :, k]
                        current_mask[:, :, i] = current_mask[:, :, i] > 0
        
            if random.Random().random() > validationTrainingRatio:
                channels_training.append(current_image.astype('uint8'))
                labels_training.append(current_mask.astype('int32'))
            else:
                channels_validation.append(current_image.astype('uint8'))
                labels_validation.append(current_mask.astype('int32'))

                
    if len(channels_training) < 1:
        raise ValueError("Empty train image list")

    #just to be non-empty
    if len(channels_validation) < 1:
        channels_validation += channels_training[len(channels_training)-1]
        labels_validation += channels_validation[len(channels_validation)-1]
    
    X_test = np.asarray(channels_validation).astype('float32')
    for k in range(X_test.shape[0]):
        for j in range(X_test.shape[-1]):
            X_test[k,:,:,j] = process_image(X_test[k,:,:,j], dim_norm1, dim_norm2)
    X_test = X_test[:,0:imaging_field_x,0:imaging_field_y,:]
    Y_test = np.asarray(labels_validation)[:,0:imaging_field_x,0:imaging_field_y,:]
    
    train_dict = {"channels": np.asarray(channels_training), "labels": np.asarray(labels_training)}

    return train_dict, (np.asarray(X_test).astype('float32'), np.asarray(Y_test).astype('int32'))


def random_sample_generator(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2, dim_norm1, dim_norm2):

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
            patch_y = np.asarray(patch_y)
            patch_y = patch_y.astype('int32')

            # image normalization
            patch_x = patch_x.astype('float32')
            for j in range(patch_x.shape[-1]):
                patch_x[:,:,j] = process_image(patch_x[:,:,j], dim_norm1, dim_norm2)

            # save image to buffer
            x[k, :, :, :] = patch_x
            y[k, :, :, :] = patch_y
            cpt += 1
            
        # return the buffer
        yield(x, y)


def GenerateRandomImgaugAugmentation(
        pAugmentationLevel=5,           # number of augmentations
        pEnableFlipping1=True,          # enable x flipping
        pEnableFlipping2=True,          # enable y flipping
        pEnableRotation90=True,           # enable rotation
        pEnableRotation=True,           # enable rotation
        pMaxRotationDegree=15,             # maximum rotation degree
        pEnableShearX=True,             # enable x shear
        pEnableShearY=True,             # enable y shear
        pMaxShearDegree=15,             # maximum shear degree
        pEnableDropOut=True,            # enable pixel dropout
        pMaxDropoutPercentage=0.1,     # maximum dropout percentage
        pEnableBlur=True,               # enable gaussian blur
        pBlurSigma=.25,                  # maximum sigma for gaussian blur
        pEnableSharpness=True,          # enable sharpness
        pSharpnessFactor=0.1,           # maximum additional sharpness
        pEnableEmboss=True,             # enable emboss
        pEmbossFactor=0.1,              # maximum emboss
        pEnableBrightness=True,         # enable brightness
        pBrightnessFactor=0.1,         # maximum +- brightness
        pEnableRandomNoise=True,        # enable random noise
        pMaxRandomNoise=0.1,           # maximum random noise strength
        pEnableInvert=True,             # enables color invert
        pEnableContrast=True,           # enable contrast change
        pContrastFactor=0.1,            # maximum +- contrast
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
        randomNumber = random.Random().randint(1,3)
        aug = iaa.Rot90(randomNumber)
        augmentationMap.append(aug)

    if pEnableRotation:
        if random.Random().randint(0, 1)==1:
            randomRotation = random.Random().random()*pMaxRotationDegree
        else:
            randomRotation = -random.Random().random()*pMaxRotationDegree
        aug = iaa.Rotate(randomRotation)
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


def random_sample_generator_dataAugmentation(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2, dim_norm1, dim_norm2):

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
            img_index = arr[cpt%n_images]

            # open images
            x_big = x_init[img_index]
            y_big = y_init[img_index]

            # augmentation
            segmap = SegmentationMapsOnImage(y_big, shape=x_big.shape)
            augmentationMap = GenerateRandomImgaugAugmentation()
            x_aug, segmap = augmentationMap(image=x_big, segmentation_maps=segmap)
            y_aug = segmap.get_arr()
            
            # image normalization
            x_norm = x_aug.astype('float32')
            for j in range(x_norm.shape[-1]):
                x_norm[:,:,j] = process_image(x_norm[:,:,j], dim_norm1, dim_norm2)
                
            # get random crop
            start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            patch_x = x_norm[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_y = y_aug[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_x = np.asarray(patch_x)
            patch_y = np.asarray(patch_y)

            # save image to buffer
            x[k, :, :, :] = patch_x
            y[k, :, :, :] = patch_y


            cpt += 1
        
        # return the buffer
        yield(x, y)


"""
Training convnets
"""
def weighted_crossentropy_3classes(weight1, weight2, weight3):

    def func(y_true, y_pred):
        class_weights = ([[[[weight1, weight2, weight3]]]])
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        weighted_losses = weights * unweighted_losses
        return tf.reduce_mean(weighted_losses)

    return func


def train_model_sample(model = None, dataset_training = None,  dataset_validation = None,
                       optimizer = None, expt = "", batch_size = 5, n_epoch = 100, 
                       imaging_field_x = 256, imaging_field_y = 256, 
                       direc_save = "./trained_classifiers/",
                       lr_sched = rate_scheduler(lr = 0.01, decay = 0.95), nb_augmentations = 1, 
                       normalizing_window_size_x = 64, normalizing_window_size_y = 64,
                       validationTrainingRatio = 0.1, class_to_dilate = [0,0,0], dil_radius = 1):

    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name_save = os.path.join(direc_save, todays_date + "_" + expt + ".h5")
    file_name_save_loss = os.path.join(direc_save, todays_date + "_" + expt + ".npz")

    train_dict, (X_test, Y_test) = get_data_sample(dataset_training, dataset_validation, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y, nb_augmentations = nb_augmentations, dim_norm1 = normalizing_window_size_x, dim_norm2 = normalizing_window_size_y,  validationTrainingRatio = 0.1, class_to_dilate = class_to_dilate, dil_radius = dil_radius)

    # data information (one way for the user to check if the training dataset makes sense)
    print(train_dict["channels"].shape[0], 'training images')
    print(X_test.shape[0], 'validation images')

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
    model.compile(loss = weighted_crossentropy_3classes(weight1 = w1, weight2 = w2, weight3 = w3), optimizer = optimizer, metrics=['accuracy'])

    # prepare the generation of data
    if nb_augmentations == 1:
        train_generator = random_sample_generator(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_x, normalizing_window_size_x, normalizing_window_size_y) 
    else:
        train_generator = random_sample_generator_dataAugmentation(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_x, normalizing_window_size_x, normalizing_window_size_y) 
        
    # fit the model
    loss_history = model.fit_generator(train_generator,
                                       steps_per_epoch = int(nb_augmentations*len(train_dict["labels"])/batch_size), 
                                       epochs=n_epoch, validation_data=(X_test,Y_test), 
                                       callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only = True),LearningRateScheduler(lr_sched)])
    
    np.savez(file_name_save_loss, loss_history = loss_history.history)



"""
Executing convnets
"""

def get_image_sizes(data_location):
    img_list = []
    img_list += [getfiles(data_location)]
    img_temp = get_image(os.path.join(data_location, img_list[0][0]))

    return img_temp.shape

def get_images_from_directory(data_location):
    img_list = []
    img_list += [getfiles(data_location)]

    img_temp = get_image(os.path.join(data_location, img_list[0][0]))
    
    n_channels = 1
    if len(img_temp.shape) > 2:
        n_channels = len(img_temp.shape[2])
    all_images = []

    for stack_iteration in range(len(img_list[0])):
        all_channels = np.zeros((1, img_temp.shape[0],img_temp.shape[1], n_channels), dtype = 'float32')
        for j in range(n_channels):
            channel_img = get_image(os.path.join(data_location, img_list[j][stack_iteration]))
            all_channels[0,:,:,j] = channel_img
        all_images += [all_channels]

    return all_images

def run_model(img, model, imaging_field_x = 256, imaging_field_y = 256, normalizing_window_size_x = 64, normalizing_window_size_y = 64):
    
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


def run_models_on_directory(data_location, output_location, model, normalizing_window_size_x = 64, normalizing_window_size_y = 64):

    # determine the number of channels and classes as well as the imaging field dimensions
    input_shape = model.layers[0].output_shape
    n_channels = input_shape[-1]
    imaging_field_x = input_shape[1]
    imaging_field_y = input_shape[2]
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    # determine the image size
    image_size_x, image_size_y = get_image_sizes(data_location)

    # process images
    counter = 0

    image_list = get_images_from_directory(data_location)
    processed_image_list = []

    for img in image_list:
        print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
        processed_image = run_model(img, model, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y, normalizing_window_size_x = normalizing_window_size_x, normalizing_window_size_y = normalizing_window_size_y)
        processed_image_list += [processed_image]
  
        # Save images
        for feat in range(n_classes):
            current_class = processed_image[:,:,feat]
            cnnout_name = os.path.join(output_location, 'feature_' + str(feat) + "_frame_" + str(counter) + '.tif')
            tiff.imsave(cnnout_name,current_class)
        counter += 1

    return processed_image_list
