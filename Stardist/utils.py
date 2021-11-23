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
import skimage

import sys
import os
from scipy import ndimage
from scipy.misc import bytescale
import threading
from threading import Thread, Lock
import h5py
import csv
import json

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from skimage.io import imread, imsave
import skimage as sk
import tifffile as tiff
import cv2
  
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import SGD, RMSprop
    
import datetime

import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

from keras.utils import np_utils

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D, Config3D, StarDist3D
from csbdeep.utils import normalize
import scipy.ndimage

"""
Interfaces
"""


def training_parameters_stardist_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    validation_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    learning_rate = np.zeros([nb_trainings], HBox)
    nb_epochs = np.zeros([nb_trainings], HBox)
    data_augmentation = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    train_to_val_ratio = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training directory")
        training_dir[i] = FileChooser('./datasets')
        display(training_dir[i])
        print('\x1b[1m'+"Validation directory")
        validation_dir[i] = FileChooser('./datasets')
        display(validation_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./models')
        display(output_dir[i])

        label_layout = Layout(width='180px',height='30px')

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        learning_rate[i] = HBox([Label('Learning rate:', layout=label_layout), widgets.FloatText(
            value=1e-4, description='', disabled=False)])
        display(learning_rate[i])

        nb_epochs[i] = HBox([Label('Number of epochs:', layout=label_layout), widgets.IntText(
            value=400, description='', disabled=False)])
        display(nb_epochs[i])

        data_augmentation[i] = HBox([Label('Data augmentation:', layout=label_layout), widgets.Checkbox(
            value=True, description='', disabled=False)])
        display(data_augmentation[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(batch_size[i])

        train_to_val_ratio[i] = HBox([Label('Ratio of training in validation:', layout=label_layout), widgets.BoundedFloatText(
            value=0.2, min=0.01, max=0.99, step=0.01, description='', disabled=False, color='black'
        )])
        display(train_to_val_ratio[i])

    parameters.append(training_dir)
    parameters.append(validation_dir)
    parameters.append(output_dir)
    parameters.append(nb_channels)
    parameters.append(learning_rate)
    parameters.append(nb_epochs)
    parameters.append(data_augmentation)
    parameters.append(batch_size)
    parameters.append(train_to_val_ratio)
    
    return parameters  


def running_parameters_stardist_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    prob_th = np.zeros([nb_trainings], HBox)
    nms_th = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser('./datasets')
        display(input_dir[i])
        print('\x1b[1m'+"Input model")
        input_classifier[i] = FileChooser('./models')
        display(input_classifier[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./datasets')
        display(output_dir[i])

        label_layout = Layout(width='220px',height='30px')

        prob_th[i] = HBox([Label('Probability threshold:', layout=label_layout), widgets.FloatText(
            value=0.65, description='', disabled=False)])
        display(prob_th[i])
        nms_th[i] = HBox([Label('Non-maximum suppresion threshold:', layout=label_layout), widgets.FloatText(
            value=0.3, description='', disabled=False)])
        display(nms_th[i])
        
    parameters.append(input_dir)
    parameters.append(input_classifier)
    parameters.append(output_dir)
    parameters.append(prob_th)
    parameters.append(nms_th)
        
    return parameters  


        
"""
Training and processing calling functions 
"""

def training_Stardist(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[2][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained model")
    
        
        if parameters[6][i].children[1].value==True:
            model_name = "StarDist_"+str(parameters[3][i].children[1].value)+"ch_lr_"+str(parameters[4][i].children[1].value)+"_withDA_"+str(parameters[5][i].children[1].value)+"ep_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            model_name = "StarDist_"+str(parameters[3][i].children[1].value)+"ch_lr_"+str(parameters[4][i].children[1].value)+"_withoutDA_"+str(parameters[5][i].children[1].value)+"ep_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        train_model_sample_Stardist(parameters[0][i].selected, parameters[1][i].selected, model_name,
                                    parameters[3][i].children[1].value, parameters[7][i].children[1].value,
                                    parameters[5][i].children[1].value, parameters[2][i].selected,
                                    parameters[4][i].children[1].value, parameters[6][i].children[1].value,
                                    parameters[8][i].children[1].value)


def running_stardist(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to run your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        model_path = parameters[1][i].selected
        model = StarDist2D(None, name = os.path.split(os.path.dirname(model_path))[-1], basedir = os.path.abspath(os.path.join(model_path,os.pardir)))
        with open(os.path.join(model_path, 'config.json')) as jsondata:
            data = json.load(jsondata)
        n_channels = data["n_channel_in"]
        run_stardist_models_on_directory(parameters[0][i].selected, parameters[2][i].selected, n_channels, model, parameters[3][i].children[1].value, parameters[4][i].children[1].value)
        del model
        
"""
Useful functions 
"""
def search_label_for_image_in_file(file_name, image_to_search, nb_classes):
    labels = np.zeros((nb_classes), 'int32')
    found_image = False
    # Open the file in read only mode
    with open(file_name, newline='') as csvfile:
        read_obj = csv.reader(csvfile)
        for line in read_obj:
            if line[0]==image_to_search:
                found_image = True
                if int(line[1]) == -1:
                    labels[0] = 1
                else:
                    labels[int(line[1])] = 1
    return found_image, labels

def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn

def process_image(img):
    
    if img.shape[2] == 1:
        output = np.zeros((img.shape[0], img.shape[1], 1), 'float32')
        
        high = np.percentile(img, 99.9)
        low = np.percentile(img, 1)

        output = np.minimum(high, img)
        output = np.maximum(low, output)

        output = (output - low) / (high - low)

    else:
        output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), 'float32')
        for i in range(img.shape[2]):
            percentile = 99.9
            high = np.percentile(img[:,:,i], percentile)
            low = np.percentile(img[:,:,i], 100-percentile)
            
            output[:,:,i] = np.minimum(high, img[:,:,i])
            output[:,:,i] = np.maximum(low, output[:,:,i])
            
            if high>low:
                output[:,:,i] = (output[:,:,i] - low) / (high - low)

    return output

def getfiles(direc_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if ('.png'  in i ) or ('.jpg'  in i ) or ('.tif' in i) or ('tiff' in i)]

    imgfiles = imgfiles
    return imgfiles

def get_image(file_name):
    if ('.tif' in file_name) or ('tiff' in file_name):
        im = tiff.imread(file_name)
        im = np.float32(im)
    else:
        im = cv2.imread(file_name) 
        #im = np.float32(imread(file_name))
        
    if len(im.shape) < 3:
        output_im = np.zeros((im.shape[0], im.shape[1], 1))
        output_im[:, :, 0] = im
        im = output_im
    else:
        if im.shape[0]<im.shape[2]:
            output_im = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
            for i in range(im.shape[0]):
                output_im[:, :, i] = im[i, :, :]
            im = output_im
    
    return im

def get_image_byte(file_name):
    if ('.tif' in file_name) or ('tiff' in file_name):
        im = tiff.imread(file_name)
        #im = bytescale(im)
        im = np.float32(im)
    else:
        #im = cv2.imread(file_name) 
        im = np.float32(imread(file_name))
        
    if len(im.shape) < 3:
        output_im = np.zeros((im.shape[0], im.shape[1], 1))
        output_im[:, :, 0] = im
        im = output_im
    else:
        if im.shape[0]<im.shape[2]:
            output_im = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
            for i in range(im.shape[0]):
                output_im[:, :, i] = im[i, :, :]
            im = output_im
    
    return im

"""
Data generator for training_data
"""
def get_data_sample_Stardist(training_directory, validation_directory, nb_channels = 1, validation_training_ratio = 0.1):

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

        imageValFileList = [f for f in os.listdir(imglist_validation_directory) if ('.png'  in f ) or ('.tif' in f) or ('tiff' in f)]
        
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            
            if os.path.exists(os.path.join(masklist_validation_directory, baseName + ".png")):
                maskPath = os.path.join(masklist_validation_directory, baseName + ".png")
            elif os.path.exists(os.path.join(masklist_validation_directory, baseName + ".tif")):
                maskPath = os.path.join(masklist_validation_directory, baseName + ".tif")
            elif os.path.exists(os.path.join(masklist_validation_directory, baseName + ".tiff")):
                maskPath = os.path.join(masklist_validation_directory, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            current_mask_image = get_image(maskPath)
                
            current_mask_image_index_max = np.max(current_mask_image)
            new_mask_indices = np.zeros([int(current_mask_image_index_max)], dtype=np.uint32)
            for y in range(current_mask_image.shape[0]):
                for x in range(current_mask_image.shape[1]):
                    index = int(current_mask_image[y,x]) - 1
                    if index >= 0:
                        new_mask_indices[index] = 1
            count = 0
            for i in range(int(current_mask_image_index_max)):
                if new_mask_indices[i] > 0:
                    new_mask_indices[i] = count
                    count += 1
        
            current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
            for y in range(current_mask_image.shape[0]):
                for x in range(current_mask_image.shape[1]):
                    index = int(current_mask_image[y,x]) - 1
                    if index >= 0:
                        current_mask[y,x] = new_mask_indices[index]
    
            labels_validation.append(current_mask.astype('int32'))
            
            imagePath = os.path.join(imglist_validation_directory, imageFile)
            
            current_image = get_image_byte(imagePath)
            if current_image.shape[0]!=current_mask_image.shape[0]:
                sys.exit("The image " + baseName + " has a different x dimension than its corresponding mask")
            if current_image.shape[1]!=current_mask_image.shape[1]:
                sys.exit("The image " + baseName + " has a different y dimension than its corresponding mask")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            channels_validation.append(process_image(current_image))

        imageFileList = [f for f in os.listdir(imglist_training_directory) if ('.png'  in f ) or ('.tif' in f) or ('tiff' in f)]
        
        for imageFile in imageFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            
            if os.path.exists(os.path.join(masklist_training_directory, baseName + ".png")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".png")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tif")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tif")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tiff")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            current_mask_image = get_image(maskPath)
            
            current_mask_image_index_max = np.max(current_mask_image)
            new_mask_indices = np.zeros([int(current_mask_image_index_max)], dtype=np.uint32)
            for y in range(current_mask_image.shape[0]):
                for x in range(current_mask_image.shape[1]):
                    index = int(current_mask_image[y,x]) - 1
                    if index >= 0:
                        new_mask_indices[index] = 1
            count = 0
            for i in range(int(current_mask_image_index_max)):
                if new_mask_indices[i] > 0:
                    new_mask_indices[i] = count
                    count += 1
        
            current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
            for y in range(current_mask_image.shape[0]):
                for x in range(current_mask_image.shape[1]):
                    index = int(current_mask_image[y,x]) - 1
                    if index >= 0:
                        current_mask[y,x] = new_mask_indices[index]

            labels_training.append(current_mask.astype('int32'))
            
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image_byte(imagePath)
            if current_image.shape[0]!=current_mask_image.shape[0]:
                sys.exit("The image " + baseName + " has a different x dimension than its corresponding mask")
            if current_image.shape[1]!=current_mask_image.shape[1]:
                sys.exit("The image " + baseName + " has a different y dimension than its corresponding mask")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            channels_training.append(current_image)
            
    else:
        imageValFileList = [f for f in os.listdir(imglist_training_directory) if ('.png'  in f ) or ('.tif' in f) or ('tiff' in f)]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image_byte(imagePath)
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            if os.path.exists(os.path.join(masklist_training_directory, baseName + ".png")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".png")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tif")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tif")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tiff")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            current_mask_image = get_image(maskPath)
            if current_image.shape[0]!=current_mask_image.shape[0]:
                sys.exit("The image " + baseName + " has a different x dimension than its corresponding mask")
            if current_image.shape[1]!=current_mask_image.shape[1]:
                sys.exit("The image " + baseName + " has a different y dimension than its corresponding mask")

            current_mask_image_index_max = np.max(current_mask_image)
            new_mask_indices = np.zeros([int(current_mask_image_index_max)], dtype=np.uint32)
            for y in range(current_mask_image.shape[0]):
                for x in range(current_mask_image.shape[1]):
                    index = int(current_mask_image[y,x]) - 1
                    if index >= 0:
                        new_mask_indices[index] = 1
            count = 0
            for i in range(int(current_mask_image_index_max)):
                if new_mask_indices[i] > 0:
                    new_mask_indices[i] = count
                    count += 1
        
            current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
            for y in range(current_mask_image.shape[0]):
                for x in range(current_mask_image.shape[1]):
                    index = int(current_mask_image[y,x]) - 1
                    if index >= 0:
                        current_mask[y,x] = new_mask_indices[index]
                    
        
            if random.Random().random() > validation_training_ratio:
                channels_training.append(current_image)
                labels_training.append(current_mask.astype('int32'))
            else:
                channels_validation.append(process_image(current_image))
                labels_validation.append(current_mask.astype('int32'))

                
    if len(channels_training) < 1:
        sys.exit("Empty train image list")

    #just to be non-empty
    if len(channels_validation) < 1:
        channels_validation += channels_training[len(channels_training)-1]
        labels_validation += channels_validation[len(channels_validation)-1]
    
    
    X_test = channels_validation
    Y_test = labels_validation
        
    for i in range(len(channels_training)):
        channels_training[i] = process_image(channels_training[i])
        
    train_dict = {"channels": channels_training, "labels": labels_training}
    
    return train_dict, (X_test, Y_test)


def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img


def original_augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y



"""
Training convnets
"""
    
def train_model_sample_Stardist(dataset_training = None,  dataset_validation = None,
                                model_name = "model", n_channels = 1, batch_size = 5, n_epoch = 100, 
                                output_dir = "./models/", learning_rate = 1e-3, 
                                data_augmentation = True, train_to_val_ratio = 0.2):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    
    if dataset_training is None:
        sys.exit("The input training dataset needs to be defined")

    train_dict, (X_test, Y_test) = get_data_sample_Stardist(dataset_training, dataset_validation, nb_channels = n_channels, validation_training_ratio = train_to_val_ratio)

    # data information (one way for the user to check if the training dataset makes sense)
    print(len(train_dict["channels"]), 'training images')
    print(len(X_test), 'validation images')

    conf = Config2D (
            # number of channels in input images
            n_channel_in           = n_channels,
            # 32 is a good default choice (see 1_data.ipynb)
            n_rays                 = 32,
            # Predict on subsampled grid for increased efficiency and larger field of view
            grid                   = (2,2),
            use_gpu                = False,
            # Number of U-Net resolution levels (down/up-sampling layers)
            unet_n_depth           = 3,#6,#3,
            # Convolution kernel size for all (U-Net) convolution layers
            unet_kernel_size       = (3,3),
            # Number of convolution kernels (feature channels) for first U-Net layer
            # Doubled after each down-sampling layer
            unet_n_filter_base     = 64,
            # Maxpooling size for all (U-Net) convolution layers
            unet_pool              = (2,2),
            # Number of filters of the extra convolution layer after U-Net (0 to disable)
            net_conv_after_unet    = 128,
            # Train model to predict complete shapes for partially visible objects at image boundary
            train_shape_completion = False,
            # If 'train_shape_completion' is set to True, specify number of pixels to crop at boundary of training patches
            # Should be chosen based on (largest) object sizes
            train_completion_crop  = 32,
            # Size of patches to be cropped from provided training images
            train_patch_size       = (256,256),
            # Regularizer to encourage distance predictions on background regions to be 0
            train_background_reg   = 1e-4,
            # Fraction (0..1) of patches that will only be sampled from regions that contain foreground pixels
            train_foreground_only  = 0.9,
            # Training loss for star-convex polygon distances ('mse' or 'mae')
            train_dist_loss        = 'mae',
            # Weights for losses relating to (probability, distance)
            train_loss_weights     = (1,0.2),
            # Number of training epochs
            train_epochs           = n_epoch,
            # Number of parameter update steps per epoch
            train_steps_per_epoch  = len(train_dict["channels"])/batch_size,
            # Learning rate for training
            train_learning_rate    = learning_rate,
            # Batch size for training
            train_batch_size       = batch_size,
            # Number of patches to be extracted from validation images (``None`` = one patch per image)
            train_n_val_patches    = None,
            # Enable TensorBoard for monitoring training progress
            train_tensorboard      = True,
        )

    model = StarDist2D(conf, name=model_name, basedir=output_dir)
    if data_augmentation==True:
        augment = original_augmenter
    else:
        augment = None
    model.train(train_dict["channels"], train_dict["labels"], validation_data=(X_test,Y_test), augmenter = augment)
    try:
        model.optimize_thresholds(X_test, Y_test)
    except Exception as e:
        print('Failed to optimize thresholds')
    del model
    

"""
Executing convnets
"""

def get_image_sizes(data_location):
    width = 256
    height = 256
    nb_channels = 1
    img_list = []
    img_list += [getfiles(data_location)]
    img_temp = get_image(os.path.join(data_location, img_list[0][0]))
    if len(img_temp.shape)>2:
        if img_temp.shape[0]<img_temp.shape[2]:
            nb_channels = img_temp.shape[0]
            width = img_temp.shape[1]
            height=img_temp.shape[2]
        else:
            nb_channels = img_temp.shape[2]
            width = img_temp.shape[0]
            height=img_temp.shape[1]
    else:
        width = img_temp.shape[0]
        height=img_temp.shape[1]
    return width, height, nb_channels

def get_images_from_directory(data_location):
    img_list = []
    img_list += [getfiles(data_location)]

    all_images = []
    for stack_iteration in range(len(img_list[0])):
        current_img = get_image_byte(os.path.join(data_location, img_list[0][stack_iteration]))
        all_channels = np.zeros((1, current_img.shape[0], current_img.shape[1], current_img.shape[2]), dtype = 'float32')
        all_channels[0, :, :, :] = current_img
        all_images += [all_channels]
            
    return all_images

        
def run_stardist_models_on_directory(data_location, output_location, n_channels, model, prob_thresh=0.5, nms_thresh=0.5):

    # create output folder if it doesn't exist
    os.makedirs(name=output_location, exist_ok=True)
    
    # determine the image size
    image_size_x, image_size_y, nb_channels = get_image_sizes(data_location)
    
    if n_channels!=nb_channels:
        sys.exit("The input image has a different number of channels than indicated in the Stardist architecture")


    # process images
    counter = 0
    img_list_files = [getfiles(data_location)]

    image_list = get_images_from_directory(data_location)

    for img in image_list:
        print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
        processed_image, details = model.predict_instances(process_image(img[0, :, :, :]), n_tiles=model._guess_n_tiles(img[0, :, :, :]), prob_thresh=prob_thresh, nms_thresh=nms_thresh, show_tile_progress=False)

        # Save images
        cnnout_name = os.path.join(output_location, os.path.splitext(img_list_files[0][counter])[0] + ".tiff")
        tiff.imsave(cnnout_name, processed_image)

        counter += 1

