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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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
import tifffile as tiff
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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.engine import Layer, InputSpec
from keras.utils import np_utils
from keras.optimizers import SGD

import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

import glob, fnmatch

from models import inceptionV3 as inceptionV3

"""
Interfaces
"""
def data_preprocessing_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_classes = np.zeros([nb_trainings], HBox)
    window_size_x = np.zeros([nb_trainings], HBox)
    window_size_y = np.zeros([nb_trainings], HBox)
    normalization = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser('./trainingData')
        display(input_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./trainingDataNpz')
        display(output_dir[i])

        label_layout = Layout(width='230px',height='30px')

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        window_size_x[i] = HBox([Label('Half window size for imaging field in x:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(window_size_x[i])

        window_size_y[i] = HBox([Label('Half window size for imaging field in y:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(window_size_y[i])

        normalization[i] = HBox([Label('Normalization:', layout=label_layout), widgets.RadioButtons(
            options=['nuclei segmentation', 'marker identification'],description='', disabled=False)])
        display(normalization[i])

    parameters.append(input_dir)
    parameters.append(output_dir)
    parameters.append(nb_classes)
    parameters.append(window_size_x)
    parameters.append(window_size_y)
    parameters.append(normalization)
    
    return parameters  

def training_parameters_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    learning_rate = np.zeros([nb_trainings], HBox)
    nb_epochs = np.zeros([nb_trainings], HBox)
    augmentation = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training dataset")
        training_dir[i] = FileChooser('./trainingDataNPZ')
        display(training_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./models')
        display(output_dir[i])

        label_layout = Layout(width='200px',height='30px')

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_y[i])

        learning_rate[i] = HBox([Label('Learning rate:', layout=label_layout), widgets.FloatText(
            value=1e-2, description='', disabled=False)])
        display(learning_rate[i])

        nb_epochs[i] = HBox([Label('Number of epochs:', layout=label_layout), widgets.IntText(
            value=10, description='', disabled=False)])
        display(nb_epochs[i])

        augmentation[i] = HBox([Label('Augmentation:', layout=label_layout), widgets.Checkbox(
            value=True, description='', disabled=False)])
        display(augmentation[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(batch_size[i])

    parameters.append(training_dir)
    parameters.append(output_dir)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(learning_rate)
    parameters.append(nb_epochs)
    parameters.append(augmentation)
    parameters.append(batch_size)
    
    return parameters  

def transfer_learning_parameters_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    pretrained_model = np.zeros([nb_trainings], FileChooser)
    nb_classes_pretrained_model = np.zeros([nb_trainings], FileChooser)
    last_layer_training = np.zeros([nb_trainings], HBox)
    nb_epochs_last_layer = np.zeros([nb_trainings], HBox)
    learning_rate_last_layer = np.zeros([nb_trainings], HBox)
    all_network_training = np.zeros([nb_trainings], HBox)
    nb_epochs_all = np.zeros([nb_trainings], HBox)
    learning_rate_all = np.zeros([nb_trainings], HBox)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    augmentation = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training dataset")
        training_dir[i] = FileChooser('./trainingDataNPZ')
        display(training_dir[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./models')
        display(output_dir[i])
        print('\x1b[1m'+"Pretrained model")
        pretrained_model[i] = FileChooser('./models')
        display(pretrained_model[i])

        label_layout = Layout(width='250px',height='30px')

        nb_classes_pretrained_model[i] = HBox([Label('Number of classes in the pretrained model:', layout=label_layout), widgets.IntText(
            value=3, description='',disabled=False)])
        display(nb_classes_pretrained_model[i])

        last_layer_training[i] = HBox([Label('Training last layer only first:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(last_layer_training[i])

        nb_epochs_last_layer[i] = HBox([Label('Number of epochs for last_layer training:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_epochs_last_layer[i])

        learning_rate_last_layer[i] = HBox([Label('Learning rate for last_layer training:', layout=label_layout), widgets.FloatText(
            value=0.05, description='', disabled=False)])
        display(learning_rate_last_layer[i])

        all_network_training[i] = HBox([Label('Training all network:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(all_network_training[i])

        nb_epochs_all[i] = HBox([Label('Number of epochs for all network training:', layout=label_layout), widgets.IntText(
            value=5, description='', disabled=False)])
        display(nb_epochs_all[i])

        learning_rate_all[i] = HBox([Label('Learning rate for all network training:', layout=label_layout), widgets.FloatText(
            value=0.01, description='', disabled=False)])
        display(learning_rate_all[i])

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_y[i])

        augmentation[i] = HBox([Label('Augmentation:', layout=label_layout), widgets.Checkbox(
            value=True, description='', disabled=False)])
        display(augmentation[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(batch_size[i])

    parameters.append(training_dir)
    parameters.append(output_dir)
    parameters.append(pretrained_model)
    parameters.append(nb_classes_pretrained_model)
    parameters.append(last_layer_training)
    parameters.append(nb_epochs_last_layer)
    parameters.append(learning_rate_last_layer)
    parameters.append(all_network_training)
    parameters.append(nb_epochs_all)
    parameters.append(learning_rate_all)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(augmentation)
    parameters.append(batch_size)
    
    return parameters  

def running_parameters_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    input_masks = np.zeros([nb_trainings], FileChooser)
    nb_channels = np.zeros([nb_trainings], HBox)
    nb_classes = np.zeros([nb_trainings], HBox)
    imaging_field_x = np.zeros([nb_trainings], HBox)
    imaging_field_y = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    normalization = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser('./testingData')
        display(input_dir[i])
        print('\x1b[1m'+"Input model")
        input_classifier[i] = FileChooser('./models')
        display(input_classifier[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./testingData')
        display(output_dir[i])

        label_layout = Layout(width='150px',height='30px')

        input_masks[i] = HBox([Label('Only apply on masks:', layout=label_layout), widgets.Text(
            value='None', description='', disabled=False)])
        display(input_masks[i])

        nb_channels[i] = HBox([Label('Number of channels:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_channels[i])

        nb_classes[i] = HBox([Label('Number of classes:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_classes[i])

        imaging_field_x[i] = HBox([Label('Imaging field in x:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_x[i])

        imaging_field_y[i] = HBox([Label('Imaging field in y:', layout=label_layout), widgets.IntText(
            value=65, description='', disabled=False)])
        display(imaging_field_y[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=32, description='', disabled=False)])
        display(batch_size[i])

        normalization[i] = HBox([Label('Normalization:', layout=label_layout), widgets.RadioButtons(
            options=['nuclei segmentation', 'marker identification'],description='', disabled=False)])
        display(normalization[i])

    parameters.append(input_dir)
    parameters.append(input_classifier)
    parameters.append(output_dir)
    parameters.append(input_masks)
    parameters.append(nb_channels)
    parameters.append(nb_classes)
    parameters.append(imaging_field_x)
    parameters.append(imaging_field_y)
    parameters.append(batch_size)
    parameters.append(normalization)
    
    return parameters  


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

def getfiles_keyword(direc_name,channel_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if channel_name in i]
    
    def sorted_nicely(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key = alphanum_key)

    imgfiles = sorted_nicely(imgfiles)
    return imgfiles

def getfiles_inverse_keyword(direc_name,channel_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if channel_name not in i]
    
    def sorted_nicely(l):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key = alphanum_key)

    imgfiles = sorted_nicely(imgfiles)
    return imgfiles

def getfiles(direc_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if ('.png'  in i ) or ('.jpg'  in i ) or ('.tif' in i) or ('tiff' in i)]

    return imgfiles

def get_image(file_name):
    if '.tif' in file_name:
        im = np.float32(tiff.imread(file_name))
    else:
        im = np.float32(imread(file_name))
    return im


"""
Preprocessing, training and processing calling functions 
"""

def preprocessing(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Preprocessing #"+str(i+1)+": You need to select an input directory for preprocessing")
        if parameters[1][i].selected==None:
            sys.exit("Preprocessing #"+str(i+1)+": You need to select an output directory for preprocessed data")
    

        preprocess(parameters[0][i].selected, parameters[1][i].selected, parameters[2][i].children[1].value, parameters[3][i].children[1].value, parameters[4][i].children[1].value, parameters[5][i].children[1].value)
        

def training(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[1][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained model")

        model = inceptionV3(n_channels=parameters[2][i].children[1].value, n_features=parameters[3][i].children[1].value, dimx=parameters[4][i].children[1].value, dimy=parameters[5][i].children[1].value)

        if parameters[8][i].children[1].value==True:
            model_name = "InceptionV3_"+str(parameters[2][i].children[1].value)+"ch_"+str(parameters[3][i].children[1].value)+"cl_"+str(parameters[4][i].children[1].value)+"_"+str(parameters[5][i].children[1].value)+"_lr_"+str(parameters[6][i].children[1].value)+"_withDA_"+str(parameters[7][i].children[1].value)+"ep"
        else:
            model_name = "InceptionV3_"+str(parameters[2][i].children[1].value)+"ch_"+str(parameters[3][i].children[1].value)+"cl_"+str(parameters[4][i].children[1].value)+"_"+str(parameters[5][i].children[1].value)+"_lr_"+str(parameters[6][i].children[1].value)+"_withoutDA_"+str(parameters[7][i].children[1].value)+"ep"
            
        train_model_sample(model, parameters[0][i].selected, model_name=model_name, 
                           batch_size=parameters[9][i].children[1].value, n_epoch = parameters[7][i].children[1].value,
                           direc_save=parameters[1][i].selected, lr=parameters[6][i].children[1].value,
                           augmentation = parameters[8][i].children[1].value)
        del model
        
def transfer_learning(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[1][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained model")
        if parameters[2][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select a pretrained model for transfer learning")

        model = inceptionV3(n_channels=parameters[10][i].children[1].value, n_features=parameters[3][i].children[1].value, 
                            dimx=parameters[12][i].children[1].value, dimy=parameters[13][i].children[1].value,
                            weights_path=parameters[3][i].selected)

        if parameters[8][i].children[1].value==True:
            model_name = "InceptionV3_transfer_learning_"+str(parameters[9][i].children[1].value)+"ch_"+str(parameters[11][i].children[1].value)+"cl_"+str(parameters[12][i].children[1].value)+"_"+str(parameters[13][i].children[1].value)+"_last_layer_lr_"+str(parameters[6][i].children[1].value)+"_"+str(parameters[5][i].children[1].value)+"ep_all_network_lr_"+str(parameters[9][i].children[1].value)+"_"+str(parameters[8][i].children[1].value)+"ep_withDA"
        else:
            model_name = "InceptionV3_transfer_learning_"+str(parameters[9][i].children[1].value)+"ch_"+str(parameters[11][i].children[1].value)+"cl_"+str(parameters[12][i].children[1].value)+"_"+str(parameters[13][i].children[1].value)+"_last_layer_lr_"+str(parameters[6][i].children[1].value)+"_"+str(parameters[5][i].children[1].value)+"ep_all_network_lr_"+str(parameters[9][i].children[1].value)+"_"+str(parameters[8][i].children[1].value)+"ep_withoutDA"
        
        # remove last layer and replace with a layer corresponding to the actual number of classes
        model.layers.pop()
        lastLayer = Dense(parameters[11][i].children[1].value, activation='softmax', name='predictions')(model.layers[-1].output)
        newModel = Model(model.layers[0].output,lastLayer)
        del model
        # training last layer
        if parameters[4][i].children[1].value==True:
            for layer in newModel.layers[:307]:
                layer.trainable = False
            train_model_sample(newModel, parameters[0][i].selected, model_name=model_name, 
                               batch_size=parameters[15][i].children[1].value, n_epoch = parameters[5][i].children[1].value,
                               direc_save=parameters[1][i].selected, lr=parameters[6][i].children[1].value,
                               augmentation = parameters[14][i].children[1].value)
        # training last layer
        if parameters[6][i].children[1].value==True:
            for layer in newModel.layers[:307]:
                layer.trainable = True
            train_model_sample(newModel, parameters[0][i].selected, model_name=model_name, 
                               batch_size=parameters[15][i].children[1].value, n_epoch = parameters[8][i].children[1].value,
                               direc_save=parameters[1][i].selected, lr=parameters[9][i].children[1].value,
                               augmentation = parameters[14][i].children[1].value)
        del newModel

def running(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to process your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        print("n_features: ", parameters[5][i].children[1].value, "n_channels: ", parameters[4][i].children[1].value, "dimx: ", parameters[6][i].children[1].value, "dimy: ", parameters[7][i].children[1].value, "weights_path: ", parameters[1][i].selected)
        model = inceptionV3(n_features=parameters[5][i].children[1].value, n_channels=parameters[4][i].children[1].value,
                            dimx=parameters[6][i].children[1].value, dimy=parameters[7][i].children[1].value, 
                            weights_path=parameters[1][i].selected)
        print("data_location: ", parameters[0][i].selected, "output_location: ", parameters[2][i].selected, "bs: ", parameters[8][i].children[1].value, "mask_names: ", parameters[3][i].children[1].value, "normalization: ", parameters[9][i].children[1].value)
        run_models_on_directory(parameters[0][i].selected, parameters[2][i].selected, model, bs=parameters[8][i].children[1].value, maxDim=800, mask_names=parameters[3][i].children[1].value, normalization=parameters[9][i].children[1].value)
        del model
        
"""
Preprocessing data
"""
def preprocess(input_dir, output_dir, nb_classes, window_size_x, window_size_y, normalization):
    direc_name = input_dir
    file_name_save = os.path.join(output_dir, 'data.npz')
    training_direcs = [i for i in os.listdir(direc_name)]
    channel_names = ["image"]
    
    # maximum number of training cases
    max_training_examples = 10000000

    nb_direcs = len(training_direcs)
    nb_channels = len(channel_names)

    # variables for images
    imglist = []
    for direc in training_direcs:
        imglist += os.listdir(os.path.join(direc_name, direc))

    # Load one file to get image sizes
    img_temp = get_image(os.path.join(direc_name,training_direcs[0],imglist[0]))
    image_size_x, image_size_y = img_temp.shape

    # Initialize arrays for the training images and the classes
    channels = np.zeros((nb_direcs, image_size_x, image_size_y, nb_channels), dtype='float32')
    class_mask = np.zeros((nb_direcs, image_size_x, image_size_y, nb_classes), dtype='float32')
        
    # Load training images
    direc_counter = 0
    for direc in training_direcs:
        imglist = os.listdir(os.path.join(direc_name, direc))
        channel_counter = 0

        # Load channels
        for channel in channel_names:
            for img in imglist: 
                if fnmatch.fnmatch(img, r'*' + channel + r'*'):
                    channel_file = img
                    channel_file = os.path.join(direc_name, direc, channel_file)
                    channel_img = get_image(channel_file)
    
                    # Normalize the images
                    if normalization=="nuclei segmentation":
                        channel_img = process_image(channel_img, 2*window_size_x + 1, 2*window_size_y + 1)
                    else:
                        channel_img = process_image_onlyLocalAverageSubtraction(channel_img, 2*window_size_x + 1, 2*window_size_y + 1)

                    channels[direc_counter,:,:,channel_counter] = channel_img
                    channel_counter += 1

        # Load class mask
        for j in range(nb_classes):
            class_name = "feature_" + str(j) + r".*"
            for img in imglist:
                if fnmatch.fnmatch(img, class_name):
                    class_file = os.path.join(direc_name, direc, img)
                    class_img = get_image(class_file)

                    if np.sum(class_img) > 0:
                        class_img /= np.amax(class_img)

                    class_mask[direc_counter,:,:,j] = class_img
       
        direc_counter += 1
    
    
    class_mask_trimmed = class_mask[:,window_size_x+1:-window_size_x-1,window_size_y+1:-window_size_y-1,:] 

    class_rows = []
    class_cols = []
    class_batch = []
    class_label = []

    min_pixel_counter = np.zeros((nb_direcs))
    for j in range(class_mask_trimmed.shape[0]):
        min_pixel_counter[j] = np.Inf
        nb_pixels_class = 0
        for k in range(nb_classes):
            nb_pixels_class = np.sum(class_mask_trimmed[j,:,:,k])
            if nb_pixels_class < min_pixel_counter[j]:
                min_pixel_counter[j] = nb_pixels_class
            
            
            
    for direc in range(channels.shape[0]):

        for k in range(nb_classes):
            class_counter = 0
            class_rows_temp, class_cols_temp = np.where(class_mask[direc,:,:,k] == 1)

            if len(class_rows_temp) > 0:

                # Randomly permute index vector
                non_rand_ind = np.arange(len(class_rows_temp))
                rand_ind = np.random.choice(non_rand_ind, size = len(class_rows_temp), replace = False)

                for i in rand_ind:
                    if class_counter < min_pixel_counter[direc]:
                        if (class_rows_temp[i] - window_size_x > 0) and (class_rows_temp[i] + window_size_x < image_size_x): 
                            if (class_cols_temp[i] - window_size_y > 0) and (class_cols_temp[i] + window_size_y < image_size_y):
                                class_rows += [class_rows_temp[i]]
                                class_cols += [class_cols_temp[i]]
                                class_batch += [direc]
                                class_label += [k]
                                class_counter += 1

    class_rows = np.array(class_rows,dtype = 'int32')
    class_cols = np.array(class_cols,dtype = 'int32')
    class_batch = np.array(class_batch, dtype = 'int32')
    class_label = np.array(class_label, dtype = 'int32')

    # Randomly select training points 
    if len(class_rows) > max_training_examples:
        non_rand_ind = np.arange(len(class_rows))
        rand_ind = np.random.choice(non_rand_ind, size = max_training_examples, replace = False)

        class_rows = class_rows[rand_ind]
        class_cols = class_cols[rand_ind]
        class_batch = class_batch[rand_ind]
        class_label = class_label[rand_ind]

    # Randomize
    non_rand_ind = np.arange(len(class_rows))
    rand_ind = np.random.choice(non_rand_ind, size = len(class_rows), replace = False)

    class_rows = class_rows[rand_ind]
    class_cols = class_cols[rand_ind]
    class_batch = class_batch[rand_ind]
    class_label = class_label[rand_ind]

    np.savez(file_name_save, channels = channels, y = class_label, batch = class_batch, pixels_x = class_rows, pixels_y = class_cols, win_x = window_size_x, win_y = window_size_y)


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
        pAugmentationLevel=5,           # number of augmentations
        pEnableFlipping1=True,          # enable x flipping
        pEnableFlipping2=True,          # enable y flipping
        pEnableRotation90=True,           # enable rotation
        pEnableRotation=True,           # enable rotation
        pMaxRotationDegree=15,
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


def random_sample_generator_centralPixelClassification(img, img_ind, x_coords, y_coords, y_init, batch_size, n_channels, n_classes, win_x, win_y, augmentation = True):

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

            if augmentation:
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
def train_model_sample(model, training_data_file_name, model_name = "", batch_size = 32, n_epoch = 100,
    direc_save = "./trained_classifiers/", lr = 0.01, augmentation = True):

    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")

    file_name_save = os.path.join(direc_save, todays_date + "_" + model_name + ".h5")
    logdir = "logs/scalars/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    optimizer = SGD(lr = lr, decay = 1e-7, momentum = 0.9, nesterov = True)
    lr_sched = rate_scheduler(lr = lr, decay = 0.99)

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

    train_generator = random_sample_generator_centralPixelClassification(train_dict["channels"], train_dict["batch"], train_dict["pixels_x"], train_dict["pixels_y"], train_dict["labels"], batch_size, n_channels, n_classes, train_dict["win_x"], train_dict["win_y"], augmentation)

    # fit the model on the batches generated by datagen.flow()
    loss_history = model.fit_generator(train_generator, 
                                       steps_per_epoch = int(len(train_dict["labels"])/batch_size), 
                                       epochs = n_epoch, validation_data = (X_test,Y_test), 
                                       callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only=True), LearningRateScheduler(lr_sched),tensorboard_callback])

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

def get_image_sizes_keyword(data_location, channel_names):
    img_list_channels = []
    for channel in channel_names:       
        img_list_channels += [getfiles_keyword(data_location, channel)]
    img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

    return img_temp.shape

def get_images_from_directory(data_location):
    img_list_channels = getfiles(data_location)
    img_temp = get_image(os.path.join(data_location, img_list_channels[0]))
    n_channels = len(img_list_channels)
    all_images = []
    image_names = []

    for stack_iteration in range(len(img_list_channels)):
        all_channels = np.zeros((1, img_temp.shape[0],img_temp.shape[1], 1), dtype = 'float32')
        channel_img = get_image(os.path.join(data_location, img_list_channels[stack_iteration]))
        all_channels[0,:,:,0] = channel_img
        all_images += [all_channels]
        image_names.append(os.path.splitext(os.path.basename(img_list_channels[stack_iteration]))[0])

    return all_images, image_names

def get_images_from_directory_keyword(data_location, channel_names):
    img_list_channels = []
    for channel in channel_names:
        img_list_channels += [getfiles_keyword(data_location, channel)]

    img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

    n_channels = len(channel_names)
    all_images = []
    image_names = []

    for stack_iteration in range(len(img_list_channels[0])):
        all_channels = np.zeros((1, img_temp.shape[0],img_temp.shape[1], n_channels), dtype = 'float32')
        for j in range(n_channels):
            channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
            all_channels[0,:,:,j] = channel_img
        all_images += [all_channels]
        image_names.append(os.path.splitext(os.path.basename(img_list_channels[stack_iteration]))[0])

    return all_images, image_names

def get_images_from_directory_keyword_inverse(data_location, channel_names):
    img_list_channels = []
    for channel in channel_names:
        img_list_channels += [getfiles_inverse_keyword(data_location, channel)]

    img_temp = get_image(os.path.join(data_location, img_list_channels[0][0]))

    n_channels = len(channel_names)
    all_images = []
    image_names = []

    for stack_iteration in range(len(img_list_channels[0])):
        all_channels = np.zeros((1, img_temp.shape[0],img_temp.shape[1], n_channels), dtype = 'float32')
        for j in range(n_channels):
            channel_img = get_image(os.path.join(data_location, img_list_channels[j][stack_iteration]))
            all_channels[0,:,:,j] = channel_img
        all_images += [all_channels]
        image_names.append(os.path.splitext(os.path.basename(img_list_channels[stack_iteration]))[0])

    return all_images, image_names

# central pixel based networks
def run_model_pixByPix(img, model, win_x = 30, win_y = 30, std = False, split = True, process = True, bs=32, maxDim=800, normalization = "nuclei segmentation"):

    if normalization == "nuclei segmentation":
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

def run_model_pixByPixOnMasks(img, mask, model, win_x = 30, win_y = 30, bs=32, maxDim=800, normalization = "nuclei segmentation"):

    if normalization == "nuclei segmentation":
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


def run_model_on_directory_pixByPix(data_location, mask_names, output_location, model, win_x = 30, win_y = 30, bs=32, maxDim=800, normalization="nuclei segmentation"):
    n_classes = model.layers[-1].output_shape[-1]
    counter = 0

    if mask_names == 'None':
        image_list, image_names = get_images_from_directory(data_location)
    else:
        image_list, image_names = get_images_from_directory_inverse_keyword(data_location, mask_names)
    processed_image_list = []
    
    if mask_names == "None":
        for img in image_list:
            print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
            processed_image = run_model_pixByPix(img, model, win_x = win_x, win_y = win_y, bs=bs, maxDim=maxDim, normalization = normalization)
            processed_image_list += [processed_image]

            # Save images
            cnnout_name = os.path.join(output_location, image_names[counter] + ".tif")
            tiff.imsave(cnnout_name,processed_image)
            counter += 1

    else:
        mask_list, image_names = get_images_from_directory_keyword(data_location, mask_names)
        for img in image_list:
            print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
            processed_image = run_model_pixByPixOnMasks(img, mask_list[counter], model, win_x = win_x, win_y = win_y, bs=bs, maxDim=maxDim, normalization = normalization)
            processed_image_list += [processed_image]

            # Save images
            cnnout_name = os.path.join(output_location, image_names[counter] + ".tif")
            tiff.imsave(cnnout_name,processed_image)
            counter += 1
    
    return processed_image_list


def run_models_on_directory(data_location, output_location, model, bs=32, maxDim=800, mask_names='None', normalization="nuclei segmentation"):

    # determine the number of channels and classes as well as the imaging field dimensions
    input_shape = model.layers[0].output_shape
    n_channels = input_shape[-1]
    imaging_field_x = int((input_shape[1]-1)/2)
    imaging_field_y = int((input_shape[2]-1)/2)
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    # determine the image size
    image_size_x, image_size_y, nb_chan = get_image_sizes(data_location)
    
    # process images
    cpt = 0
    model_output = []
    processed_image_list=run_model_on_directory_pixByPix(data_location, mask_names, output_location, model, win_x = imaging_field_x, win_y = imaging_field_y, bs=bs, maxDim=maxDim, normalization = normalization)
    model_output += [np.stack(processed_image_list, axis = 0)]

    return model_output

def run_models_on_directory_previous(data_location, mask_names, output_location, model, bs=32, maxDim=800, normalization = 1):

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
    image_size_x, image_size_y, nb_channels = get_image_sizes(data_location)

    # process images
    cpt = 0
    model_output = []
    processed_image_list= run_model_on_directory_pixByPix(data_location, mask_names, output_location, model, win_x = imaging_field_x, win_y = imaging_field_y, bs=bs, maxDim=maxDim, normalization = normalization)
    model_output += [np.stack(processed_image_list, axis = 0)]

    return model_output