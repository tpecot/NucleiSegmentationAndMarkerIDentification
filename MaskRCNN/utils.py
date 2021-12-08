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

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

import os
from scipy import ndimage
from scipy.misc import bytescale
import threading
from threading import Thread, Lock
import h5py

from skimage.io import imread, imsave
import skimage as sk
import tifffile as tiff

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random

import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

import sys
sys.path.append("Mask_RCNN-2.1")
import mrcnn_model
import mrcnn_utils
sys.path.append("biomagdsb")
import mask_rcnn_additional
import additional_train
import additional_segmentation


"""
Interfaces
"""

def TensorBoard_interface():
    
    print('\x1b[1m'+"Model location")
    classifier_directory = FileChooser('./models')
    display(classifier_directory)
    
    return classifier_directory



def training_parameters_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    validation_dir = np.zeros([nb_trainings], FileChooser)
    input_model = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    heads_training = np.zeros([nb_trainings], HBox)
    nb_epochs_heads = np.zeros([nb_trainings], HBox)
    learning_rate_heads = np.zeros([nb_trainings], HBox)
    all_network_training = np.zeros([nb_trainings], HBox)
    nb_epochs_all = np.zeros([nb_trainings], HBox)
    learning_rate_all = np.zeros([nb_trainings], HBox)
    nb_augmentations = np.zeros([nb_trainings], HBox)
    image_size = np.zeros([nb_trainings], HBox)
    train_to_val_ratio = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training directory")
        training_dir[i] = FileChooser('./datasets')
        display(training_dir[i])
        print('\x1b[1m'+"Validation directory")
        validation_dir[i] = FileChooser('./datasets')
        display(validation_dir[i])
        print('\x1b[1m'+"Input model")
        input_model[i] = FileChooser('./pretrainedModels')
        display(input_model[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./models')
        display(output_dir[i])

        label_layout = Layout(width='250px',height='30px')

        heads_training[i] = HBox([Label('Training heads only first:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(heads_training[i])

        nb_epochs_heads[i] = HBox([Label('Number of epochs for heads training:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_epochs_heads[i])

        learning_rate_heads[i] = HBox([Label('Learning rate for heads training:', layout=label_layout), widgets.FloatText(
            value=0.001, description='', disabled=False)])
        display(learning_rate_heads[i])

        all_network_training[i] = HBox([Label('Training all network:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(all_network_training[i])

        nb_epochs_all[i] = HBox([Label('Number of epochs for all network training:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_epochs_all[i])

        learning_rate_all[i] = HBox([Label('Learning rate for all network training:', layout=label_layout), widgets.FloatText(
            value=0.0005, description='', disabled=False)])
        display(learning_rate_all[i])

        nb_augmentations[i] = HBox([Label('Number of augmentations:', layout=label_layout), widgets.IntText(
            value=100, description='', disabled=False)])
        display(nb_augmentations[i])

        image_size[i] = HBox([Label('Image size as seen by the network:', layout=label_layout), widgets.IntText(
            value=256, description='', disabled=False)])
        display(image_size[i])

        train_to_val_ratio[i] = HBox([Label('Ratio of training in validation:', layout=label_layout), widgets.BoundedFloatText(
            value=0.2, min=0.01, max=0.99, step=0.01, description='', disabled=False, color='black'
        )])
        display(train_to_val_ratio[i])

    parameters.append(training_dir)
    parameters.append(validation_dir)
    parameters.append(input_model)
    parameters.append(output_dir)
    parameters.append(heads_training)
    parameters.append(nb_epochs_heads)
    parameters.append(learning_rate_heads)
    parameters.append(all_network_training)
    parameters.append(nb_epochs_all)
    parameters.append(learning_rate_all)
    parameters.append(nb_augmentations)
    parameters.append(image_size)
    parameters.append(train_to_val_ratio)
    
    return parameters  

def running_parameters_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    image_size = np.zeros([nb_trainings], HBox)
    
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

        label_layout = Layout(width='215px',height='30px')

        image_size[i] = HBox([Label('Image size as seen by the network:', layout=label_layout), widgets.IntText(
            value=512, description='', disabled=False)])
        display(image_size[i])

    parameters.append(input_dir)
    parameters.append(input_classifier)
    parameters.append(output_dir)
    parameters.append(image_size)
    
    return parameters  


"""
Training and processing calling functions 
"""

def training(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[2][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input model for transfer learning")
        if parameters[3][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained model")
    
        model_name = "MaskRCNN_"+str(parameters[6][i].children[1].value)+"_lr_heads_"+str(parameters[5][i].children[1].value)+"ep_heads_"+str(parameters[9][i].children[1].value)+"_lr_all_"+str(parameters[8][i].children[1].value)+"ep_all_"+str(parameters[10][i].children[1].value)+"DA"

        if parameters[4][i].children[1].value==True and parameters[7][i].children[1].value==True:
            epoch_groups = [{"layers":"heads","epochs":str(parameters[5][i].children[1].value),"learning_rate":str(parameters[6][i].children[1].value)}, {"layers":"all","epochs":str(parameters[8][i].children[1].value),"learning_rate":str(parameters[9][i].children[1].value)}]
        else:
            if parameters[4][i].children[1].value==True:
                epoch_groups = [{"layers":"heads","epochs":str(parameters[5][i].children[1].value),"learning_rate":str(parameters[6][i].children[1].value)}]
            elif parameters[7][i].children[1].value==True:
                epoch_groups = [{"layers":"all","epochs":str(parameters[8][i].children[1].value),"learning_rate":str(parameters[9][i].children[1].value)}]
            else:
                sys.exit("Training #"+str(i+1)+": You need to train heads, all network or both")

        model = additional_train.MaskTrain(parameters[0][i].selected, parameters[1][i].selected, parameters[2][i].selected, parameters[3][i].selected, model_name, epoch_groups, parameters[10][i].children[1].value, 0, parameters[12][i].children[1].value, True, 0.5, 0.6, parameters[11][i].children[1].value)
        model.Train()
        
        

def running(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to run your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        # last parameter for memory
        # if crashes down, downsize it
        model = additional_segmentation.Segmentation(parameters[1][i].selected, 0.5, 0.35, 2000)
        model.Run(parameters[0][i].selected, parameters[2][i].selected, [parameters[3][i].children[1].value], 512, 512)
        del model
    