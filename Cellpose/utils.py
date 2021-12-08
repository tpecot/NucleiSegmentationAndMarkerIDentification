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

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from skimage.io import imread, imsave
import skimage as sk
import tifffile as tiff
import cv2
  
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

import skimage.segmentation as seg
from cellpose import models, io, utils, dynamics


"""
Interfaces
"""


def training_parameters_cellpose_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    validation_dir = np.zeros([nb_trainings], FileChooser)
    learning_rate = np.zeros([nb_trainings], HBox)
    nb_epochs = np.zeros([nb_trainings], HBox)
    batch_size = np.zeros([nb_trainings], HBox)
    diameter_mean = np.zeros([nb_trainings], HBox)
    train_to_val_ratio = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training directory")
        training_dir[i] = FileChooser('./datasets')
        display(training_dir[i])
        print('\x1b[1m'+"Validation directory")
        validation_dir[i] = FileChooser('./datasets')
        display(validation_dir[i])

        label_layout = Layout(width='180px',height='30px')

        learning_rate[i] = HBox([Label('Learning rate:', layout=label_layout), widgets.FloatText(
            value=0.2, description='', disabled=False)])
        display(learning_rate[i])

        nb_epochs[i] = HBox([Label('Number of epochs:', layout=label_layout), widgets.IntText(
            value=400, description='', disabled=False)])
        display(nb_epochs[i])

        batch_size[i] = HBox([Label('Batch size:', layout=label_layout), widgets.IntText(
            value=8, description='', disabled=False)])
        display(batch_size[i])

        diameter_mean[i] = HBox([Label('Diameter mean (pixels):', layout=label_layout), widgets.IntText(
            value=30, description='', disabled=False)])
        display(diameter_mean[i])

        train_to_val_ratio[i] = HBox([Label('Ratio of training in validation:', layout=label_layout), widgets.BoundedFloatText(
            value=0.2, min=0.01, max=0.99, step=0.01, description='', disabled=False, color='black'
        )])
        display(train_to_val_ratio[i])
        
    parameters.append(training_dir)
    parameters.append(validation_dir)
    parameters.append(learning_rate)
    parameters.append(nb_epochs)
    parameters.append(batch_size)
    parameters.append(diameter_mean)
    parameters.append(train_to_val_ratio)
    
    return parameters  


def running_parameters_cellpose_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_model = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    diameter_mean = np.zeros([nb_trainings], FileChooser)
    prob_th = np.zeros([nb_trainings], HBox)
    flow_th = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser('./datasets')
        display(input_dir[i])
        print('\x1b[1m'+"Input model")
        input_model[i] = FileChooser('./models')
        display(input_model[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./datasets')
        display(output_dir[i])
        
        label_layout = Layout(width='180px',height='30px')

        diameter_mean[i] = HBox([Label('Diameter mean (pixels):', layout=label_layout), widgets.IntText(
            value=30, description='', disabled=False)])
        display(diameter_mean[i])

        prob_th[i] = HBox([Label('Probability threshold:', layout=label_layout), widgets.FloatText(
            value=1., description='', disabled=False)])
        display(prob_th[i])
        
        flow_th[i] = HBox([Label('Flow threshold:', layout=label_layout), widgets.FloatText(
            value=2., description='', disabled=False)])
        display(flow_th[i])

    parameters.append(input_dir)
    parameters.append(input_model)
    parameters.append(output_dir)
    parameters.append(diameter_mean)
    parameters.append(prob_th)
    parameters.append(flow_th)
    
    return parameters  

"""
Pre-processing functions 
"""
    

        
"""
Training and processing calling functions 
"""

def training_cellpose(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        
        train_model_sample_cellpose(parameters[0][i].selected, parameters[1][i].selected,
                                    parameters[2][i].children[1].value, parameters[3][i].children[1].value,
                                    parameters[4][i].children[1].value, parameters[5][i].children[1].value)
        

def running_cellpose(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to process your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        model_path = parameters[1][i].selected
        model = models.CellposeModel(gpu=True, pretrained_model=model_path)
        run_cellpose_on_directory(parameters[0][i].selected, parameters[2][i].selected, model, parameters[3][i].children[1].value, parameters[4][i].children[1].value, parameters[5][i].children[1].value)
        del model

"""
Useful functions 
"""
def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn


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
def get_data_sample_cellpose(training_directory, validation_directory, validation_training_ratio = 0.1):
    
    training_images_directory = os.path.join(training_directory, 'images/')
    training_masks_directory = os.path.join(training_directory, 'masks/')
    images = []
    labels = []
    test_images = []
    test_labels = []

    # adding evaluation data into validation
    if validation_directory is not None:

        imageTrainFileList = [f for f in os.listdir(training_images_directory) if ('.png'  in f ) or ('.tif' in f) or ('tiff' in f)]
        for imageFile in imageTrainFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            if os.path.exists(os.path.join(training_masks_directory, baseName + "_flows.tiff")):
                flowPath = os.path.join(training_masks_directory, baseName + "_flows.tiff")
                current_flows = get_image(flowPath)
                if current_flows.shape[0]>4:
                    updated_current_flows = np.zeros((4, current_flows.shape[0], current_flows.shape[1]), 'float32')
                    for k in range(4):
                        updated_current_flows[k, :, :] = current_flows[:, :, k].astype('float32')
                    current_flows = updated_current_flows
            else:
                if os.path.exists(os.path.join(training_masks_directory, baseName + ".png")):
                    maskPath = os.path.join(training_masks_directory, baseName + ".png")
                elif os.path.exists(os.path.join(training_masks_directory, baseName + ".tif")):
                    maskPath = os.path.join(training_masks_directory, baseName + ".tif")
                elif os.path.exists(os.path.join(training_masks_directory, baseName + ".tiff")):
                    maskPath = os.path.join(training_masks_directory, baseName + ".tiff")
                else:
                    sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
                current_mask_image = get_image(maskPath)
                if len(current_mask_image.shape)>2:
                    updated_current_mask_image = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
                    updated_current_mask_image = current_mask_image[:, :, 0].astype('int32')
                    current_mask_image = updated_current_mask_image
                labels_list = []
                labels_list.append(current_mask_image)
                current_flows = dynamics.labels_to_flows(labels_list)[-1]#, redo_flows = True)[-1]
                tiff.imsave(os.path.join(training_masks_directory, baseName + "_flows.tiff"), current_flows.astype('float32'))
                if current_flows.shape[0]>4:
                    updated_current_flows = np.zeros((4, current_flows.shape[0], current_flows.shape[1]), 'float32')
                    for k in range(4):
                        updated_current_flows[k, :, :] = current_flows[:, :, k]
                    current_flows = updated_current_flows
    
            labels.append(current_flows)
            imagePath = os.path.join(training_images_directory, imageFile)
            current_image = get_image(imagePath)
            images.append(current_image)

        validation_images_directory = os.path.join(validation_directory, 'images/')
        validation_masks_directory = os.path.join(validation_directory, 'masks/')
        
        imageValFileList = [f for f in os.listdir(validation_images_directory) if ('.png'  in f ) or ('.tif' in f) or ('tiff' in f)]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            if os.path.exists(os.path.join(validation_masks_directory, baseName + "_flows.tiff")):
                flowPath = os.path.join(validation_masks_directory, baseName + "_flows.tiff")
                current_flows = get_image(flowPath)
                if current_flows.shape[0]>4:
                    updated_current_flows = np.zeros((4, current_flows.shape[0], current_flows.shape[1]), 'float32')
                    for k in range(4):
                        updated_current_flows[k, :, :] = current_flows[:, :, k].astype('float32')
                    current_flows = updated_current_flows
            else:
                if os.path.exists(os.path.join(validation_masks_directory, baseName + ".png")):
                    maskPath = os.path.join(validation_masks_directory, baseName + ".png")
                elif os.path.exists(os.path.join(validation_masks_directory, baseName + ".tif")):
                    maskPath = os.path.join(validation_masks_directory, baseName + ".tif")
                elif os.path.exists(os.path.join(validation_masks_directory, baseName + ".tiff")):
                    maskPath = os.path.join(validation_masks_directory, baseName + ".tiff")
                else:
                    sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
                current_mask_image = get_image(maskPath)
                if len(current_mask_image.shape)>2:
                    updated_current_mask_image = np.zeros((current_image.shape[0], current_image.shape[1]), 'int32')
                    updated_current_mask_image = current_mask_image[:, :, 0]
                    current_mask_image = updated_current_mask_image.astype('int32')
                labels_list = []
                labels_list.append(current_mask_image)
                current_flows = dynamics.labels_to_flows(labels_list)[-1]#, redo_flows = True)[-1]
                tiff.imsave(os.path.join(validation_masks_directory, baseName + "_flows.tiff"), current_flows.astype('float32'))
                if current_flows.shape[0]>4:
                    updated_current_flows = np.zeros((4, current_flows.shape[0], current_flows.shape[1]), 'float32')
                    for k in range(4):
                        updated_current_flows[k, :, :] = current_flows[:, :, k]
                    current_flows = updated_current_flows
    
            test_labels.append(current_flows)
            imagePath = os.path.join(validation_images_directory, imageFile)
            current_image = get_image(imagePath)
            test_images.append(current_image)
            
    else:
        imageTrainFileList = [f for f in os.listdir(training_images_directory) if ('.png'  in f ) or ('.tif' in f) or ('tiff' in f)]
        for imageFile in imageTrainFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            if os.path.exists(os.path.join(training_masks_directory, baseName + "_flows.tiff")):
                flowPath = os.path.join(training_masks_directory, baseName + "_flows.tiff")
                current_flows = get_image(flowPath)
                if current_flows.shape[0]>4:
                    updated_current_flows = np.zeros((4, current_flows.shape[0], current_flows.shape[1]), 'float32')
                    for k in range(4):
                        updated_current_flows[k, :, :] = current_flows[:, :, k].astype('float32')
                    current_flows = updated_current_flows
            else:
                if os.path.exists(os.path.join(training_masks_directory, baseName + ".png")):
                    maskPath = os.path.join(training_masks_directory, baseName + ".png")
                elif os.path.exists(os.path.join(training_masks_directory, baseName + ".tif")):
                    maskPath = os.path.join(training_masks_directory, baseName + ".tif")
                elif os.path.exists(os.path.join(training_masks_directory, baseName + ".tiff")):
                    maskPath = os.path.join(training_masks_directory, baseName + ".tiff")
                else:
                    sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
                current_mask_image = get_image(maskPath)
                if len(current_mask_image.shape)>2:
                    updated_current_mask_image = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
                    updated_current_mask_image = current_mask_image[:, :, 0].astype('int32')
                    current_mask_image = updated_current_mask_image
                labels_list = []
                labels_list.append(current_mask_image)
                current_flows = dynamics.labels_to_flows(labels_list)[-1]#, redo_flows = True)[-1]
                tiff.imsave(os.path.join(training_masks_directory, baseName + "_flows.tiff"), current_flows.astype('float32'))
                if current_flows.shape[0]>4:
                    updated_current_flows = np.zeros((4, current_flows.shape[0], current_flows.shape[1]), 'float32')
                    for k in range(4):
                        updated_current_flows[k, :, :] = current_flows[:, :, k]
                    current_flows = updated_current_flows
                    
            imagePath = os.path.join(training_images_directory, imageFile)
            current_image = get_image(imagePath)
    
            if random.Random().random() > validation_training_ratio:
                labels.append(current_flows)
                images.append(current_image)
            else:
                test_labels.append(current_flows)
                test_images.append(current_image)


    if len(images) < 1:
        sys.exit("Empty train image list")

    #just to be non-empty
    if len(test_images) < 1:
        test_images += images[len(images)-1]
        test_labels += labels[len(labels)-1]
    
    return images, labels, test_images, test_labels



"""
Training convnets
"""
    
def train_model_sample_cellpose(dataset_training = None,  dataset_validation = None,
                                learning_rate = 0.2, n_epoch = 400, batch_size = 8, 
                                diameter_mean = 30, train_to_val_ratio = 0.2):

    if dataset_training is None:
        sys.exit("The input training dataset needs to be defined")

    model = models.CellposeModel(gpu=True, model_type=None, pretrained_model=None, diam_mean=diameter_mean)
    
    images, labels, test_images, test_labels = get_data_sample_cellpose(dataset_training, dataset_validation, validation_training_ratio = train_to_val_ratio)


    model.train(images, labels, test_data = test_images, test_labels = test_labels, 
            channels = [0], n_epochs = n_epoch, learning_rate = learning_rate, 
            momentum=0.9, weight_decay=0.00001,
            save_path = "./", batch_size = batch_size)
    
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

        
def run_cellpose_on_directory(data_location, output_location, model, diameter, prob_thresh, flow_thresh):

    # process images
    channels = [[0,0]]
    counter = 0
    img_list_files = [getfiles(data_location)]

    image_list = get_images_from_directory(data_location)

    for img in image_list:
        print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
        mask, flow, style = model.eval(img[0, :, :, :], do_3D=False, diameter=diameter, channels=channels, flow_threshold=flow_thresh, cellprob_threshold=prob_thresh, batch_size=1) 
    
        # Save images
        cnnout_name = os.path.join(output_location, os.path.splitext(img_list_files[0][counter])[0] + ".tiff")
        tiff.imsave(cnnout_name, mask)

        counter += 1
