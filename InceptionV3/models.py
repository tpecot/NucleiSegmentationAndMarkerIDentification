# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

'''
Model bank - deep convolutional neural network architectures
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, concatenate, Dense, Activation, Convolution2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import ZeroPadding2D, UpSampling2D
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils

from keras.models import Model

import os
import datetime
import h5py

## Inception networks
def conv2d_bn(x, filters, num_row, num_col, border_mode='same', strides=(1, 1), data_format='channels_last', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Convolution2D(filters=filters, kernel_size=(num_row,num_col), strides=strides, 
                      padding=border_mode, data_format=data_format,name=conv_name)(x)
    if data_format=='channels_last':    
        x = BatchNormalization(axis=-1, name=bn_name)(x)
    else:
        x = BatchNormalization(axis=1, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

  
def inceptionV3(n_features = 3, n_channels = 2, dimx = 33, dimy = 33, reg = 1e-3, init = 'he_normal', weights_path = None):
    input = Input(shape=(dimx,dimy,n_channels))
    
    channel_axis = 3
    format = 'channels_last'
    include_top = True
    
    x = conv2d_bn(input, 32, 3, 3, border_mode='same', data_format=format)
    x = conv2d_bn(x, 64, 3, 3, data_format=format)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), data_format=format)(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid', data_format=format)
    x = conv2d_bn(x, 192, 3, 3, border_mode='same', data_format=format)
    x = MaxPooling2D((3, 3), strides=(2, 2), data_format=format)(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, data_format=format)

    branch5x5 = conv2d_bn(x, 48, 1, 1, data_format=format)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, data_format=format)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1,1), padding='same', data_format=format)(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, data_format=format)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, data_format=format)

    branch5x5 = conv2d_bn(x, 48, 1, 1, data_format=format)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, data_format=format)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=format)(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, data_format=format)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, data_format=format)

    branch5x5 = conv2d_bn(x, 48, 1, 1, data_format=format)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, data_format=format)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=format)(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, data_format=format)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), border_mode='valid', data_format=format)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, data_format=format)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), border_mode='valid', data_format=format)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), data_format=format)(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, data_format=format)

    branch7x7 = conv2d_bn(x, 128, 1, 1, data_format=format)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, data_format=format)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, data_format=format)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, data_format=format)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=format)(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, data_format=format)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, data_format=format)

        branch7x7 = conv2d_bn(x, 160, 1, 1, data_format=format)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, data_format=format)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, data_format=format)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, data_format=format)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, data_format=format)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, data_format=format)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, data_format=format)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, data_format=format)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=format)(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, data_format=format)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, data_format=format)

    branch7x7 = conv2d_bn(x, 192, 1, 1, data_format=format)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, data_format=format)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, data_format=format)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, data_format=format)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, data_format=format)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=format)(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, data_format=format)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, data_format=format)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), border_mode='valid', data_format=format)

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, data_format=format)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, data_format=format)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, data_format=format)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), border_mode='valid', data_format=format)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), data_format=format)(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, data_format=format)

        branch3x3 = conv2d_bn(x, 384, 1, 1, data_format=format)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3, data_format=format)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1, data_format=format)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, data_format=format)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3, data_format=format)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3, data_format=format)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1, data_format=format)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same', data_format=format)(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, data_format=format)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed' + str(9 + i))
    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool', data_format=format)(x)
        x = Dense(n_features, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(data_format=format)(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(data_format=format)(x)

    # Create model.
    model = Model(input, x, name='inception_v3')

    if weights_path != None:
        model.load_weights(weights_path)
        
    return model
