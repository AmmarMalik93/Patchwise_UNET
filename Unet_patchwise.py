# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:38:43 2018

@author: ammarmalik
"""
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout

def Unet_patchwise(img_rows,img_cols):
    num_channels = 1
    
    inputs = Input((img_rows, img_cols,num_channels))
    
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    up4 = (UpSampling2D(size = (2,2))(pool3))
    merge4 = concatenate([conv3,up4], axis = 3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge4)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    
    up5 = (UpSampling2D(size = (2,2))(conv4))
    merge5 = concatenate([conv2,up5], axis = 3)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    
    up6 = (UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv1,up6], axis = 3)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    
    conv7 = Conv2D(4, 1, activation = 'softmax')(conv6)
    model = Model(inputs = inputs, outputs = conv7)
    model.compile(optimizer=SGD(lr=0.001, momentum=0.99), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model