# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:03:49 2018

@author: ammarmalik
"""

import os
import glob
import nibabel as nib
import numpy as np

def extractslices():
    # original window size
    rows = 256 
    cols = 256
    def label2class(label):
        x = np.zeros([rows, cols, 4])
        for i in range(rows):
            for j in range(cols):
                x[i, j, int(label[i][j])] = 1
        return x
    
    current = os.getcwd()
    masks_path = (current+"/"+"Masks"+"/")        
    masks = glob.glob(masks_path+'*_masked_gfc_fseg.img')
    
    image_path = (current+"/"+"Raw"+"/")        
    images = glob.glob(image_path+'*_masked_gfc.img')
    
    subjects = 50
    slices = 48
    maskdatas = np.ndarray((subjects*slices,rows,cols,4), dtype='uint16')
    imagedatas = np.ndarray((subjects*slices,rows,cols,1), dtype='uint16')
    i = 0
    x = 0 
    
    
    for f in masks[0:subjects]:
        x += 1
        Mask = nib.load(f)
        Mask_data = Mask.get_data()
        Mask_data = Mask_data.reshape(Mask_data.shape[0],Mask_data.shape[1],Mask_data.shape[2])
        for j in range(10,152,3): # extract slices starting from 10 and ending at 152 with interval of 3 slices in between
            ROI = Mask_data[:,:,j].T
            ROI = np.pad(ROI,((24,24),(40,40)),'constant', constant_values=(0, 0)) # zero-padding to make the dimensions 256 x 256
            ROI = label2class(ROI) # converting multi-class ground truth segmentation map to multi-channel binary segmentation maps
            maskdatas[i] = ROI
            i += 1
            
    i = 0
    x = 0
    for f in images[0:subjects]:
        x += 1
        image = nib.load(f)
        image_data = image.get_data()
        image_data = image_data.reshape(image_data.shape[0],image_data.shape[1],image_data.shape[2])
        for j in range(10,152,3): # extract slices starting from 10 and ending at 152 with interval of 3 slices in between
            Selected = image_data[:,:,j] 
            ROI = Selected.T        
            ROI = np.pad(ROI,((24,24),(40,40)),'constant', constant_values=(0, 0)) # zero-padding to make the dimensions 256 x 256
            ROI = ROI.reshape(ROI.shape[0], ROI.shape[1], 1)
            imagedatas[i] = ROI
            i += 1
            
    return (imagedatas, maskdatas)