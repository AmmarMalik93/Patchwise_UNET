# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:07:00 2018

@author: ammarmalik
"""
import numpy as np
import matplotlib.pyplot as plt 
def displayimages(pred,orig,subject_no,slice_no):
    end_patch = subject_no*slice_no*4
    start_patch = end_patch-4
    Final_Pred = (pred[start_patch:end_patch,:,:,1]*1)+(pred[start_patch:end_patch,:,:,2]*2)+(pred[start_patch:end_patch,:,:,3]*3)
    Final_Orig = (orig[start_patch:end_patch,:,:,1]*1)+(orig[start_patch:end_patch,:,:,2]*2)+(orig[start_patch:end_patch,:,:,3]*3)
    
    def combine_patches(mat):
        row1 = np.hstack((mat[0,:,:],mat[1,:,:]))
        row2 = np.hstack((mat[2,:,:],mat[3,:,:]))
        combine =  np.concatenate((row1,row2))
        combine = np.flipud(combine[24:232, 41:217])
        return combine
    
    f,(ax1,ax2) = plt.subplots(1, 2, sharey=True)
    
    ax1.imshow(combine_patches(Final_Orig), cmap = 'gray', interpolation = 'bicubic')
 #   ax1.xticks([]), plt.yticks([])
    ax1.set_title('Original')
    
    ax2.imshow(combine_patches(Final_Pred), cmap = 'gray', interpolation = 'bicubic')
 #   ax2.xticks([]), plt.yticks([])
    ax2.set_title('Predicted')