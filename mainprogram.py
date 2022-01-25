# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:29:38 2018

@author: ammarmalik
"""
from extractslices import extractslices
from extractpatches import extractpatches
from Unet_patchwise import Unet_patchwise
from keras.callbacks import EarlyStopping
from displayresults import displayresults
from displayimages import displayimages
(imagedatas, maskdatas) = extractslices()

windowsize_r = 128
windowsize_c = 128
num_patch = 4

(train_img,train_mask,test_img,test_mask) = extractpatches(imagedatas, maskdatas, windowsize_r,windowsize_c,num_patch)

model = Unet_patchwise(windowsize_r,windowsize_c)

es = EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.001, restore_best_weights=True)
model.fit(train_img, train_mask, batch_size=1, epochs=2, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=[es])


masks_pred = model.predict(test_img, batch_size=1, verbose=1)
gm_dsc,gm_jc,wm_dsc,wm_jc,csf_dsc,csf_jc = displayresults(masks_pred,test_mask)

print("Gray Matter:",gm_dsc,gm_jc)
print("White Matter:",wm_dsc,wm_jc)
print("CSF:",csf_dsc,csf_jc)

subject_no = 1
slice_no = 10

displayimages(masks_pred,test_mask,subject_no,slice_no)



#print(train_img.shape)
#print(train_mask.shape)
#print(test_img.shape)
#print(test_mask.shape)