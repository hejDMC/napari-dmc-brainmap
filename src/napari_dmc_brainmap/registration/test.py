#%%
from sharpy_track.model.AtlasModel import AtlasModel

atlasModel = AtlasModel('C:\\Users\\xiao\\GitHub\\sharpy_track\\sharpy_track\\')
atlasModel.loadAnnot()
atlasModel.loadVolume()


#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

ap = 750

# slice annotation volume
slice_annot = atlasModel.annot[ap,:,:].astype(np.int32) # convert to int32 for contour detection
slice_vol = atlasModel.vol[ap,:,:] # volume 8 bit

contours,_ = cv2.findContours(slice_annot, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
# create canvas
empty = np.zeros((800,1140),dtype=np.uint8)
# draw contours
outline = cv2.drawContours(empty,contours,-1,color=1)
# convert to bool index
outline_bool = outline.astype(bool) # return this
slice_vol[outline_bool] = 255
plt.imshow(slice_vol,cmap='gray')