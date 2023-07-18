from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import *
import numpy as np
import json


def angleSlice(MLangle,DVangle,AP):
    # load volume
    vol = np.load('atlas\\template_volume_8bit.npy')
    # calculate from ml and dv angle, the plane of current slice
    ml_shift = int(np.tan(np.deg2rad(MLangle)) * 570)
    dv_shift = int(np.tan(np.deg2rad(DVangle)) * 400)
    # pick up slice
    center = np.array([get_ap(AP),400,570])
    c_right = np.array([get_ap(AP)+ml_shift,400,1139])
    c_top = np.array([get_ap(AP)-dv_shift,0,570]) 
    # calculate plane normal vector
    vec_1 = c_right-center
    vec_2 = c_top-center
    vec_n = np.cross(vec_1,vec_2)
    # calculate ap matrix
    grid,r_grid_x,r_grid_y = calculateImageGrid()
    ap_mat = (-vec_n[1]*(grid[:,:,0]-center[1])-vec_n[2]*(grid[:,:,1]-center[2]))/vec_n[0] + center[0]
    ap_flat = ap_mat.astype(int).ravel()
    # within volume check
    outside_vol = np.argwhere((ap_flat<0)|(ap_flat>1319)) # outside of volume index
    if outside_vol.size == 0: # if outside empty, inside of volume
        # index volume with ap_mat and grid
        slice = vol[ap_mat.astype(int).ravel(),r_grid_y,r_grid_x].reshape(800,1140)
    else: # if not empty, show black image with warning
        slice = np.zeros((800,1140),dtype=np.uint8)
        cv2.putText(slice, "Slice out of volume!", (400,400), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3, cv2.LINE_AA)
    return slice

def calculateImageGrid(): # one time calculation
    dv = np.arange(800)
    ml = np.arange(1140)
    grid_x,grid_y = np.meshgrid(ml,dv)
    r_grid_x = grid_x.ravel()
    r_grid_y = grid_y.ravel()
    grid = np.stack([grid_y,grid_x],axis=2)
    return grid,r_grid_x,r_grid_y