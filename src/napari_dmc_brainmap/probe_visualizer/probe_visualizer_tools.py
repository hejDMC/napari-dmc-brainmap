
import pandas as pd
import numpy as np


def get_primary_axis(direction_vector): # get primary axis from direction vector
    direction_comp = np.abs(direction_vector) # get component absolute value
    direction_comp[1] += 1e-10 # 1st default, DV axis
    direction_comp[0] += 1e-11 # 2nd default, AP axis

    primary_axis = direction_comp.argmax() # select biggest component as primary axis
    # axis_name = ['zpixel', 'ypixel', 'xpixel']
    # print('Insertion axis is: ', axis_name[primary_axis])
    return(primary_axis) # 0:AP, 1:DV, 2:ML


def get_voxelized_coord(ax_primary,line_object): # get voxelized line from primary axis, line fit
    if ax_primary == 1: # DV-axis
        dv = np.arange(800) # estimate along DV-axis
        lamb = (dv - line_object.point[1])/line_object.direction[1] # DV = dv_point + lambda * dv_direction, get lambda
        ap = (line_object.point[0] + lamb * line_object.direction[0]).astype(int) # AP = ap_point + lambda * ap_direction, get AP
        ml = (line_object.point[2] + lamb * line_object.direction[2]).astype(int) # ML = ml_point + lambda * ml_direction, get ML

    elif ax_primary == 0: # AP-axis
        ap = np.arange(1320)
        lamb = (ap - line_object.point[0])/line_object.direction[0] # AP = ap_point + lambda * ap_direction, get lambda
        ml = (line_object.point[2] + lamb * line_object.direction[2]).astype(int) # ML = ml_point + lambda * ml_direction, get ML
        dv = (line_object.point[1] + lamb * line_object.direction[1]).astype(int) # DV = dv_point + lambda * dv_direction, get DV

    else: # ML-axis
        ml = np.arange(1140)
        lamb = (ml - line_object.point[2])/line_object.direction[2] # ML = ml_point + lambda * ml_direction, get lambda
        dv = (line_object.point[1] + lamb * line_object.direction[1]).astype(int) # DV = dv_point + lambda * dv_direction, get DV
        ap = (line_object.point[0] + lamb * line_object.direction[0]).astype(int) # AP = ap_point + lambda * ap_direction, get AP

    return ap ,dv, ml