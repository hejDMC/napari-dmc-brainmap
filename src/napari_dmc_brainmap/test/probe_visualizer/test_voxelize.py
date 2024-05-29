#%% test two different functions for voxelizing a probe
import numpy as np
from skspatial.objects import Line, Point, Vector

# new function
def get_voxelized_coord_new(primary_axis_idx, line_object, atlas): # get voxelized line from primary axis, line fit

    z = np.arange(atlas.shape[primary_axis_idx])
    lamb = (z - line_object.point[primary_axis_idx])/line_object.direction[primary_axis_idx]

    y_idx, x_idx = atlas.space.index_pairs[primary_axis_idx]
    x = (line_object.point[x_idx] + lamb * line_object.direction[x_idx]).astype(int)
    y = (line_object.point[y_idx] + lamb * line_object.direction[y_idx]).astype(int)
    # todo this is weird
    if primary_axis_idx == 0:
        a = z
        b = y
        c = x
    elif primary_axis_idx == 1:
        a = y
        b = z
        c = x
    else:
        a = y
        b = x
        c = z
    x[x >= atlas.shape[x_idx]] = atlas.shape[x_idx] - 1 # todo check if necessary?
    y[y >= atlas.shape[y_idx]] = atlas.shape[y_idx] - 1
    return a, b, c  # return points in order of atlas

# old function
def get_voxelized_coord_old(ax_primary,line_object): # get voxelized line from primary axis, line fit
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
    return ap,dv,ml

primary_axis = 1
line_fit = Line(point=Point([356.61904762, 347.85714286, 687.23809524]), direction=Vector([0.0466727 , 0.99859611, 0.02504918]))

# create comparision dataframe
import pandas as pd
df_compare = pd.DataFrame()
ap,dv,ml = get_voxelized_coord_old(primary_axis, line_fit)
df_compare[["ap_old","dv_old","ml_old"]] = pd.DataFrame({"ap_old":ap,"dv_old":dv,"ml_old":ml})
# get atlas
from bg_atlasapi import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_10um")

a,b,c = get_voxelized_coord_new(primary_axis, line_fit, atlas)
df_compare[["ap_new","dv_new","ml_new"]] = pd.DataFrame({"ap_new":a,"dv_new":b,"ml_new":c})
# reoder columns to put old and new together
df_compare = df_compare[["ap_old","ap_new","dv_old","dv_new","ml_old","ml_new"]]
# new and old output identical length and values