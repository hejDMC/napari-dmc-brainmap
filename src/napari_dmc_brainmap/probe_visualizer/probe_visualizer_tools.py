
import pandas as pd
import numpy as np
import distinctipy
import matplotlib.pyplot as plt

from napari_dmc_brainmap.utils import get_animal_id

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


def get_certainty_list(probe_tract, annot):
    # calculate certainty value
    check_size = 3  # [-3,-2,-1,0,1,2,3] # check neibouring (n*2+1)**3 voxels , only odd numbers here

    d_ap, d_dv, d_ml = np.meshgrid(np.arange(-check_size, check_size + 1, 1),
                                np.arange(-check_size, check_size + 1, 1),
                                np.arange(-check_size, check_size + 1, 1))

    certainty_list = []

    for row in range(len(probe_tract)):
        nAP = probe_tract['Voxel_AP'][row] + d_ap.ravel()
        nDV = probe_tract['Voxel_DV'][row] + d_dv.ravel()
        nML = probe_tract['Voxel_ML'][row] + d_ml.ravel()
        # handle outlier voxels
        outlierAP = np.where((nAP < 0) | (nAP > 1319))[0].tolist()
        outlierDV = np.where((nDV < 0) | (nDV > 799))[0].tolist()
        outlierML = np.where((nML < 0) | (nML > 1139))[0].tolist()
        if len(outlierAP) + len(outlierDV) + len(outlierML) == 0:  # no outlier voxel
            voxel_reduce = 0
        else:  # has out of range voxels
            i_to_remove = np.unique(outlierAP + outlierDV + outlierML)  # get voxel to remove index
            nAP = np.delete(nAP, i_to_remove)  # remove
            nDV = np.delete(nDV, i_to_remove)  # from
            nML = np.delete(nML, i_to_remove)  # index lists
            voxel_reduce = len(i_to_remove)  # reduce demoninator at certainty

        structures_neighbor = annot[nAP, nDV, nML]  # get structure_ids of all neighboring voxels, except outliers
        structure_id = annot[probe_tract['Voxel_AP'][row],
                         probe_tract['Voxel_DV'][row],
                         probe_tract['Voxel_ML'][row]]  # center voxel
        uni, count = np.unique(structures_neighbor, return_counts=True)  # summarize neibouring structures
        try:
            certainty = dict(zip(uni, count))[structure_id] / (
                    (check_size * 2 + 1) ** 3 - voxel_reduce)  # calculate certainty score
            certainty_list.append(certainty)
        except KeyError:
            certainty_list.append(0)

    return certainty_list


def check_probe_insert(probe_insert, linefit, surface_vox):

    # get probe tip coordinate
    # get direction unit vector
    direction_vec = linefit['direction'].values  # line direction vector
    direction_unit = direction_vec / np.linalg.norm(direction_vec)  # scale direction vector to length 1

    # # read probe depth from histology evidence  todo delete this?
    # if probe_insert is None:
    #     print('manipulator readout not provided, using histology.')
    #     df_3Dscatter = pd.read_csv('step2_output_probevoxelcoordinates.csv',
    #                                index_col=0)  # read probe 3D coordinates from step2
    #     scatter_vox = np.array(df_3Dscatter[['AP', 'DV', 'ML']].values)
    #
    #     # calculate scatter projection on fit line
    #     projection_online = np.matmul(
    #         np.expand_dims(np.dot(scatter_vox - linefit['point'].values, direction_unit), 1),
    #         np.expand_dims(direction_unit, 0)) + linefit['point'].values
    #
    #     # calculate distance of projection from surface
    #     dist_to_surface = ((projection_online.T[0] - surface_vox[0]) ** 2 + \
    #                        (projection_online.T[1] - surface_vox[1]) ** 2 + \
    #                        (projection_online.T[2] - surface_vox[2]) ** 2) ** 0.5
    #
    #     furthest_um = np.max(dist_to_surface) * 10  # convert voxel to um, 10um/voxel
    #     probe_insert = int(furthest_um)
    # else:
    #     pass

    return probe_insert, direction_unit


def save_probe_tract_fig(input_path, probe, save_path, probe_tract):


    animal_id = get_animal_id(input_path)

    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(5, 20))
    # plot probe tip
    ax[0].fill([32, 64, 0, 32], [0, 175, 175, 0], 'k', zorder=0)  # bottom layer
    # plot shank 384
    ax[0].fill([0, 64, 64, 0, 0], [175, 175, 4015, 4015, 175], 'b', zorder=1)  # middle layer
    # plot electrodes
    electrode_x = np.tile(np.array([8, 40, 24, 56]), 96)
    electrode_y = np.repeat(np.arange(0, 192 * 20, 20) + 185, 2)
    # mark outside of brain electrodes
    inside_brain_bool = np.repeat(probe_tract['Inside_Brain'].values, 2)
    # skip electrodes outside of brain
    ax[0].scatter(electrode_x[inside_brain_bool], electrode_y[inside_brain_bool], marker='s', s=10, c='yellow',
                  zorder=2)  # top layer
    ax[0].set_xlim(0, 64)
    ax[0].set_ylim(0, 4015)
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_aspect(1)
    ax[0].set_ylabel('Probe Lenth From Tip (um)', fontsize=15)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # plot electrode channels configuration
    text_x = np.tile(np.array([0, 64]), 192)
    for chan, tx in zip(range(384), text_x):
        ax[1].text(tx, electrode_y[chan] - 5, str(chan + 1), fontsize=8)

    ax[1].set_xlim(0, 120)
    ax[1].set_ylim(0, 4015)
    ax[1].set_aspect(1)

    # get electrode brain reigon
    for row in range(len(probe_tract)):
        acro_y = probe_tract.iloc[row, :]['Distance_To_Tip(um)']
        acro_text = probe_tract.iloc[row, :]['Acronym']
        ax[2].text(0, acro_y - 5, acro_text, fontsize=8)

    ax[2].set_xlim(0, 100)
    ax[2].set_ylim(0, 4015)
    ax[2].set_aspect(1)

    ## get certainty value, color code with each brain region
    # get unique regions
    region_unique = np.unique(probe_tract['structure_id'].values)
    # generate visually distinct colors
    colors = distinctipy.get_colors(len(region_unique))
    reg_color_dict = dict(zip(region_unique, colors))

    region_split = np.split(probe_tract['structure_id'].values, np.where(np.diff(probe_tract['structure_id'].values))[0] + 1)

    chan_row_n = 0
    for r in region_split:
        acro_text = probe_tract['Acronym'].values[chan_row_n]
        # fill color
        ax[3].fill_betweenx(probe_tract['Distance_To_Tip(um)'].values[chan_row_n:chan_row_n + len(r)] + 5,
                            probe_tract['Certainty'].values[chan_row_n:chan_row_n + len(r)], color=reg_color_dict[r[0]])
        # add text
        ax[4].text(0, (probe_tract['Distance_To_Tip(um)'].values[chan_row_n] + probe_tract['Distance_To_Tip(um)'].values[
            chan_row_n + len(r) - 1]) / 2 - 5, acro_text)

        chan_row_n += len(r)

    ax[3].set_xlim(0, 1)
    ax[3].set_ylim(0, 4015)
    ax[3].get_yaxis().set_visible(False)
    ax[3].set_xlabel('Certainty', fontsize=8)
    ax[3].tick_params(direction="in", pad=-16)
    # ax[3].set_xticks([])
    ax[3].spines['left'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)

    ax[4].set_ylim(0, 4015)
    ax[4].axis('off')

    # add x labels
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_xlabel('Electrodes', fontsize=8)
    ax[1].set_xticks([])
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)

    ax[2].get_yaxis().set_visible(False)
    ax[2].set_xlabel('Acronym', fontsize=8)
    ax[2].set_xticks([])
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)

    fig.suptitle('Animal: ' + animal_id, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    save_fn = save_path.joinpath(probe + '.svg')
    fig.savefig(save_fn, dpi=400)

