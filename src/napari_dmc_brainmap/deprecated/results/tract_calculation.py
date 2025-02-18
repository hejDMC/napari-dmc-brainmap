import numpy as np
import pandas as pd
import distinctipy
import matplotlib.pyplot as plt
from skspatial.objects import Line, Points # scikit-spatial package: https://scikit-spatial.readthedocs.io/en/stable/
from napari_dmc_brainmap.utils import get_animal_id, get_info, get_bregma, coord_mm_transform

def get_primary_axis_idx(direction_vector): # get primary axis from direction vector
    direction_comp = np.abs(direction_vector) # get component absolute value
    direction_comp[1] += 1e-10 # 1st default, DV axis
    direction_comp[0] += 1e-11 # 2nd default, AP axis

    primary_axis_idx = direction_comp.argmax() # select biggest component as primary axis
    # axis_name = ['zpixel', 'ypixel', 'xpixel']
    # print('Insertion axis is: ', axis_name[primary_axis])
    return primary_axis_idx  # 0:a, 1:b, 2:c


def get_voxelized_coord(primary_axis_idx, line_object, atlas): # get voxelized line from primary axis, line fit

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


def get_certainty_list(probe_tract, annot, col_names):
    # calculate certainty value
    check_size = 3  # [-3,-2,-1,0,1,2,3] # check neibouring (n*2+1)**3 voxels , only odd numbers here

    d_a, d_b, d_c = np.meshgrid(np.arange(-check_size, check_size + 1, 1),
                                np.arange(-check_size, check_size + 1, 1),
                                np.arange(-check_size, check_size + 1, 1))

    certainty_list = []

    for row in range(len(probe_tract)):
        nA = probe_tract[col_names[0]][row] + d_a.ravel()
        nB = probe_tract[col_names[1]][row] + d_b.ravel()
        nC = probe_tract[col_names[2]][row] + d_c.ravel()
        # handle outlier voxels
        outlierA = np.where((nA < 0) | (nA > (annot.shape[0]-1)))[0].tolist()
        outlierB = np.where((nB < 0) | (nB > (annot.shape[1]-1)))[0].tolist()
        outlierC = np.where((nC < 0) | (nC > (annot.shape[2]-1)))[0].tolist()
        if len(outlierA) + len(outlierB) + len(outlierC) == 0:  # no outlier voxel
            voxel_reduce = 0
        else:  # has out of range voxels
            i_to_remove = np.unique(outlierA + outlierB + outlierC)  # get voxel to remove index
            nA = np.delete(nA, i_to_remove)  # remove
            nB = np.delete(nB, i_to_remove)  # from
            nC = np.delete(nC, i_to_remove)  # index lists
            voxel_reduce = len(i_to_remove)  # reduce demoninator at certainty

        structures_neighbor = annot[nA, nB, nC]  # get structure_ids of all neighboring voxels, except outliers
        structure_id = annot[probe_tract[col_names[0]][row],
                         probe_tract[col_names[1]][row],
                         probe_tract[col_names[2]][row]]  # center voxel
        uni, count = np.unique(structures_neighbor, return_counts=True)  # summarize neibouring structures
        try:
            certainty = dict(zip(uni, count))[structure_id] / (
                    (check_size * 2 + 1) ** 3 - voxel_reduce)  # calculate certainty score
            certainty_list.append(certainty)
        except KeyError:
            certainty_list.append(0)

    return certainty_list


def estimate_confidence(v_coords, atlas_resolution_um, annot):
    # calculate r<=10 sphere
    cube_10 = np.array(np.meshgrid(np.arange(-10,11,1),
                        np.arange(-10,11,1),
                        np.arange(-10,11,1))).T.reshape(9261,3)
    d = np.sqrt((cube_10 ** 2).sum(axis=1))
    sphere_10 = cube_10[d<=10] # filter with r<=10, 4169 voxels

    confidence_list = []
    for _,row in v_coords.iterrows():
        c1,c2,c3 = row.values
        current_id = annot[c1,c2,c3] # electrode structure_id
        # restrict view to r=10 voxels sphere space
        within_sphere = np.tile(np.array([c1,c2,c3]),(4169,1)) + sphere_10
        sphere_struct = annot[within_sphere.T[0],within_sphere.T[1],within_sphere.T[2]]
        struct_else = (sphere_struct != current_id)
        if np.sum(struct_else) == 0:
            confidence_list.append(10*atlas_resolution_um)
        else:
            confidence_list.append(np.sqrt((((within_sphere[struct_else] - np.tile(np.array([c1,c2,c3]),(np.sum(struct_else),1)))) ** 2).sum(axis=1)).min() * atlas_resolution_um)
    confidence_list = np.array(confidence_list,dtype=np.uint8)
    return confidence_list


def check_probe_insert(probe_df, probe_insert, linefit, surface_vox, resolution,ax_primary):

    # get probe tip coordinate
    # get direction unit vector
    direction_vec = linefit['direction'].values  # line direction vector
    direction_unit = direction_vec / np.linalg.norm(direction_vec)  # scale direction vector to length 1
    # Rules to flip direction unit
    if ax_primary == 1: # DV axis is primary axis
        if direction_unit[1] < 0:
            direction_unit = -direction_unit
        else: # future add more here
            pass
    else:
        pass

    # read probe depth from histology evidence  todo delete this?
    if not probe_insert:
        print('manipulator readout not provided, using histology.')
        scatter_vox = np.array(probe_df[['a_coord', 'b_coord', 'c_coord']].values)

        # calculate scatter projection on fit line
        projection_online = np.matmul(
            np.expand_dims(np.dot(scatter_vox - linefit['point'].values, direction_unit), 1),
            np.expand_dims(direction_unit, 0)) + linefit['point'].values

        # calculate distance of projection from surface
        dist_to_surface = ((projection_online.T[0] - surface_vox[0]) ** 2 +
                           (projection_online.T[1] - surface_vox[1]) ** 2 +
                           (projection_online.T[2] - surface_vox[2]) ** 2) ** 0.5

        furthest_um = np.max(dist_to_surface) * resolution  # convert voxel to um, 10um/voxel
        probe_insert = int(furthest_um)
    else:
        pass

    return probe_insert, direction_unit


def load_probe_data(results_dir, probe, atlas):

    data_dir = results_dir.joinpath(probe)
    data_fn = list(data_dir.glob('*csv'))[0]
    probe_df = pd.read_csv(data_fn)
    name_dict = {
        'ap': 'ap_coords',
        'si': 'dv_coords',
        'rl': 'ml_coords'
    }
    abc_list = ['a_coord', 'b_coord', 'c_coord']
    # append columns (a,b,c) in case atlas ordering is different that ap, dv, ml (e.g. dv, ml, ap)
    for atlas_ax, abc_name in zip(atlas.space.axes_description, abc_list):
        probe_df[abc_name] = probe_df[name_dict[atlas_ax]].copy()
    print(probe_df)
    return probe_df


def get_linefit3d(probe_df, atlas):


    points = Points(probe_df[['a_coord', 'b_coord', 'c_coord']].values)
    line = Line.best_fit(points)  # fit 3D line
    linefit = pd.DataFrame()
    linefit['point'] = line.point  # add point coordinates to dataframe
    linefit['direction'] = line.direction  # add direction vector coordinates to dataframe
    primary_axis_idx = get_primary_axis_idx(line.direction)  # get primary axis index: 0:a, 1:b, 2:c
    voxel_line = np.array(get_voxelized_coord(primary_axis_idx, line, atlas)).T  # voxelize

    linevox = pd.DataFrame(voxel_line, columns=['a_coord', 'b_coord', 'c_coord'])  # add to dataframe
    return linefit, linevox, primary_axis_idx


def get_probe_tract(input_path, atlas, seg_type, ax_primary, probe_df, probe, probe_insert, linefit, linevox):

    # find brain surface
    annot = atlas.annotation
    bregma = get_bregma(atlas.atlas_name)
    structure_id_list = annot[linevox['a_coord'], linevox['b_coord'], linevox['c_coord']]
    structure_split = np.split(structure_id_list, np.where(np.diff(structure_id_list))[0] + 1)
    surface_index = len(structure_split[0])  # get index of first non-root structure
    surface_vox = linevox.iloc[surface_index, :].values  # get brain surface voxel coordinates
    probe_insert, direction_unit = check_probe_insert(probe_df, probe_insert, linefit, surface_vox,
                                                      atlas.resolution[ax_primary],ax_primary)
    # create probe information dataframe
    probe_tract = pd.DataFrame()
    # create electrode left and right channel columns
    # by default only process "bank 0", unless changed here by user
    bank = 0 # specify recording bank here, if other than 0
    if bank == 0:
        l_chan = np.arange(1,384,2)
        r_chan = np.arange(2,385,2)
        dtt_offset = 0
    elif bank == 1:
        l_chan = np.arange(385,768,2)
        r_chan = np.arange(386,769,2)
        dtt_offset = 3840
    elif bank == 2:
        l_chan = np.arange(769,1152,2)
        r_chan = np.arange(770,1153,2)
        dtt_offset = 7680
    else:
        print('Invalid bank number, using bank 0')
        l_chan = np.arange(1,384,2)
        r_chan = np.arange(2,385,2)
        dtt_offset = 0


    probe_tract['channel_l'] = l_chan
    probe_tract['channel_r'] = r_chan
    probe_tract['distance_to_tip(um)'] = np.arange(192)*20 + 175 + dtt_offset #
    probe_tract['depth(um)'] = probe_insert - probe_tract['distance_to_tip(um)']
    probe_tract['inside_brain'] = (probe_tract['depth(um)'] >= -5)
    name_dict = {
        'ap': 'ap',
        'si': 'dv',
        'rl': 'ml'
    }
    col_names = []
    col_names.extend([name_dict[n] + '_coords' for n in atlas.space.axes_description])
    probe_tract[col_names] = pd.DataFrame(np.round(
        np.dot(
            np.expand_dims(
                (probe_insert - probe_tract['distance_to_tip(um)'].values) / 10, axis=1),
            # manipulator_readout - [distance_to_tip] = distance_from_surface
            np.expand_dims(direction_unit, axis=0))
        + surface_vox).astype(int))
    probe_tract[[name_dict[n] + '_mm' for n in atlas.space.axes_description]] = \
        probe_tract.apply(lambda row: pd.Series(coord_mm_transform([row[col_names[0]],
                                                                    row[col_names[1]],
                                                                    row[col_names[2]]],
                                                                   bregma, atlas.space.resolution)), axis=1)

    probe_tract['structure_id'] = annot[probe_tract[col_names[0]], probe_tract[col_names[1]], probe_tract[col_names[2]]]

    # read df_tree
    df_tree = atlas.structures

    probe_tract['acronym'] = [df_tree.data[i]['acronym'] if i > 0 else 'root' for i in probe_tract['structure_id']]
    probe_tract['name'] = [df_tree.data[i]['name'] if i > 0 else 'root' for i in probe_tract['structure_id']]

    #certainty_list = _get_certainty_list(probe_tract, annot, col_names)
    probe_tract['distance_to_nearest_structure(um)'] = estimate_confidence(v_coords=probe_tract[[col_names[0],
                                                                           col_names[1],
                                                                           col_names[2]]],
                                                    atlas_resolution_um=atlas.resolution[0], # todo: atlases with different resolution?
                                                    annot=annot)
    if seg_type == "neuropixels_probe":
        save_probe_tract_fig(input_path, seg_type, probe, probe_tract, bank)
    else:
        probe_tract['depth(um)'] += probe_tract['distance_to_tip(um)']
        probe_tract = probe_tract.drop(columns=['channel_l', 'channel_r', 'distance_to_tip(um)'])
    return probe_tract, col_names

def save_probe_tract_fig(input_path, seg_type, probe, probe_tract, bank):

    animal_id = get_animal_id(input_path)
    results_dir = get_info(input_path, 'results', seg_type=seg_type, only_dir=True)

    df_plot = probe_tract.copy()
    fig, ax = plt.subplots(ncols=5, nrows=1, figsize=(5, 20))
    # if recording from bank 0, show probe tip, otherwise, hide tip
    if bank == 0:
        # plot probe tip
        ax[0].fill([32, 64, 0, 32], [0, 175, 175, 0], 'k', zorder=0)  # bottom layer
    else:
        pass
    # plot shank 384
    ax[0].fill([0, 64, 64, 0, 0], [175, 175, 4015, 4015, 175], 'b', zorder=1)  # middle layer
    # plot electrodes
    electrode_x = np.tile(np.array([8, 40, 24, 56]), 96)
    electrode_y = np.repeat(np.arange(0, 192 * 20, 20) + 185, 2)
    # skip electrodes outside of brain
    ax[0].scatter(electrode_x, electrode_y, marker='s', s=10, c='yellow',
                  zorder=2)  # top layer
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_aspect(1)
    ax[0].set_ylabel('depth (um)', fontsize=15)
    yticklabels = (np.arange(np.ceil(df_plot['depth(um)'].max()/1000)) * 1000).astype(int)
    if df_plot["depth(um)"].min() > 5:
        yticklabels = np.concatenate(([int(df_plot["depth(um)"].min())], yticklabels))
        overlapping_tick = None
        overlapping_tick_idx = 1
        for tl in yticklabels[1:]:
            if np.abs(tl - yticklabels[0]) < 50:
                overlapping_tick = tl
                break
            overlapping_tick_idx += 1

        if overlapping_tick is not None:
            yticklabels = yticklabels.tolist()
            del yticklabels[overlapping_tick_idx]
            yticklabels = np.array(yticklabels)

    yticks = (df_plot["distance_to_tip(um)"] + df_plot["depth(um)"]).values[0] - yticklabels + 15
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticklabels)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_xlim(0, 64)
    ax[0].set_ylim(0, 4015)
    # plot electrode channels configuration
    text_x = np.tile(np.array([0, 64]), 192)
    for chan, tx in zip(range(384), text_x):
        ax[1].text(tx, electrode_y[chan] - 5, str(chan + 1), fontsize=8)

    ax[1].set_xlim(0, 120)
    ax[1].set_ylim(0, 4015)
    ax[1].set_aspect(1)

    # get electrode brain reigon
    for row in range(len(df_plot)):
        acro_y = df_plot.iloc[row, :]['distance_to_tip(um)']
        acro_text = df_plot.iloc[row, :]['acronym']
        ax[2].text(0, acro_y + 5, acro_text, fontsize=8)

    ax[2].set_xlim(0, 100)
    ax[2].set_ylim(0, 4015)
    ax[2].set_aspect(1)

    # get certainty value, color code with each brain region
    # get unique regions
    region_unique = np.unique(df_plot['structure_id'].values)
    # generate visually distinct colors
    colors = distinctipy.get_colors(len(region_unique))
    reg_color_dict = dict(zip(region_unique, colors))

    region_split = np.split(df_plot['structure_id'].values, np.where(np.diff(df_plot['structure_id'].values))[0] + 1)

    chan_row_n = 0
    for r in region_split:
        acro_text = df_plot['acronym'].values[chan_row_n]
        # fill color
        ax[3].fill_betweenx(df_plot['distance_to_tip(um)'].values[chan_row_n:chan_row_n + len(r)] + 15,
                            df_plot['distance_to_nearest_structure(um)'].values[chan_row_n:chan_row_n + len(r)], color=reg_color_dict[r[0]])
        # add text
        ax[4].text(0, (df_plot['distance_to_tip(um)'].values[chan_row_n] + df_plot['distance_to_tip(um)'].values[
            chan_row_n + len(r) - 1]) / 2 - 5, acro_text)

        chan_row_n += len(r)

    ax[3].set_xlim(0, 100)
    ax[3].set_ylim(0, 4015)
    ax[3].get_yaxis().set_visible(False)
    ax[3].set_xlabel('Confidence(um)', fontsize=8)
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
    save_fn = results_dir.joinpath(probe + '.svg')
    fig.savefig(save_fn, dpi=400)