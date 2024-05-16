import pandas as pd
import numpy as np
import json
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from natsort import natsorted
from napari_dmc_brainmap.utils import get_info, clean_results_df, get_bregma


def get_ipsi_contra(df):
    '''
    Function to add a column specifying if cells if ipsi or contralateral to injection side
    ml_mm values of <0 are on the 'left' hemisphere, >0 are on the 'right hemisphere
    :param df: dataframe with results for animal, not the merged across animals
    :return:
    '''

    df['ipsi_contra'] = ['ipsi'] * len(df)  # add a column defaulting to 'ipsi'
    # change values to contra with respect to the location of the injection side
    if df['injection_side'][0] == 'left':
        df.loc[(df['ml_mm'] < 0), 'ipsi_contra'] = 'contra'
    elif df['injection_side'][0] == 'right':
        df.loc[(df['ml_mm'] > 0), 'ipsi_contra'] = 'contra'
    return df


def get_tgt_data_only(df, atlas, tgt_list):

    tgt_only_data = pd.DataFrame()
    for tgt in tgt_list:
        tgt_list_childs = atlas.get_structure_descendants(tgt)
        if tgt_list_childs:
            dummy_df = df[df['acronym'].isin(tgt_list_childs)]
        else:
            dummy_df = df[df['acronym'].isin([tgt])]
        dummy_df['tgt_name'] = [tgt] * len(dummy_df)
        tgt_only_data = pd.concat([tgt_only_data, dummy_df])

    return tgt_only_data

def resort_df(tgt_data_to_plot, tgt_list, index_sort=False):
    # function to resort brain areas from alphabetic to tgt_list sorting
    # create list of len brain areas
    if not index_sort:
        sort_list = tgt_list * len(tgt_data_to_plot['animal_id'].unique())  # add to list for each animal
        sort_index = dict(zip(sort_list, range(len(sort_list))))
        tgt_data_to_plot['tgt_name_sort'] = tgt_data_to_plot['tgt_name'].map(sort_index)
    else:
        sort_list = tgt_list
        sort_index = dict(zip(sort_list, range(len(sort_list))))
        tgt_data_to_plot['tgt_name_sort'] = tgt_data_to_plot.index.map(sort_index)
    tgt_data_to_plot = tgt_data_to_plot.sort_values(['tgt_name_sort'])
    tgt_data_to_plot.drop('tgt_name_sort', axis=1, inplace=True)

    return tgt_data_to_plot

def get_ipsi_contra(df):
    '''
    Function to add a column specifying if cells if ipsi or contralateral to injection side
    ML_location values of <0 are on the 'left' hemisphere, >0 are on the 'right hemisphere
    :param df: dataframe with results for animal, not the merged across animals
    :return:
    '''

    df['ipsi_contra'] = ['ipsi'] * len(df)  # add a column defaulting to 'ipsi'
    # change values to contra with respect to the location of the injection side
    if df['injection_side'][0] == 'left':
        df.loc[(df['ml_mm'] < 0), 'ipsi_contra'] = 'contra'
    elif df['injection_side'][0] == 'right':
        df.loc[(df['ml_mm'] > 0), 'ipsi_contra'] = 'contra'
    return df

def load_data(input_path, atlas, animal_list, channels, data_type='cells'):


    #  loop over animal_ids
    results_data_merged = pd.DataFrame()  # initialize merged dataframe
    for animal_id in animal_list:
        # for animal_idx, animal_id in enumerate(animal_list):
        if data_type == "optic_fiber" or data_type == "neuropixels_probe":
            seg_super_dir = get_info(input_path.joinpath(animal_id), 'results', seg_type=data_type, only_dir=True)
            channels = natsorted([f.parts[-1] for f in seg_super_dir.iterdir() if f.is_dir()])
        for channel in channels:
            results_dir = get_info(input_path.joinpath(animal_id), 'results', seg_type=data_type, channel=channel,
                                    only_dir=True)
            results_file = results_dir.joinpath(animal_id + '_' + data_type + '.csv')

            if results_file.exists():
                results_data = pd.read_csv(results_file)  # load the data
                results_data['ml_mm'] *= (-1)  # so that negative values are left hemisphere
                results_data['animal_id'] = [animal_id] * len(
                    results_data)  # add the animal_id as a column for later identification
                if (data_type == "optic_fiber" or data_type == "neuropixels_probe") and len(animal_list) > 1:
                    results_data['channel'] = [animal_id + '_' + channel] * len(results_data)
                else:
                    results_data['channel'] = [channel] * len(results_data)
                # add the injection hemisphere stored in params.json file
                params_file = input_path.joinpath(animal_id, 'params.json')  # directory of params.json file
                with open(params_file) as fn:  # load the file
                    params_data = json.load(fn)
                try:
                    injection_side = params_data['general']['injection_side']  # add the injection_side as a column
                except KeyError:
                    # injection_side = input("no injection side specified in params.json file for " + animal_id +
                    #                        ", please enter manually: ")
                    print("WARNING: no injection side specified in params files, defaulting to right hemisphere")
                    injection_side = 'right'

                try:
                    genotype = params_data['general']['genotype']
                except KeyError:
                    # print("warning, no genotype specified for " + animal_id +
                    #       " this could lead to problems down the line, "
                    #       "use the create params.json function to enter genotype")
                    genotype = 0
                try:
                    group = params_data['general']['group']
                except KeyError:
                    # print(
                    #     "warning, no experimental group specified for " + animal_id +
                    #     " this could lead to problems down the line, "
                    #     "use the create params.json function to enter experimental group")
                    group = 0

                results_data['injection_side'] = [injection_side] * len(results_data)
                results_data = get_ipsi_contra(results_data)
                results_data['genotype'] = [genotype] * len(results_data)
                results_data['group'] = [group] * len(results_data)
                # add if the location of a cell is ipsi or contralateral to the injection side
                results_data = get_ipsi_contra(results_data)
                results_data_merged = pd.concat([results_data_merged, results_data])
        print("loaded data from " + animal_id)
        results_data_merged = clean_results_df(results_data_merged, atlas)
        results_data_merged = results_data_merged.reset_index(drop=True)
    return results_data_merged

def coord_mm_transform(df, bregma, to_coord = True):  # todo delete this and use only the one in utils?
    """
    Function to calculate atlas coordinates into mm and vice versa
    Inserted df needs to have specified columns
    """

    if to_coord:
        df['ap_mm'] = -(df['ap_mm'] / 0.01 - bregma[0]).astype(int)
        df['dv_mm'] = -(df['dv_mm'] / 0.01).astype(int)
        df['ml_mm'] = (df['ml_mm'] / 0.01 + bregma[2]).astype(int)
    elif not to_coord:
        df['ap_mm'] = (-df['ap_mm'] + bregma[0])*0.01
        df['dv_mm'] = -(df['dv_mm'] * 0.01)
        df['ml_mm'] = (df['ml_mm'] - bregma[2]) * 0.01
    return df


def match_lists(list1, list2, item):
    if len(list1) == len(list2):
        return list1, list2
    elif len(list1) > len(list2):
        diff = len(list1) - len(list2)
        if item == 'color':
            list2.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
        elif item == 'transparency':
            list2.append(list2[-1])
        return list1, list2
    elif len(list1) < len(list2):
        list2 = list2[:len(list1)]
        return list1, list2


def brain_region_color(plotting_params, atlas):
    color_dict_regions = {}
    brain_areas = plotting_params['brain_areas']
    brain_areas_color = plotting_params['brain_areas_color']
    if brain_areas_color:
        if 'ATLAS' in brain_areas_color:
            brain_areas_color = []
            for b in brain_areas:
                b_acronym = atlas.structures.acronym_to_id_map[b]
                brain_areas_color.append(tuple([c / 255 for c in atlas.structures[b_acronym]['rgb_triplet']]))
    brain_areas_transparency = plotting_params['brain_areas_transparency']
    if brain_areas_transparency:
        brain_areas_transparency = [int(b) for b in brain_areas_transparency]
    else:
        brain_areas_transparency = [255]
    brain_areas, brain_areas_color = match_lists(brain_areas, brain_areas_color, 'color')
    brain_areas, brain_areas_transparency = match_lists(brain_areas, brain_areas_transparency, 'transparency')

    return brain_areas, brain_areas_color, brain_areas_transparency


def plot_brain_schematic(atlas, slice_idx, orient_idx, plotting_params, unilateral_target=False, transparent=True):
    """
    # todo orientation for plot
    Function to plot brain schematics as colored plots

    :param annot_section: 2d array with brain section
    :param structure_tree:
    :param target_region_list: LIST of target brain regions to plot  # todo this also for abbr. not only names
    :param target_color_list: LIST of colors for target brain regions
    :param target_transparency: LIST of transparency values for target brain regions
    :param unilateral_target: BOOLEAN if target should only be plotted on one hemisphere -- TO BE IMPLEMENTED
    :param transparent: BOOLEAN for setting white pixels to transparent (e.g. plotting on black background)
    :return: annot_section in RGBA values on x-y coordintaes for plotting
    """

    if orient_idx == 0:
        annot_section = atlas.annotation[slice_idx, :, :].copy()
    elif orient_idx == 1:
        annot_section = atlas.annotation[:, slice_idx, :].copy()
    else:
        annot_section = atlas.annotation[:, :, slice_idx].copy()
    if plotting_params['plot_outline']:
        # extract contours of brain areas
        # annot_section_u8 = annot_section.astype(np.uint8)
        # annot_section_contours = cv2.Canny(annot_section_u8, 0, 255)
        # annot_section_contours[annot_section_contours > 0] = 1
        contour = plt.contour(annot_section, levels=np.unique(annot_section), colors=['gray'],
                              linewidths=0.2)
        plt.close()
        contour_lines = []
        for collection in contour.collections:
            paths = collection.get_paths()
            for path in paths:
                contour_lines.append(path.vertices)
        annot_section_contours = np.zeros_like(annot_section)
        for line in contour_lines:
            line = np.round(line).astype(int)
            annot_section_contours[line[:, 1], line[:, 0]] = 1


        cmap_contours = ['white', 'gainsboro']
        cmap_contours = np.array([[int(x * 255) for x in list(mcolors.to_rgba(c))] for c in cmap_contours])
        cmap_contours[0][-1] = 0
        annot_section_contours = cmap_contours[annot_section_contours]
    else:
        annot_section_contours = np.array(False)

    annot_section[annot_section > 0] = 1  # set all brain areas to 1

    cmap_brain = ['white', 'linen', 'lightgray',
                  'lightcyan']  # colormap for the brain outline (white: empty space,
                                # linen=brain, lightgray=root, lightcyan=ventricles)

    if plotting_params['brain_areas']:  # if target region list exists, check if len of tgt regions and colors and transparencies is same
        brain_areas, brain_areas_color, brain_areas_transparency = brain_region_color(plotting_params, atlas)
        # add colors list of brain regions to cmap for plotting
        cmap_brain += brain_areas_color
    else:
        brain_areas = False

    cmap_brain = np.array(
        [[int(x * 255) for x in list(mcolors.to_rgba(c))] for c in cmap_brain])  # transfer colors to RGBA for imshow function

    # set the transparency values based on input from list
    if brain_areas:
        for idx in range(len(brain_areas_transparency)):
            cmap_brain[(idx-len(brain_areas_transparency))][-1] = brain_areas_transparency[idx]
        tgt_list = ['fiber tracts', 'VS'] + brain_areas
    else:
        tgt_list = ['fiber tracts', 'VS']

    if transparent:
        cmap_brain[0][-1] = 0  # set alpha on white pixels transparent
    for n, tgt in enumerate(tgt_list):
        if orient_idx == 0:
            tgt_mask = atlas.get_structure_mask(tgt)[slice_idx, :, :]
        elif orient_idx == 1:
            tgt_mask = atlas.get_structure_mask(tgt)[:, slice_idx, :]
        else:
            tgt_mask = atlas.get_structure_mask(tgt)[:, :, slice_idx]
        annot_section[tgt_mask > 0] = n + 2  # for setting color, 0 = background, 1 = non target brain, 2 = fibers, 3 = ventricles, >3 tgt structures
    #
    #
    #
    #
    # # get the idx for fibre structures (gray) and ventricles)
    # gray_idx = []
    # fiber_tracts_childs = atlas.get_structure_descendants('fiber tracts')
    # fiber_tracts_ids = [st[i]['id'] for i in fiber_tracts_childs]
    #
    # ventr_idx = []
    # ventr_tracts_childs = atlas.get_structure_descendants('VS')
    # ventr_tracts_ids = [st[i]['id'] for i in ventr_tracts_childs]
    #
    #
    # ventr_tracts_path = st[st['name'] == 'ventricular systems']['structure_id_path'].iloc[0]
    # for item in np.unique(annot_section):
    #     if (item > 0) & (item != 1):  # todo changed 997 to 1 here
    #         if st[st['sphinx_id'] == item]['structure_id_path'].iloc[0].startswith(fiber_tracts_path):
    #             gray_idx.append(item)
    #         elif st[st['sphinx_id'] == item]['structure_id_path'].iloc[0].startswith(ventr_tracts_path):
    #             ventr_idx.append(item)
    #
    # # get indices for tgt_area as well, iterative stuff is likely quite slow...
    # if target_region_list:
    #     tgt_idx_list = {}
    #     for idx, target_region in enumerate(target_region_list):
    #         tgt_idx = []
    #         tgt_path = st[st['name'] == target_region]['structure_id_path'].iloc[0]
    #         for item in np.unique(annot_section):
    #             if (item > 0) & (item != 1):
    #                 if st[st['sphinx_id'] == item]['structure_id_path'].iloc[0].startswith(tgt_path):
    #                     tgt_idx.append(item)
    #         tgt_idx_list[idx] = tgt_idx
    #     dummy_list = []  # dummy list of all target idx for loop below
    #     for i in tgt_idx_list:
    #         dummy_list += tgt_idx_list[i]
    #     # change values in annot slice accordingly
    #     # 0 (= nothing there) stays 0
    #     for idx_r, row in enumerate(annot_section):  # todo: there must be a better option than this loop...
    #         for idx_c, col in enumerate(row):
    #             # brain stuff set to 1
    #             if (col != 0) & (col not in gray_idx) & (col not in ventr_idx) & (col not in dummy_list):
    #                 annot_section[idx_r, idx_c] = 1
    #             # fibres to 2
    #             elif col in gray_idx:
    #                 annot_section[idx_r, idx_c] = 2
    #             # ventricles to 3
    #             elif col in ventr_idx:
    #                 annot_section[idx_r, idx_c] = 3
    #             # set target values to increasing values accordingly
    #             elif col in dummy_list:
    #                 for tgt in tgt_idx_list:
    #                     if col in tgt_idx_list[tgt]:
    #                         annot_section[idx_r, idx_c] = tgt + 4
    # else:
    #     # change values in annot slice accordingly
    #     # 0 (= nothing there) stays 0
    #     for idx_r, row in enumerate(annot_section):  # todo: there must be a better option than this loop...
    #         for idx_c, col in enumerate(row):
    #             # brain stuff set to 1
    #             if (col != 0) & (col not in gray_idx) & (col not in ventr_idx):
    #                 annot_section[idx_r, idx_c] = 1
    #             # fibres to 2
    #             elif col in gray_idx:
    #                 annot_section[idx_r, idx_c] = 2
    #             # ventricles to 3
    #             elif col in ventr_idx:
    #                 annot_section[idx_r, idx_c] = 3

    # transfer to RGB values and return annot_section
    annot_section_plt = cmap_brain[annot_section]
    return annot_section_plt, annot_section_contours
