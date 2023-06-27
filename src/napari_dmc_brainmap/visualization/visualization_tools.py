import pandas as pd
import numpy as np
import json
import matplotlib.colors as mcolors
from natsort import natsorted
from napari_dmc_brainmap.utils import get_animal_id, get_info, split_strings_layers, clean_results_df
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.find_structure import sliceHandle


def dummy_load_allen_structure_tree():
    s = sliceHandle()
    st = s.df_tree
    return st

def dummy_load_allen_annot():
    s = sliceHandle(load_annot=True)
    annot = s.annot  # todo not sure this works
    return annot

def get_bregma():
    s = sliceHandle()
    bregma = s.bregma
    return bregma

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


def get_tgt_data_only(df, tgt_list):
    tgt_path_list = []
    st = dummy_load_allen_structure_tree()
    for acr in tgt_list:
        curr_idx = st.index[st['acronym'] == acr].tolist()
        tgt_path_list.append(st['structure_id_path'].iloc[curr_idx])

    for idx, tgt_path in enumerate(tgt_path_list):
        tgt_path = tgt_path.to_list()
        if idx == 0:
            tgt_only_data = df[df['path_list'].str.contains(tgt_path[0])].copy()
            tgt_only_data['tgt_name'] = [tgt_list[idx]] * len(tgt_only_data)
        else:
            dummy_data = df[df['path_list'].str.contains(tgt_path[0])].copy()
            dummy_data['tgt_name'] = [tgt_list[idx]] * len(dummy_data)
            tgt_only_data = pd.concat([tgt_only_data, dummy_data])
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

def load_data(input_path, animal_list, channels, data_type='cells'):
    st = dummy_load_allen_structure_tree()
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
                results_data['sphinx_id'] -= 1  # correct for indices starting at 1
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
                    injection_side = input("no injection side specified in params.json file for " + animal_id +
                                           ", please enter manually: ")

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
        results_data_merged = clean_results_df(results_data_merged, st)
        results_data_merged = results_data_merged.reset_index(drop=True)
    return results_data_merged

def coord_mm_transform(df, to_coord = True):
    """
    Function to calculate atlas coordinates into mm and vice versa
    Inserted df needs to have specified columns
    """
    bregma = get_bregma()
    if to_coord:
        df['ap_mm'] = -(df['ap_mm'] / 0.01 - bregma[0]).astype(int)
        df['dv_mm'] = -(df['dv_mm'] / 0.01).astype(int)
        df['ml_mm'] = (df['ml_mm'] / 0.01 + bregma[2]).astype(int)
    elif not to_coord:
        df['ap_mm'] = (-df['ap_mm'] + bregma[0])*0.01
        df['dv_mm'] = -(df['dv_mm'] * 0.01)
        df['ml_mm'] = (df['ml_mm'] - bregma[2]) * 0.01
    return df


def plot_brain_schematic(annot_section, st, target_region_list=False, target_color_list=['plum'], target_transparency=[255],
                         unilateral_target=False, transparent=True):
    """
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
    annot_section[annot_section == 1] = 0  # set non brain values to 1

    if target_region_list:  # if target region list exists, check if len of tgt regions and colors and transparencies is same
        target_color_list = target_len_check(target_region_list, target_color_list, "color")
        target_transparency = target_len_check(target_region_list, target_transparency, "transparency")

    cmap_brain = ['white', 'linen', 'lightgray',
                  'lightcyan']  # colormap for the brain outline (white: empty space,
                                # linen=brain, lightgray=root, lightcyan=ventricles)

    if target_region_list:  # add colors list of brain regions to cmap for plotting
        cmap_brain += target_color_list

    cmap_brain = np.array(
        [[int(x * 255) for x in list(mcolors.to_rgba(c))] for c in cmap_brain])  # transfer colors to RGBA for imshow function

    # set the transparency values based on input from list
    if target_region_list:
        for idx in range(len(target_transparency)):
            cmap_brain[(idx-len(target_transparency))][-1] = target_transparency[idx]

    if transparent:
        cmap_brain[0][-1] = 0  # set alpha on white pixels transparent

    # get the idx for fibre structures (gray) and ventricles)
    gray_idx = []
    fiber_tracts_path = st[st['name'] == 'fiber tracts']['structure_id_path'].iloc[0]

    ventr_idx = []
    ventr_tracts_path = st[st['name'] == 'ventricular systems']['structure_id_path'].iloc[0]
    for item in np.unique(annot_section):
        if (item > 0) & (item != 1):  # todo changed 997 to 1 here
            if st[st['sphinx_id'] == item]['structure_id_path'].iloc[0].startswith(fiber_tracts_path):
                gray_idx.append(item)
            elif st[st['sphinx_id'] == item]['structure_id_path'].iloc[0].startswith(ventr_tracts_path):
                ventr_idx.append(item)

    # get indices for tgt_area as well, iterative stuff is likely quite slow...
    if target_region_list:
        tgt_idx_list = {}
        for idx, target_region in enumerate(target_region_list):
            tgt_idx = []
            tgt_path = st[st['name'] == target_region]['structure_id_path'].iloc[0]
            for item in np.unique(annot_section):
                if (item > 0) & (item != 1):
                    if st[st['sphinx_id'] == item]['structure_id_path'].iloc[0].startswith(tgt_path):
                        tgt_idx.append(item)
            tgt_idx_list[idx] = tgt_idx
        dummy_list = []  # dummy list of all target idx for loop below
        for i in tgt_idx_list:
            dummy_list += tgt_idx_list[i]
        # change values in annot slice accordingly
        # 0 (= nothing there) stays 0
        for idx_r, row in enumerate(annot_section):  # todo: there must be a better option than this loop...
            for idx_c, col in enumerate(row):
                # brain stuff set to 1
                if (col != 0) & (col not in gray_idx) & (col not in ventr_idx) & (col not in dummy_list):
                    annot_section[idx_r, idx_c] = 1
                # fibres to 2
                elif col in gray_idx:
                    annot_section[idx_r, idx_c] = 2
                # ventricles to 3
                elif col in ventr_idx:
                    annot_section[idx_r, idx_c] = 3
                # set target values to increasing values accordingly
                elif col in dummy_list:
                    for tgt in tgt_idx_list:
                        if col in tgt_idx_list[tgt]:
                            annot_section[idx_r, idx_c] = tgt + 4
    else:
        # change values in annot slice accordingly
        # 0 (= nothing there) stays 0
        for idx_r, row in enumerate(annot_section):  # todo: there must be a better option than this loop...
            for idx_c, col in enumerate(row):
                # brain stuff set to 1
                if (col != 0) & (col not in gray_idx) & (col not in ventr_idx):
                    annot_section[idx_r, idx_c] = 1
                # fibres to 2
                elif col in gray_idx:
                    annot_section[idx_r, idx_c] = 2
                # ventricles to 3
                elif col in ventr_idx:
                    annot_section[idx_r, idx_c] = 3

    # transfer to RGB values and return annot_section
    annot_section_plt = cmap_brain[annot_section]
    return annot_section_plt


def target_len_check(target_region_list, target_list_to_comapare, target_type):
    """
    Related to plot_brain_schematic() -- check length of list for colors and transparency for target regions
    Add or delete default values if length differs

    :param target_region_list: LIST with target regions
    :param target_list_to_comapare: LIST with colors/transparency values
    :param target_type: needs to be either 'color' or 'transparency'
    :return: corrected list of colors or transparency values

    """
    if len(target_region_list) == len(target_list_to_comapare):
        pass
    elif len(target_region_list) > len(target_list_to_comapare):
        print("WARNING -- more target regions than target " + target_type + " --> setting missing to default " + target_type)
        for i in range(len(target_region_list) - len(target_list_to_comapare)):
            if target_type == 'color':
                target_list_to_comapare.append('plum')
            elif target_type == 'transparency':
                target_list_to_comapare.append(255)
            else:
                print("ERROR -- target_type not correctly defined -- set to either 'color' or 'transparency'")
    else:
        print("WARNING -- more target " + target_type + " than target  " + target_type + " --> deleting colors to match number of " + target_type)
        for i in range(len(target_list_to_comapare) - len(target_region_list)):
            target_list_to_comapare.pop()
    return target_list_to_comapare