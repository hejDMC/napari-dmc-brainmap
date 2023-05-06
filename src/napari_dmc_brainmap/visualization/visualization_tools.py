import pandas as pd
import numpy as np
import json
import matplotlib.colors as mcolors
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


def load_data(input_path, animal_list, channels):
    st = dummy_load_allen_structure_tree()
    #  loop over animal_ids
    results_data_merged = pd.DataFrame()  # initialize merged dataframe
    for animal_id in animal_list:
        # for animal_idx, animal_id in enumerate(animal_list):
        for channel in channels:
            results_dir = get_info(input_path.joinpath(animal_id), 'results', seg_type='cells', channel=channel,
                                    only_dir=True)
            results_file = results_dir.joinpath(animal_id + '_cells.csv')

            if results_file.exists():
                results_data = pd.read_csv(results_file)  # load the data
                results_data['sphinx_id'] -= 1  # correct for indices starting at 1
                results_data['animal_id'] = [animal_id] * len(
                    results_data)  # add the animal_id as a column for later identification
                results_data['channel'] = [channel] * len(results_data)
                # add the injection hemisphere stored in params.json file
                params_file = input_path.joinpath(animal_id, 'params.json')  # directory of params.json file
                with open(params_file) as fn:  # load the file
                    params_data = json.load(fn)
                try:
                    injection_side = params_data['general']['injection_side']  # add the injection_side as a column
                except KeyError:
                    injection_side = input("no injection side specified in params.json file, please enter manually: ")
                # todo add genotype or exp_group from params.json
                results_data['injection_side'] = [injection_side] * len(results_data)
                # add if the location of a cell is ipsi or contralateral to the injection side
                results_data = get_ipsi_contra(results_data)
                results_data_merged = pd.concat([results_data_merged, results_data])
        print("loaded data from " + animal_id)
        results_data_merged = clean_results_df(results_data_merged, st)
    return results_data_merged



def plot_brain_schematic(annot_section, structure_tree, target_region_list=False, target_color_list=['plum'], target_transparency=[255],
                         unilateral_target=False, transparent=True):
    """
    Function to plot brain schematics as colored plots

    :param annot_section: 2d array from allensdk with brain section
    :param structure_tree: structure tree data from allensdk
    :param target_region_list: LIST of target brain regions to plot
    :param target_color_list: LIST of colors for target brain regions
    :param target_transparency: LIST of transparency values for target brain regions
    :param unilateral_target: BOOLEAN if target should only be plotted on one hemisphere -- TO BE IMPLEMENTED
    :param transparent: BOOLEAN for setting white pixels to transparent (e.g. plotting on black background)
    :return: annot_section in RGBA values on x-y coordintaes for plotting
    """
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
    ventr_idx = []
    for item in np.unique(annot_section):
        if (item > 0) & (item != 997):
            if structure_tree.ancestors([item])[0][-2]['name'] == 'fiber tracts':
                gray_idx.append(item)
            elif structure_tree.ancestors([item])[0][-2]['name'] == 'ventricular systems':
                ventr_idx.append(item)

    # get indices for tgt_area as well, iterative stuff is likely quite slow...
    if target_region_list:
        tgt_idx_list = {}
        for idx, target_region in enumerate(target_region_list):
            tgt_idx = []
            for item in np.unique(annot_section):
                if (item > 0) & (item != 997):
                    for st_level in structure_tree.ancestors([item])[0]:  # iterate over the levels and check if it contains the target brain structure
                        if st_level['name'] == target_region:
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