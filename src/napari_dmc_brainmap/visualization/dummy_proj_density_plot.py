import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
    annot_section[annot_section==1] = 0  # set non brain values to 1

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
    fiber_tracts_path = st[st['name']=='fiber tracts']['structure_id_path'].iloc[0]

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

#%%

section_list = [1.6, 1.4, 0.7]
section_range = 0.05
bregma = [540, 0, 570]
st = pd.read_csv(r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dmc-brainmap\napari-dmc-brainmap\src\napari_dmc_brainmap\registration\sharpy_track\sharpy_track\atlas\structure_tree_safe_2017.csv')
annot = np.load(r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dmc-brainmap\napari-dmc-brainmap\src\napari_dmc_brainmap\registration\sharpy_track\sharpy_track\atlas\annotation_volume_10um_by_index.npy')
seg_data = pd.read_csv(r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dopamine\analysis\anatomy\data\DP-117\results\projections\cy3\DP-117_projections.csv')
bin_size = 5
vmax = 1000.0

fig, ax = plt.subplots(1, 3)

section_dict = {}
cnt = 0
for section in section_list:
    target_ap = [section + section_range, section - section_range]
    target_ap = [int(-(target / 0.01 - bregma[0])) for target in target_ap]
    slice_idx = int(-(section / 0.01 - bregma[0]))
    annot_section = annot[slice_idx, :, :].copy()
    annot_section_plt = plot_brain_schematic(annot_section, st)
    ax[cnt].imshow(annot_section_plt)
    ax[cnt].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
               colors=['gainsboro'],
               linewidths=0.2)
    # select data in bin
    section_dict[cnt] = seg_data[(seg_data['zpixel'] >= target_ap[0]) & (seg_data['zpixel'] <= target_ap[1])]
    sns.histplot(ax=ax[cnt],
                 data=section_dict[cnt], x="xpixel", y="ypixel", binwidth=bin_size,cmap='Reds',
                 vmax=vmax)
    cnt += 1
plt.show()
# get min-max for scaling and plot



#%%


#sns.kdeplot(ax=ax,
#    data=seg_data_temp, x="xpixel", y="ypixel", fill=True,
#)
sns.histplot(ax=ax,
    data=seg_data_temp, x="xpixel", y="ypixel", cbar='Blues'
)
plt.show()


#%%

def find_common_suffix(image_list, folder='unknown'):
    if len(image_list) > 1:
        for i in range(len(image_list[0])):
            if i > 0:
                if image_list[0][-i] == image_list[1][-i]:
                    continue
                else:
                    break
        common_suffix = image_list[0][-i + 1:]
        # print("estimated common_suffix for " + folder + " folder: " + common_suffix)
    elif len(image_list) == 1:
        print('only one image in ' + folder + ' folder: ' + image_list[0])
        common_suffix = input("please, manually enter suffix: ")
    else:
        common_suffix = []
    return common_suffix
import pandas as pd
import os
import numpy as np
import cv2

rgb_dir = r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dopamine\analysis\anatomy\data\DP-117\rgb'
seg_dir = r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dopamine\analysis\anatomy\data\DP-117\segmentation\projections\cy3'


im_list = os.listdir(seg_dir)
com_suf = find_common_suffix(im_list)

for im in im_list:
    fn_base = im[:-len(com_suf)]
    im_p = os.path.join(seg_dir, im)
    df = pd.read_csv(im_p)
    img = cv2.imread(os.path.join(rgb_dir, fn_base+'_RGB.tif'))
    y_coord = np.shape(img)[0]
    df['Position Y'] = y_coord-df['Position Y']
    df.to_csv(im_p)


#%%

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
                results_data['ml_mm'] *= (-1)  # so that negative values are left hemisphere
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
                    injection_side = input("no injection side specified in params.json file for " + animal_id +
                                           ", please enter manually: ")

                try:
                    genotype = params_data['general']['genotype']
                except KeyError:
                    print("warning, no genotype specified for " + animal_id +
                          " this could lead to problems down the line, "
                          "use the create params.json function to enter genotype")
                    genotype = 0
                try:
                    group = params_data['general']['group']
                except KeyError:
                    print(
                        "warning, no experimental group specified for " + animal_id +
                        " this could lead to problems down the line, "
                        "use the create params.json function to enter experimental group")
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



