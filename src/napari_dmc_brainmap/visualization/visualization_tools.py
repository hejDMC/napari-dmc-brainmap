import pandas as pd
import numpy as np
import json
from pathlib import Path
import random
from skimage import measure
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from natsort import natsorted
from napari_dmc_brainmap.utils import get_info, clean_results_df, get_bregma, get_xyz


def get_ipsi_contra(df):
    '''
    Function to add a column specifying if cells if ipsi or contralateral to injection site
    ml_mm values of <0 are on the 'left' hemisphere, >0 are on the 'right hemisphere
    :param df: dataframe with results for animal, not the merged across animals
    :return:
    '''

    df['ipsi_contra'] = ['ipsi'] * len(df)  # add a column defaulting to 'ipsi'
    # change values to contra with respect to the location of the injection site
    if df['injection_site'][0] == 'left':
        df.loc[(df['ml_mm'] < 0), 'ipsi_contra'] = 'contra'
    elif df['injection_site'][0] == 'right':
        df.loc[(df['ml_mm'] > 0), 'ipsi_contra'] = 'contra'
    return df


# def get_tgt_data_only(df, atlas, tgt_list):
#
#     tgt_only_data = pd.DataFrame()
#     for tgt in tgt_list:
#         tgt_list_childs = atlas.get_structure_descendants(tgt)
#         if tgt_list_childs:
#             dummy_df = df[df['acronym'].isin(tgt_list_childs)]
#         else:
#             dummy_df = df[df['acronym'].isin([tgt])]
#         dummy_df['tgt_name'] = [tgt] * len(dummy_df)
#         tgt_only_data = pd.concat([tgt_only_data, dummy_df])
#
#     return tgt_only_data
def get_tgt_data_only(df, atlas, tgt_list, negative=False, use_na=False):
    # get IDs of all regions
    # ids = [atlas.structures[reg]['id'] for reg in tgt_list]
    ids = []
    for reg in tgt_list:
        try:
            ids.append(atlas.structures[reg]['id'])
        except KeyError:
            print(f'No region called >> {reg} <<, skipping that region.')
            pass
    # get child IDs
    ids_child = [atlas.hierarchy.is_branch(id) for id in ids]
    ids_child = [c for c_g in ids_child for c in c_g]
    ids += ids_child
    if 'NA' in tgt_list and use_na:
        ids.append(-42)
        df.loc[~df.structure_id.isin(ids), 'structure_id'] = -42
    # delete non tgt cells from df
    if negative:
        tgt_data = df[~df.structure_id.isin(ids)].reset_index(drop=True)
    else:
        tgt_data = df[df.structure_id.isin(ids)].reset_index(drop=True)
    tgt_data['tgt_name'] = [get_tgt_name(s, atlas, tgt_list) for s in tgt_data['structure_id']]
    # tgt_data['tgt_name'] = [atlas.structures[s]['acronym'] if atlas.structures[s]['acronym'] in tgt_list
    #                   else list(set(atlas.get_structure_ancestors(s)) & set(tgt_list))[0] for s in tgt_data['structure_id']]  # add plotting acronyms
    return tgt_data


def get_tgt_name(s, atlas, tgt_list):
    try:
        acronym = atlas.structures[s]['acronym']
        if acronym in tgt_list:
            return acronym
        else:
            ancestors = list(set(atlas.get_structure_ancestors(s)) & set(tgt_list))
            if ancestors:
                return ancestors[0]
            else:
                return 'NA'
    except KeyError:
        return 'NA'


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


def get_unique_filename(data_fn):
    """
    Generate a unique filename by appending a suffix if the file already exists.
    """
    counter = 1
    data_fn_old = data_fn
    while data_fn.exists():
        data_fn = data_fn_old.with_name(f"{data_fn_old.stem}_{counter:03d}{data_fn_old.suffix}")
        counter += 1

    return data_fn


def load_data(input_path, atlas, animal_list, channels, data_type='cells', hemisphere='both'):


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
                if data_type in ["optic_fiber", "neuropixels_probe"]:
                    results_data = results_data[results_data['inside_brain']]
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
                    injection_site = params_data['general']['injection_site']  # add the injection_site as a column
                except KeyError:
                    # injection_site = input("no injection site specified in params.json file for " + animal_id +
                    #                        ", please enter manually: ")
                    print("WARNING: no injection site specified in params files, defaulting to right hemisphere")
                    injection_site = 'right'

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

                results_data['injection_site'] = [injection_site] * len(results_data)
                results_data['genotype'] = [genotype] * len(results_data)
                results_data['group'] = [group] * len(results_data)
                # add if the location of a cell is ipsi or contralateral to the injection site
                results_data = get_ipsi_contra(results_data)
                results_data_merged = pd.concat([results_data_merged, results_data])
        print("loaded data from " + animal_id)
        results_data_merged = clean_results_df(results_data_merged, atlas)
        if hemisphere == 'ipsi':
            results_data_merged = results_data_merged[results_data_merged['ipsi_contra'] == 'ipsi']
        elif hemisphere == 'contra':
            results_data_merged = results_data_merged[results_data_merged['ipsi_contra'] == 'contra']
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
        for d in range(diff):
            if item == 'color':
                list2.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
                print("warning: no/too few brain region colors stated, random colors will be chosen. "
                      "Colors differ between sections.")
            elif item == 'transparency':
                list2.append(list2[-1])
        return list1, list2
    elif len(list1) < len(list2):
        list2 = list2[:len(list1)]
        return list1, list2


def brain_region_color(plotting_params, atlas):
    brain_areas = plotting_params['brain_areas']
    brain_areas_color = plotting_params['brain_areas_color']
    if brain_areas_color:
        if 'ATLAS' in brain_areas_color:
            brain_areas_color = []
            for b in brain_areas:
                b_acronym = atlas.structures.acronym_to_id_map[b]
                brain_areas_color.append(tuple([c / 255 for c in atlas.structures[b_acronym]['rgb_triplet']]))
    else:
        brain_areas_color = [random.choice(list(mcolors.CSS4_COLORS.keys()))]
    brain_areas_transparency = plotting_params['brain_areas_transparency']
    if brain_areas_transparency:
        brain_areas_transparency = [int(b) for b in brain_areas_transparency]
    else:
        brain_areas_transparency = [255]
    brain_areas, brain_areas_color = match_lists(brain_areas, brain_areas_color, 'color')
    brain_areas, brain_areas_transparency = match_lists(brain_areas, brain_areas_transparency, 'transparency')
    print(brain_areas)
    print(brain_areas_color)
    print(brain_areas_transparency)
    return brain_areas, brain_areas_color, brain_areas_transparency

def brain_region_color_genes(df, cmap, atlas, plot_type):
    if plot_type == 'clusters':
        count_clusters = df.groupby(['acronym', 'structure_id', 'cluster_id']).size().reset_index(name='count')
        brain_region_colors = count_clusters.loc[count_clusters.groupby('acronym')['count'].idxmax()]
        brain_region_colors['brain_areas_color'] = brain_region_colors['cluster_id'].map(cmap)
        # brain_region_colors = create_color_ids(brain_region_colors)
    else:
        # calculate average expression levels of genes by brain region
        brain_region_colors = df.groupby('acronym')['gene_expression_norm'].mean().reset_index()
        # add structure_id
        brain_region_colors['structure_id'] = [atlas.structures.acronym_to_id_map[a] for a in brain_region_colors['acronym']]
        # add color
        curr_cmap = plt.get_cmap(cmap)
        brain_region_colors['brain_areas_color'] = [curr_cmap(g) for g in brain_region_colors['gene_expression_norm']]  # todo check rgb code
    # drop colors if brain region has descendants otherwise these parent structures will overwrite descendants
    # vs_list = get_descendants(['VS'], atlas)
    # mask = brain_region_colors['acronym'].apply(lambda tgt: check_descendants(tgt, atlas, vs_list))
    # brain_region_colors = brain_region_colors[mask]
    brain_region_colors['len'] = brain_region_colors['structure_id'].apply(
        lambda a: len(atlas.structures.data[a]['structure_id_path']))
    # exclude root, fiber tracts and ventricles
    # drop_list = ['root']
    # drop_list += get_descendants(['VS'], atlas)
    # drop_list += get_descendants(['fiber tracts'], atlas)
    # brain_region_colors = brain_region_colors[~brain_region_colors['acronym'].isin(drop_list)]
    brain_region_colors.sort_values(by='len', inplace=True)
    brain_areas = brain_region_colors.acronym.to_list()
    brain_areas_colors = brain_region_colors.brain_areas_color.to_list()
    brain_areas_transparency = [255] * len(brain_areas)
    return [brain_areas, brain_areas_colors, brain_areas_transparency]



def plot_brain_schematic(atlas, slice_idx, orient_idx, plotting_params, gene_color=False, transparent=True, voronoi=False):
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
    # get indices for brain outline
    contours = measure.find_contours(annot_section, level=0.5)
    brain_outline_idx = []
    for contour in contours:
        brain_outline_idx.append(contour.astype(int))
    if gene_color or voronoi:
        cmap_brain = ['white', 'white', 'lightgray',
                      'white']  # colormap for the brain outline (white: empty space,
        # linen=brain, lightgray=root, lightcyan=ventricles)
    else:
         cmap_brain = ['white', 'linen', 'lightgray',
                  'lightcyan']  # colormap for the brain outline (white: empty space,
                                # linen=brain, lightgray=root, lightcyan=ventricles)

    if plotting_params['brain_areas']:  # if target region list exists, check if len of tgt regions and colors and transparencies is same
        brain_areas, brain_areas_color, brain_areas_transparency = brain_region_color(plotting_params, atlas)
        # add colors list of brain regions to cmap for plotting
        cmap_brain += brain_areas_color
    elif gene_color:
        brain_areas, brain_areas_color, brain_areas_transparency = gene_color
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
    if voronoi:
        voronoi[0][annot_section == 0] = 0
        voronoi[0][annot_section == 2] = 2
        voronoi[0][annot_section == 3] = 3
        annot_section = voronoi[0]
        n = annot_section.max() - 2
        cmap_brain = np.append(cmap_brain, np.array([[int(x * 255) for x in list(v)] for v in voronoi[1]]), axis=0)

    annot_section[brain_outline_idx[0][:, 0], brain_outline_idx[0][:, 1]] = n + 3
    cmap_brain = np.append(cmap_brain, np.array([[int(x * 255) for x in mcolors.to_rgba('black')]]), axis=0)
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
    unilateral = plotting_params['unilateral']
    if unilateral in ['left', 'right'] and orient_idx < 2:
        bregma = get_bregma(atlas.atlas_name)
        if unilateral == 'left':
            annot_section_plt = annot_section_plt[:, bregma[atlas.space.axes_description.index('rl')]:]
            if plotting_params['plot_outline']:
                annot_section_contours = annot_section_contours[:, bregma[atlas.space.axes_description.index('rl')]:]
        else:
            annot_section_plt = annot_section_plt[:, :bregma[atlas.space.axes_description.index('rl')]]
            if plotting_params['plot_outline']:
                annot_section_contours = annot_section_contours[:, :bregma[atlas.space.axes_description.index('rl')]]

    return annot_section_plt, annot_section_contours


def create_cmap(animal_dict, plotting_params, clr_id, df=pd.DataFrame(), hue_id='channel'):

    cmap = {}
    if not df.empty:
        group_ids = list(df[hue_id].unique())
        if hue_id == 'cluster_id':
            group_ids = natsorted(group_ids)
    else:
        group_ids = list(animal_dict.keys())
    cmap_groups = plotting_params[clr_id]
    if not isinstance(cmap_groups, list):
        print("not possible to combine c:cmap with colors, random colors will be assigned, use comma seperate list of colors instead")
        cmap_groups = []
    num_groups = len(group_ids)
    if cmap_groups:
        num_colors = len(cmap_groups)
    else:
        num_colors = 0
        cmap_groups = []
    if num_groups > num_colors:  # more groups than colors
        print("warning: " + str(num_groups) + " channels/groups/genotypes provided, but only " + str(num_colors) +
              " cmap groups --> adding random colors")
        diff = num_groups - num_colors
        if clr_id == 'colors_projections':
            colormaps = [cc for cc in cm.datad]
            for d in range(diff):
                cmap_groups.append(random.choice(colormaps))
        else:
            if num_groups > 148:
                print("Number of groups exceeds matplotlib standard colors (148) using xkcd colors instead, overriding "
                      "input of colors. If you want to provide input use hex keys of xkcd colors instead: https://xkcd.com/color/rgb/" )
                # load xkcd.json file

                with open(Path(__file__).resolve().parent.joinpath('xkcd.json')) as fn:
                    xcol_data = json.load(fn)
                xcol_list = []
                for i in xcol_data['colors']:
                    # xcol_list.append(f"xkcd:{i['color']}")
                    xcol_list.append(f"{i['hex']}")
                cmap_groups = random.sample(xcol_list, num_groups)
            else:
                diff_cmap = random.sample(list(mcolors.CSS4_COLORS.keys()), diff)
                cmap_groups.extend(diff_cmap)
            # for d in range(diff):
                # cmap_groups.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
    elif num_groups < num_colors:  # less groups than colors
        print("warning: " + str(num_groups) + " channels/groups/genotypes, but  " + str(len(cmap_groups)) +
              " cmap groups --> dropping colors")
        diff = num_colors - num_groups
        cmap_groups = cmap_groups[:-diff]
    cmap_groups = [mcolors.to_rgba(c) for c in cmap_groups]
    for g, c in zip(group_ids, cmap_groups):
        cmap[g] = c
    return cmap

def get_descendants(tgt_list, atlas):
    tgt_layer_list = []
    for tgt in tgt_list:
        descendents = atlas.get_structure_descendants(tgt)
        if not descendents:  # if no descendents found, return tgt
            descendents = [tgt]
        tgt_layer_list += descendents
    return tgt_layer_list

def get_voronoi_mask(df, cmap, atlas, plotting_params, orient_mapping):
    df = df.reset_index(drop=True)
    xyz_dict = get_xyz(atlas, plotting_params['section_orient'])
    matrix = np.zeros((xyz_dict['y'][1], xyz_dict['x'][1]))
    points = np.array([[x, y] for x, y in zip(df[orient_mapping['x_plot']], df[orient_mapping['y_plot']])])

    if plotting_params['plot_gene'] == 'clusters':
        df.loc[:,'voronoi_colors'] = df['cluster_id'].map(cmap)
        # voronoi_colors = df['voronoi_colors'].to_list()
        # create mapping for colors
        df = create_color_ids(df)
        # values = np.array(df['clr_id'].to_list()).astype('int')
    else:
        curr_cmap = plt.get_cmap(cmap)
        df.loc[:, 'voronoi_colors'] = [curr_cmap(g) for g in df['gene_expression_norm'].to_list()]
        df.loc[:, 'clr_id'] = np.arange(4, len(df) + 4)
        # values = np.array(df['clr_id'].to_list())
    xx, yy = np.meshgrid(np.arange(xyz_dict['x'][1]), np.arange(xyz_dict['y'][1]))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    distances = cdist(grid_points, points)
    nearest_point_index = np.argmin(distances, axis=1)
    # npi = np.vectorize(lambda x: int(df.loc[x, 'clr_id']))(nearest_point_index)
    # matrix.ravel()[np.arange(matrix.size)] = values[nearest_point_index]
    matrix.ravel()[np.arange(matrix.size)] = nearest_point_index
    matrix = matrix.reshape((xyz_dict['y'][1], xyz_dict['x'][1])).astype('int')
    matrix = np.vectorize(lambda x: int(df.loc[x, 'clr_id']))(matrix)
    voronoi_colors = [tuple(df[df['clr_id'] == clr_id]['voronoi_colors'].unique())[0] for clr_id in natsorted(df['clr_id'].unique())]

    return [matrix, voronoi_colors]

def create_color_ids(df):
    unique_clusters = natsorted(df['cluster_id'].unique())
    new_ids = np.arange(4, len(unique_clusters) + 4)
    map_dict = {}
    for u, n in zip(unique_clusters, new_ids):
        map_dict[u] = int(n)
    df.loc[:, 'clr_id'] = df['cluster_id'].map(map_dict)
    return df


# def check_descendants(tgt, atlas, vs_list):
#     tgt_layer_list = get_descendants([tgt], atlas)
#     if any(t in vs_list for t in tgt_layer_list):
#         return False
#     else:
#         return len(tgt_layer_list) == 1