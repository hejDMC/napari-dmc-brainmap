import math
import random
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from napari_dmc_brainmap.utils import split_to_list, split_strings_layers, load_group_dict
from napari_dmc_brainmap.visualization.visualization_tools import get_bregma, dummy_load_allen_structure_tree, \
    dummy_load_allen_annot, plot_brain_schematic, coord_mm_transform

# results_data_merged = coord_mm_transform(results_data_merged)
#
# results_dir = '/home/felix/Academia/DMC-lab/Projects/Dopamine/Analyses/Anatomy/results/plots_orb-manuscript/'  # to store figs
# section_list = [2.5, 2.0, 1.5, 1.0]  # in mm AP coordinates for coronal sections relative to bregma -- [ORB, AUD]
# # target_list = ['Orbital area', 'Prelimbic area', 'Infralimbic area', 'Anterior cingulate area', 'Secondary motor area', 'Agranular insular area']
# # target_colors = ['green', 'blue', 'yellow', 'pink', 'brown', 'orange']
# target_list = ['Orbital area, lateral part', 'Orbital area, ventrolateral part', 'Orbital area, medial part',
#                'Prelimbic area', 'Infralimbic area',
#                'Anterior cingulate area, dorsal part', 'Anterior cingulate area, ventral part',
#                'Agranular insular area, dorsal part', 'Agranular insular area, ventral part',
#                'Secondary motor area']
# target_colors = ['darkgreen', 'green', 'lightgreen',
#                  'navy', 'royalblue',
#                  'darkorchid', 'plum',
#                  'olive', 'yellowgreen',
#                  'indigo']
# target_transparency = [100] * len(target_list)
# for section in section_list:
#     slice_idx = int(-(section / 0.01 - bregma[0]))
#     annot_section = annot[slice_idx, :, 570:].copy()
#     annot_section_plt = plot_brain_schematic(annot_section, st, target_list, target_colors, target_transparency)
#     fig, ax = plt.subplots(figsize=(10, 6))  # len(section_list), figsize=(18,5))
#     im = ax.imshow(annot_section_plt)
#     ax.contour(annot[slice_idx, :, 570:], levels=np.unique(annot[slice_idx, :, 570:]), colors=['gainsboro'],
#                linewidths=0.2)
#     ax.axis('off')

def get_brain_section_params(brainsec_widget):
    plotting_params = {
        "section_list": [float(i) for i in brainsec_widget.section_list.value.split(',')],
        "groups": brainsec_widget.groups.value,
        "cmap_groups": split_to_list(brainsec_widget.cmap_groups.value),
        "section_range": float(brainsec_widget.section_range.value),
        "save_name": brainsec_widget.save_name.value,
        "save_fig": brainsec_widget.save_fig.value,
    }
    return plotting_params

def get_rows_cols(section_list):
    n_sec = len(section_list)
    n_cols = int(np.ceil(math.sqrt(n_sec)))
    if (n_cols ** 2 - n_sec) >= n_cols:
        n_rows = n_cols -1
    else:
        n_rows = n_cols
    return n_rows, n_cols


def create_geno_cmap(animal_dict, plotting_params, df=False):

    geno_cmap = {}
    if df:
        group_ids = list(df['channel'].unique())
    else:
        group_ids = list(animal_dict.keys())
    cmap_groups = plotting_params["cmap_groups"]
    num_groups = len(group_ids)
    num_colors = len(cmap_groups)
    if num_groups > num_colors:  # more groups than colors
        print(str(len(group_ids)) + " groups provided, but only " + str(len(cmap_groups)) +
              " cmap groups, adding random colors")
        diff = num_groups - num_colors
        for d in range(diff):
            cmap_groups.append(random.choice(list(mcolors.CSS4_COLORS.keys())))  # todo this doesn't work yet
    elif num_groups < num_colors:  # less groups than colors
        print(str(len(group_ids)) + " groups provided, but  " + str(len(cmap_groups)) +
              " cmap groups, dropping colors")
        diff = num_colors - num_groups
        cmap_groups = cmap_groups[:-diff]
    for g, c in zip(group_ids, cmap_groups):
        geno_cmap[g] = c

    return geno_cmap


def do_brain_section_plot(input_path, df, animal_list, plotting_params, brain_section_widget, save_path):

    if plotting_params["groups"] in ['genotype', 'group']:
        animal_dict = load_group_dict(input_path, animal_list, group_id=plotting_params["groups"])
        geno_cmap = create_geno_cmap(animal_dict, plotting_params)
        single_color=False
    elif plotting_params["groups"] == 'channel':
        geno_cmap = create_geno_cmap([], plotting_params, df=df)
        single_color = False
    else:
        single_color=True
    # tgt_channel = plotting_params["tgt_channel"] # this not from params file todo option for more than one channel
    section_list = plotting_params["section_list"]  # in mm AP coordinates for coronal sections
    n_rows, n_cols = get_rows_cols(section_list)
    figsize = [int(i) for i in brain_section_widget.plot_size.value.split(',')]
    mpl_widget = FigureCanvas(Figure(figsize=(figsize)))
    static_ax = mpl_widget.figure.subplots(n_rows, n_cols)
    n_row = 0
    n_col = 0
    section_range = plotting_params["section_range"]
    bregma = get_bregma()
    st = dummy_load_allen_structure_tree()
    annot = dummy_load_allen_annot()
    df = coord_mm_transform(df)
    for section in section_list:
        target_ap = [section + section_range, section - section_range]
        target_ap = [int(-(target / 0.01 - bregma[0])) for target in target_ap]
        slice_idx = int(-(section / 0.01 - bregma[0]))
        annot_section = annot[slice_idx, :, :].copy()
        annot_section_plt = plot_brain_schematic(annot_section, st)  # todo fix this one
        df_to_plot = df[(df['ap_mm'] >= target_ap[0]) & (df['ap_mm'] <= target_ap[1])]
        if len(section_list) > 2:
            im = static_ax[n_row, n_col].imshow(annot_section_plt)
            static_ax[n_row, n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                                            colors=['gainsboro'],
                                            linewidths=0.2)
            if single_color:
                sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=df_to_plot,
                                color=plotting_params["cmap_groups"][0])
            else:
                sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=df_to_plot,
                                hue=plotting_params["groups"], palette=geno_cmap)
            static_ax[n_row, n_col].title.set_text('bregma - ' + str(-(slice_idx - bregma[0]) * 0.01) + ' mm')
            static_ax[n_row, n_col].axis('off')
        elif len(section_list) == 2:
            im = static_ax[n_col].imshow(annot_section_plt)
            static_ax[n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                                            colors=['gainsboro'],
                                            linewidths=0.2)
            if single_color:
                sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=df_to_plot,
                                color=plotting_params["cmap_groups"][0])
            else:
                sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=df_to_plot,
                                hue=plotting_params["groups"], palette=geno_cmap)
            static_ax[n_col].title.set_text('bregma - ' + str(-(slice_idx - bregma[0]) * 0.01) + ' mm')
            static_ax[n_col].axis('off')
        else:
            im = static_ax.imshow(annot_section_plt)
            static_ax.contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                                            colors=['gainsboro'],
                                            linewidths=0.2)
            if single_color:
                sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=df_to_plot,
                                color=plotting_params["cmap_groups"][0])
            else:
                sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=df_to_plot,
                                hue=plotting_params["groups"], palette=geno_cmap)
            static_ax.title.set_text('bregma - ' + str(-(slice_idx - bregma[0]) * 0.01) + ' mm')
            static_ax.axis('off')
        if n_col < n_cols-1:
            n_col += 1
        else:
            n_col = 0
            n_row += 1
    if plotting_params["save_fig"]:
        static_ax.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
    return mpl_widget