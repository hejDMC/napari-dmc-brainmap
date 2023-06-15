import math
import random
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from napari_dmc_brainmap.utils import split_to_list, load_group_dict
from napari_dmc_brainmap.visualization.visualization_tools import get_bregma, dummy_load_allen_structure_tree, \
    dummy_load_allen_annot, plot_brain_schematic, coord_mm_transform


def get_brain_section_params(brainsec_widget):
    plotting_params = {
        "plot_item": brainsec_widget.plot_item.value,
        "section_list": [float(i) for i in brainsec_widget.section_list.value.split(',')],
        "section_range": float(brainsec_widget.section_range.value),
        "groups": brainsec_widget.groups.value,
        "cmap_groups": split_to_list(brainsec_widget.cmap_groups.value),
        "bin_width": int(brainsec_widget.bin_width.value),
        "vmax": int(brainsec_widget.vmax.value),
        "cmap_projection": brainsec_widget.cmap_projection.value,
        "color_inj": [str(i) for i in brainsec_widget.color_inj.value.split(',')],
        "color_optic": [str(i) for i in brainsec_widget.color_optic.value.split(',')],
        "color_npx": [str(i) for i in brainsec_widget.color_npx.value.split(',')],
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


def create_geno_cmap(animal_dict, plotting_params, df=pd.DataFrame()):

    geno_cmap = {}
    if not df.empty:
        group_ids = list(df['channel'].unique())
    else:
        group_ids = list(animal_dict.keys())
    cmap_groups = plotting_params["cmap_groups"]
    num_groups = len(group_ids)
    if cmap_groups[0]:
        num_colors = len(cmap_groups)
    else:
        num_colors = 0
        cmap_groups = []
    if num_groups > num_colors:  # more groups than colors
        print("warning: " + str(num_groups) + " channels/groups/genotypes provided, but only " + str(num_colors) +
              " cmap groups --> adding random colors")
        diff = num_groups - num_colors
        for d in range(diff):
            cmap_groups.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
    elif num_groups < num_colors:  # less groups than colors
        print("warning: " + str(num_groups) + " channels/groups/genotypes, but  " + str(len(cmap_groups)) +
              " cmap groups --> dropping colors")
        diff = num_colors - num_groups
        cmap_groups = cmap_groups[:-diff]
    for g, c in zip(group_ids, cmap_groups):
        geno_cmap[g] = c
    return geno_cmap

def create_color_dict(input_path, animal_list, data_dict, plotting_params):

    color_dict = {}
    for item in plotting_params['plot_item']:
        # todo use this?? including the other items as well? and rename geno_cmap variable
        if item == 'cells':
            color_dict[item] = {}
            if plotting_params["groups"] in ['genotype', 'group']:
                animal_dict = load_group_dict(input_path, animal_list, group_id=plotting_params["groups"])
                geno_cmap = create_geno_cmap(animal_dict, plotting_params)
                single_color = False
            elif plotting_params["groups"] == 'channel':
                geno_cmap = create_geno_cmap([], plotting_params, df=data_dict['cells'])
                single_color = False
            else:
                single_color = True
                if not plotting_params["cmap_groups"][0]:  # todo this is bad
                    geno_cmap = random.choice(list(mcolors.CSS4_COLORS.keys()))
                else:
                    geno_cmap = plotting_params["cmap_groups"][0]
        elif item == 'projections':
            color_dict[item] = {}
            geno_cmap = plotting_params["cmap_projection"]
            single_color = True
        elif item == 'injection_side':
            color_dict[item] = {}
            geno_cmap = plotting_params["color_inj"]
            single_color = True
        elif item == "optic_fiber":
            color_dict[item] = {}
            if len(plotting_params["color_optic"]) == 1:
                single_color = True
            else:
                single_color = False
            geno_cmap = plotting_params["color_optic"]
        elif item == "neuropixels_probe":
            color_dict[item] = {}
            if len(plotting_params["color_npx"]) == 1:
                single_color = True
            else:
                single_color = False
            geno_cmap = plotting_params["color_npx"]
        # todo this will fill the dict wrongly for items nto in elif clause
        color_dict[item]['geno_cmap'] = geno_cmap
        color_dict[item]['single_color'] = single_color
    return color_dict

def do_brain_section_plot(input_path, data_dict, animal_list, plotting_params, brain_section_widget, save_path):

    color_dict = create_color_dict(input_path, animal_list, data_dict, plotting_params)
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
    for item in data_dict:
        data_dict[item] = coord_mm_transform(data_dict[item])

    for section in section_list:

        plot_dict = {}
        target_ap = [section + section_range, section - section_range]
        target_ap = [int(-(target / 0.01 - bregma[0])) for target in target_ap]
        slice_idx = int(-(section / 0.01 - bregma[0]))
        annot_section = annot[slice_idx, :, :].copy()
        annot_section_plt = plot_brain_schematic(annot_section, st)  # todo fix this one
        for item in data_dict:
            plot_dict[item] = data_dict[item][(data_dict[item]['ap_mm'] >= target_ap[0])
                                              & (data_dict[item]['ap_mm'] <= target_ap[1])]


        cnt = 0
        for item in plot_dict:
            if len(section_list) > 2:
                if cnt < 1:
                    static_ax[n_row, n_col].imshow(annot_section_plt)
                    static_ax[n_row, n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                                                    colors=['gainsboro'],
                                                    linewidths=0.2)

                if item == 'cells':
                    if color_dict[item]['single_color']:
                            sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=plot_dict['cells'],
                                            color=color_dict[item]["geno_cmap"])
                    else:
                            sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=plot_dict['cells'],
                                            hue=plotting_params["groups"], palette=color_dict[item]["geno_cmap"])

                elif item == 'projections':
                    sns.histplot(ax=static_ax[n_row, n_col], data=plot_dict['projections'], x="xpixel", y="ypixel",
                                 cmap=color_dict[item]["geno_cmap"], binwidth=plotting_params['bin_width'],
                                 vmax=plotting_params['vmax'])

                elif item == 'injection_side':
                    if color_dict[item]['single_color']:
                        sns.kdeplot(ax=static_ax[n_row, n_col], data=plot_dict[item], x="xpixel", y="ypixel", fill=True,
                                    color=color_dict[item]["geno_cmap"][0])
                else:
                    if color_dict[item]["single_color"]:
                        sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=plot_dict[item],
                                        color=color_dict[item]["geno_cmap"][0], s=20)
                        sns.regplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm',
                                    data=plot_dict[item],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["geno_cmap"][0]),
                                    scatter=None, ci=None)
                    else:
                        sns.scatterplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm', data=plot_dict[item],
                                        hue='channel', palette=color_dict[item]["geno_cmap"], s=20)

                        for i, c in enumerate(plot_dict[item]['channel'].unique()):
                            sns.regplot(ax=static_ax[n_row, n_col], x='ml_mm', y='dv_mm',
                                        data=plot_dict[item][plot_dict[item]['channel'] == c],
                                        line_kws=dict(alpha=0.7, color=color_dict[item]["geno_cmap"][i]),
                                        scatter=None, ci=None)

                static_ax[n_row, n_col].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[0]) * 0.01), 1)) + ' mm')
                static_ax[n_row, n_col].axis('off')

            elif len(section_list) == 2:
                if cnt < 1:
                    static_ax[n_col].imshow(annot_section_plt)
                    static_ax[n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                                                    colors=['gainsboro'],
                                                    linewidths=0.2)

                if item == 'cells':
                    if color_dict[item]["single_color"]:
                            sns.scatterplot(ax=static_ax[n_col], x='ml_mm', y='dv_mm', data=plot_dict['cells'],
                                            color=color_dict[item]["geno_cmap"])
                    else:
                            sns.scatterplot(ax=static_ax[n_col], x='ml_mm', y='dv_mm', data=plot_dict['cells'],
                                            hue=plotting_params["groups"], palette=color_dict[item]["geno_cmap"])

                elif item == 'projections':
                    sns.histplot(ax=static_ax[n_col], data=plot_dict['projections'], x="xpixel", y="ypixel",
                                 cmap=color_dict[item]["geno_cmap"], binwidth=plotting_params['bin_width'],
                                 vmax=plotting_params['vmax'])

                elif item == 'injection_side':
                    if color_dict[item]['single_color']:
                        sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x="xpixel", y="ypixel", fill=True,
                                    color=color_dict[item]["geno_cmap"][0])
                else:
                    if color_dict[item]["single_color"]:
                        sns.scatterplot(ax=static_ax[n_col], x='ml_mm', y='dv_mm', data=plot_dict[item],
                                        color=color_dict[item]["geno_cmap"][0], s=20)
                        sns.regplot(ax=static_ax[n_col], x='ml_mm', y='dv_mm',
                                    data=plot_dict[item],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["geno_cmap"][0]),
                                    scatter=None, ci=None)
                    else:
                        sns.scatterplot(ax=static_ax[n_col], x='ml_mm', y='dv_mm', data=plot_dict[item],
                                        hue='channel', palette=color_dict[item]["geno_cmap"], s=20)

                        for i, c in enumerate(plot_dict[item]['channel'].unique()):
                            sns.regplot(ax=static_ax[n_col], x='ml_mm', y='dv_mm',
                                        data=plot_dict[item][plot_dict[item]['channel'] == c],
                                        line_kws=dict(alpha=0.7, color=color_dict[item]["geno_cmap"][i]),
                                        scatter=None, ci=None)

                static_ax[n_col].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[0]) * 0.01), 1)) + ' mm')
                static_ax[n_col].axis('off')
            else:
                if cnt < 1:
                    static_ax.imshow(annot_section_plt)
                    static_ax.contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                                                    colors=['gainsboro'],
                                                    linewidths=0.2)

                if item == 'cells':
                    if color_dict[item]["single_color"]:
                            sns.scatterplot(ax=static_ax, x='ml_mm', y='dv_mm', data=plot_dict[item],
                                            color=color_dict[item]["geno_cmap"])
                    else:
                            sns.scatterplot(ax=static_ax, x='ml_mm', y='dv_mm', data=plot_dict[item],
                                            hue=plotting_params["groups"], palette=color_dict[item]["geno_cmap"])

                elif item == 'projections':
                    sns.histplot(ax=static_ax, data=plot_dict[item], x="xpixel", y="ypixel",
                                 cmap=color_dict[item]["geno_cmap"], binwidth=plotting_params['bin_width'],
                                 vmax=plotting_params['vmax'])

                elif item == 'injection_side':
                    if color_dict[item]['single_color']:
                        sns.kdeplot(ax=static_ax, data=plot_dict[item], x="xpixel", y="ypixel", fill=True,
                                    color=color_dict[item]["geno_cmap"][0])

                else:
                    if color_dict[item]["single_color"]:
                        sns.scatterplot(ax=static_ax, x='ml_mm', y='dv_mm', data=plot_dict[item],
                                        color=color_dict[item]["geno_cmap"][0], s=20)
                        sns.regplot(ax=static_ax, x='ml_mm', y='dv_mm',
                                    data=plot_dict[item],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["geno_cmap"][0]),
                                    scatter=None, ci=None)
                    else:
                        sns.scatterplot(ax=static_ax, x='ml_mm', y='dv_mm', data=plot_dict[item],
                                        hue='channel', palette=color_dict[item]["geno_cmap"], s=20)

                        print(plot_dict)
                        print(color_dict)
                        for i, c in enumerate(plot_dict[item]['channel'].unique()):
                            sns.regplot(ax=static_ax, x='ml_mm', y='dv_mm',
                                        data=plot_dict[item][plot_dict[item]['channel'] == c],
                                        line_kws=dict(alpha=0.7, color=color_dict[item]["geno_cmap"][i]),
                                        scatter=None, ci=None)
                static_ax.title.set_text('bregma - ' + str(round((-(slice_idx - bregma[0]) * 0.01), 1)) + ' mm')
                static_ax.axis('off')

            cnt += 1
        if n_col < n_cols-1:
            n_col += 1
        else:
            n_col = 0
            n_row += 1
    if plotting_params["save_fig"]:
        mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
    return mpl_widget