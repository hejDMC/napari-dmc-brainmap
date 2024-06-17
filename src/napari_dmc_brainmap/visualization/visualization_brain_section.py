import math
import random
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from napari_dmc_brainmap.utils import split_to_list, load_group_dict, get_xyz
from napari_dmc_brainmap.visualization.visualization_tools import get_bregma, plot_brain_schematic, create_cmap


def get_brain_section_params(brainsec_widget):
    plotting_params = {
        "section_orient": brainsec_widget.section_orient.value,
        "plot_outline": brainsec_widget.plot_outline.value,
        "plot_item": brainsec_widget.plot_item.value,
        "hemisphere": brainsec_widget.hemisphere.value,
        "brain_areas": split_to_list(brainsec_widget.brain_areas.value),
        "brain_areas_color": split_to_list(brainsec_widget.brain_areas_color.value),
        "brain_areas_transparency": split_to_list(brainsec_widget.brain_areas_transparency.value, out_format='int'),
        "section_list": split_to_list(brainsec_widget.section_list.value, out_format='float'),
        "section_range": float(brainsec_widget.section_range.value),
        "groups": brainsec_widget.groups.value,
        "dot_size": int(brainsec_widget.dot_size.value),
        "bin_width": int(brainsec_widget.bin_width.value),
        "vmin": int(brainsec_widget.vmin.value),
        "vmax": int(brainsec_widget.vmax.value),
        "smooth_proj": brainsec_widget.smooth_proj.value,
        "smooth_thresh": float(brainsec_widget.smooth_thresh.value),
        "color_cells_atlas": brainsec_widget.color_cells_atlas.value,
        "color_cells": split_to_list(brainsec_widget.color_cells.value),
        "color_projections": split_to_list(brainsec_widget.cmap_projection.value),
        "color_injection_side": split_to_list(brainsec_widget.color_inj.value),
        "color_optic_fiber": split_to_list(brainsec_widget.color_optic.value),
        "color_neuropixels_probe": split_to_list(brainsec_widget.color_npx.value),
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



def create_color_dict(input_path, animal_list, data_dict, plotting_params):

    color_dict = {}
    for item in plotting_params['plot_item']:
        color_dict[item] = {}
        clr_id = 'color_' + item
        if item in ['cells', 'projections', 'injection_site']:
            if plotting_params["groups"] in ['genotype', 'group']:
                animal_dict = load_group_dict(input_path, animal_list, group_id=plotting_params["groups"])
                cmap = create_cmap(animal_dict, plotting_params, clr_id)
                single_color = False
            elif plotting_params["groups"] in ['channel', 'animal_id']:
                cmap = create_cmap([], plotting_params, clr_id, df=data_dict[item], hue_id=plotting_params["groups"])
                single_color = False
            else:
                single_color = True
                if not plotting_params[clr_id]:
                    cmap = random.choice(list(mcolors.CSS4_COLORS.keys()))
                else:
                    cmap = plotting_params[clr_id][0]
        else:
            num_probe = len(data_dict[item]['channel'].unique())
            if num_probe == 1:
                single_color = True
                if not plotting_params[clr_id]:
                    cmap = random.choice(list(mcolors.CSS4_COLORS.keys()))
                else:
                    cmap = plotting_params[clr_id][0]
            else:
                single_color = False
                cmap = create_cmap([], plotting_params, clr_id, df=data_dict[item])

        color_dict[item]['cmap'] = cmap
        color_dict[item]['single_color'] = single_color
    return color_dict

def get_orient_map(atlas, plotting_params):
    orient_dict = {
        'ap': 'ap_coords',
        'rl': 'ml_coords',
        'si': 'dv_coords'
    }
    xyz_dict = get_xyz(atlas, plotting_params['section_orient'])

    orient_mapping = {
        'z_plot': [orient_dict[xyz_dict['z'][0]], atlas.space.axes_description.index(xyz_dict['z'][0]), xyz_dict['z'][2]/1000],
        'x_plot': orient_dict[xyz_dict['x'][0]],
        'y_plot': orient_dict[xyz_dict['y'][0]]
    }
    return orient_mapping

def do_brain_section_plot(input_path, atlas, data_dict, animal_list, plotting_params, brain_section_widget, save_path):

    orient_mapping = get_orient_map(atlas, plotting_params)

    color_dict = create_color_dict(input_path, animal_list, data_dict, plotting_params)
    section_list = plotting_params["section_list"]  # in mm AP coordinates for coronal sections
    n_rows, n_cols = get_rows_cols(section_list)
    figsize = [int(i) for i in brain_section_widget.plot_size.value.split(',')]
    mpl_widget = FigureCanvas(Figure(figsize=figsize))
    static_ax = mpl_widget.figure.subplots(n_rows, n_cols)
    n_row = 0
    n_col = 0
    section_range = plotting_params["section_range"]
    bregma = get_bregma(atlas.atlas_name)
    # annot = atlas.annotation
    #for item in data_dict:
    #    data_dict[item] = coord_mm_transform(data_dict[item], bregma)

    for section in section_list:

        plot_dict = {}
        target_z = [section + section_range, section - section_range]
        target_z = [int(-(target / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
                    for target in target_z]
        slice_idx = int(-(section / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
        annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx, orient_mapping['z_plot'][1], plotting_params)
        for item in data_dict:
            plot_dict[item] = data_dict[item][(data_dict[item][orient_mapping['z_plot'][0]] >= target_z[0])
                                              & (data_dict[item][orient_mapping['z_plot'][0]] <= target_z[1])]


        cnt = 0
        if not plot_dict:
            plot_dict = {'dummy'}  # plot only contours
            print("no plotting item selected, plotting only contours of brain section")

        for item in plot_dict:
            if len(section_list) > 2:
                if cnt < 1:
                    static_ax[n_row, n_col].imshow(annot_section_plt)
                    if annot_section_contours.any():
                        static_ax[n_row, n_col].imshow(annot_section_contours)
                    # if orient_mapping['z_plot'][1] == 0:
                    #     static_ax[n_row, n_col].contour(annot[slice_idx, :, :],
                    #                                     levels=np.unique(annot[slice_idx, :, :]),
                    #                                     colors=['gainsboro'],
                    #                                     linewidths=0.2)
                    # elif orient_mapping['z_plot'][1] == 1:
                    #     static_ax[n_row, n_col].contour(annot[:, slice_idx, :],
                    #                                     levels=np.unique(annot[:, slice_idx, :]),
                    #                                     colors=['gainsboro'],
                    #                                     linewidths=0.2)
                    # else:
                    #     static_ax[n_row, n_col].contour(annot[:, :, slice_idx],
                    #                                     levels=np.unique(annot[:, :, slice_idx]),
                    #                                     colors=['gainsboro'],
                    #                                     linewidths=0.2)



                if item == 'cells':
                    if plotting_params['color_cells_atlas']:
                        palette = {}
                        for s in plot_dict[item].structure_id.unique():
                            palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])

                        sns.scatterplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'],
                                        y=orient_mapping['y_plot'], data=plot_dict[item],
                                        hue='structure_id', palette=palette,
                                        s=plotting_params["dot_size"], legend=False)
                    else:
                        if color_dict[item]['single_color']:
                                sns.scatterplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                                color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        else:
                                sns.scatterplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                                hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                elif item == 'projections':
                    if plot_dict[item].empty:
                        pass
                    else:
                        if plotting_params['smooth_proj']:
                            sns.kdeplot(ax=static_ax[n_row, n_col], data=plot_dict[item], x=orient_mapping['x_plot'],
                                        y=orient_mapping['y_plot'],
                                        cmap=color_dict[item]["cmap"],
                                        thresh=plotting_params['smooth_thresh'], fill=True)
                        else:
                            sns.histplot(ax=static_ax[n_row, n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                     cmap=color_dict[item]["cmap"], binwidth=plotting_params['bin_width'], vmin=plotting_params['vmin'],
                                     vmax=plotting_params['vmax'])

                elif item == 'injection_site':
                    if color_dict[item]['single_color']:
                        sns.kdeplot(ax=static_ax[n_row, n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                    color=color_dict[item]["cmap"])
                    else:
                        sns.kdeplot(ax=static_ax[n_row, n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                    hue=plotting_params["groups"], palette=color_dict[item]["cmap"])
                elif item in ['optic_fiber', 'neuropixels_probe']:
                    if color_dict[item]["single_color"]:
                        sns.scatterplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                        color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        sns.regplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                    data=plot_dict[item],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
                                    scatter=None, ci=None)
                    else:
                        sns.scatterplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                        hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                        for c in plot_dict[item]['channel'].unique():
                            sns.regplot(ax=static_ax[n_row, n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                        data=plot_dict[item][plot_dict[item]['channel'] == c],
                                        line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
                                        scatter=None, ci=None)

                static_ax[n_row, n_col].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
                static_ax[n_row, n_col].axis('off')

            elif len(section_list) == 2:
                if cnt < 1:
                    static_ax[n_col].imshow(annot_section_plt)
                    if annot_section_contours.any():
                        static_ax[n_col].imshow(annot_section_contours)
                    # if orient_mapping['z_plot'][1] == 0:
                    #     static_ax[n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                    #                              colors=['gainsboro'],
                    #                              linewidths=0.2)
                    # elif orient_mapping['z_plot'][1] == 1:
                    #     static_ax[n_col].contour(annot[:, slice_idx, :], levels=np.unique(annot[:, slice_idx, :]),
                    #                              colors=['gainsboro'],
                    #                              linewidths=0.2)
                    # else:
                    #     static_ax[n_col].contour(annot[:, :, slice_idx], levels=np.unique(annot[:, :, slice_idx]),
                    #                              colors=['gainsboro'],
                    #                              linewidths=0.2)


                if item == 'cells':
                    if plotting_params['color_cells_atlas']:
                        palette = {}
                        for s in plot_dict[item].structure_id.unique():
                            palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])

                        sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'],
                                        y=orient_mapping['y_plot'], data=plot_dict[item],
                                        hue='structure_id', palette=palette,
                                        s=plotting_params["dot_size"], legend=False)
                    else:
                        if color_dict[item]["single_color"]:
                                sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict['cells'],
                                                color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        else:
                                sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict['cells'],
                                                hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                elif item == 'projections':
                    if plot_dict[item].empty:
                        pass
                    else:
                        if plotting_params['smooth_proj']:
                            sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'],
                                        y=orient_mapping['y_plot'],
                                        cmap=color_dict[item]["cmap"],
                                        thresh=plotting_params['smooth_thresh'], fill=True)
                        else:
                            sns.histplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'],
                                         y=orient_mapping['y_plot'], cmap=color_dict[item]["cmap"],
                                         binwidth=plotting_params['bin_width'], vmin=plotting_params['vmin'],
                                         vmax=plotting_params['vmax'])

                elif item == 'injection_site':
                    if color_dict[item]['single_color']:
                        sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                    color=color_dict[item]["cmap"])
                    else:
                        sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                    hue=plotting_params["groups"], palette=color_dict[item]["cmap"])
                elif item in ['optic_fiber', 'neuropixels_probe']:
                    if color_dict[item]["single_color"]:
                        sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                        color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        sns.regplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                    data=plot_dict[item],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
                                    scatter=None, ci=None)
                    else:
                        sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                        hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                        for c in plot_dict[item]['channel'].unique():
                            sns.regplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                        data=plot_dict[item][plot_dict[item]['channel'] == c],
                                        line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
                                        scatter=None, ci=None)

                static_ax[n_col].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
                static_ax[n_col].axis('off')
            else:
                if cnt < 1:
                    static_ax.imshow(annot_section_plt)
                    if annot_section_contours.any():
                        static_ax.imshow(annot_section_contours)
                    # if orient_mapping['z_plot'][1] == 0:
                    #     static_ax.contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
                    #                       colors=['gainsboro'],
                    #                       linewidths=0.2)
                    # elif orient_mapping['z_plot'][1] == 1:
                    #     static_ax.contour(annot[:, slice_idx, :], levels=np.unique(annot[:, slice_idx, :]),
                    #                       colors=['gainsboro'],
                    #                       linewidths=0.2)
                    # else:
                    #     static_ax.contour(annot[:, :, slice_idx], levels=np.unique(annot[:, :, slice_idx]),
                    #                       colors=['gainsboro'],
                    #                       linewidths=0.2)

                if item == 'cells':
                    if plotting_params['color_cells_atlas']:
                        palette = {}
                        for s in plot_dict[item].structure_id.unique():
                            palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])

                        sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'],
                                        y=orient_mapping['y_plot'], data=plot_dict[item],
                                        hue='structure_id', palette=palette,
                                        s=plotting_params["dot_size"], legend=False)
                    else:
                        if color_dict[item]["single_color"]:
                                sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                                color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        else:
                                sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                                hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                elif item == 'projections':
                    if plot_dict[item].empty:
                        pass
                    else:
                        if plotting_params['smooth_proj']:
                            sns.kdeplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'],
                                        y=orient_mapping['y_plot'],
                                        cmap=color_dict[item]["cmap"],
                                        thresh=plotting_params['smooth_thresh'], fill=True)
                        else:
                            sns.histplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                     cmap=color_dict[item]["cmap"], binwidth=plotting_params['bin_width'],
                                     vmax=plotting_params['vmax'], vmin=plotting_params['vmin'])

                elif item == 'injection_site':
                    if color_dict[item]['single_color']:
                        sns.kdeplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                    color=color_dict[item]["cmap"])
                    else:
                        sns.kdeplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                    hue=plotting_params["groups"], palette=color_dict[item]["cmap"])

                elif item in ['optic_fiber', 'neuropixels_probe']:
                    if color_dict[item]["single_color"]:
                        sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                        color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        sns.regplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                    data=plot_dict[item],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
                                    scatter=None, ci=None)
                    else:
                        sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                        hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                        for c in plot_dict[item]['channel'].unique():
                            sns.regplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                        data=plot_dict[item][plot_dict[item]['channel'] == c],
                                        line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
                                        scatter=None, ci=None)
                static_ax.title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
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


# todo started with some stuff to merge line of brain section plot, but removing elements doesn't work..plus the child function is making things quite big, so prob just appending the g_element
# import xml.etree.ElementTree as ET
#
# def merge_layers(svg_file_path):
#     tree = ET.parse(svg_file_path)
#     root = tree.getroot()
#
#     # Find all <g> elements
#     g_elements = root.findall('.//{http://www.w3.org/2000/svg}g')
#
#     if len(g_elements) > 1:
#         # Create a new <g> element for the merged layers
#         merged_g = ET.Element('{http://www.w3.org/2000/svg}g')
#
#         # Iterate through each <g> element and move its children to the merged layer
#         elements_to_remove = []
#         for g_element in g_elements:
#             try:
#                 if g_element.get('id').startswith('LineCollection'):
#                     for child in g_element:
#                         merged_g.append(child)
#                     elements_to_remove.append(g_element)
#             except AttributeError:
#                 pass
#
#         # Create a list of elements to remove
#         # elements_to_remove = list(g_elements)
#
#         # Remove the existing <g> elements
#         for element in elements_to_remove:
#             root.remove(element)
#
#         # Add the merged <g> element to the root
#         root.append(merged_g)
#
#     # Save the modified SVG file
#     tree.write(svg_file_path, xml_declaration=True)
#
# # Example usage
# svg_file_path = '/home/felix/Desktop/BM-001/plots/test.svg'
# merge_layers(svg_file_path)