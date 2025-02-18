import concurrent.futures
import math
import random
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'
import matplotlib.path as mpath
from shapely.geometry import LineString
from napari_dmc_brainmap.utils import split_to_list, load_group_dict, get_xyz
from napari_dmc_brainmap.visualization.visualization_tools import get_bregma, plot_brain_schematic, create_cmap, \
    brain_region_color_genes, get_voronoi_mask, calculate_density, calculate_heatmap, calculate_heatmap_difference


def get_brain_section_params(brainsec_widget):
    plotting_params = {
        "section_orient": brainsec_widget.section_orient.value,
        "plot_outline": brainsec_widget.plot_outline.value,
        "plot_item": brainsec_widget.plot_item.value,
        "hemisphere": brainsec_widget.hemisphere.value,
        "unilateral": brainsec_widget.unilateral.value,
        "brain_areas": split_to_list(brainsec_widget.brain_areas.value),
        "brain_areas_color": split_to_list(brainsec_widget.brain_areas_color.value),
        "brain_areas_transparency": split_to_list(brainsec_widget.brain_areas_transparency.value, out_format='int'),
        "color_brain_density": brainsec_widget.color_brain_density.value,
        "section_list": split_to_list(brainsec_widget.section_list.value, out_format='float'),
        "section_range": float(brainsec_widget.section_range.value),
        "groups": brainsec_widget.groups.value,
        "dot_size": int(brainsec_widget.dot_size.value),
        "color_cells_atlas": brainsec_widget.color_cells_atlas.value,
        "color_cells": split_to_list(brainsec_widget.color_cells.value),
        "show_cbar": brainsec_widget.show_cbar.value,
        "color_cells_density": split_to_list(brainsec_widget.cmap_cells.value),
        "bin_size_cells": int(brainsec_widget.bin_size_cells.value),
        "vmin_cells": int(brainsec_widget.vmin_cells.value),
        "vmax_cells": int(brainsec_widget.vmax_cells.value),
        "group_diff_cells": brainsec_widget.group_diff_cells.value,
        "group_diff_items_cells": brainsec_widget.group_diff_items_cells.value.split('-'),
        "color_projections": split_to_list(brainsec_widget.cmap_projection.value),
        "bin_size_proj": int(brainsec_widget.bin_size_cells.value),
        "vmin_proj": int(brainsec_widget.vmin_proj.value),
        "vmax_proj": int(brainsec_widget.vmax_proj.value),
        "group_diff_proj": brainsec_widget.group_diff_proj.value,
        "group_diff_items_proj": brainsec_widget.group_diff_items_proj.value.split('-'),
        # "smooth_proj": brainsec_widget.smooth_proj.value,
        # "smooth_thresh_proj": float(brainsec_widget.smooth_thresh_proj.value),
        "color_injection_site": split_to_list(brainsec_widget.color_inj.value),
        "color_optic_fiber": split_to_list(brainsec_widget.color_optic.value),
        "color_neuropixels_probe": split_to_list(brainsec_widget.color_npx.value),
        "plot_gene": brainsec_widget.plot_gene.value,
        "color_genes": split_to_list(brainsec_widget.color_genes.value),
        "gene": brainsec_widget.gene.value,
        "color_brain_genes": brainsec_widget.color_brain_genes.value,
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
        if item in ['cells', 'cells_density', 'projections', 'injection_site', 'genes']:
            if plotting_params["groups"] in ['genotype', 'group']:
                animal_dict = load_group_dict(input_path, animal_list, group_id=plotting_params["groups"])
                cmap = create_cmap(animal_dict, plotting_params, clr_id)
                single_color = False
            elif plotting_params["groups"] in ['channel', 'animal_id']:
                cmap = create_cmap([], plotting_params, clr_id, df=data_dict[item], hue_id=plotting_params["groups"])
                single_color = False
            elif item == 'genes':
                if plotting_params["plot_gene"] == 'clusters':
                    cmap = create_cmap([], plotting_params, clr_id, df=data_dict[item], hue_id='cluster_id')
                    single_color = False
                else:
                    if plotting_params['color_genes']:
                        cmap = plotting_params['color_genes']
                        if not cmap in plt.colormaps():
                            print(f'{cmap} not found in list of colormaps...')
                            cmap = random.choice(plt.colormaps())
                            print(f'... randomly selected {cmap} as colormap.')
                    else:
                        print('No colormap for gene expression data specified...')
                        cmap = random.choice(plt.colormaps())
                        print(f'... randomly selected {cmap} as colormap.')
                    single_color = True
            else:
                single_color = True
                if not plotting_params[clr_id]:
                    cmap = random.choice(list(mcolors.CSS4_COLORS.keys()))
                else:
                    cmap = plotting_params[clr_id][0]
                if item == 'projections':
                    try:
                        cmap = plt.get_cmap(cmap)
                    except ValueError:
                        if not '-' in cmap:
                            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap',
                                                                             ['white', cmap])
                        else:
                            cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap',
                                                                             [cmap.split('-')[0], cmap.split('-')[1]])

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

def plot_section(data_dict, color_dict, atlas, plotting_params, orient_mapping, bregma, section, section_range,
                 density):
    plot_dict = {}
    target_z = [section + section_range, section - section_range]
    target_z = [int(-(target / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
                for target in target_z]
    slice_idx = int(-(section / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
    if plotting_params['color_brain_genes'] in ['brain_areas', 'voronoi']:
        pass  # don't get brain section to plot if brain areas are colored according to clusters
    elif plotting_params['area_density']:
        annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx, orient_mapping['z_plot'][1],
                                                                         plotting_params, density=density)
    else:
        annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx, orient_mapping['z_plot'][1],
                                                                         plotting_params)
    for item in data_dict:
        plot_dict[item] = data_dict[item][(data_dict[item][orient_mapping['z_plot'][0]] >= target_z[0])
                                          & (data_dict[item][orient_mapping['z_plot'][0]] <= target_z[1])]
        if item == 'genes' and plotting_params['color_brain_genes'] in ['brain_areas', 'voronoi']:
            # calculate colors according to number of cluster_ids in brain regions
            if plotting_params['color_brain_genes'] == 'brain_areas':
                gene_color = brain_region_color_genes(plot_dict[item], color_dict[item]['cmap'], atlas,
                                                      plotting_params['plot_gene'])
                annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx,
                                                                                 orient_mapping['z_plot'][1],
                                                                                 plotting_params, gene_color=gene_color)
            else:
                voronoi = get_voronoi_mask(plot_dict[item], color_dict[item]['cmap'], atlas, plotting_params,
                                           orient_mapping)
                annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx,
                                                                                 orient_mapping['z_plot'][1],
                                                                                 plotting_params,
                                                                                 voronoi=voronoi)

        if plotting_params['unilateral'] in ['left', 'right'] and orient_mapping['z_plot'][1] < 2:
            if plotting_params['unilateral'] == 'left':
                # delete values on other hemisphere if still in dataset
                plot_dict[item] = plot_dict[item][
                    plot_dict[item]['ml_coords'] > bregma[atlas.space.axes_description.index('rl')]]
                # adjust ml_coords
                plot_dict[item].loc[:, 'ml_coords'] -= bregma[atlas.space.axes_description.index('rl')]
            else:
                # delete values on other hemisphere if still in dataset
                plot_dict[item] = plot_dict[item][
                    plot_dict[item]['ml_coords'] < bregma[atlas.space.axes_description.index('rl')]]
            plot_dict[item] = plot_dict[item].reset_index(drop=True)
        print(f"collected data for section at: {section} mm", flush=True)
    return (annot_section_plt, annot_section_contours, plot_dict, slice_idx)


def do_brain_section_plot(input_path, atlas, data_dict, animal_list, plotting_params, brain_section_widget, save_path):

    orient_mapping = get_orient_map(atlas, plotting_params)
    color_dict = create_color_dict(input_path, animal_list, data_dict, plotting_params)
    section_list = plotting_params["section_list"]  # in mm AP coordinates for coronal sections
    n_rows, n_cols = get_rows_cols(section_list)
    figsize = [int(i) for i in brain_section_widget.plot_size.value.split(',')]
    mpl_widget = FigureCanvas(Figure(figsize=figsize))
    static_ax = mpl_widget.figure.subplots(n_rows, n_cols)
    if len(section_list) == 1:
        static_ax = np.array([static_ax])
    static_ax = static_ax.ravel()
    section_range = plotting_params["section_range"]
    bregma = get_bregma(atlas.atlas_name)
    if plotting_params['color_brain_density']:
        if any([c in data_dict.keys() for c in ['cells', 'projections']]):
            plotting_params['area_density'] = [i for i in data_dict.keys() if i in ['cells', 'projections']][0]
            print(f"color brain areas according to {plotting_params['area_density']}")
            density = calculate_density(data_dict[plotting_params['area_density']],
                                        color_dict[plotting_params['area_density']], atlas, plotting_params)

        else:
            plotting_params['area_density'] = False
            density = False
    else:
        plotting_params['area_density'] = False
        density=False

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for section in section_list:
            futures.append(executor.submit(plot_section, data_dict, color_dict, atlas, plotting_params, orient_mapping,
                                           bregma, section, section_range, density))
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    print('...done')
    for s, (annot_section_plt, annot_section_contours, plot_dict, slice_idx) in enumerate(results):
        if not plot_dict:
            plot_dict = {'dummy'}  # plot only contours
            print("no plotting item selected, plotting only contours of brain section")
        static_ax[s].imshow(annot_section_plt)

        for item in plot_dict:
            if item == 'cells':
                if plotting_params['color_cells_atlas']:
                    palette = {}
                    for s in plot_dict[item].structure_id.unique():
                        palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])

                    sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'],
                                    y=orient_mapping['y_plot'], data=plot_dict[item],
                                    hue='structure_id', palette=palette,
                                    s=plotting_params["dot_size"], legend=False)
                else:
                    if color_dict[item]['single_color']:
                            sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                            color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                    else:
                            sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                            hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
            elif item == 'cells_density':
                if plot_dict[item].empty:
                    pass
                else:
                    bin_size = plotting_params['bin_size_cells']
                    x_dim = annot_section_plt.shape[1]
                    y_dim = annot_section_plt.shape[0]
                    x_bins = np.arange(0, x_dim + bin_size, bin_size)
                    y_bins = np.arange(0, y_dim + bin_size, bin_size)
                    if plotting_params['group_diff_cells'] == '':
                        heatmap_data, mask = calculate_heatmap(annot_section_plt, plot_dict[item], orient_mapping,
                                                               y_bins, x_bins, bin_size)
                    else:
                        heatmap_data, mask = calculate_heatmap_difference(annot_section_plt, plot_dict[item], plotting_params, orient_mapping,
                                                                    y_bins, x_bins, bin_size, 'group_diff_cells', 'group_diff_items_cells')

                    sns.heatmap(ax=static_ax[s], data=heatmap_data, mask=mask, cbar=plotting_params['show_cbar'], cbar_kws={'shrink': 0.5},
                                cmap=color_dict[item]["cmap"],
                                vmin=plotting_params['vmin_cells'], vmax=plotting_params['vmax_cells'],
                                )
            elif item == 'projections':
                if plot_dict[item].empty:
                    pass
                else:
                    bin_size = plotting_params['bin_size_proj']
                    x_dim = annot_section_plt.shape[1]
                    y_dim = annot_section_plt.shape[0]
                    x_bins = np.arange(0, x_dim + bin_size, bin_size)
                    y_bins = np.arange(0, y_dim + bin_size, bin_size)
                    if plotting_params['group_diff_proj'] == '':
                        heatmap_data, mask = calculate_heatmap(annot_section_plt, plot_dict[item], orient_mapping,
                                                               y_bins, x_bins, bin_size)
                    else:
                        heatmap_data, mask = calculate_heatmap_difference(annot_section_plt, plot_dict[item],
                                                                          plotting_params, orient_mapping,
                                                                          y_bins, x_bins, bin_size, 'group_diff_proj',
                                                                          'group_diff_items_proj')

                    sns.heatmap(ax=static_ax[s], data=heatmap_data, mask=mask, cbar=plotting_params['show_cbar'], cbar_kws={'shrink': 0.5},
                                cmap=color_dict[item]["cmap"],
                                vmin=plotting_params['vmin_proj'], vmax=plotting_params['vmax_proj'],
                                )

            elif item == 'injection_site':
                if color_dict[item]['single_color']:
                    sns.kdeplot(ax=static_ax[s], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                color=color_dict[item]["cmap"])
                else:
                    sns.kdeplot(ax=static_ax[s], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
                                hue=plotting_params["groups"], palette=color_dict[item]["cmap"])
            elif item in ['optic_fiber', 'neuropixels_probe']:
                if color_dict[item]["single_color"]:
                    # sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                    #                 color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                    sns.regplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                data=plot_dict[item],
                                line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
                                scatter=None, ci=None)
                else:
                    # sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                    #                 hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])

                    for c in plot_dict[item]['channel'].unique():
                        sns.regplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                                    data=plot_dict[item][plot_dict[item]['channel'] == c],
                                    line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
                                    scatter=None, ci=None)
            elif item == 'genes':
                if plotting_params['color_cells_atlas']:
                    palette = {}
                    for s in plot_dict[item].structure_id.unique():
                        palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])

                    sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'],
                                    y=orient_mapping['y_plot'], data=plot_dict[item],
                                    hue='structure_id', palette=palette,
                                    s=plotting_params["dot_size"], legend=False)
                else:
                    if plotting_params["plot_gene"] == 'clusters':
                        if color_dict[item]["single_color"]:
                                sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                                color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                        else:
                                sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
                                                hue="cluster_id", palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
                                               # edgecolors='lightgray')
                    else:
                        # sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
                        #                 data=plot_dict[item],
                        #                 hue="gene_expression_norm", palette=color_dict[item]["cmap"],
                        #                 s=plotting_params["dot_size"])
                        static_ax[s].scatter(plot_dict[item][orient_mapping['x_plot']],
                                    plot_dict[item][orient_mapping['y_plot']],
                                    c=plot_dict[item]['gene_expression_norm'], cmap=color_dict[item]["cmap"],
                                          vmin=0, vmax=1, s=plotting_params["dot_size"])
                        static_ax[s].collections[0].set_clim(0, 1)


        static_ax[s].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
        static_ax[s].axis('off')
    if plotting_params["save_fig"]:
        mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
    return mpl_widget


# def do_brain_section_plot(input_path, atlas, data_dict, animal_list, plotting_params, brain_section_widget, save_path):
#
#     orient_mapping = get_orient_map(atlas, plotting_params)
#     color_dict = create_color_dict(input_path, animal_list, data_dict, plotting_params)
#     section_list = plotting_params["section_list"]  # in mm AP coordinates for coronal sections
#     n_rows, n_cols = get_rows_cols(section_list)
#     figsize = [int(i) for i in brain_section_widget.plot_size.value.split(',')]
#     mpl_widget = FigureCanvas(Figure(figsize=figsize))
#     static_ax = mpl_widget.figure.subplots(n_rows, n_cols)
#     if len(section_list) == 1:
#         static_ax = np.array([static_ax])
#     static_ax = static_ax.ravel()
#     section_range = plotting_params["section_range"]
#     bregma = get_bregma(atlas.atlas_name)
#
#
#     for s, section in enumerate(section_list):
#
#         plot_dict = {}
#         target_z = [section + section_range, section - section_range]
#         target_z = [int(-(target / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
#                     for target in target_z]
#         slice_idx = int(-(section / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
#         if plotting_params['color_brain_genes'] in ['brain_areas', 'voronoi']:
#             pass  # don't get brain section to plot if brain areas are colored according to clusters
#         else:
#             annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx, orient_mapping['z_plot'][1], plotting_params)
#         for item in data_dict:
#             plot_dict[item] = data_dict[item][(data_dict[item][orient_mapping['z_plot'][0]] >= target_z[0])
#                                               & (data_dict[item][orient_mapping['z_plot'][0]] <= target_z[1])]
#             if item == 'genes' and plotting_params['color_brain_genes'] in ['brain_areas', 'voronoi']:
#                 # calculate colors according to number of cluster_ids in brain regions
#                 if plotting_params['color_brain_genes'] == 'brain_areas':
#                     gene_color = brain_region_color_genes(plot_dict[item], color_dict[item]['cmap'], atlas,
#                                                           plotting_params['plot_gene'])
#                     annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx,
#                                                                                      orient_mapping['z_plot'][1],
#                                                                                      plotting_params, gene_color=gene_color)
#                 else:
#                     voronoi = get_voronoi_mask(plot_dict[item], color_dict[item]['cmap'], atlas, plotting_params,
#                                                     orient_mapping)
#                     annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx,
#                                                                                      orient_mapping['z_plot'][1],
#                                                                                      plotting_params,
#                                                                                      voronoi=voronoi)
#
#             if plotting_params['unilateral'] in ['left', 'right'] and orient_mapping['z_plot'][1] < 2:
#                 if plotting_params['unilateral'] == 'left':
#                     # delete values on other hemisphere if still in dataset
#                     plot_dict[item] = plot_dict[item][plot_dict[item]['ml_coords'] > bregma[atlas.space.axes_description.index('rl')]]
#                     # adjust ml_coords
#                     plot_dict[item].loc[:,'ml_coords'] -= bregma[atlas.space.axes_description.index('rl')]
#                 else:
#                     # delete values on other hemisphere if still in dataset
#                     plot_dict[item] = plot_dict[item][
#                         plot_dict[item]['ml_coords'] < bregma[atlas.space.axes_description.index('rl')]]
#                 plot_dict[item] = plot_dict[item].reset_index(drop=True)
#
#
#         cnt = 0
#         if not plot_dict:
#             plot_dict = {'dummy'}  # plot only contours
#             print("no plotting item selected, plotting only contours of brain section")
#
#         for item in plot_dict:
#             if cnt < 1:
#                 static_ax[s].imshow(annot_section_plt)
#                 if annot_section_contours.any():
#                     static_ax[s].imshow(annot_section_contours)
#                 # if orient_mapping['z_plot'][1] == 0:
#                 #     static_ax[s].contour(annot[slice_idx, :, :],
#                 #                                     levels=np.unique(annot[slice_idx, :, :]),
#                 #                                     colors=['gainsboro'],
#                 #                                     linewidths=0.2)
#                 # elif orient_mapping['z_plot'][1] == 1:
#                 #     static_ax[s].contour(annot[:, slice_idx, :],
#                 #                                     levels=np.unique(annot[:, slice_idx, :]),
#                 #                                     colors=['gainsboro'],
#                 #                                     linewidths=0.2)
#                 # else:
#                 #     static_ax[s].contour(annot[:, :, slice_idx],
#                 #                                     levels=np.unique(annot[:, :, slice_idx]),
#                 #                                     colors=['gainsboro'],
#                 #                                     linewidths=0.2)
#
#             if item == 'cells':
#                 if plotting_params['color_cells_atlas']:
#                     palette = {}
#                     for s in plot_dict[item].structure_id.unique():
#                         palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])
#
#                     sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'],
#                                     y=orient_mapping['y_plot'], data=plot_dict[item],
#                                     hue='structure_id', palette=palette,
#                                     s=plotting_params["dot_size"], legend=False)
#                 else:
#                     if color_dict[item]['single_color']:
#                             sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#                                             color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#                     else:
#                             sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#                                             hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#
#             elif item == 'projections':
#                 if plot_dict[item].empty:
#                     pass
#                 else:
#                     if plotting_params['smooth_proj']:
#                         sns.kdeplot(ax=static_ax[s], data=plot_dict[item], x=orient_mapping['x_plot'],
#                                     y=orient_mapping['y_plot'],
#                                     cmap=color_dict[item]["cmap"],
#                                     thresh=plotting_params['smooth_thresh'], fill=True)
#                     else:
#                         sns.histplot(ax=static_ax[s], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#                                  cmap=color_dict[item]["cmap"], binwidth=plotting_params['bin_width'], vmin=plotting_params['vmin'],
#                                  vmax=plotting_params['vmax'])
#
#             elif item == 'injection_site':
#                 if color_dict[item]['single_color']:
#                     sns.kdeplot(ax=static_ax[s], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
#                                 color=color_dict[item]["cmap"])
#                 else:
#                     sns.kdeplot(ax=static_ax[s], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
#                                 hue=plotting_params["groups"], palette=color_dict[item]["cmap"])
#             elif item in ['optic_fiber', 'neuropixels_probe']:
#                 if color_dict[item]["single_color"]:
#                     # sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#                     #                 color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#                     sns.regplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#                                 data=plot_dict[item],
#                                 line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
#                                 scatter=None, ci=None)
#                 else:
#                     # sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#                     #                 hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#
#                     for c in plot_dict[item]['channel'].unique():
#                         sns.regplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#                                     data=plot_dict[item][plot_dict[item]['channel'] == c],
#                                     line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
#                                     scatter=None, ci=None)
#             elif item == 'genes':
#                 if plotting_params['color_cells_atlas']:
#                     palette = {}
#                     for s in plot_dict[item].structure_id.unique():
#                         palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])
#
#                     sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'],
#                                     y=orient_mapping['y_plot'], data=plot_dict[item],
#                                     hue='structure_id', palette=palette,
#                                     s=plotting_params["dot_size"], legend=False)
#                 else:
#                     if plotting_params["plot_gene"] == 'clusters':
#                         if color_dict[item]["single_color"]:
#                                 sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#                                                 color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#                         else:
#                                 sns.scatterplot(ax=static_ax[s], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#                                                 hue="cluster_id", palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#                                                # edgecolors='lightgray')
#                     else:
#                         # sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#                         #                 data=plot_dict[item],
#                         #                 hue="gene_expression_norm", palette=color_dict[item]["cmap"],
#                         #                 s=plotting_params["dot_size"])
#                         static_ax[s].scatter(plot_dict[item][orient_mapping['x_plot']],
#                                     plot_dict[item][orient_mapping['y_plot']],
#                                     c=plot_dict[item]['gene_expression_norm'], cmap=color_dict[item]["cmap"],
#                                           vmin=0, vmax=1, s=plotting_params["dot_size"])
#                         static_ax[s].collections[0].set_clim(0, 1)
#
#
#             static_ax[s].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
#             static_ax[s].axis('off')
#
#         #     elif len(section_list) == 2:
#         #         if cnt < 1:
#         #             static_ax[n_col].imshow(annot_section_plt)
#         #             if annot_section_contours.any():
#         #                 static_ax[n_col].imshow(annot_section_contours)
#         #             # if orient_mapping['z_plot'][1] == 0:
#         #             #     static_ax[n_col].contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
#         #             #                              colors=['gainsboro'],
#         #             #                              linewidths=0.2)
#         #             # elif orient_mapping['z_plot'][1] == 1:
#         #             #     static_ax[n_col].contour(annot[:, slice_idx, :], levels=np.unique(annot[:, slice_idx, :]),
#         #             #                              colors=['gainsboro'],
#         #             #                              linewidths=0.2)
#         #             # else:
#         #             #     static_ax[n_col].contour(annot[:, :, slice_idx], levels=np.unique(annot[:, :, slice_idx]),
#         #             #                              colors=['gainsboro'],
#         #             #                              linewidths=0.2)
#         #
#         #
#         #         if item == 'cells':
#         #             if plotting_params['color_cells_atlas']:
#         #                 palette = {}
#         #                 for s in plot_dict[item].structure_id.unique():
#         #                     palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])
#         #
#         #                 sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'],
#         #                                 y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                 hue='structure_id', palette=palette,
#         #                                 s=plotting_params["dot_size"], legend=False)
#         #             else:
#         #                 if color_dict[item]["single_color"]:
#         #                         sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict['cells'],
#         #                                         color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                 else:
#         #                         sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict['cells'],
#         #                                         hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #
#         #         elif item == 'projections':
#         #             if plot_dict[item].empty:
#         #                 pass
#         #             else:
#         #                 if plotting_params['smooth_proj']:
#         #                     sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'],
#         #                                 y=orient_mapping['y_plot'],
#         #                                 cmap=color_dict[item]["cmap"],
#         #                                 thresh=plotting_params['smooth_thresh'], fill=True)
#         #                 else:
#         #                     sns.histplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'],
#         #                                  y=orient_mapping['y_plot'], cmap=color_dict[item]["cmap"],
#         #                                  binwidth=plotting_params['bin_width'], vmin=plotting_params['vmin'],
#         #                                  vmax=plotting_params['vmax'])
#         #
#         #         elif item == 'injection_site':
#         #             if color_dict[item]['single_color']:
#         #                 sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
#         #                             color=color_dict[item]["cmap"])
#         #             else:
#         #                 sns.kdeplot(ax=static_ax[n_col], data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
#         #                             hue=plotting_params["groups"], palette=color_dict[item]["cmap"])
#         #         elif item in ['optic_fiber', 'neuropixels_probe']:
#         #             if color_dict[item]["single_color"]:
#         #                 # sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                 #                 color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                 sns.regplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                             data=plot_dict[item],
#         #                             line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
#         #                             scatter=None, ci=None)
#         #             else:
#         #                 # sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                 #                 hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #
#         #                 for c in plot_dict[item]['channel'].unique():
#         #                     sns.regplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                                 data=plot_dict[item][plot_dict[item]['channel'] == c],
#         #                                 line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
#         #                                 scatter=None, ci=None)
#         #         elif item == 'genes':
#         #             if plotting_params['color_cells_atlas']:
#         #                 palette = {}
#         #                 for s in plot_dict[item].structure_id.unique():
#         #                     palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])
#         #
#         #                 sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'],
#         #                                 y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                 hue='structure_id', palette=palette,
#         #                                 s=plotting_params["dot_size"], legend=False)
#         #             else:
#         #                 if plotting_params["plot_gene"] == 'clusters':
#         #                     if color_dict[item]["single_color"]:
#         #                             sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                             color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                     else:
#         #                             sns.scatterplot(ax=static_ax[n_col], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                             hue="cluster_id", palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                                            # edgecolors='lightgray')
#         #                 else:
#         #                     # sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                     #                 data=plot_dict[item],
#         #                     #                 hue="gene_expression_norm", palette=color_dict[item]["cmap"],
#         #                     #                 s=plotting_params["dot_size"])
#         #                     static_ax[n_col].scatter(plot_dict[item][orient_mapping['x_plot']],
#         #                                 plot_dict[item][orient_mapping['y_plot']],
#         #                                 c=plot_dict[item]['gene_expression_norm'], cmap=color_dict[item]["cmap"],
#         #                                       vmin=0, vmax=1, s=plotting_params["dot_size"])
#         #                     static_ax[n_col].collections[0].set_clim(0, 1)
#         #
#         #
#         #         static_ax[n_col].title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
#         #         static_ax[n_col].axis('off')
#         #     else:
#         #         if cnt < 1:
#         #             static_ax.imshow(annot_section_plt)
#         #             if annot_section_contours.any():
#         #                 static_ax.imshow(annot_section_contours)
#         #             # if orient_mapping['z_plot'][1] == 0:
#         #             #     static_ax.contour(annot[slice_idx, :, :], levels=np.unique(annot[slice_idx, :, :]),
#         #             #                       colors=['gainsboro'],
#         #             #                       linewidths=0.2)
#         #             # elif orient_mapping['z_plot'][1] == 1:
#         #             #     static_ax.contour(annot[:, slice_idx, :], levels=np.unique(annot[:, slice_idx, :]),
#         #             #                       colors=['gainsboro'],
#         #             #                       linewidths=0.2)
#         #             # else:
#         #             #     static_ax.contour(annot[:, :, slice_idx], levels=np.unique(annot[:, :, slice_idx]),
#         #             #                       colors=['gainsboro'],
#         #             #                       linewidths=0.2)
#         #
#         #         if item == 'cells':
#         #             if plotting_params['color_cells_atlas']:
#         #                 palette = {}
#         #                 for s in plot_dict[item].structure_id.unique():
#         #                     palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])
#         #
#         #                 sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'],
#         #                                 y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                 hue='structure_id', palette=palette,
#         #                                 s=plotting_params["dot_size"], legend=False)
#         #             else:
#         #                 if color_dict[item]["single_color"]:
#         #                         sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                         color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                 else:
#         #                         sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                         hue=plotting_params["groups"], palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #
#         #         elif item == 'projections':
#         #             if plot_dict[item].empty:
#         #                 pass
#         #             else:
#         #                 if plotting_params['smooth_proj']:
#         #                     sns.kdeplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'],
#         #                                 y=orient_mapping['y_plot'],
#         #                                 cmap=color_dict[item]["cmap"],
#         #                                 thresh=plotting_params['smooth_thresh'], fill=True)
#         #                 else:
#         #                     sns.histplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                              cmap=color_dict[item]["cmap"], binwidth=plotting_params['bin_width'],
#         #                              vmax=plotting_params['vmax'], vmin=plotting_params['vmin'])
#         #
#         #         elif item == 'injection_site':
#         #             if color_dict[item]['single_color']:
#         #                 sns.kdeplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
#         #                             color=color_dict[item]["cmap"])
#         #             else:
#         #                 sns.kdeplot(ax=static_ax, data=plot_dict[item], x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], fill=True,
#         #                             hue=plotting_params["groups"], palette=color_dict[item]["cmap"])
#         #
#         #         elif item in ['optic_fiber', 'neuropixels_probe']:
#         #             if color_dict[item]["single_color"]:
#         #                 # sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                 #                 color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                 sns.regplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                             data=plot_dict[item],
#         #                             line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"]),
#         #                             scatter=None, ci=None)
#         #             else:
#         #                 # sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                 #                 hue='channel', palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #
#         #                 for c in plot_dict[item]['channel'].unique():
#         #                     sns.regplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                                 data=plot_dict[item][plot_dict[item]['channel'] == c],
#         #                                 line_kws=dict(alpha=0.7, color=color_dict[item]["cmap"][c]),
#         #                                 scatter=None, ci=None)
#         #         elif item == 'genes':
#         #             if plotting_params['color_cells_atlas']:
#         #                 palette = {}
#         #                 for s in plot_dict[item].structure_id.unique():
#         #                     palette[s] = tuple([c / 255 for c in atlas.structures[s]['rgb_triplet']])
#         #
#         #                 sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'],
#         #                                 y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                 hue='structure_id', palette=palette,
#         #                                 s=plotting_params["dot_size"], legend=False)
#         #             else:
#         #                 if plotting_params["plot_gene"] == 'clusters':
#         #                     if color_dict[item]["single_color"]:
#         #                             sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                             color=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                     else:
#         #                             sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'], data=plot_dict[item],
#         #                                             hue="cluster_id", palette=color_dict[item]["cmap"], s=plotting_params["dot_size"])
#         #                                            # edgecolors='lightgray')
#         #                 else:
#         #                     # sns.scatterplot(ax=static_ax, x=orient_mapping['x_plot'], y=orient_mapping['y_plot'],
#         #                     #                 data=plot_dict[item],
#         #                     #                 hue="gene_expression_norm", palette=color_dict[item]["cmap"],
#         #                     #                 s=plotting_params["dot_size"])
#         #                     static_ax.scatter(plot_dict[item][orient_mapping['x_plot']],
#         #                                 plot_dict[item][orient_mapping['y_plot']],
#         #                                 c=plot_dict[item]['gene_expression_norm'], cmap=color_dict[item]["cmap"],
#         #                                       vmin=0, vmax=1, s=plotting_params["dot_size"])
#         #                     static_ax.collections[0].set_clim(0, 1)
#         #         static_ax.title.set_text('bregma - ' + str(round((-(slice_idx - bregma[orient_mapping['z_plot'][1]]) * orient_mapping['z_plot'][2]), 1)) + ' mm')
#         #         static_ax.axis('off')
#         #
#         #     cnt += 1
#         # if n_col < n_cols-1:
#         #     n_col += 1
#         # else:
#         #     n_col = 0
#         #     n_row += 1
#         print(f"collected data for section at: {section} mm")
#     if plotting_params["save_fig"]:
#         mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
#     return mpl_widget


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