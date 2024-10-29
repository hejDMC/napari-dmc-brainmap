import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'

from napari_dmc_brainmap.utils import split_to_list, split_strings_layers
from napari_dmc_brainmap.visualization.visualization_tools import get_tgt_data_only, resort_df, get_bregma, \
    get_unique_filename, get_descendants

def calculate_percentage_heatmap_plot(df_all, atlas, plotting_params, animal_list, tgt_list, sub_list):

    df = get_tgt_data_only(df_all, atlas, tgt_list, use_na=True)
    interval_labels = plotting_params["interval_labels"]
    intervals = plotting_params["intervals"]
    df['bin'] = pd.cut(df['ap_mm'], intervals, labels=interval_labels)

    absolute_numbers = plotting_params["absolute_numbers"]  # use absolute numbers

    if sub_list:  # sub data to calculate relative count for
        sub_data = get_tgt_data_only(df_all, atlas, sub_list)
        absolute_numbers = False
        rel_percentage = False
        print("sub list for normalization provided, overriding input for percentage/absolute numbers")
    if plotting_params['expression']:
        if plotting_params["descendants"]:
            df = df.pivot_table(values=plotting_params['gene'], index='bin', columns='acronym', aggfunc='mean').fillna(0)

        else:
            df = df.pivot_table(values=plotting_params['gene'], index='bin', columns='tgt_name', aggfunc='mean').fillna(0)
        df = df.transpose()  # this is just stupid
    else:
        if plotting_params["descendants"]:
            df = df.pivot_table(index='acronym', columns=['animal_id', 'bin'],
                                                aggfunc='count').fillna(0)
        else:
            df = df.pivot_table(index='tgt_name', columns=['animal_id', 'bin'],
                                aggfunc='count').fillna(0)
    # add "missing" brain structures -- brain structures w/o cells
    if len(df.index.values.tolist()) > 0:
        miss_areas = list(set(df.index.values.tolist()) ^ set(tgt_list))
    else:
        miss_areas = tgt_list
    if len(miss_areas) > 0:  # todo this fix does not work yet, if all areas are missing --> no column names
        # create df with zeros and miss areas as rows and columsn as df
        dd = pd.DataFrame(0, index=miss_areas, columns=df.columns.values)
        # concat dataframes
        df = pd.concat([df, dd])
    # sort indices according to tgt_list
    df = resort_df(df, tgt_list, index_sort=True)
    if plotting_params["expression"]:
        return df

    # normalize numbers for each animal respectively
    df_plot = pd.DataFrame(np.zeros((len(tgt_list), len(interval_labels))), index=tgt_list, columns=interval_labels)
    for animal_id in animal_list:
        # todo here can be the bug by dividing by zero!
        if absolute_numbers == 'absolute':
            data = df['ap_coords'][animal_id]
        elif absolute_numbers == 'percentage_selection':
            data = (df['ap_coords'][animal_id] / df['ap_coords'][animal_id].sum().sum()) * 100
        elif sub_list:
            data = (df['ap_coords'][animal_id] / len(sub_data[sub_data['animal_id']==animal_id])) * 100
        else:  # percentage relative to all cells in animal
            data = (df['ap_coords'][animal_id] / len(df_all[df_all['animal_id'] == animal_id])) * 100
        data = data.fillna(0)
        df_plot += data
    # calculate average
    df_plot /= len(animal_list)
    return df_plot

def get_interval_labels(intervals):
    intervals = sorted(intervals)  # assure ascending order
    interval_labels = []
    for i in range(len(intervals)):
        if i < len(intervals)-1:
            interval_labels.append(str(intervals[i]) + " to " + str(intervals[i+1]))
    return interval_labels

# def get_layer_list(tgt_list, atlas):
#
#     # for now only for isocortical areas, not sure about olfactory areas
#     new_tgt_list = []
#     for tgt in tgt_list:
#         if 'Isocortex' in atlas.get_structure_ancestors(tgt):
#             new_tgt_list.append(atlas.get_structure_descendants(tgt))
#         else:
#             new_tgt_list.append(tgt)
#     new_tgt_list = [l for sublist in new_tgt_list for l in sublist]  # flatten list
#     #     tgt_path = st[st['acronym'] == tgt]['structure_id_path'].to_list()[0]
#     #     try:
#     #         new_tgt_list += st[st['structure_id_path'].str.contains(tgt_path)]['acronym'].to_list()[1:]  # ignore first entry -- this is parent
#     #     except IndexError:
#     #         pass
#     # # for some cortical regions, a layer 2 and layer 2/3 exist --> ignore the layer 2
#     # new_tgt_list = [t for t in new_tgt_list if not split_strings_layers(t)[1] == '2']
#     return new_tgt_list

def check_brain_area_in_bin(df, atlas):
    # for heatmap plotting, check if brain area exists in bin
    print('checking for existence of brain area in bin ...')
    bregma = get_bregma(atlas.atlas_name)
    ap_idx = atlas.space.axes_description.index('ap')
    annot = atlas.annotation
    for bin in df.columns:
        print(bin)
        b_start, b_end = bin.split(' to ')
        b_start_coord = int(-(float(b_start)/0.01 - bregma[ap_idx]))
        b_end_coord = int(-(float(b_end) / 0.01 - bregma[ap_idx]))
        if b_start_coord > b_end_coord:
            b0 = b_end_coord
            b1 = b_start_coord
        else:
            b0 = b_start_coord
            b1 = b_end_coord
        if ap_idx == 0:
            ids_in_bin = np.unique(annot[b0:b1, :, :])
        elif ap_idx == 1:
            ids_in_bin = np.unique(annot[:, b0:b1, :])
        else:
            ids_in_bin = np.unique(annot[:, :, b0:b1])

        area_list = df.index
        for area in area_list:
            area_descendants = get_descendants([area], atlas)
            area_ids = [atlas.structures[a]['id'] for a in area_descendants]
            if not any(area_id in ids_in_bin for area_id in area_ids):
                df[bin][area] = np.nan
    return df

def get_heatmap_params(heatmap_widget):
    plotting_params = {
        "group_diff": heatmap_widget.group_diff.value,
        "group_diff_items": heatmap_widget.group_diff_items.value.split('-'),
        "expression": heatmap_widget.expression.value,
        "gene": heatmap_widget.gene.value,
        # "xlabel": [heatmap_widget.xlabel.value, int(heatmap_widget.xlabel_size.value)],  # 0: label, 1: fontsize
        "ylabel": [heatmap_widget.ylabel.value, int(heatmap_widget.ylabel_size.value)],
        "tick_size": [int(heatmap_widget.xticklabel_size.value), int(heatmap_widget.yticklabel_size.value)],
        # "rotate_xticks": int(heatmap_widget.rotate_xticks.value),  # set to False of no rotation
        # "title": [heatmap_widget.title.value, int(heatmap_widget.title_size.value)],
        "subtitle_size": int(heatmap_widget.subtitle_size.value),
        "style": heatmap_widget.style.value,
        "color": heatmap_widget.color.value,
        "cmap": split_to_list(heatmap_widget.cmap.value),
        "cbar_label": heatmap_widget.cbar_label.value,
        "cmap_min_max":  split_to_list(heatmap_widget.cmap_min_max.value, out_format='float'),
        "intervals": sorted(split_to_list(heatmap_widget.intervals.value, out_format='float')),  # assure ascending order
        "interval_labels": get_interval_labels(split_to_list(heatmap_widget.intervals.value, out_format='float')),
        "descendants": heatmap_widget.descendants.value,
        "transpose": heatmap_widget.transpose.value,
        "save_name": heatmap_widget.save_name.value,
        "save_fig": heatmap_widget.save_fig.value,
        "save_data_name": heatmap_widget.save_data_name.value,
        "save_data": heatmap_widget.save_data.value,
        "absolute_numbers": heatmap_widget.absolute_numbers.value
    }
    return plotting_params

def do_heatmap(df, atlas, animal_list, tgt_list, plotting_params, heatmap_widget, save_path, sub_list=False):

    if plotting_params["descendants"]:
        new_tgt_list = get_descendants(tgt_list, atlas)
    else:
        new_tgt_list = tgt_list
    if plotting_params['group_diff'] == '':
        tgt_data_to_plot = calculate_percentage_heatmap_plot(df, atlas, plotting_params, animal_list, new_tgt_list, sub_list)
        # put not existing areas to -1 for plotting
        tgt_data_to_plot = check_brain_area_in_bin(tgt_data_to_plot, atlas)
    else:
        group_list = df[plotting_params['group_diff']].unique()
        # check that items to calculate difference from exist
        if all([i in group_list for i in plotting_params['group_diff_items']]):
            diff_data = []
            for i_d in plotting_params['group_diff_items']:
                animal_sub_list = df[df[plotting_params['group_diff']] == i_d]['animal_id'].unique()
                sub_data_to_plot = calculate_percentage_heatmap_plot(df[df[plotting_params['group_diff']] == i_d],
                                                                     atlas, plotting_params, animal_sub_list,
                                                                     new_tgt_list, sub_list)
                sub_data_to_plot = check_brain_area_in_bin(sub_data_to_plot, atlas)
                diff_data.append(sub_data_to_plot)
            tgt_data_to_plot = diff_data[0] - diff_data[1]

        else:
            print(f"selected items to calculate difference not found: {plotting_params['group_diff_items']}  \n"
                  f"check if items exists, also check params file if items are stated \n"
                  f"--> plotting regular heatmap")
            tgt_data_to_plot = calculate_percentage_heatmap_plot(df, atlas, plotting_params, animal_list, new_tgt_list,
                                                                 sub_list)
            # put not existing areas to -1 for plotting
            tgt_data_to_plot = check_brain_area_in_bin(tgt_data_to_plot, atlas)


    if plotting_params["transpose"]:
        tgt_data_to_plot = tgt_data_to_plot.transpose()
    if plotting_params["save_data"]:
        data_fn = save_path.joinpath(plotting_params["save_data_name"])
        if data_fn.exists():
            data_fn = get_unique_filename(data_fn)
        tgt_data_to_plot.to_csv(data_fn)
    sns.set(style=plotting_params["style"])
    if plotting_params["style"] == 'white':
        mask_cbar = 'binary'
    else:
        mask_cbar = 'binary_r'
    if plotting_params["cmap_min_max"] == 'auto':
        max_range = tgt_data_to_plot.max().max()
        if plotting_params['group_diff'] != '':
            min_range = tgt_data_to_plot.min().min()
            vmin = min_range * 0.75
        else:
            vmin = -1
        vmax = max_range * 0.75
    else:
        vmin = plotting_params["cmap_min_max"][0]
        vmax = plotting_params["cmap_min_max"][1]
    cmap = plotting_params["cmap"]
    figsize = [int(i) for i in heatmap_widget.plot_size.value.split(',')]
    mpl_widget = FigureCanvas(Figure(figsize=figsize))
    static_ax = mpl_widget.figure.subplots(1, (len(tgt_list) + 1),
                                           gridspec_kw={'width_ratios': [1] * len(tgt_list) + [0.15]}) #, figsize=figsize)
    #for a in range(len(static_ax)):
     #   static_ax[0].get_shared_y_axes().join(static_ax[a])
    for t, tgt in enumerate(tgt_list):
        if plotting_params["descendants"]:
            tgt_col = get_descendants([tgt], atlas)
            i_start = tgt_data_to_plot.columns.get_loc(tgt_col[0])
            i_end = tgt_data_to_plot.columns.get_loc(tgt_col[-1])
            sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end+1], cmap=cmap, cbar=False, vmin=vmin,
                        vmax=vmax, linewidths=1)
            # sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end + 1], cmap=mask_cbar, vmin=vmin,
            #             vmax=vmax, mask=tgt_data_to_plot.iloc[:, i_start:i_end + 1] > -1, cbar=False)
            if t + 1 == len(tgt_list):
                sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end+1], cmap=cmap, vmin=vmin,
                            vmax=vmax, linewidths=1, cbar_ax=static_ax[t + 1], cbar_kws={'label': plotting_params["cbar_label"]})
                # sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end + 1], cmap=mask_cbar, vmin=vmin,
                #             vmax=vmax, mask=tgt_data_to_plot.iloc[:, i_start:i_end + 1] > -1, cbar=False)
        else:
            # i_col = tgt_data_to_plot.columns.get_loc(tgt)
            # sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=cmap, cbar=False, vmin=cmap_min,
            #             vmax=max_range * cmap_max, linewidths=1)
            # sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=mask_cbar, vmin=cmap_min,
            #             vmax=max_range * cmap_max, mask=tgt_data_to_plot.iloc[:, i_col + 1] > -1, cbar=False)
            # if t + 1 == len(tgt_list):
            #     sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=cmap, vmin=cmap_min, vmax=max_range * cmap_max,
            #                 linewidths=1, cbar_ax=static_ax[t + 1], cbar_kws={'label': plotting_params["cbar_label"]})
            #     sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=mask_cbar, vmin=cmap_min,
            #                 vmax=max_range * cmap_max, mask=tgt_data_to_plot.iloc[:, i_col + 1] > -1, cbar=False)
            i_col = tgt_data_to_plot.columns.get_loc(tgt)
            sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col:i_col + 1], cmap=cmap, cbar=False, vmin=vmin,
                        vmax=vmax, linewidths=1)
            # sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col:i_col + 1], cmap=mask_cbar, vmin=vmin,
            #             vmax=vmax, mask=tgt_data_to_plot.iloc[:, i_col:i_col + 1] > -1, cbar=False)
            if t + 1 == len(tgt_list):
                sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col:i_col + 1], cmap=cmap, vmin=vmin,
                            vmax=vmax,
                            linewidths=1, cbar_ax=static_ax[t + 1], cbar_kws={'label': plotting_params["cbar_label"]})
                # sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col:i_col + 1], cmap=mask_cbar, vmin=vmin,
                #             vmax=vmax, mask=tgt_data_to_plot.iloc[:, i_col:i_col + 1] > -1, cbar=False)

        static_ax[t].set_title(tgt, fontsize=plotting_params['subtitle_size'])
        static_ax[t].set_ylabel('')
        static_ax[t].set_xlabel('')
        if t > 0:
            static_ax[t].set_yticks([])
        if plotting_params["descendants"]:
            tl = [split_strings_layers(t, atlas_name=atlas.metadata['name'], return_str=True)[1] for t in tgt_col]
            # tl = [tgt]
            static_ax[t].set_xticks(np.arange(len(tl)))
            static_ax[t].set_xticklabels(tl, rotation=45, fontsize=plotting_params["tick_size"][0])

        static_ax[t].xaxis.tick_top()
        static_ax[t].xaxis.set_label_position('top')
        tly = static_ax[t].get_yticklabels()
        static_ax[t].set_yticklabels(tly, rotation=45, fontsize=plotting_params["tick_size"][1])
        static_ax[t].tick_params(left=False, bottom=False, top=False)
        static_ax[t].invert_yaxis()
        static_ax[t].figure.axes[-1].yaxis.label.set_color(plotting_params["color"])
        # cbar = ax.collections[0].colorbar
        # ax[t].collections[0].colorbar.ax.tick_params(colors=plotting_params["color"])  # todo this one is commented, but needs to be fixed, prob. calling it after loop or give color in last heatmap
        static_ax[t].spines['bottom'].set_color(plotting_params["color"])
        static_ax[t].spines['left'].set_color(plotting_params["color"])
        static_ax[t].xaxis.label.set_color(plotting_params["color"])
        static_ax[t].yaxis.label.set_color(plotting_params["color"])
        static_ax[t].tick_params(colors=plotting_params["color"])

    static_ax[0].set_ylabel(plotting_params["ylabel"][0], fontsize=plotting_params["ylabel"][1])
    if plotting_params["save_fig"]:
        mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
    return mpl_widget

