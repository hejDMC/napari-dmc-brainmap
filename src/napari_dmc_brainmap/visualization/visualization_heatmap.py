import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

from napari_dmc_brainmap.utils import split_to_list, split_strings_layers
from napari_dmc_brainmap.visualization.visualization_tools import get_tgt_data_only, resort_df, get_bregma, \
    get_unique_filename

def calculate_percentage_heatmap_plot(df_all, atlas, plotting_params, animal_list, tgt_list, sub_list):

    df = get_tgt_data_only(df_all, atlas, tgt_list)
    interval_labels = plotting_params["interval_labels"]
    intervals = plotting_params["intervals"]
    df['bin'] = pd.cut(df['ap_mm'], intervals, labels=interval_labels)

    absolute_numbers = plotting_params["absolute_numbers"]  # use absolute numbers
    if absolute_numbers:
        rel_percentage = False
    else:
        rel_percentage = True

    if sub_list:  # sub data to calculate relative count for
        sub_data = get_tgt_data_only(df_all, sub_list)
        absolute_numbers = False
        rel_percentage = False
        print("sub list for normalization provided, overriding input for percentage/absolute numbers")

    df = df.pivot_table(index='acronym', columns=['animal_id', 'bin'],
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

    # normalize numbers for each animal respectively, to the total number of PFC cells
    for animal_idx, animal_id in enumerate(animal_list):
        if animal_idx == 0:
            if absolute_numbers:
                df_plot = df['ap_mm'][animal_id]
            elif rel_percentage:
                df_plot = (df['ap_mm'][animal_id] / df['ap_mm'][animal_id].sum().sum()) * 100
            elif sub_list:
                df_plot = (df['ap_mm'][animal_id] / len(sub_data[sub_data['animal_id']==animal_id])) * 100
            else:  # percentage relative to all cells in animal
                df_plot = (df['ap_mm'][animal_id] / len(df_all[df_all['animal_id']==animal_id])) * 100
        else:
            if absolute_numbers:
                df_plot += df['ap_mm'][animal_id]
            elif rel_percentage:
                df_plot += (df['ap_mm'][animal_id] / df['ap_mm'][animal_id].sum().sum()) * 100
            elif sub_list:
                df_plot += (df['ap_mm'][animal_id] / len(sub_data[sub_data['animal_id']==animal_id])) * 100
            else:  # percentage relative to all cells in animal
                df_plot += (df['ap_mm'][animal_id] / len(df_all[df_all['animal_id'] == animal_id])) * 100
    # calculate average
    df_plot /= len(animal_list)
    return df_plot

def get_interval_labels(intervals):
    interval_labels = []
    for i in range(len(intervals)):
        if i < len(intervals)-1:
            interval_labels.append(str(intervals[i]) + " to " + str(intervals[i+1]))
    return interval_labels

def get_layer_list(tgt_list, atlas):

    # for now only for isocortical areas, not sure about olfactory areas
    new_tgt_list = []
    for tgt in tgt_list:
        if 'Isocortex' in atlas.get_structure_ancestors(tgt):
            new_tgt_list.append(atlas.get_structure_descendants(tgt))
        else:
            new_tgt_list.append(tgt)
    new_tgt_list = [l for sublist in new_tgt_list for l in sublist]  # flatten list
    #     tgt_path = st[st['acronym'] == tgt]['structure_id_path'].to_list()[0]
    #     try:
    #         new_tgt_list += st[st['structure_id_path'].str.contains(tgt_path)]['acronym'].to_list()[1:]  # ignore first entry -- this is parent
    #     except IndexError:
    #         pass
    # # for some cortical regions, a layer 2 and layer 2/3 exist --> ignore the layer 2
    # new_tgt_list = [t for t in new_tgt_list if not split_strings_layers(t)[1] == '2']
    return new_tgt_list

def check_brain_area_in_bin(df, atlas):
    # for heatmap plotting, check if brain area exists in bin
    print('checking for existence of brain area in bin ...')
    bregma = get_bregma(atlas.atlas_name)
    ap_idx = atlas.space.axes_description.index('ap')
    st = atlas.structures
    annot = atlas.annotation
    for bin in df.columns:
        print(bin)
        b_start, b_end = bin.split(' to ')
        b_start_coord = int(-(float(b_start)/0.01 - bregma[ap_idx]))
        b_end_coord = int(-(float(b_end) / 0.01 - bregma[ap_idx]))
        if b_start_coord > b_end_coord:
            ids_in_bin = np.unique(annot[b_end_coord:b_start_coord, :, :])
        else:  # todo here's a future warning: FutureWarning: Calling int on a single element Series is deprecated and will raise a TypeError in the future. Use int(ser.iloc[0]) instead!

            ids_in_bin = np.unique(annot[b_start_coord:b_end_coord, :, :])
        for area in df.index:
            area_id = st[area].data['id']
            # area_id = int(st[st['acronym'] == area]['sphinx_id'])
            if area_id not in ids_in_bin:
                df[bin][area] = -1
    return df

def get_heatmap_params(heatmap_widget):
    plotting_params = {
        "hemisphere": heatmap_widget.hemisphere.value,
        "xlabel": [heatmap_widget.xlabel.value, int(heatmap_widget.xlabel_size.value)],  # 0: label, 1: fontsize
        "ylabel": [heatmap_widget.ylabel.value, int(heatmap_widget.ylabel_size.value)],
        # "tick_size": int(heatmap_widget.tick_size.value),  # for now only y and x same size
        # "rotate_xticks": int(heatmap_widget.rotate_xticks.value),  # set to False of no rotation
        "title": [heatmap_widget.title.value, int(heatmap_widget.title_size.value)],
        "style": heatmap_widget.style.value,
        "color": heatmap_widget.color.value,
        "cmap": split_to_list(heatmap_widget.cmap.value),
        "cbar_label": heatmap_widget.cbar_label.value,
        "cmap_min_max":  [float(i) for i in heatmap_widget.cmap_min_max.value.split(',')],  # [0] is absolute value for vmin, [1] is value to multiply max_range value with
        "intervals": [float(i) for i in heatmap_widget.intervals.value.split(',')],
        "interval_labels": get_interval_labels(split_to_list(heatmap_widget.intervals.value)),
        "include_layers": True,  # heatmap_widget.include_layers.value,
        "transpose": heatmap_widget.transpose.value,
        "save_name": heatmap_widget.save_name.value,
        "save_fig": heatmap_widget.save_fig.value,
        "save_data_name": heatmap_widget.save_data_name.value,
        "save_data": heatmap_widget.save_data.value,
        "absolute_numbers": heatmap_widget.absolute_numbers.value
    }
    return plotting_params

def do_heatmap(df, atlas, animal_list, tgt_list, plotting_params, heatmap_widget, save_path, sub_list=False):
    # if applicable only get the ipsi or contralateral cells
    hemisphere = plotting_params['hemisphere']
    if hemisphere == 'ipsi':
        df = df[df['ipsi_contra'] == 'ipsi']
    elif hemisphere == 'contra':
        df = df[df['ipsi_contra'] == 'contra']

    if plotting_params["include_layers"]:
        new_tgt_list = get_layer_list(tgt_list, atlas)
        print(new_tgt_list)
    else:
        new_tgt_list = tgt_list

    tgt_data_to_plot = calculate_percentage_heatmap_plot(df, atlas, plotting_params, animal_list, new_tgt_list, sub_list)
    # put not existing areas to -1 for plotting
    tgt_data_to_plot = check_brain_area_in_bin(tgt_data_to_plot, atlas)
    if plotting_params["transpose"]:
        tgt_data_to_plot = tgt_data_to_plot.transpose()
    if plotting_params["save_data"]:
        data_fn = save_path.joinpath(plotting_params["save_data_name"])
        if data_fn.exists():
            data_fn = get_unique_filename(data_fn)
        tgt_data_to_plot.to_csv(data_fn)
    max_range = tgt_data_to_plot.max().max()
    interval_labels = plotting_params["interval_labels"]
    sns.set(style=plotting_params["style"])
    if plotting_params["style"] == 'white':
        mask_cbar = 'binary'
    else:
        mask_cbar = 'binary_r'
    cmap_min = plotting_params["cmap_min_max"][0]
    cmap_max = plotting_params["cmap_min_max"][1]
    cmap = plotting_params["cmap"]
    figsize = [int(i) for i in heatmap_widget.plot_size.value.split(',')]
    mpl_widget = FigureCanvas(Figure(figsize=(figsize)))
    static_ax = mpl_widget.figure.subplots(1, (len(tgt_list) + 1),
                                           gridspec_kw={'width_ratios': [1] * len(tgt_list) + [0.15]}) #, figsize=figsize)
    #fig, ax = plt.subplots(1, (len(tgt_list) + 1),
    #                       gridspec_kw={'width_ratios': [1] * len(tgt_list) + [0.15]}, figsize=figsize)
    for a in range(len(static_ax)):
        static_ax[0].get_shared_y_axes().join(static_ax[a])
    for t, tgt in enumerate(tgt_list):
        if plotting_params["include_layers"]:
            tgt_col = get_layer_list([tgt], atlas)
            print(tgt_col)
            i_start = tgt_data_to_plot.columns.get_loc(tgt_col[0])
            i_end = tgt_data_to_plot.columns.get_loc(tgt_col[-1])
            sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end+1], cmap=cmap, cbar=False, vmin=cmap_min,
                        vmax=max_range * cmap_max, linewidths=1)
            sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end + 1], cmap=mask_cbar, vmin=cmap_min,
                        vmax=max_range * cmap_max, mask=tgt_data_to_plot.iloc[:, i_start:i_end + 1] > -1, cbar=False)
            if t + 1 == len(tgt_list):
                sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end+1], cmap=cmap, vmin=cmap_min,
                            vmax=max_range * cmap_max, linewidths=1, cbar_ax=static_ax[t + 1], cbar_kws={'label': plotting_params["cbar_label"]})
                sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_start:i_end + 1], cmap=mask_cbar, vmin=cmap_min,
                            vmax=max_range * cmap_max, mask=tgt_data_to_plot.iloc[:, i_start:i_end + 1] > -1, cbar=False)
        else:
            i_col = tgt_data_to_plot.columns.get_loc(tgt)
            sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=cmap, cbar=False, vmin=cmap_min,
                        vmax=max_range * cmap_max, linewidths=1)
            sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=mask_cbar, vmin=cmap_min,
                        vmax=max_range * cmap_max, mask=tgt_data_to_plot.iloc[:, i_col + 1] > -1, cbar=False)
            if t + 1 == len(tgt_list):
                sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=cmap, vmin=cmap_min, vmax=max_range * cmap_max,
                            linewidths=1, cbar_ax=static_ax[t + 1], cbar_kws={'label': plotting_params["cbar_label"]})
                sns.heatmap(ax=static_ax[t], data=tgt_data_to_plot.iloc[:, i_col], cmap=mask_cbar, vmin=cmap_min,
                            vmax=max_range * cmap_max, mask=tgt_data_to_plot.iloc[:, i_col + 1] > -1, cbar=False)
        static_ax[t].set_title(tgt_list[t], fontsize=18)
        static_ax[t].set_ylabel('')
        static_ax[t].set_xlabel('')
        if t > 0:
            static_ax[t].set_yticks([])
        if plotting_params["include_layers"]:
            tl = [split_strings_layers(t)[1] for t in tgt_col]
        else:
            tl = tgt
        static_ax[t].set_xticklabels(tl, rotation=45)
        static_ax[t].xaxis.tick_top()
        static_ax[t].xaxis.set_label_position('top')
        tly = static_ax[t].get_yticklabels()
        static_ax[t].set_yticklabels(tly, rotation=45)
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

