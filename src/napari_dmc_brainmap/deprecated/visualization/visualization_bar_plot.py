import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'
from napari_dmc_brainmap.utils import split_to_list
from napari_dmc_brainmap.visualization.visualization_tools import get_tgt_data_only, resort_df, get_unique_filename, \
    create_cmap

def calculate_percentage_bar_plot(df_all, atlas, animal_list, tgt_list, plotting_params):

    absolute_numbers = plotting_params["absolute_numbers"]
    df = get_tgt_data_only(df_all, atlas, tgt_list, use_na=True)
    if plotting_params["groups"] in ["channel", "ipsi_contra"]:
        df = df.pivot_table(index='tgt_name', columns=['animal_id', plotting_params["groups"]],
                                                            aggfunc='count').fillna(0)
    elif plotting_params["expression"]:
        df = pd.DataFrame(df.groupby('tgt_name')[plotting_params["gene_list"]].mean().fillna(0))
    else:
        df = df.pivot_table(index='tgt_name', columns=['animal_id'],
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
    if plotting_params["expression"]:
        df.reset_index(inplace=True)
        df['animal_id'] = animal_list[0]
        if len(plotting_params['gene_list']) > 1:
            df.rename(columns={'index': 'tgt_name'}, inplace=True)
            df = pd.melt(df, id_vars=['tgt_name', 'animal_id'],
                            value_vars=plotting_params['gene_list'],
                            var_name='genes', value_name='percent')
        else:
            df.rename(columns={'index': 'tgt_name', plotting_params['gene_list'][0]: 'percent'}, inplace=True)
        return df
    # calculate percentages
    df_to_plot = pd.DataFrame()
    for animal_id in animal_list:
       # genotype = df_geno[df_geno['animal_id'] == animal_id]['genotype'].unique()[0]
        if absolute_numbers == 'absolute': # if absolute numbers
            dummy_df = pd.DataFrame(df['ap_mm'][animal_id])
        elif absolute_numbers == 'percentage_selection': # if relative percentage for cells in tgt_regions
            if plotting_params["groups"] in ["channel", "ipsi_contra"]:
                dummy_df = pd.DataFrame((df['ap_mm'][animal_id] / df['ap_mm'][
                    animal_id].sum().sum()) * 100)
            else:
                dummy_df = pd.DataFrame((df['ap_mm'][animal_id] / df['ap_mm'][
                    animal_id].sum()) * 100)
        else: # to percentage for all cells
            dummy_df = pd.DataFrame((df['ap_mm'][animal_id] /
                                     len(df_all[df_all['animal_id'] == animal_id]))*100)
        if plotting_params["groups"] in ["channel", "ipsi_contra"]:
            dummy_df = dummy_df.stack().reset_index()
            dummy_df = dummy_df.rename(columns={plotting_params["groups"]: "groups", 0: 'percent'})
        else:
            dummy_df = dummy_df.rename(columns={animal_id: "percent"})
        dummy_df['animal_id'] = [animal_id] * len(dummy_df)
        if plotting_params["groups"] in ['group', 'genotype']:
            dummy_df['groups'] = [df_all[df_all['animal_id'] == animal_id][plotting_params["groups"]].unique()[0]] * len(dummy_df)
        df_to_plot = pd.concat([df_to_plot, dummy_df])

    if not plotting_params["groups"] in ["channel", "ipsi_contra"]:
        df_to_plot.index.name = 'tgt_name'
        df_to_plot.reset_index(inplace=True)

    return df_to_plot

def get_bar_plot_params(barplot_widget):
    plotting_params = {
        "expression": barplot_widget.expression.value,
        "gene_list": split_to_list(barplot_widget.gene_list.value),
        "groups": barplot_widget.groups.value,
        "horizontal": barplot_widget.orient.value,
        "xlabel": [barplot_widget.xlabel.value, int(barplot_widget.xlabel_size.value)],  # 0: label, 1: fontsize
        "ylabel": [barplot_widget.ylabel.value, int(barplot_widget.ylabel_size.value)],
        "tick_size": int(barplot_widget.tick_size.value),  # for now only y and x same size
        "rotate_xticks": int(barplot_widget.rotate_xticks.value),  # set to False of no rotation
        "title": [barplot_widget.title.value, int(barplot_widget.title_size.value)],
        "alphabetic": barplot_widget.alphabetic.value,
        "style": barplot_widget.style.value,
        "color": barplot_widget.color.value,
        "bar_palette": split_to_list(barplot_widget.tgt_colors.value),
        "scatter_palette": split_to_list(barplot_widget.scatter_palette.value),
        "scatter_hue": barplot_widget.scatter_hue.value,
        "scatter_size": int(barplot_widget.scatter_size.value),
        # "legend_hide": barplot_widget.legend_hide.value,
        "save_name": barplot_widget.save_name.value,
        "save_fig": barplot_widget.save_fig.value,
        "save_data": barplot_widget.save_data.value,
        "save_data_name": barplot_widget.save_data_name.value,
        "absolute_numbers": barplot_widget.absolute_numbers.value
    }
    return plotting_params


def do_bar_plot(df, atlas, plotting_params, animal_list, tgt_list, barplot_widget, save_path):


    if plotting_params["horizontal"] == "horizontal":
        plot_orient = 'h'
        x_var = "percent"
        y_var = "tgt_name"
    else:
        plot_orient = 'v'
        y_var = "percent"
        x_var = "tgt_name"
    # get re-structured dataframe for plotting
    tgt_data_to_plot = calculate_percentage_bar_plot(df, atlas, animal_list, tgt_list, plotting_params)
    if plotting_params["save_data"]:
        data_fn = save_path.joinpath(plotting_params["save_data_name"])
        if data_fn.exists():
            data_fn = get_unique_filename(data_fn)
        tgt_data_to_plot.to_csv(data_fn)
    if not plotting_params[
        "alphabetic"]:  # re-structuring of df creates alphabetic order of brain areas, if tgt_list order should be kept do resort
        tgt_data_to_plot = resort_df(tgt_data_to_plot, tgt_list)
    figsize = [int(i) for i in barplot_widget.plot_size.value.split(',')]
    mpl_widget = FigureCanvas(Figure(figsize=figsize))
    static_ax = mpl_widget.figure.subplots()
    sns.set(style=plotting_params["style"])  # set style todo: dark style not really implemented
    if plotting_params["groups"] == '' and not plotting_params['expression']:
        sns.barplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot, palette=plotting_params["bar_palette"],
                    capsize=.1, errorbar=None, orient=plot_orient)  # do the barplot
        if plotting_params["scatter_hue"]:  # color code dots by animals
            sns.swarmplot(ax=static_ax, x=x_var, y=y_var, hue='animal_id', data=tgt_data_to_plot,
                          palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
                          orient=plot_orient, legend=False)
    elif plotting_params['expression']:
        if len(plotting_params['gene_list']) > 1:
            cmap = create_cmap([], plotting_params, "bar_palette", df=tgt_data_to_plot, hue_id='genes')
            sns.barplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot, hue='genes', palette=cmap,
                        capsize=.1, errorbar=None, orient=plot_orient)
        else:
            sns.barplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot, palette=plotting_params["bar_palette"],
                    capsize=.1, errorbar=None, orient=plot_orient)  # do the barplot
    else:
        if plotting_params["groups"] == 'animal_id':
            hue = 'animal_id'
        else:
            hue = 'groups'
        cmap = create_cmap([], plotting_params, "bar_palette", df=tgt_data_to_plot, hue_id=hue)
        sns.barplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot, hue=hue, palette=cmap,
                    capsize=.1, errorbar=None, orient=plot_orient)  # do the barplot
        if plotting_params["scatter_hue"]:  # color code dots by animals
            sns.swarmplot(ax=static_ax, x=x_var, y=y_var, hue='animal_id', data=tgt_data_to_plot,
                          palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
                          dodge=True, orient=plot_orient, legend=False)
    # else:
    #     sns.swarmplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot,
    #                   palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
    #                   orient=plot_orient)
    if plot_orient == 'v':
        static_ax.set_xlabel(plotting_params["xlabel"][0], fontsize=plotting_params["xlabel"][1])
        static_ax.set_ylabel(plotting_params["ylabel"][0], fontsize=plotting_params["ylabel"][1])
    else:
        static_ax.set_ylabel(plotting_params["xlabel"][0], fontsize=plotting_params["xlabel"][1])
        static_ax.set_xlabel(plotting_params["ylabel"][0], fontsize=plotting_params["ylabel"][1])

    static_ax.set_title(plotting_params["title"][0], fontsize=plotting_params["title"][1])
    static_ax.spines['top'].set_visible(False)
    static_ax.spines['right'].set_visible(False)
    if plotting_params["groups"] != '':  # adjust color for legend
        leg = static_ax.get_legend()
        leg.set_title(plotting_params['groups'])
        leg.get_title().set_color(plotting_params["color"])
        frame = leg.get_frame()
        frame.set_alpha(None)
        frame.set_facecolor((0, 0, 1, 0))
        for text in leg.get_texts():
            text.set_color(plotting_params["color"])
        # if plotting_params["scatter_legend_hide"]:  # remove legend from scatter plot
        #     static_ax.legend_.remove()

    static_ax.spines['bottom'].set_color(plotting_params["color"])
    static_ax.spines['left'].set_color(plotting_params["color"])
    static_ax.xaxis.label.set_color(plotting_params["color"])
    static_ax.yaxis.label.set_color(plotting_params["color"])
    static_ax.tick_params(colors=plotting_params["color"], labelsize=plotting_params["tick_size"])
    if plotting_params["rotate_xticks"]:  # rotate x-ticks
        static_ax.set_xticklabels(static_ax.get_xticklabels(), rotation=plotting_params["rotate_xticks"])
    if plotting_params["save_fig"]:
        mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
    return mpl_widget

