import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.find_structure import sliceHandle

def get_ipsi_contra(df):
    '''
    Function to add a column specifying if cells if ipsi or contralateral to injection side
    ML_location values of <0 are on the 'left' hemisphere, >0 are on the 'right hemisphere
    :param df: dataframe with results for animal, not the merged across animals
    :return:
    '''

    df['ipsi_contra'] = ['ipsi'] * len(df)  # add a column defaulting to 'ipsi'
    # change values to contra with respect to the location of the injection side
    if df['injection_side'][0] == 'left':
        df.loc[(df['ML_location'] < 0), 'ipsi_contra'] = 'contra'
    elif df['injection_side'][0] == 'right':
        df.loc[(df['ML_location'] > 0), 'ipsi_contra'] = 'contra'
    return df


def get_tgt_data_only(df, tgt_list):
    tgt_path_list = []
    s = sliceHandle()
    st = s.df_tree
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

def bar_plot_areas(df, animal_list, tgt_list, plotting_params, hemisphere=False):
    """
    Function to plot bar plot of retro-tracing data from merged dataframe
    :param df: merged dataframe with tracing data (rows are cells)
    :param animal_list: list of animal_ids (animal_id needs to be one identifier for cells
    :param tgt_list: list with structures to plot (at the correct level already)
    :param plotting_params: dict with plotting params
    :param hemisphere: ipsi or contra relative to injection side, if not specified will use data from both hemispheres
    :return:
    """
    # if applicable only get the ipsi or contralateral cells
    if hemisphere == 'ipsi':
        df = df[df['ipsi_contra'] == 'ipsi']
    elif hemisphere == 'contra':
        df = df[df['ipsi_contra'] == 'contra']

    if plotting_params["horizontal"]:
        plot_orient = 'h'
        x_var = "percent_cells"
        y_var = "tgt_name"
    else:
        plot_orient = 'v'
        y_var = "percent_cells"
        x_var = "tgt_name"
    # get re-structured dataframe for plotting
    tgt_data_to_plot = calculate_percentage_bar_plot(df, animal_list, tgt_list, plotting_params)

    if not plotting_params["alphabetic"]:  # re-structuring of df creates alphabetic order of brain areas, if tgt_list order should be kept do resort
        tgt_data_to_plot = resort_df(tgt_data_to_plot, tgt_list)
    # if plotting_params["style"] == "black":
    #     pass
    # elif plotting_params["style"] == "white":
    sns.set(style=plotting_params["style"])  # set style todo: dark style not really implemented
    ax = sns.barplot(x=x_var, y=y_var, data=tgt_data_to_plot, palette=plotting_params["bar_palette"],
                         capsize=.1, ci=None, orient=plot_orient)  # do the barplot
    # ax = sns.barplot(x="tgt_name", y="percent_cells", data=tgt_bar_plot_df, palette=palette_blue, capsize=.1, ci=None) # order=pfc_list,
    # palette = sns.color_palette("GnBu_d",  n_colors=2)
    # palette = sns.cubehelix_palette(len(animal_list))
    if plotting_params["scatter_hue"]:  # color code dots by animals
        ax = sns.swarmplot(x=x_var, y=y_var, hue='animal_id', data=tgt_data_to_plot,
                           palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
                           orient=plot_orient)
    else:
        ax = sns.swarmplot(x=x_var, y=y_var, data=tgt_data_to_plot,
                           palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
                           orient=plot_orient
                           ) #, legend=plotting_params["scatter_legend"])  # , alpha=.35) hue='animal_id',
    ax.set_xlabel(plotting_params["xlabel"][0], fontsize=plotting_params["xlabel"][1])
    ax.set_ylabel(plotting_params["ylabel"][0], fontsize=plotting_params["ylabel"][1])
    ax.set_title(plotting_params["title"][0], fontsize=plotting_params["title"][1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plotting_params["scatter_hue"]:  # adjust color for legend
        leg = ax.get_legend()
        leg.get_title().set_color(plotting_params["color"])
        frame = leg.get_frame()
        frame.set_alpha(None)
        frame.set_facecolor((0, 0, 1, 0))
        for text in leg.get_texts():
            text.set_color(plotting_params["color"])
    if plotting_params["scatter_legend_hide"]:  # remove legend from scatter plot
        ax.legend_.remove()
    ax.spines['bottom'].set_color(plotting_params["color"])
    ax.spines['left'].set_color(plotting_params["color"])
    ax.xaxis.label.set_color(plotting_params["color"])
    ax.yaxis.label.set_color(plotting_params["color"])
    ax.tick_params(colors=plotting_params["color"], labelsize=plotting_params["tick_size"])
    if plotting_params["rotate_xticks"]:  # rotate x-ticks
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.show()
    if plotting_params["save_fig"]:
        plt.savefig(plotting_params["save_dir"])

# def calculate_count_bar_plot(df, animal_list, tgt_list):


def calculate_percentage_bar_plot(df_all, animal_list, tgt_list, plotting_params):

    absolute_numbers = plotting_params["absolute_numbers"]
    rel_percentage = plotting_params["rel_percentage"]
    if absolute_numbers:
        rel_percentage = False
        print("using absolute numbers, overriding input for percentage")
    df = get_tgt_data_only(df_all, tgt_list)
    df_geno = df.copy() # copy df to extract genotype of mice later on
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
    # calculate percentages
    df_to_plot = pd.DataFrame()
    for animal_id in animal_list:
        genotype = df_geno[df_geno['animal_id'] == animal_id]['genotype'].unique()[0]
        if absolute_numbers: # if absolute numbers
            dummy_df = pd.DataFrame(df['AP_location'][animal_id])
        elif rel_percentage: # if relative percentage for cells in tgt_regions
            dummy_df = pd.DataFrame((df['AP_location'][animal_id] / df['AP_location'][
                animal_id].sum()) * 100)
        else: # to percentage for all cells
            dummy_df = pd.DataFrame((df['AP_location'][animal_id] /
                                     len(df_all[df_all['animal_id']==animal_id]))*100)
        dummy_df = dummy_df.rename(columns={animal_id: "percent_cells"})
        dummy_df['animal_id'] = [animal_id] * len(tgt_list)
        dummy_df['genotype'] = [genotype] * len(tgt_list)
        df_to_plot = pd.concat([df_to_plot, dummy_df])

    df_to_plot.index.name = 'tgt_name'
    df_to_plot.reset_index(inplace=True)

    return df_to_plot

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