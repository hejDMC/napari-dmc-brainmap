import pandas as pd

def get_ipsi_contra(df):
    '''
    Function to add a column specifying if cells if ipsi or contralateral to injection side
    ml_mm values of <0 are on the 'left' hemisphere, >0 are on the 'right hemisphere
    :param df: dataframe with results for animal, not the merged across animals
    :return:
    '''

    df['ipsi_contra'] = ['ipsi'] * len(df)  # add a column defaulting to 'ipsi'
    # change values to contra with respect to the location of the injection side
    if df['injection_side'][0] == 'left':
        df.loc[(df['ml_mm'] < 0), 'ipsi_contra'] = 'contra'
    elif df['injection_side'][0] == 'right':
        df.loc[(df['ml_mm'] > 0), 'ipsi_contra'] = 'contra'
    return df


def get_tgt_data_only(df, tgt_list):
    tgt_path_list = []
    #s = sliceHandle()
    #st = s.df_tree
    st = pd.read_csv(r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dmc-brainmap\napari-dmc-brainmap\src\napari_dmc_brainmap\registration\sharpy_track\sharpy_track\atlas\structure_tree_safe_2017.csv')
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

def calculate_percentage_bar_plot(df_all, animal_list, tgt_list, plotting_params):

    absolute_numbers = plotting_params["absolute_numbers"]
    if absolute_numbers:
        rel_percentage = False
    else:
        rel_percentage = True
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
       # genotype = df_geno[df_geno['animal_id'] == animal_id]['genotype'].unique()[0]
        if absolute_numbers: # if absolute numbers
            dummy_df = pd.DataFrame(df['ap_mm'][animal_id])
        elif rel_percentage: # if relative percentage for cells in tgt_regions
            dummy_df = pd.DataFrame((df['ap_mm'][animal_id] / df['ap_mm'][
                animal_id].sum()) * 100)
        else: # to percentage for all cells
            dummy_df = pd.DataFrame((df['ap_mm'][animal_id] /
                                     len(df_all[df_all['animal_id']==animal_id]))*100)
        dummy_df = dummy_df.rename(columns={animal_id: "percent_cells"})
        dummy_df['animal_id'] = [animal_id] * len(tgt_list)
        # dummy_df['genotype'] = [genotype] * len(tgt_list)
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