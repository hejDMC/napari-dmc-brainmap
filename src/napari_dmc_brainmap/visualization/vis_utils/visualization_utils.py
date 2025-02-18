import random
from natsort import natsorted
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
from typing import List, Any, Tuple
from pathlib import Path
from bg_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info

def get_descendants(tgt_list: List[str], atlas: BrainGlobeAtlas) -> List[str]:
    """
    Retrieve all descendant regions for each target in the target list.

    Parameters:
        tgt_list (List[str]): List of target regions.
        atlas (BrainGlobeAtlas): BrainGlobeAtlas object for retrieving region hierarchy.

    Returns:
        List[str]: List of descendant regions.
    """
    tgt_layer_list = []
    for tgt in tgt_list:
        descendents = atlas.get_structure_descendants(tgt)
        if not descendents:  # if no descendents found, return tgt
            descendents = [tgt]
        tgt_layer_list += descendents
    return tgt_layer_list

def get_ancestors(tgt_list: List[str], atlas: BrainGlobeAtlas) -> List[str]:
    """
    Retrieve the top-level ancestor for each target in the target list.

    Parameters:
        tgt_list (List[str]): List of target regions.
        atlas (BrainGlobeAtlas): BrainGlobeAtlas object for retrieving region hierarchy.

    Returns:
        List[str]: List of top-level ancestors for each target.
    """
    tgt_ancestor_list = []
    for tgt in tgt_list:
        ancestor = atlas.get_structure_ancestors(tgt)[-1]
        if not ancestor in tgt_ancestor_list:
            tgt_ancestor_list.append(ancestor)
    return tgt_ancestor_list

def match_lists(list1: List[Any], list2: List[Any]) -> Tuple[List[Any], List[Any]]:
    """
    Match two lists by length, padding the shorter list with random colors if needed.

    Parameters:
        list1 (List[Any]): First list.
        list2 (List[Any]): Second list.

    Returns:
        Tuple[List[Any], List[Any]]: Adjusted lists with matched lengths.
    """
    if len(list1) != len(list2):
        diff = len(list1) - len(list2)
        if diff > 0:  # list1 is longer
            for d in range(diff):
                list2.append(random.choice(list(mcolors.CSS4_COLORS.keys())))
            show_info("Warning: Fewer colors provided than brain regions. Random colors will be used.")
        else:  # list2 is longer
            list2 = list2[:len(list1)]
    return list1, list2

def get_unique_folder(data_fn: Path) -> Path:
    """
    Generate a unique folder path by appending a counter if the folder already exists.

    Parameters:
        data_fn (Path): Initial folder path.

    Returns:
        Path: Unique folder path.
    """
    counter = 1
    data_fn_old = data_fn
    while data_fn.exists():
        data_fn = data_fn_old.with_name(f"{data_fn_old.stem}_{counter:03d}{data_fn_old.suffix}")
        counter += 1

    return data_fn

def resort_df(tgt_data_to_plot: pd.DataFrame, tgt_list: List[str], index_sort: bool = False) -> pd.DataFrame:
    """
    Resort the dataframe based on a target list of regions.

    Parameters:
        tgt_data_to_plot (pd.DataFrame): Dataframe containing target data.
        tgt_list (List[str]): Target list of regions to sort by.
        index_sort (bool): Whether to sort by index. Defaults to False.

    Returns:
        pd.DataFrame: Resort dataframe.
    """
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
