from typing import List, Union, Dict, Tuple

import numpy as np
from napari.utils.notifications import show_info
from bg_atlasapi import BrainGlobeAtlas


def get_bregma(atlas_id: str) -> List[int]:
    """
    Definition of bregma coordinates for different atlases.

    Parameters:
        atlas_id (str): The atlas ID.

    Returns:
        List[int]: The bregma coordinates pre-defined for the popular atlases or estimated from the atlas dimensions.

    Example:
        >>> get_bregma('allen_mouse_10um')
        [540, 0, 570]
    """
    bregma_dict = {
        "allen_mouse_10um": [540, 0, 570],
        "whs_sd_rat_39um": [371, 72, 266],
        "azba_zfish_4um": [360, 0, 335]
    }
    if atlas_id in bregma_dict.keys():
        bregma = bregma_dict[atlas_id]
        return bregma
    else:
        show_info(f'no bregma coordinates specified for {atlas_id} \n estimating bregma from atlas dimensions')
        show_info("loading reference atlas...")
        atlas = BrainGlobeAtlas(atlas_id)
        bregma = list(atlas.shape)
        for i in range(len(bregma)):
           if i in atlas.space.index_pairs[atlas.space.axes_description.index('si')]:
               bregma[i] = int(bregma[i] / 2)
           else:
               bregma[i] = 0
        # raise NotImplementedError("Please specify Bregma coordinates for the selected atlas: ", atlas_id,
        #                           " in the utils.atlas_utils.get_bregma function : bregma_dict dictionary. And try again.")
        return bregma

def xyz_atlas_transform(triplet: List[int], regi_dict: Dict[str, Dict[str, List[int]]], atlas_tuple: Tuple[str, str, str]) -> List[
    int]:
    """
    Transpose xyz triplet to match atlas orientation.

    Parameters:
        triplet (List[int]): The xyz triplet to transform.
        regi_dict (Dict[str, Dict[str, List[int]]]): The registration information dictionary.
        atlas_tuple (Tuple[str, str, str]): The atlas orientation tuple.

    Returns:
        List[int]: The transformed xyz triplet.
    """
    xyz_tuple = tuple([regi_dict['xyz_dict']['x'][0], regi_dict['xyz_dict']['y'][0], regi_dict['xyz_dict']['z'][0]])
    index_match = [xyz_tuple.index(e) for e in atlas_tuple]

    triplet_new = [triplet[i] for i in index_match]

    return triplet_new


def coord_mm_transform(triplet: List[Union[int, float]], bregma: List[int], resolution_tuple: List[float],
                       mm_to_coord: bool = False) -> Union[int, list[int], list[float]]:
    """
    Transform coordinates from mm to pixel or vice versa.

    Parameters:
        triplet (List[Union[int, float]]): The coordinate to transform.
        bregma (List[int]): The bregma coordinates.
        resolution_tuple (List[float]): The resolution tuple.
        mm_to_coord (bool): Whether to transform from mm to pixel or vice versa.

    Returns:
        List[Union[int, float]]: The transformed coordinates.
    """
    decimal_list = get_decimal(resolution_tuple)

    if mm_to_coord:
        triplet_new = [round(-coord / (res / 1000)) + br_coord for coord, br_coord, res in
                       zip(triplet, bregma, resolution_tuple)]
    else:
        triplet_new = []
        for coord, br_coord, res, decimal in zip(triplet, bregma, resolution_tuple, decimal_list):
            triplet_new.append(round((br_coord - coord) * (res / 1000), decimal))

    if len(triplet_new) == 1:
        return triplet_new[0]
    else:
        return triplet_new


def sort_ap_dv_ml(triplet: List[Union[int, float]], atlas_tuple: Tuple[str, str, str]) -> List[float]:
    """
    Reorder the input triplet to match the atlas orientation.

    Parameters:
        triplet (List[float]): The xyz triplet to reorder.
        atlas_tuple (Tuple[str, str, str]): The atlas orientation tuple.

    Returns:
        List[float]: The reordered triplet.
    """
    tgt_tuple = ('ap', 'si', 'rl')
    index_match = [atlas_tuple.index(e) for e in tgt_tuple]
    triplet_new = [triplet[i] for i in index_match]
    return triplet_new


def get_xyz(atlas: BrainGlobeAtlas, section_orient: str) -> Dict[str, List[Union[str, int, float]]]:
    """
    Get the xyz dictionary from the atlas information.

    Parameters:
        atlas (BrainGlobeAtlas): The BrainGlobeAtlas object.
        section_orient (str): The section orientation.

    Returns:
        Dict[str, List[Union[str, int, float]]]: The xyz dictionary containing information about each axis.
    """
    orient_dict = {
        'coronal': 'frontal',
        'horizontal': 'horizontal',
        'sagittal': 'sagittal'
    }
    orient_idx = atlas.space.sections.index(orient_dict[section_orient])
    resolution_idx = atlas.space.index_pairs[orient_idx]
    xyz_dict = {
        'x': [atlas.space.axes_description[resolution_idx[1]], atlas.space.shape[resolution_idx[1]],
              atlas.space.resolution[resolution_idx[1]]],
        'y': [atlas.space.axes_description[resolution_idx[0]], atlas.space.shape[resolution_idx[0]],
              atlas.space.resolution[resolution_idx[0]]],
        'z': [atlas.space.axes_description[orient_idx], atlas.space.shape[orient_idx],
              atlas.space.resolution[orient_idx]]
    }

    return xyz_dict


def get_orient_map(atlas: BrainGlobeAtlas, plotting_params: Dict[str, str]) -> Dict[
    str, Union[str, List[Union[str, float]]]]:
    """
    Get the orientation mapping for plotting.

    Parameters:
        atlas (BrainGlobeAtlas): The BrainGlobeAtlas object.
        plotting_params (Dict[str, str]): Parameters for plotting orientation.

    Returns:
        Dict[str, Union[str, List[Union[str, float]]]]: The orientation mapping dictionary.
    """
    orient_dict = {
        'ap': 'ap_coords',
        'rl': 'ml_coords',
        'si': 'dv_coords'
    }
    xyz_dict = get_xyz(atlas, plotting_params['section_orient'])

    orient_mapping = {
        'z_plot': [orient_dict[xyz_dict['z'][0]], atlas.space.axes_description.index(xyz_dict['z'][0]),
                   xyz_dict['z'][2] / 1000],
        'x_plot': orient_dict[xyz_dict['x'][0]],
        'y_plot': orient_dict[xyz_dict['y'][0]]
    }
    return orient_mapping


def get_decimal(res_tup: List[float]) -> List[int]:
    """
    Get decimal number for displaying accurate z-step size in registration widget.

    Parameters:
        res_tup (List[float]): The resolution tuple.

    Returns:
        List[int]: The list of decimal places for each resolution.
    """
    decimal_list = []
    for r in res_tup:
        step_float = r / 1000
        decimal = 2
        while np.abs(np.round(step_float, decimal) - step_float) >= 0.01 * step_float:
            decimal += 1
        decimal_list.append(decimal)
    return decimal_list
