from enum import Enum
from typing import Dict

from brainglobe_atlasapi.list_atlases import get_all_atlases_lastversions
from skimage import filters as filters


def get_available_atlases() -> Dict[str, str]:
    """
    Get the available BrainGlobe atlases.

    This function fetches the available atlases from the BrainGlobe's configuration URL
    and rearranges the list to make "allen_mouse_10um" the default and move
    "example_mouse_100um" to the end.

    Returns:
        Dict[str, str]: A dictionary where keys and values are atlas names and versions.
    """
    available_atlases = get_all_atlases_lastversions()

    preferred_atlas = "allen_mouse_10um"
    example_atlas = "example_mouse_100um"

    ordered_atlases = {
        name: available_atlases[name]
        for name in (preferred_atlas,)
        if name in available_atlases
    }
    ordered_atlases.update(
        {
            name: version
            for name, version in available_atlases.items()
            if name not in (preferred_atlas, example_atlas)
        }
    )
    if example_atlas in available_atlases:
        ordered_atlases[example_atlas] = available_atlases[example_atlas]

    return ordered_atlases


def get_atlas_dropdown() -> Enum:
    """
    Generate a dropdown menu for selecting atlases.

    Returns:
        Enum: An enumeration where each key and value represent available atlas names.
    """
    atlas_dict = {}
    for i, k in enumerate(get_available_atlases().keys()):
        atlas_dict.setdefault(k, k)
    atlas_keys = Enum("atlas_key", atlas_dict)
    return atlas_keys


def get_threshold_dropdown() -> Enum:
    """
    Generate a dropdown menu for selecting thresholding functions from skimage.filters.

    This function prioritizes the "threshold_yen" function by moving it to the top of the list.

    Returns:
        Enum: An enumeration where each key and value represent available thresholding function names.
    """
    func_list = dir(filters)
    func_list = [f for f in func_list if f.startswith('threshold')]

    # Move "threshold_yen" to the top of the list
    idx_yen = func_list.index('threshold_yen')
    func_list = [func_list.pop(idx_yen)] + func_list

    func_dict = {}
    for f in func_list:
        func_dict.setdefault(f, f)
    func_keys = Enum("func_key", func_dict)
    return func_keys
