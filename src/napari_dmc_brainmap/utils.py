import json
import re
from natsort import natsorted
import pandas as pd
from mergedeep import merge
from enum import Enum
from bg_atlasapi.list_atlases import descriptors, utils
from bg_atlasapi import BrainGlobeAtlas
import numpy as np
import skimage.filters as filters
from napari.utils.notifications import show_info
import pathlib
from typing import List, Union

def get_animal_id(input_path: pathlib.PosixPath | pathlib.WindowsPath) -> str:
    """
    Get the animal ID from the input path.
    Parameters:
    input_path (pathlib.PosixPath | pathlib.WindowsPath): The path to the animal data.
    Returns:
    str: The animal ID.

    Example:
    >>> from napari_dmc_brainmap.utils import get_animal_id
    >>> windows_path = pathlib.PureWindowsPath('C:\\Users\\username\\histology\\animal_id')
    >>> get_animal_id(windows_path)
    'animal_id'
    >>> posix_path = pathlib.PurePosixPath('/home/username/histology/animal_ID')
    >>> get_animal_id(posix_path)
    'animal_ID'
    """
    animal_id = input_path.parts[-1]
    return animal_id


def get_info(input_path, folder_id, channel=False, seg_type=False, create_dir=False, only_dir=False):
    if not seg_type:
        if channel:
            data_dir = input_path.joinpath(folder_id, channel)
        else:
            data_dir = input_path.joinpath(folder_id)
        if data_dir.exists():
            data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.tif')])
        else:
            data_list = []
    else:
        if channel:
            data_dir = input_path.joinpath(folder_id, seg_type, channel)
        else:
            data_dir = input_path.joinpath(folder_id, seg_type)
        if data_dir.exists():
            data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.csv')])
        else:
            data_list = []
    if create_dir:
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            show_info(f'creating folder under: {str(data_dir)}')
    if only_dir:
        return data_dir
    else:
        if len(data_list) > 0:
            data_suffix = find_common_suffix(data_list, input_path=input_path, folder=folder_id, im_list_present=True)
        else:
            data_suffix = ''
        return data_dir, data_list, data_suffix


def find_common_suffix(image_list, input_path=False, folder='unknown', im_list_present=False):
    # if image list is present, load and get suffix
    if im_list_present:
        im0 = image_list[0]
        im_list = get_image_list(input_path, folder_id=folder)
        im1 = [i for i in im_list if im0.startswith(i)]
        if len(im1) == 1:
            im1 = im1[0]
            common_suffix = im0[len(im1):]
        elif len(im1) == 2:
            im1 = im1[1]
            common_suffix = im0[len(im1):]
        else:
            common_suffix = input("please, manually enter suffix: ")
    else:
        if len(image_list) > 1:
            for i in range(len(image_list[0])):
                if i > 0:
                    if image_list[0][-i] == image_list[1][-i]:
                        continue
                    else:
                        break
            common_suffix = image_list[0][-i + 1:]
            show_info("estimated common_suffix for " + folder + " folder: " + common_suffix)
        elif len(image_list) == 1:
            show_info(f'only one file in {folder} folder: {image_list[0]}')
            show_info(
                'in DMC-BrainMap an image name has a base_string and a suffix. For an image named *animal1_obj1_1_stitched.tif* '
                '\n the base_string is animal1_obj1_1 and the suffix is _stitched.tif')
            common_suffix = input("please, manually enter suffix: ")
        else:
            common_suffix = []
    return common_suffix


def get_image_list(input_path, folder_id='stitched', file_id='*.tif'):
    im_list_name = input_path.joinpath('image_names.csv')
    if im_list_name.exists():
        image_list = pd.read_csv(im_list_name)
        image_list = image_list['0'].to_list()

    else:
        data_dir = get_info(input_path, folder_id, only_dir=True)
        if folder_id == 'confocal':
            image_list = natsorted([f.parts[-1] for f in data_dir.glob(file_id)])
        else:
            filter_dir = [f for f in data_dir.glob('**/*') if f.is_dir()][0]  # just take the first folder
            image_list = natsorted([f.parts[-1] for f in filter_dir.glob(file_id)])
        common_suffix = find_common_suffix(image_list, input_path=input_path,folder=folder_id)
        image_list = [image[:-len(common_suffix)] for image in image_list]  # delete the common_suffix

        # store data as .csv file
        image_list_store = pd.DataFrame(image_list)
        image_list_store.to_csv(im_list_name)

    return image_list

def chunk_list(input_list:List[str], chunk_size: int=4) -> List[List[str]]:
    """
    Split a list into chunks of a specified size.
    Parameters:
    input_list (List[str]): The list to split.
    chunk_size (int): The size of each chunk.
    Returns:
    List[List[str]]: A list of chunks.
    Example:
    >>> input_list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    >>> chunk_list(input_list, 4)
    [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h'], ['i', 'j']]
    >>> chunk_list(input_list, 1)
    [['a'], ['b'], ['c'], ['d'], ['e'], ['f'], ['g'], ['h'], ['i'], ['j']]
    >>> chunk_list(input_list, 12)
    [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']]
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def load_params(input_path: pathlib.PosixPath | pathlib.WindowsPath) -> dict:
    """
    Load the params.json file from the specified input path.
    Parameters:
    input_path (pathlib.PosixPath | pathlib.WindowsPath): Path to the directory where params.json should be located.
    Returns:
    dict: The loaded params.json data.
    Raises:
    FileNotFoundError: If the params.json file is missing.
    Example:
    >>> input_path = pathlib.Path('path/to/animal_id')
    >>> load_params(input_path)
    {'key': 'value'}
    """
    params_fn = input_path.joinpath('params.json')
    if params_fn.exists():
        with open(params_fn) as fn:
            params_dict = json.load(fn)
        return params_dict
    else:
        raise FileNotFoundError(" ['Params.json'] file missing for " + get_animal_id(input_path) + " \n"
                                "Check Data Integrity at folder: {} \n"
                                "and try again!".format(input_path))
    
# def load_params(input_path, parent_widget=None):
#     """
#     Loads the params.json file from the specified input path.
#
#     Parameters:
#     input_path (Path): Path to the directory where params.json should be located.
#     parent_widget (QWidget, optional): Parent widget to attach dialogs.
#
#     Returns:
#     dict: The loaded params.json data.
#     """
#     params_fn = input_path.joinpath('params.json')
#
#     try:
#         with open(params_fn, 'r') as fn:
#             params = json.load(fn)
#             return params
#     except FileNotFoundError:
#         msg_box = QMessageBox(parent_widget)
#         msg_box.setIcon(QMessageBox.Critical)
#         msg_box.setText(f"params.json not found in {str(params_fn)}")
#         msg_box.setWindowTitle("File Not Found Error")
#         msg_box.exec_()
#         return {}
#     except json.JSONDecodeError:
#         msg_box = QMessageBox(parent_widget)
#         msg_box.setIcon(QMessageBox.Critical)
#         msg_box.setText("The params.json file is corrupted and could not be decoded.")
#         msg_box.setWindowTitle("JSON Decode Error")
#         msg_box.exec_()
#         return {}

def clean_params_dict(params_dict:dict, key:str) -> dict:
    """
    Remove empty keys and processes that have not run from the params dictionary.
    Parameters:
    params_dict (dict): The params dictionary.
    key (str): The key to clean.
    Returns:
    dict: The cleaned params dictionary.
    Example:
    >>> params_dict = {
    ...     'processes': {
    ...         'proc1': True,
    ...         'proc2': False,
    ...         'proc3': None,
    ...         'proc4': 'value',
    ...         'proc5': ''
    ...     },
    ...     'proc2_params': {'param1': 'value1'},
    ...     'proc3_params': {'param2': 'value2'}
    ... }
    >>> key = 'processes'
    >>> clean_params_dict(params_dict, key)
    {'processes': {'proc1': True, 'proc4': 'value'}}
    """
    # remove empty keys and processes that have not run
    del_list = []
    for k in params_dict[key]:
        if not params_dict[key][k]:
            del_list.append(k)
    for d in del_list:
        del params_dict[key][d]
        try:
            del params_dict[f"{d}_params"]
        except KeyError:
            pass
    return params_dict


def update_params_dict(input_path:pathlib.Path, params_dict:dict, create:bool=False) -> dict:
    """
    Update the params.json file with the specified dictionary.
    Parameters:
    input_path (Path): Path to the directory where params.json should be located.
    params_dict (dict): The dictionary to update the params.json file with.
    create (bool, optional): Whether to create the params.json file if it does not exist.
    Returns:
    dict: The updated params dictionary.
    Raises:
    FileNotFoundError: If the params.json file is missing and create is False.
    Example:
    >>> input_path = pathlib.Path('path/to/animal_id')
    >>> old_params_dict = {'key': 'value'}
    >>> new_params_dict = {'new_key': 'new_value'}
    >>> update_params_dict(input_path, params_dict)
    {'key': 'value', 'new_key': 'new_value'}
    """
    params_fn = input_path.joinpath('params.json')
    if params_fn.exists():
        show_info("params.json exists -- overriding existing values")
        with open(params_fn) as fn:
            params_dict_old = json.load(fn)
        params_dict_new = merge(params_dict_old, params_dict)

        with open(params_fn, 'w') as fn:
            json.dump(params_dict_new, fn, indent=4)

        return params_dict_new
    elif create:
        with open(params_fn, 'w') as fn:
            json.dump(params_dict, fn, indent=4)
        return params_dict
    else:
        raise FileNotFoundError(" ['Params.json'] file missing for " + get_animal_id(input_path) + " \n"
                                "Check Data Integrity at folder: {} \n"
                                "and try again!".format(input_path))


# def split_strings_layers(s):
#     # from: https://stackoverflow.com/questions/430079/how-to-split-strings-into-text-and-number
#     if s.startswith('TEa'):  # special case due to small 'a', will otherwise split 'TE' + 'a1', not 'TEa' + '1'
#         head = s[:3]
#         tail = s[3:]
#     elif s.startswith('CA'):
#         head = s
#         tail = []
#     else:
#         head = s.rstrip('0123456789/ab')
#         tail = s[len(head):]
#     return head, tail

def split_strings_layers(s, atlas_name, return_str=False):
    # likely not working for other atlases than ABA
    if atlas_name == 'allen_mouse':
        if s.startswith('CA'):
            head = s
            tail = []
        else:
            match = re.match(r"([A-Za-z-]+)(\d+.*)", s)  # re.match(r"([A-Za-z]+)(\d+.*)", s)
            if match:
                head = match.group(1)
                tail = match.group(2)
            else:
                head = s
                tail = []
    else:
        head = s
        tail = []
    if return_str:
        if tail == []:
            tail = head
    return head, tail

# def get_parent(a, st):
#     # dummy function to get parent id for quick and dirty quantification of injection site
#     level_dict = {
#         7: ['Isocortex', 'HPF', 'TH'],
#         6: ['OLF', 'HY'],
#         5: ['CTXsp', 'CNU', 'MB', 'CB'],
#         4: ['HB']
#     }
#     a_parent = a
#     for k, v in level_dict.items():
#         for i in v:
#             if st[st['acronym'] == i]['structure_id_path'].iloc[0] in a:
#                 if len(a.split('/')) > (k + 3):
#                     a_parent = st[st['structure_id_path'] == '/'.join(a.split('/')[:k + 2]) + '/']['acronym'].iloc[0]
#                 break

#     return a_parent


def clean_results_df(df:pd.DataFrame, atlas:BrainGlobeAtlas) -> pd.DataFrame:
    """
    Clean the results DataFrame by removing unwanted brain structures and thier decendants.
    Parameters:
    df (DataFrame): The results DataFrame.
    atlas (BrainGlobeAtlas): The BrainGlobeAtlas object.
    Returns:
    DataFrame: The cleaned results DataFrame.
    Example:
    >>> data = {'acronym': ['root', 'other', 'cm', 'VL']}
    >>> df = pd.DataFrame(data)
    >>> atlas = BrainGlobeAtlas('allen_mouse_10um')
    >>> expected_data = {'acronym': ['other']}
    >>> clean_results_df(df, atlas) == pd.DataFrame(expected_data)
    >>> True
    """
    list_delete = ['root']
    for item in ['fiber tracts', 'VS']:
        list_delete.extend(atlas.get_structure_descendants(item))  
    df = df.drop(df[df['acronym'].isin(list_delete)].index)
    df = df.reset_index(drop=True)
    return df


def split_to_list(input_str: Union[None,str], out_format: str='str'
                  ) -> Union[bool,str,List[str],List[float],List[int]]:
    """
    Split a user input string into a list of strings, floats, or integers.
    Parameters:
    input_str (str): The user input string.
    out_format (str): The output format ('str', 'float', 'int').
    Returns:
    bool: False if the input string is empty.
    str: 'auto' if the input string is 'auto'.
    List[str]: A list of strings.
    List[float]: A list of floats.
    List[int]: A list of integers.
    Example:
    >>> split_to_list("a,b,c,d")
    ['a', 'b', 'c', 'd']
    >>> split_to_list("1.1,2.2,3.3", 'float')
    [1.1, 2.2, 3.3]
    """
    if not input_str:
        output_list = False
    elif input_str == 'auto':
        output_list = 'auto'
    else:
        if input_str.startswith('c:'):
            return input_str[2:]
        else:
            if out_format == 'str':
                output_list = [i for i in input_str.split(',')]
            elif out_format == 'float':
                output_list = [float(i) for i in input_str.split(',')]
            elif out_format == 'int':
                output_list = [int(i) for i in input_str.split(',')]
            else:
                output_list = [i for i in input_str.split(',')]

    return output_list


def load_group_dict(input_path: pathlib.Path, 
                    animal_list: List[str], 
                    group_id: str='genotype'
                    ) -> dict:
    """
    Collect the group_id information from the specified input path.
    Parameters:
    input_path (Path): The path to the input data.
    animal_list (List[str]): The list of animal IDs.
    group_id (str): The group ID to collect.
    Returns:
    dict: The group dictionary.
    Example:
    >>> input_path = pathlib.Path('path/to/animal_id')
    >>> animal_list = ['animal1', 'animal2', 'animal3', 'animal4']
    >>> load_group_dict(input_path, animal_list)
    {'genotype_1': ['animal1', 'animal_2'], 'genotype_2': ['animal3', 'animal4']}
    """
    dict = {}
    for animal_id in animal_list:
        data_dir = input_path.joinpath(animal_id)
        params_fn = data_dir.joinpath('params.json')
        if params_fn.exists():
            with open(params_fn) as fn:
                params_dict = json.load(fn)
            try:
                g_id = params_dict['general'][group_id]
                if g_id in dict.keys():
                    dict[g_id].append(animal_id)
                else:
                    dict[g_id] = [animal_id]
            except KeyError:
                show_info(f"no group_id value (* {group_id}*) specified for {animal_id}")
                show_info(f"    --> skipping {animal_id}")
                pass

        else:
            show_info(f"No params.json file under {str(params_fn)}")
            show_info(f"    --> skipping {animal_id}")

    return dict


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
        # Ref: https://github.com/cortex-lab/allenCCF/blob/master/Browsing%20Functions/allenCCFbregma.m
        "whs_sd_rat_39um": [371, 72, 266],
        # Ref: Papp EA. Neuroimage. 2014 Aug 15;97:374-86. doi: 10.1016/j.neuroimage.2014.04.001.
        "azba_zfish_4um": [360, 0, 335]
        # Ref: no bregma for zebrafish
    }
    if atlas_id in bregma_dict.keys():
        bregma = bregma_dict[atlas_id]
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
    return bregma


def create_regi_dict(input_path: pathlib.Path, 

                     regi_chan: str) -> dict:
    """
    Create a registration information dictionary from the specified input path.
    
    Parameters:
    -----------
    input_path : pathlib.Path
        The path to the input directory containing the necessary files.
    regi_chan : str
        The registration channel to be used for obtaining registration information.
    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - 'input_path': The input path provided.
        - 'regi_dir': The directory containing registration information.
        - 'atlas': The atlas information from the parameters.
        - 'orientation': The orientation information from the parameters.
        - 'xyz_dict': The xyz dictionary from the atlas information.
    """
    regi_dir = get_info(input_path, 'sharpy_track', channel=regi_chan, only_dir=True)
    params_dict = load_params(input_path)

    regi_dict = {
        'input_path': input_path,
        'regi_dir': regi_dir,
        'atlas': params_dict['atlas_info']['atlas'],
        'orientation': params_dict['atlas_info']['orientation'],
        'xyz_dict': params_dict['atlas_info']['xyz_dict']
    }

    return regi_dict


def xyz_atlas_transform(triplet: List[int], 
                        regi_dict: dict, 
                        atlas_tuple: List[str]) -> List[int]:
    """
    Transpose xyz triplet to match atlas orientation.
    Parameters:
    -----------
    triplet : List[int]
        The xyz triplet to transform.
    regi_dict : dict
        The registration information dictionary.
    atlas_tuple : List[str]
        The atlas orientation tuple.
    Returns:
    --------
    List[int]
        The transformed xyz triplet.
    """
    # change indices of xyz triplet tuple to match atlas
    # list with [x,y,z] triplet
    xyz_tuple = tuple([regi_dict['xyz_dict']['x'][0], regi_dict['xyz_dict']['y'][0], regi_dict['xyz_dict']['z'][0]])
    index_match = [xyz_tuple.index(e) for e in atlas_tuple]

    triplet_new = [triplet[i] for i in index_match]

    return triplet_new


def get_decimal(res_tup: List[float]) -> List[float]:
    """
    Get decimal number for displaying accurate z-step size in registration widget.
    Parameters:
    -----------
    res_tup : List[float]
        The resolution tuple.
    Returns:
    --------
    List[float]
        The decimal list.
    """
    decimal_list = []
    for r in res_tup:
        step_float = r / 1000
        # by default keep 2 decimals
        decimal = 2
        while np.abs(np.round(step_float, decimal) - step_float) >= 0.01 * step_float:
            decimal += 1
        decimal_list.append(decimal)
    return decimal_list



def coord_mm_transform(triplet: Union[int,float], 
                       bregma: List[int], 
                       resolution_tuple: List[float], 
                       mm_to_coord: bool=False
                       ) -> Union[int,float]:
    """
    Transform coordinates from mm to pixel or vice versa.
    Parameters:
    -----------
    triplet : Union[int,float]
        The coordinate to transform.
    bregma : List[int]
        The bregma coordinates.
    resolution_tuple : List[float]
        The resolution tuple.
    mm_to_coord : bool
        Whether to transform from mm to pixel or vice versa.
    Returns:
    --------
    Union[int,float]
    """
    decimal_list = get_decimal(resolution_tuple)

    if mm_to_coord:
        triplet_new = [round(- coord / (res / 1000)) + br_coord for coord, br_coord, res in
                        zip(triplet, bregma, resolution_tuple)]
    else:
        triplet_new = []
        for coord, br_coord, res, decimal in zip(triplet, bregma, resolution_tuple, decimal_list):
            triplet_new.append(round((br_coord - coord) * (res / 1000), decimal))

    if len(triplet_new) == 1:
        return triplet_new[0]
    else:
        return triplet_new


def sort_ap_dv_ml(triplet: List[float], atlas_tuple: List[str]) -> List[float]:
    """
    Reorder the input triplet to match the atlas orientation.
    Parameters:
    -----------
    triplet : List[float]
        The xyz triplet to reorder.
    atlas_tuple : List[str]
        The atlas orientation tuple.
    Returns:
    --------
    List[float]
    """
    # sort input triplet in respective atlas convention to new tipled in [ap, dv, ml] order
    tgt_tuple = ('ap', 'si', 'rl')  # bg naming convention
    index_match = [atlas_tuple.index(e) for e in tgt_tuple]
    triplet_new = [triplet[i] for i in index_match]
    return triplet_new


def get_xyz(atlas: BrainGlobeAtlas, section_orient: str) -> dict:
    """
    Get the xyz dictionary from the atlas information.
    Parameters:
    -----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object.
    section_orient : str
        The section orientation.
    Returns:
    --------
    dict
    """
    # resolution tuple (width, height)
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


def get_available_atlases():
    """
    from: https://github.com/brainglobe/brainreg-segment  -- July 2023
    Get the available brainglobe atlases
    :return: Dict of available atlases (["name":version])
    """
    available_atlases = utils.conf_from_url(
        descriptors.remote_url_base.format("last_versions.conf")
    )
    available_atlases = dict(available_atlases["atlases"])
    # move "example_mouse_100um" to the back of the list
    available_atlases = {k: available_atlases[k] for k in available_atlases if k != 'example_mouse_100um'} \
                        | {k: available_atlases[k] for k in ['example_mouse_100um'] if k in available_atlases}
    return available_atlases


def get_atlas_dropdown():
    atlas_dict = {}
    for i, k in enumerate(get_available_atlases().keys()):
        atlas_dict.setdefault(k, k)
    atlas_keys = Enum("atlas_key", atlas_dict)
    return atlas_keys

def get_threshold_dropdown():
    func_list = dir(filters)
    func_list = [f for f in func_list if f.startswith('threshold')]
    idx_yen = func_list.index('threshold_yen')
    func_list = [func_list.pop(idx_yen)] + func_list
    func_dict = {}
    for f in func_list:
        func_dict.setdefault(f,f)
    func_keys = Enum("func_key", func_dict)
    return func_keys

def find_key_by_value(d: dict, 
                      target_value: Union[str,int]
                      ) -> Union[str,None]:
    """
    Get the key from a dictionary by its value.
    Assumes that the values are unique.

    Parameters:
    -----------
    d : dict
        The dictionary to search.
    target_value : Union[str,int]
        The value to search for.
    Returns:
    --------
    Union[str,None]
    """
    return next((key for key, value in d.items() if value == target_value), None)


if __name__ == "__main__":
    pass