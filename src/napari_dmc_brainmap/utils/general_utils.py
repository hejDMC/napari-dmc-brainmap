import pathlib
import re
from typing import Union, List, Dict, Optional, Tuple

from napari_dmc_brainmap.utils.params_utils import load_params


def split_strings_layers(s: str, atlas_name: str, return_str: bool = False) -> Tuple[str, Union[str, int]]:
    """
    Split a string into its head and tail components based on atlas-specific rules.

    Parameters:
        s (str): The input string to split.
        atlas_name (str): The name of the atlas (e.g., 'allen_mouse').
        return_str (bool): Whether to return the tail as a string if it's empty.

    Returns:
        [str, str]: The head and tail components of the string.
    """
    if atlas_name == 'allen_mouse':
        if s.startswith('CA'):
            head = s
            tail = []
        else:
            match = re.match(r"([A-Za-z-]+)(\d+.*)", s)
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


def split_to_list(input_str: Optional[str], out_format: str = 'str') -> Union[bool, str, List[Union[str, float, int]]]:
    """
    Split a user input string into a list of strings, floats, or integers.

    Parameters:
        input_str (Optional[str]): The user input string.
        out_format (str): The desired output format ('str', 'float', 'int').

    Returns:
        Union[bool, str, List[Union[str, float, int]]]: The processed output:
            - False if the input string is empty.
            - 'auto' if the input string is 'auto'.
            - A list of strings, floats, or integers depending on the specified format.

    Examples:
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


def find_key_by_value(d: Dict[str, Union[str, int]], target_value: Union[str, int]) -> Optional[str]:
    """
    Find the key in a dictionary corresponding to a given value.

    Parameters:
        d (Dict[str, Union[str, int]]): The dictionary to search.
        target_value (Union[str, int]): The value to look for.

    Returns:
        Optional[str]: The key associated with the target value, or None if not found.
    """
    return next((key for key, value in d.items() if value == target_value), None)


def get_animal_id(input_path: pathlib.Path) -> str:
    """
    Extract the animal ID from the input path.

    Parameters:
        input_path (pathlib.Path): The path to the animal data directory.

    Returns:
        str: The animal ID derived from the last part of the path.
    """
    animal_id = input_path.parts[-1]
    return animal_id


def create_regi_dict(input_path: pathlib.Path, regi_dir: pathlib.Path) -> Dict[str, Union[pathlib.Path, Dict]]:
    """
    Create a registration information dictionary from the specified input path.

    Parameters:
        input_path (pathlib.Path): The path to the input directory containing necessary files.
        regi_dir (pathlib.Path): The directory containing registration information.

    Returns:
        Dict[str, Union[pathlib.Path, Dict]]: A dictionary containing:
            - 'input_path': The provided input path.
            - 'regi_dir': The directory for registration information.
            - 'atlas': Atlas information from the parameters.
            - 'orientation': Orientation information from the parameters.
            - 'xyz_dict': XYZ dictionary from the atlas information.
    """
    params_dict = load_params(input_path)

    regi_dict = {
        'input_path': input_path,
        'regi_dir': regi_dir,
        'atlas': params_dict['atlas_info']['atlas'],
        'orientation': params_dict['atlas_info']['orientation'],
        'xyz_dict': params_dict['atlas_info']['xyz_dict']
    }

    return regi_dict
