import json
import pathlib
from typing import Dict
from mergedeep import merge
from napari.utils.notifications import show_info


def load_params(input_path: pathlib.Path) -> Dict:
    """
    Load the `params.json` file from the specified input path.

    Parameters:
        input_path (pathlib.Path): Path to the directory where `params.json` should be located.

    Returns:
        Dict: The loaded `params.json` data.

    Raises:
        FileNotFoundError: If the `params.json` file is missing.

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
        raise FileNotFoundError(
            " ['Params.json'] file missing for " + input_path.parts[-1] + " \n"
            "Check Data Integrity at folder: {} \n"
            "and try again!".format(input_path)
        )


def clean_params_dict(params_dict: Dict, key: str) -> Dict:
    """
    Remove empty keys and processes that have not run from the params dictionary.

    Parameters:
        params_dict (Dict): The params dictionary.
        key (str): The key to clean within the params dictionary.

    Returns:
        Dict: The cleaned params dictionary.

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


def update_params_dict(input_path: pathlib.Path, params_dict: Dict, create: bool = False) -> Dict:
    """
    Update the `params.json` file with the specified dictionary.

    Parameters:
        input_path (pathlib.Path): Path to the directory where `params.json` should be located.
        params_dict (Dict): The dictionary to update the `params.json` file with.
        create (bool, optional): Whether to create the `params.json` file if it does not exist. Defaults to False.

    Returns:
        Dict: The updated params dictionary.

    Raises:
        FileNotFoundError: If the `params.json` file is missing and `create` is False.

    Example:
        >>> input_path = pathlib.Path('path/to/animal_id')
        >>> old_params_dict = {'key': 'value'}
        >>> new_params_dict = {'new_key': 'new_value'}
        >>> update_params_dict(input_path, new_params_dict)
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
        raise FileNotFoundError(
            " ['Params.json'] file missing for " + input_path.parts[-1] + " \n"
            "Check Data Integrity at folder: {} \n"
            "and try again!".format(input_path)
        )
