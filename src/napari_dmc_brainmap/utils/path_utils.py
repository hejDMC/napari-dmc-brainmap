from pathlib import Path
from typing import List, Tuple, Union
from natsort import natsorted
import pandas as pd
from napari.utils.notifications import show_info


def construct_path(input_path: Path, *sub_paths: str) -> Path:
    """
    Construct a path by joining the base path with sub-paths.

    Parameters:
        input_path (Path): The base path.
        sub_paths (str): Additional sub-paths to append.

    Returns:
        Path: The constructed path.
    """
    return input_path.joinpath(*sub_paths)


def get_data_dir(input_path: Path, folder_id: str, channel: str = None, seg_type: str = None) -> Path:
    """
    Construct the directory path based on provided parameters.

    Parameters:
        input_path (Path): The base path.
        folder_id (str): The folder identifier.
        channel (str, optional): The channel name. Defaults to None.
        seg_type (str, optional): The segmentation type. Defaults to None.

    Returns:
        Path: The constructed directory path.
    """
    sub_paths = [folder_id]
    if seg_type:
        sub_paths.append(seg_type)
    if channel:
        sub_paths.append(channel)
    return construct_path(input_path, *sub_paths)


def create_directory(directory: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters:
        directory (Path): The directory to create.
    """
    if not directory.exists():
        directory.mkdir(parents=True)
        show_info(f'Creating folder under: {directory}')


def get_data_list(data_dir: Path, pattern: str) -> List[str]:
    """
    Fetch a sorted list of files matching a pattern.

    Parameters:
        data_dir (Path): The directory to search.
        pattern (str): The glob pattern to match files.

    Returns:
        List[str]: A sorted list of file names.
    """
    if data_dir.exists():
        return natsorted([f.name for f in data_dir.glob(pattern)])
    return []


def get_info(input_path: Path,
             folder_id: str,
             channel: str = None,
             seg_type: str = None,
             create_dir: bool = False,
             only_dir: bool = False) -> Union[Path, Tuple[Path, List[str], str]]:
    """
    Main function to retrieve directory, file list, and suffix information.

    Parameters:
        input_path (Path): The base path.
        folder_id (str): The folder identifier.
        channel (str, optional): The channel name. Defaults to None.
        seg_type (str, optional): The segmentation type. Defaults to None.
        create_dir (bool, optional): Whether to create the directory if it doesn't exist. Defaults to False.
        only_dir (bool, optional): Whether to return only the directory. Defaults to False.

    Returns:
        Union[Path, Tuple[Path, List[str], str]]: The directory path, file list, and common suffix (or just the directory).
    """
    data_dir = get_data_dir(input_path, folder_id, channel, seg_type)
    if create_dir:
        create_directory(data_dir)
    if only_dir:
        return data_dir

    file_extension = '*.csv' if seg_type else '*.tif'
    data_list = get_data_list(data_dir, file_extension)

    data_suffix = ''
    if data_list:
        data_suffix = find_common_suffix(data_list)

    return data_dir, data_list, data_suffix


def find_common_suffix(file_list: List[str]) -> str:
    """
    Find the common suffix across multiple file names.

    Parameters:
        file_list (List[str]): A list of file names.

    Returns:
        str: The common suffix, or an empty string if no files exist.
    """
    if len(file_list) < 2:
        show_info(f'Only one file in folder: {file_list}')
        show_info(
            'In DMC-BrainMap, an image name has a base string and a suffix. '
            'For an image named *animal1_obj1_1_stitched.tif*, the base string is animal1_obj1_1, and the suffix is _stitched.tif.'
        )
        return input("Please, manually enter suffix: ") if file_list else ''

    suffix_length = 1
    while all(f[-suffix_length:] == file_list[0][-suffix_length:] for f in file_list):
        suffix_length += 1
    return file_list[0][-suffix_length + 1:]


def get_image_list(input_path: Path, chan: str, folder_id: str = 'stitched', file_id: str = '*.tif') -> List[str]:
    """
    Fetch the image list and remove the common suffix.

    Parameters:
        input_path (Path): The base path.
        chan (str): The channel name.
        folder_id (str, optional): The folder containing the images. Defaults to 'stitched'.
        file_id (str, optional): The file pattern to search for. Defaults to '`*.tif`'.

    Returns:
        List[str]: A list of image names with the common suffix removed.
    """
    im_list_file = input_path / 'image_names.csv'
    existing_list = []
    if im_list_file.exists():
        existing_list = pd.read_csv(im_list_file)['0'].tolist()

    data_dir = get_info(input_path, folder_id, only_dir=True)
    if not data_dir.parts[-1] == 'rgb':
        data_dir = data_dir / chan
        # data_dir = next((f for f in data_dir.glob('**/*') if f.is_dir()), None)
    data_list = get_data_list(data_dir, file_id)
    common_suffix = find_common_suffix(data_list)
    data_list = [f[:-len(common_suffix)] for f in data_list]
    # pd.DataFrame(data_list).to_csv(im_list_file, index=False)
    if set(existing_list) != set(data_list):
        merged_list = sorted(set(existing_list + data_list))
        pd.DataFrame(merged_list).to_csv(im_list_file, index=False)
        return merged_list
    return existing_list or data_list
