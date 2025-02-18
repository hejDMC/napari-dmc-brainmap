from pathlib import Path
import tifffile
import numpy as np
import json
import cv2
from skimage.exposure import rescale_intensity
from typing import List, Tuple, Optional, Dict, Union
from napari.utils.notifications import show_info


def load_meta(section_dir: Path) -> Dict:
    """
    Load metadata from a .tif file in the specified directory.

    Parameters:
        section_dir (Path): Directory containing the .tif file.

    Returns:
        Dict: Metadata as a dictionary.
    """
    path_to_tiff = section_dir.joinpath([f.parts[-1] for f in section_dir.glob('*.tif')][0])
    with tifffile.TiffFile(path_to_tiff) as tif:
        meta_data = json.loads(tif.imagej_metadata['Info'])
    return meta_data


def get_size_json(pos_list: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Calculate the width and height of the grid from a list of positions.

    Parameters:
        pos_list (List[Tuple[int, int]]): List of positions as tuples of (x, y).

    Returns:
        Tuple[int, int]: Width and height of the grid.
    """
    pos = np.array(pos_list)
    pos_x = pos[:, 0]
    height = np.sum(np.abs(np.diff(pos_x).astype(int)) < 13) + 1
    width = int(len(pos_x) / height)
    return width, height


def map_loc(width: int, height: int) -> Dict[int, int]:
    """
    Generate a mapping of tile locations for constructing a stitched image.

    Parameters:
        width (int): Width of the grid.
        height (int): Height of the grid.

    Returns:
        Dict[int, int]: Dictionary mapping original tile indices to new locations.
    """
    total = int(width * height)
    new_loc = np.array([])

    if height % 2 == 0:  # even number of rows
        snake = 0  # natural row scanning direction
        range_max = total
        for i in range(height):
            if snake == 0:  # left to right
                this_row = np.arange(range_max - width, range_max, 1)
                new_loc = np.concatenate((new_loc, this_row))
                range_max = range_max - width - 1
                snake += 1
            else:
                this_row = np.arange(range_max, range_max - width, -1)
                new_loc = np.concatenate((new_loc, this_row))
                range_max = range_max - width + 1
                snake -= 1
    else:  # odd number of rows
        snake = 1  # reversed row scanning direction
        range_max = total - 1
        for i in range(height):
            if snake == 1:  # right to left
                this_row = np.arange(range_max, range_max - width, -1)
                new_loc = np.concatenate((new_loc, this_row))
                range_max = range_max - width + 1
                snake -= 1
            else:
                this_row = np.arange(range_max - width, range_max, 1)
                new_loc = np.concatenate((new_loc, this_row))
                range_max = range_max - width - 1
                snake += 1

    return {i: int(new_loc[i]) for i in range(total)}


def get_canvas(width: int,
               height: int,
               overlap: int = 205,
               c_size: int = 2048) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Create a blank canvas and generate a tile location map for stitching.

    Parameters:
        width (int): Width of the grid.
        height (int): Height of the grid.
        overlap (int, optional): Overlap between tiles. Defaults to 205.
        c_size (int, optional): Size of each tile. Defaults to 2048.

    Returns:
        Tuple[np.ndarray, Dict[int, int]]: Tuple containing the blank canvas and location map.
    """
    canvas_w = int(c_size * width) - int(overlap * (width - 1))
    canvas_h = int(c_size * height) - int(overlap * (height - 1))
    stitch_canvas = np.zeros((canvas_h, canvas_w), np.uint16)
    loc_map = map_loc(width, height)
    return stitch_canvas, loc_map

def fill_canvas(width: int,
                height: int,
                stitch_canvas: np.ndarray,
                loc_map: Dict[int, int],
                data_dict: Dict,
                overlap: int = 205,
                c_size: int = 2048,
                stack: bool = True) -> np.ndarray:
    """
    Fill the stitching canvas with image tiles.

    Parameters:
        width (int): Width of the grid.
        height (int): Height of the grid.
        stitch_canvas (np.ndarray): The canvas to be filled.
        loc_map (Dict[int, int]): Mapping of tile locations.
        data_dict (Dict): Dictionary containing image data.
        overlap (int, optional): Overlap between tiles. Defaults to 205.
        c_size (int, optional): Size of each tile. Defaults to 2048.
        stack (bool, optional): Whether to use a stack or individual sections. Defaults to True.

    Returns:
        np.ndarray: Filled stitching canvas.
    """
    if stack:
        whole_stack = data_dict['whole_stack']
    else:
        section_dir = data_dict['section_dir']
        data_list = data_dict['data_list']

    for j in range(height):
        for i in range(width):
            d_left, d_up = (0, 0)
            if stack:
                img = whole_stack[loc_map[int(width * j) + i]]
            else:
                img = cv2.imread(str(section_dir.joinpath(data_list[loc_map[int(width * j) + i]])), cv2.IMREAD_ANYDEPTH)

            d_left = int(overlap * i) if i > 0 else 0
            d_up = int(overlap * j) if j > 0 else 0

            try:
                stitch_canvas[
                    int(j * c_size) - d_up:int((j + 1) * c_size) - d_up,
                    int(i * c_size) - d_left:int((i + 1) * c_size) - d_left
                ] = img
            except:
                show_info("Image damaged during stitching.")
    return stitch_canvas


def stitch_stack(pos_list: List[Tuple[int, int]], whole_stack: np.ndarray, overlap: int, stitched_path: str,
                 params: dict, chan: str, downsampled_path: Optional[Path] = False,
                 resolution: Optional[Tuple[int, int]] = False) -> None:
    """
    Stitch a stack of images into a single image.
    Parameters:
        pos_list (List[Tuple[int, int]]): List of positions as tuples of (x, y).
        whole_stack (np.ndarray): Stack of images to be stitched.
        overlap (int): Overlap between tiles.
        stitched_path(str): Path to save the stitched image.
        params (dict): Dictionary of parameters.
        chan (str): Channel information.
        downsampled_path (Optional[Path]): Path to save the downsampled image (optional).
        resolution (Optional[Tuple[int, int]]): Resolution for padding (optional).
    """
    width, height = get_size_json(pos_list)
    pop_img = int(width * height)
    stitch_canvas, loc_map = get_canvas(width, height, overlap=overlap)
    data_dict = {'whole_stack': whole_stack}
    stitch_canvas = fill_canvas(width, height, stitch_canvas, loc_map, data_dict, overlap=overlap, stack=True)
    stitch_canvas = padding_for_atlas(stitch_canvas, resolution)

    tifffile.imwrite(stitched_path, stitch_canvas)

    if downsampled_path:
        contrast_tuple = tuple(params['sharpy_track_params'][chan])
        im_ds = downsample_image(stitch_canvas, resolution, contrast_tuple)
        tifffile.imwrite(downsampled_path, im_ds)


def stitch_folder(section_dir: Path, overlap: int, stitched_path: Path, params: dict, chan: str,
                  downsampled_path: Optional[Path] = False, resolution: Optional[Tuple[int, int]] = False) -> None:
    """
    Stitch images from a folder into a single image.
    Parameters:
        section_dir (Path): Directory containing the images.
        overlap (int): Overlap between tiles.
        stitched_path (Path): Path to save the stitched image.
        params (dict): Dictionary of parameters.
        chan (str): Channel information.
        downsampled_path (Optional[Path]): Path to save the downsampled image (optional).
        resolution (Optional[Tuple[int, int]]): Resolution for padding (optional).
    """
    meta_data = load_meta(section_dir)
    data_list = [meta_data['Prefix'] + "_MMStack_" + d['Label'] + '.ome.tif' for d in meta_data['StagePositions']]

    width = max([i['GridCol'] for i in meta_data['StagePositions']]) + 1
    height = max([i['GridRow'] for i in meta_data['StagePositions']]) + 1

    stitch_canvas, loc_map = get_canvas(width, height, overlap=overlap)
    data_dict = {'section_dir': section_dir, 'data_list': data_list}
    stitch_canvas = fill_canvas(width, height, stitch_canvas, loc_map, data_dict, overlap=overlap, stack=False)

    stitch_canvas = padding_for_atlas(stitch_canvas, resolution)

    tifffile.imwrite(stitched_path, stitch_canvas)

    if downsampled_path:
        contrast_tuple = tuple(params['sharpy_track_params'][chan])
        im_ds = downsample_image(stitch_canvas, resolution, contrast_tuple)
        tifffile.imwrite(downsampled_path, im_ds)

def downsample_image(input_tiff: Union[str, np.ndarray],
                     size_tuple: Tuple[int, int],
                     contrast_tuple: Tuple[int, int]) -> np.ndarray:
    """
    Downsample an image and adjust its brightness.

    Parameters:
        input_tiff (Union[str, np.ndarray]): Input image (file path or image matrix).
        size_tuple (Tuple[int, int]): Target size for downsampling.
        contrast_tuple (Tuple[int, int]): Contrast adjustment parameters.

    Returns:
        np.ndarray: Downsampled image as a NumPy array.
    """
    img = cv2.imread(input_tiff, cv2.IMREAD_ANYDEPTH) if isinstance(input_tiff, str) else input_tiff
    img_down = cv2.resize(img, size_tuple)
    img_down = rescale_intensity(img_down, contrast_tuple)
    img_8 = (img_down >> 8).astype('uint8')
    return cv2.cvtColor(img_8, cv2.COLOR_GRAY2RGB)


def padding_for_atlas(input_array: np.ndarray, resolution: Optional[Tuple[int, int]]) -> np.ndarray:
    """
    Apply padding to an image for atlas registration.

    Parameters:
        input_array (np.ndarray): Input image as a NumPy array.
        resolution (Optional[Tuple[int, int]]): Desired resolution for padding.

    Returns:
        np.ndarray: Padded image as a NumPy array.
    """
    if resolution:
        x, y = resolution
        tgt_ratio = x / y
        h, w = input_array.shape
        ratio = w / h
        if ratio == tgt_ratio:
            output_array = input_array
        elif ratio < tgt_ratio:
            dest_w = round(h / y * x)
            if (dest_w % 2) != 0:
                dest_w += 1
            d_w = int((dest_w - w) / 2)
            output_array = np.pad(input_array, ((0, 0), (d_w, d_w)), 'constant', constant_values=0)
        else:
            dest_h = round(w / x * y)
            if (dest_h % 2) != 0:
                dest_h += 1
            d_h = int((dest_h - h) / 2)
            output_array = np.pad(input_array, ((d_h, d_h), (0, 0)), 'constant', constant_values=0)
    else:
        output_array = input_array

    return output_array
