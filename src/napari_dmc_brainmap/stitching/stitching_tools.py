import json
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import tifffile
from napari.utils.notifications import show_info
from skimage.exposure import rescale_intensity

from napari_dmc_brainmap.stitching.layout import (
    TILE_OVERLAP_PX,
    TILE_SIZE_PX,
    StitchLayout,
    layout_from_grid_metadata,
    layout_from_stage_positions,
)


def load_meta(section_dir: Path) -> Dict:
    """
    Load metadata from a .tif file in the specified directory.

    Parameters:
        section_dir (Path): Directory containing the .tif file.

    Returns:
        Dict: Metadata as a dictionary.
    """
    path_to_tiff = section_dir.joinpath([f.parts[-1] for f in section_dir.glob('*.tif') if not f.name.startswith('._')][0])
    with tifffile.TiffFile(path_to_tiff) as tif:
        meta_data = json.loads(tif.imagej_metadata['Info'])
    return meta_data


def get_atlas_padding(
    shape: Tuple[int, int],
    resolution: Optional[Tuple[int, int]],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return symmetric ``((top, bottom), (left, right))`` atlas padding."""
    if not resolution:
        return (0, 0), (0, 0)

    height, width = shape
    target_x, target_y = resolution
    target_ratio = target_x / target_y
    ratio = width / height

    if ratio == target_ratio:
        return (0, 0), (0, 0)
    if ratio < target_ratio:
        destination_width = round(height / target_y * target_x)
        if destination_width % 2:
            destination_width += 1
        horizontal = (destination_width - width) // 2
        return (0, 0), (horizontal, horizontal)

    destination_height = round(width / target_x * target_y)
    if destination_height % 2:
        destination_height += 1
    vertical = (destination_height - height) // 2
    return (vertical, vertical), (0, 0)


def fill_layout_canvas(
    stitch_canvas: np.ndarray,
    layout: StitchLayout,
    tile_loader: Callable[[int], Optional[np.ndarray]],
    *,
    overlap: int = TILE_OVERLAP_PX,
    c_size: int = TILE_SIZE_PX,
    offset: Tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Place tiles using rigid grid cells and deterministic row-major writes."""
    stride = c_size - overlap
    offset_y, offset_x = offset

    for placement in sorted(
        layout.placements,
        key=lambda tile: (tile.row, tile.column),
    ):
        image = tile_loader(placement.source_index)
        if image is None or image.shape != (c_size, c_size):
            show_info(
                "Tile:{} data corrupted or has an unexpected shape. "
                "Leaving its grid cell empty.".format(placement.source_index)
            )
            continue

        top = offset_y + placement.row * stride
        left = offset_x + placement.column * stride
        stitch_canvas[top:top + c_size, left:left + c_size] = image

    return stitch_canvas


def _open_output_canvas(
    stitched_path: Union[str, Path],
    layout: StitchLayout,
    resolution: Optional[Tuple[int, int]],
    *,
    overlap: int,
    c_size: int,
) -> tuple[np.memmap, Tuple[int, int]]:
    base_shape = (
        c_size * layout.height - overlap * (layout.height - 1),
        c_size * layout.width - overlap * (layout.width - 1),
    )
    vertical_padding, horizontal_padding = get_atlas_padding(
        base_shape,
        resolution,
    )
    output_shape = (
        base_shape[0] + sum(vertical_padding),
        base_shape[1] + sum(horizontal_padding),
    )
    canvas = tifffile.memmap(
        stitched_path,
        shape=output_shape,
        dtype=np.uint16,
        photometric="minisblack",
    )
    return canvas, (vertical_padding[0], horizontal_padding[0])


def stitch_stack(
    pos_list: Sequence[Sequence[float]],
    whole_stack: np.ndarray,
    overlap: int,
    stitched_path: str,
    params: dict,
    chan: str,
    downsampled_path: Optional[Path] = False,
    resolution: Optional[Tuple[int, int]] = False,
    *,
    c_size: int = TILE_SIZE_PX,
) -> StitchLayout:
    """
    Stitch a stack of images into a single image.
    Parameters:
        pos_list: Sequential XY stage positions.
        whole_stack (np.ndarray): Stack of images to be stitched.
        overlap (int): Overlap between tiles.
        stitched_path(str): Path to save the stitched image.
        params (dict): Dictionary of parameters.
        chan (str): Channel information.
        downsampled_path (Optional[Path]): Path to save the downsampled image (optional).
        resolution (Optional[Tuple[int, int]]): Resolution for padding (optional).
    """
    layout = layout_from_stage_positions(pos_list)
    if len(whole_stack) < layout.tile_count:
        raise ValueError(
            f"Expected {layout.tile_count} tiles, found {len(whole_stack)}."
        )
    stitch_canvas, offset = _open_output_canvas(
        stitched_path,
        layout,
        resolution,
        overlap=overlap,
        c_size=c_size,
    )
    try:
        fill_layout_canvas(
            stitch_canvas,
            layout,
            lambda source_index: whole_stack[source_index],
            overlap=overlap,
            c_size=c_size,
            offset=offset,
        )
        stitch_canvas.flush()

        if downsampled_path:
            contrast_tuple = tuple(params['sharpy_track_params'][chan])
            im_ds = downsample_image(stitch_canvas, resolution, contrast_tuple)
            tifffile.imwrite(downsampled_path, im_ds)
    finally:
        del stitch_canvas
    return layout


def stitch_folder(
    section_dir: Path,
    overlap: int,
    stitched_path: Path,
    params: dict,
    chan: str,
    downsampled_path: Optional[Path] = False,
    resolution: Optional[Tuple[int, int]] = False,
    *,
    c_size: int = TILE_SIZE_PX,
) -> StitchLayout:
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
    stage_positions = meta_data['StagePositions']
    layout = layout_from_grid_metadata(stage_positions)
    data_list = [
        meta_data['Prefix'] + "_MMStack_" + position['Label'] + '.ome.tif'
        for position in stage_positions
    ]
    stitch_canvas, offset = _open_output_canvas(
        stitched_path,
        layout,
        resolution,
        overlap=overlap,
        c_size=c_size,
    )

    def load_tile(source_index: int) -> Optional[np.ndarray]:
        return cv2.imread(
            str(section_dir.joinpath(data_list[source_index])),
            cv2.IMREAD_ANYDEPTH,
        )

    try:
        fill_layout_canvas(
            stitch_canvas,
            layout,
            load_tile,
            overlap=overlap,
            c_size=c_size,
            offset=offset,
        )
        stitch_canvas.flush()

        if downsampled_path:
            contrast_tuple = tuple(params['sharpy_track_params'][chan])
            im_ds = downsample_image(stitch_canvas, resolution, contrast_tuple)
            tifffile.imwrite(downsampled_path, im_ds)
    finally:
        del stitch_canvas
    return layout

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
    padding = get_atlas_padding(input_array.shape, resolution)
    if padding == ((0, 0), (0, 0)):
        return input_array
    return np.pad(input_array, padding, 'constant', constant_values=0)
