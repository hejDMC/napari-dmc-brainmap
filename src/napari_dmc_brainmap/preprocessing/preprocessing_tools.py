from pathlib import Path
import importlib
import cv2
import math
import numpy as np
from skimage.exposure import rescale_intensity
import tifffile
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.path_utils import get_info
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


RGB_CHANNELS = ('cy3', 'green', 'dapi')
AUTOMATIC_RGB_PROFILES = {
    'balanced_cells': {
        'label': 'Balanced (cells)',
        'low_clip_fraction': 0.00175,
        'high_clip_fraction': 0.00175,
    },
    'preserve_strong_signal_probe': {
        'label': 'Preserve strong signal (probe)',
        'low_clip_fraction': 0.00175,
        'section_high_quantile': 0.9999,
        'dataset_high_quantile': 0.99,
    },
}


def chunk_list(input_list: List[str], chunk_size: int = 4) -> List[List[str]]:
    """
    Split a list into smaller chunks of a specified size.

    Parameters:
        input_list (List[str]): The list to be divided into chunks.
        chunk_size (int): The maximum size of each chunk. Default is 4.

    Returns:
        List[List[str]]: A list containing smaller lists (chunks).
    """
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def create_dirs(params: Dict[str, Union[str, dict]], input_path: Union[str, Path]) -> Dict[str, Path]:
    """
    Create directories for saving processed images based on given parameters.

    Parameters:
        params (Dict[str, Union[str, dict]]): Preprocessing parameters including operations and channels.
        input_path (Union[str, Path]): The base path to the input directory.

    Returns:
        Dict[str, Path]: Dictionary with operation names as keys and created directory paths as values.
    """
    save_dirs = {}
    if 'operations' in params.keys():
        operation_list = list(params['operations'].keys())
        for operation in operation_list:
            if params['operations'][operation]:
                if operation in ['rgb', 'stack']:
                    data_dir = get_info(input_path, operation, create_dir=True, only_dir=True)
                    save_dirs[operation] = data_dir
                else:
                    chan_list = params[f"{operation}_params"]['channels']
                    filter_list = params['general']['chans_imaged']
                    chans = select_chans(chan_list, filter_list, operation)
                    for chan in chans:
                        data_dir = get_info(input_path, operation, channel=chan, create_dir=True, only_dir=True)
                        save_dirs[operation] = data_dir.parent
    return save_dirs


def get_channels(params: Dict[str, Union[str, dict]]) -> List[str]:
    """
    Extract a unique list of channels from preprocessing parameters.

    Parameters:
        params (Dict[str, Union[str, dict]]): Preprocessing parameters including operations and channels.

    Returns:
        List[str]: List of unique channels to be processed.
    """
    channels = []
    if 'operations' in params.keys():
        operation_list = list(params['operations'].keys())
        for operation in operation_list:
            channels.extend(params[f"{operation}_params"]["channels"])
    return list(dict.fromkeys(channels))


def load_stitched_images(input_path: Union[str, Path], chan: str, image: str) -> Union[np.ndarray, bool]:
    """
    Load a stitched image file from the specified directory.

    Parameters:
        input_path (Union[str, Path]): Path to the directory containing images.
        chan (str): Channel name.
        image (str): Image name (excluding suffix).

    Returns:
        Union[np.ndarray, bool]: Loaded image as a NumPy array, or False if the image is not found.
    """
    im_fn = input_path.joinpath('stitched', chan, f"{image}_stitched.tif")
    if not im_fn.exists():
        raise FileNotFoundError(f"WARNING: No stitched images named {image}_stitched.tif found in {im_fn}. "
        "Do padding on images if _stitched.tif suffix is missing."
        "Ensure images have the '_stitched.tif' suffix and are single-channel 16-bit."
        "Please restart the preprocessing widget to continue.")
    try:
        return tifffile.memmap(im_fn, mode="r")
    except (OSError, TypeError, ValueError):
        # Compressed TIFF data cannot be memory-mapped. Keep the existing
        # readers as compatibility fallbacks for compressed and unusual TIFFs.
        pass

    try:
        img = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError("cv2.imread returned None")
        return img
    except Exception as e:
        print(f"[INFO] OpenCV failed: {e}\nFalling back to tifffile...")
        return tifffile.imread(im_fn)


def find_shared_content_bounds(
    images: Sequence[np.ndarray],
) -> Union[Tuple[slice, slice], None]:
    """Return the bounding box outside shared all-channel black padding."""
    if not images:
        return None

    shape = images[0].shape
    if len(shape) != 2:
        raise ValueError(f"Expected a 2D stitched image, got shape {shape}")

    row_has_data = np.zeros(shape[0], dtype=bool)
    column_has_data = np.zeros(shape[1], dtype=bool)
    for image in images:
        if image.shape != shape:
            raise ValueError(
                "All channels for a section must have the same shape; "
                f"found {shape} and {image.shape}."
            )
        nonzero = image != 0
        if np.issubdtype(image.dtype, np.floating):
            nonzero &= np.isfinite(image)
        row_has_data |= np.any(nonzero, axis=1)
        column_has_data |= np.any(nonzero, axis=0)

    rows = np.flatnonzero(row_has_data)
    columns = np.flatnonzero(column_has_data)
    if not rows.size or not columns.size:
        return None
    return (
        slice(int(rows[0]), int(rows[-1]) + 1),
        slice(int(columns[0]), int(columns[-1]) + 1),
    )


def image_histogram(
    image: np.ndarray,
    bounds: Tuple[slice, slice],
    chunk_rows: int = 512,
) -> np.ndarray:
    """Calculate an exact uint8/uint16 histogram inside ``bounds``."""
    if image.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise TypeError(
            "Automatic RGB contrast supports uint8 and uint16 stitched "
            f"images, not {image.dtype}."
        )

    row_slice, column_slice = bounds
    row_start = 0 if row_slice.start is None else row_slice.start
    row_stop = image.shape[0] if row_slice.stop is None else row_slice.stop
    histogram = np.zeros(65536, dtype=np.uint64)
    for start in range(row_start, row_stop, chunk_rows):
        stop = min(start + chunk_rows, row_stop)
        chunk = np.asarray(image[start:stop, column_slice]).reshape(-1)
        histogram += np.bincount(chunk, minlength=65536).astype(
            np.uint64, copy=False
        )
    return histogram


def histogram_quantile(histogram: np.ndarray, quantile: float) -> int:
    """Return an inverse-CDF quantile from histogram counts or weights."""
    if not 0 <= quantile <= 1:
        raise ValueError(f"Quantile must be in [0, 1], got {quantile}.")
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("Cannot estimate contrast from an empty histogram.")
    cumulative = np.cumsum(histogram, dtype=np.float64)
    if quantile == 1:
        return int(np.flatnonzero(histogram > 0)[-1])
    return int(
        np.searchsorted(cumulative, quantile * total, side='right')
    )


def average_normalized_histograms(
    histograms: Sequence[np.ndarray],
) -> np.ndarray:
    """Combine histograms while giving every eligible section equal weight."""
    if not histograms:
        raise ValueError("At least one section histogram is required.")
    combined = np.zeros(65536, dtype=np.float64)
    for histogram in histograms:
        total = float(histogram.sum())
        if total > 0:
            combined += histogram / total
    total = combined.sum()
    if total <= 0:
        raise ValueError("All section histograms are empty.")
    return combined / total


def _stable_histogram_range(
    histogram: np.ndarray,
    low: int,
    high: int,
) -> List[int]:
    """Return a non-degenerate integer range supported by the histogram."""
    if high > low:
        return [low, high]
    occupied = np.flatnonzero(histogram > 0)
    if occupied.size > 1:
        return [int(occupied[0]), int(occupied[-1])]
    value = int(occupied[0]) if occupied.size else 0
    return [value, value + 1] if value < 65535 else [65534, 65535]


def estimate_automatic_rgb_ranges(
    input_path: Union[str, Path],
    image_names: Sequence[str],
    source_channels: Sequence[str],
    target_profiles: Mapping[str, str],
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Dict[str, object]:
    """Estimate one reproducible automatic contrast range per RGB channel."""
    source_channels = list(dict.fromkeys(source_channels))
    if not source_channels:
        raise ValueError("At least one imaged source channel is required.")
    if not image_names:
        raise ValueError("No stitched images were found for calibration.")

    unknown_channels = set(target_profiles) - set(RGB_CHANNELS)
    if unknown_channels:
        raise ValueError(
            f"Unsupported RGB channels: {sorted(unknown_channels)}"
        )
    missing_sources = set(target_profiles) - set(source_channels)
    if missing_sources:
        raise ValueError(
            "Automatic RGB channels must be present in imaged channels: "
            f"{sorted(missing_sources)}"
        )
    for channel, profile in target_profiles.items():
        if profile not in AUTOMATIC_RGB_PROFILES:
            raise ValueError(
                f"Unknown automatic profile {profile!r} for {channel}."
            )

    combined_histograms = {
        channel: np.zeros(65536, dtype=np.float64)
        for channel in target_profiles
    }
    sections_used = {channel: 0 for channel in target_profiles}
    section_highs = {channel: [] for channel in target_profiles}
    padding_fractions = []

    total_sections = len(image_names)
    for section_index, image_name in enumerate(image_names, start=1):
        images = {
            channel: load_stitched_images(input_path, channel, image_name)
            for channel in source_channels
        }
        bounds = find_shared_content_bounds(list(images.values()))
        if bounds is None:
            if progress_callback is not None:
                progress_callback(
                    {
                        'event': 'section',
                        'section_index': section_index,
                        'total_sections': total_sections,
                        'image_name': image_name,
                        'status': 'skipped_all_black',
                        'padding_fraction': 1.0,
                        'channel_cutoffs': {},
                    }
                )
            continue

        shape = next(iter(images.values())).shape
        valid_area = (
            (bounds[0].stop - bounds[0].start)
            * (bounds[1].stop - bounds[1].start)
        )
        padding_fraction = 1.0 - valid_area / np.prod(shape)
        padding_fractions.append(padding_fraction)
        channel_cutoffs = {}

        for channel, profile in target_profiles.items():
            histogram = image_histogram(images[channel], bounds)
            if not histogram.sum():
                continue
            combined_histograms[channel] += histogram / histogram.sum()
            sections_used[channel] += 1
            profile_config = AUTOMATIC_RGB_PROFILES[profile]
            section_low = histogram_quantile(
                histogram, profile_config['low_clip_fraction']
            )
            if profile == 'preserve_strong_signal_probe':
                section_high = histogram_quantile(
                    histogram,
                    profile_config['section_high_quantile'],
                )
                section_highs[channel].append(section_high)
            else:
                section_high = histogram_quantile(
                    histogram,
                    1.0 - profile_config['high_clip_fraction'],
                )
            channel_cutoffs[channel] = _stable_histogram_range(
                histogram, section_low, section_high
            )

        if progress_callback is not None:
            progress_callback(
                {
                    'event': 'section',
                    'section_index': section_index,
                    'total_sections': total_sections,
                    'image_name': image_name,
                    'status': 'processed',
                    'padding_fraction': float(padding_fraction),
                    'channel_cutoffs': channel_cutoffs,
                }
            )

    ranges = {}
    clipped_percent = {}
    for channel, profile in target_profiles.items():
        if not sections_used[channel]:
            raise ValueError(
                f"No valid histogram pixels were found for {channel}."
            )
        combined = combined_histograms[channel] / sections_used[channel]
        profile_config = AUTOMATIC_RGB_PROFILES[profile]
        low = histogram_quantile(
            combined, profile_config['low_clip_fraction']
        )
        if profile == 'balanced_cells':
            high = histogram_quantile(
                combined, 1.0 - profile_config['high_clip_fraction']
            )
        else:
            high = int(
                np.quantile(
                    section_highs[channel],
                    profile_config['dataset_high_quantile'],
                    method='higher',
                )
            )
        low, high = _stable_histogram_range(combined, low, high)
        ranges[channel] = [low, high]
        total = combined.sum()
        clipped_percent[channel] = {
            'low': float(combined[:low].sum() / total * 100),
            'high': float(combined[high + 1:].sum() / total * 100),
        }
    calibration = {
        'automatic_ranges': ranges,
        'automatic_profiles': dict(target_profiles),
        'automatic_clipped_percent': clipped_percent,
        'automatic_calibration': {
            'method': 'equal_section_histogram',
            'padding_exclusion': 'shared_all_channel_black_border',
            'source_channels': source_channels,
            'sections_requested': len(image_names),
            'sections_used': sections_used,
            'mean_padding_fraction': (
                float(np.mean(padding_fractions))
                if padding_fractions else 0.0
            ),
        },
    }
    if progress_callback is not None:
        progress_callback(
            {
                'event': 'complete',
                'section_index': total_sections,
                'total_sections': total_sections,
                'automatic_ranges': ranges,
            }
        )
    return calibration


def resolve_automatic_rgb_params(
    input_path: Union[str, Path],
    image_names: Sequence[str],
    params: Dict[str, object],
) -> Dict[str, object]:
    """Populate automatic ranges in RGB parameters when they are not cached."""
    rgb_params = params['rgb_params']
    if rgb_params.get('contrast_mode') != 'automatic':
        return {}

    channels = rgb_params['channels']
    profiles = rgb_params['automatic_profiles']
    ranges = rgb_params.get('automatic_ranges', {})
    if not all(channel in ranges for channel in channels):
        calibration = estimate_automatic_rgb_ranges(
            input_path,
            image_names,
            params['general']['chans_imaged'],
            {channel: profiles[channel] for channel in channels},
        )
        rgb_params.update(calibration)
    else:
        calibration = {
            key: rgb_params[key]
            for key in (
                'automatic_ranges',
                'automatic_profiles',
                'automatic_clipped_percent',
                'automatic_calibration',
            )
            if key in rgb_params
        }

    for channel in channels:
        rgb_params[channel] = list(rgb_params['automatic_ranges'][channel])
    rgb_params['contrast_adjustment'] = True
    return calibration


def downsample_and_adjust_contrast(
    image: np.ndarray,
    params: Dict[str, Union[str, list]],
    scale_key: str,
    contrast_key: str
) -> np.ndarray:
    """
    Downsample and adjust contrast of an image.

    Parameters:
        image (np.ndarray): Input image to process.
        params (Dict[str, Union[str, list]]): Parameters for scaling and contrast adjustment.
        scale_key (str): Key to retrieve downsampling factor.
        contrast_key (str): Key to retrieve contrast limits.

    Returns:
        np.ndarray: Processed image after downsampling and contrast adjustment.
    """
    if params[scale_key] > 1:
        scale_factor = params[scale_key]
        size_tuple = (math.floor(image.shape[1] / scale_factor), math.floor(image.shape[0] / scale_factor))
        image = cv2.resize(image, size_tuple)

    contrast_limits = params.get(contrast_key, [])
    adjust_contrast = params.get(
        'contrast_adjustment', bool(contrast_limits)
    )
    if adjust_contrast:
        if not contrast_limits:
            raise ValueError(
                f"Contrast adjustment is enabled, but no limits were "
                f"provided for channel '{contrast_key}'."
            )
        contrast_tuple = tuple(contrast_limits)
        image = rescale_intensity(image, contrast_tuple)

    return image


def do_8bit(data: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit image to 8-bit format.

    Parameters:
        data (np.ndarray): Input image in 16-bit or 8-bit format.

    Returns:
        np.ndarray: Converted 8-bit image.

    Raises:
        TypeError: If the input data is neither uint16 nor uint8.
    """
    if data.dtype == 'uint16':
        data = data.astype(int)
        return (data >> 8).astype('uint8')
    if data.dtype == 'uint8':
        return data
    raise TypeError(f"Unsupported data type for conversion to 8-bit: {data.dtype}")


def save_zstack(path: Union[str, Path], stack_dict: Dict[str, np.ndarray]) -> None:
    """
    Save a z-stack of images to a file.

    Parameters:
        path (Union[str, Path]): File path to save the z-stack.
        stack_dict (Dict[str, np.ndarray]): Dictionary of z-stack images.
    """
    with tifffile.TiffWriter(path) as tif:
        for value in stack_dict.values():
            tif.write(value)


def make_rgb(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, Path], resolution_tuple) -> None:
    """
    Create an RGB image from a stack of different channel images.

    Parameters:
        stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
        params (Dict[str, Union[str, dict]]): Parameters for processing.
        im (str): Image name.
        save_dirs (Dict[str, Path]): Save directories for processed images.
        resolution_tuple: Tuple indicating the resolution.
    """
    rgb_list = ['cy3', 'green', 'dapi']  # channels for R(ed)G(reen)B(lue) images
    missing_channels = list(set(rgb_list) - set(stack_dict.keys()))

    for chan in stack_dict.keys():
        stack_dict[chan] = downsample_and_adjust_contrast(stack_dict[chan], params['rgb_params'], 'downsampling', chan)

    image_size = stack_dict[next(iter(stack_dict))].shape  # get the shape of the images
    default_dtype = stack_dict[next(iter(stack_dict))].dtype  # get the default data type of the images
    
    for missing_chan in missing_channels:
        stack_dict[missing_chan] = np.zeros(image_size, dtype=default_dtype)

    rgb_stack = np.dstack((stack_dict['cy3'], stack_dict['green'], stack_dict['dapi'])).astype(default_dtype)  # create a stack of all three channels
    rgb_stack_8bit = do_8bit(rgb_stack)  # convert to 8bit (RGB is 0-255)

    rgb_fn = im + '_RGB.tif'
    rgb_save_dir = save_dirs['rgb'].joinpath(rgb_fn)
    tifffile.imwrite(str(rgb_save_dir), rgb_stack_8bit)


def make_single_channel(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, Path], resolution_tuple) -> None:
    """
    Create single-channel images from a stack of channel images.

    Parameters:
        stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
        params (Dict[str, Union[str, dict]]): Parameters for processing.
        im (str): Image name.
        save_dirs (Dict[str, Path]): Save directories for processed images.
        resolution_tuple: Tuple indicating the resolution.
    """
    for chan in stack_dict.keys():
        single_channel_image = downsample_and_adjust_contrast(stack_dict[chan], params['single_channel_params'],
                                                              'downsampling', chan)
        single_fn = im + '_single.tif'
        single_save_dir = save_dirs['single_channel'].joinpath(chan, single_fn)
        tifffile.imwrite(str(single_save_dir), single_channel_image)


def make_sharpy_track(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, str], resolution_tuple) -> None:
    """
    Create Sharpy-track images from a stack of channel images.

    Parameters:
    - stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
    - params (Dict[str, Union[str, dict]]): Parameters for processing.
    - im (str): Image name.
    - save_dirs (Dict[str, str]): Save directories for processed images.
    - resolution_tuple: Tuple indicating the resolution.
    """
    for chan in stack_dict.keys():
        sharpy_image = cv2.resize(stack_dict[chan], resolution_tuple)
        if params['sharpy_track_params']['contrast_adjustment']:
            contrast_tuple = tuple(params['sharpy_track_params'][chan])
            sharpy_image = rescale_intensity(sharpy_image, contrast_tuple)
        sharpy_image = do_8bit(sharpy_image)
        ds_image_name = im + '_downsampled.tif'
        ds_image_path = save_dirs['sharpy_track'].joinpath(chan, ds_image_name)
        tifffile.imwrite(str(ds_image_path), sharpy_image)


def make_stack(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, Path], resolution_tuple) -> None:
    """
    Create a z-stack of images from a dictionary of channel images.

    Parameters:
        stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
        params (Dict[str, Union[str, dict]]): Parameters for processing.
        im (str): Image name.
        save_dirs (Dict[str, Path]): Save directories for processed images.
        resolution_tuple: Tuple indicating the resolution.
    """
    for chan in stack_dict.keys():
        stack_dict[chan] = downsample_and_adjust_contrast(stack_dict[chan], params['stack_params'],
                                                          'downsampling', chan)
    save_stack_name = im + '_stack.tif'
    save_stack_path = save_dirs['stack'].joinpath(save_stack_name)
    save_zstack(save_stack_path, stack_dict)


def make_binary(
    stack_dict: Dict[str, np.ndarray],
    params: Dict[str, Union[str, dict]],
    im: str,
    save_dirs: Dict[str, Path],
    resolution_tuple: Tuple[int, int]
) -> None:
    """
    Create binary images for each channel based on a threshold.

    Parameters:
        stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
        params (Dict[str, Union[str, dict]]): Parameters for processing.
        im (str): Image name.
        save_dirs (Dict[str, Path]): Directories for saving processed images.
        resolution_tuple (Tuple[int, int]): Desired resolution for the output image.
    """
    for chan in stack_dict.keys():
        if params['binary_params']['downsampling'] > 1:
            scale_factor = params['binary_params']['downsampling']
            size_tuple = (math.floor(stack_dict[chan].shape[1] / scale_factor),
                          math.floor(stack_dict[chan].shape[0] / scale_factor))
            image = cv2.resize(stack_dict[chan], size_tuple)
        if params['binary_params']['manual_threshold']:
            threshold = params['binary_params'][chan]
        else:
            threshold_method = params['binary_params']['thresh_method']
            module = importlib.import_module('skimage.filters')
            threshold_func = getattr(module, threshold_method)
            threshold = threshold_func(image)

        binary_image = np.zeros_like(image, dtype=np.uint8)
        binary_image[image >= threshold] = 255

        # Save binary image
        binary_path = save_dirs['binary'].joinpath(chan, f"{im}_binary.tif")
        tifffile.imwrite(str(binary_path), binary_image)

def select_chans(chan_list: List[str], filter_list: List[str], operation: str) -> List[str]:
    """
    Select valid channels for a given operation.

    Parameters:
        chan_list (List[str]): List of requested channels.
        filter_list (List[str]): List of available channels.
        operation (str): Name of the operation.

    Returns:
        List[str]: List of selected channels.
    """
    if chan_list == ['all']:
        chans = filter_list
    else:
        non_match = list(set(chan_list).difference(filter_list))
        for n in non_match:
            show_info(f"WARNING -- selected {n} channel for {operation} not found in imaged channels!")
        chans = list(set(chan_list) & set(filter_list))
    return chans


PROCESSING_STEPS = {
    'sharpy_track': make_sharpy_track,
    'rgb': make_rgb,
    'single_channel': make_single_channel,
    'stack': make_stack,
    'binary': make_binary,
}


def preprocess_images(im: str, channels: List[str], input_path, params: Dict[str, Union[str, dict]], save_dirs: Dict[str, str], resolution_tuple) -> None:
    """
    Preprocess images for a given set of operations and channels.

    Parameters:
        im (str): Image name.
        channels (List[str]): List of channels to process.
        input_path (Union[str, Path]): Path to the input directory containing images.
        params (Dict[str, Union[str, dict]]): Parameters for preprocessing operations.
        save_dirs (Dict[str, Path]): Directories for saving processed images.
        resolution_tuple (Tuple[int, int]): Desired resolution for the output images.
    """
    stack_dict = {chan: load_stitched_images(input_path, chan, im) for chan in channels}

    # Skip processing if all channels are missing
    if all(stack is False for stack in stack_dict.values()):
        return

    # Filter present channels
    present_channels = [chan for chan, stack in stack_dict.items() if isinstance(stack, np.ndarray)]

    for operation, func in PROCESSING_STEPS.items():
        if operation in params['operations']:
            selected_channels = select_chans(
                params[f"{operation}_params"]['channels'], present_channels, operation
            )
            func({chan: stack_dict[chan] for chan in selected_channels}, params, im, save_dirs, resolution_tuple)
