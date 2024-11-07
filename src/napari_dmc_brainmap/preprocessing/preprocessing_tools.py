import importlib
import cv2
import math
import numpy as np
from skimage.exposure import rescale_intensity
import tifffile as tiff
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils import get_info
from typing import Dict, List, Union


def create_dirs(params: Dict[str, Union[str, dict]], input_path) -> Dict[str, str]:
    """
    Create directories for saving processed images based on given parameters.

    Parameters:
    - params (Dict[str, Union[str, dict]]): Preprocessing parameters including operations and channels.
    - input_path: The path to the input directory.

    Returns:
    - Dict[str, str]: Dictionary containing paths to save directories for each operation.
    """
    save_dirs = {}
    if 'operations' in params.keys():
        operation_list = list(params['operations'].keys())
        for operation in operation_list:
            if params['operations'][operation]:
                if operation == 'rgb' or operation == 'stack':
                    data_dir = get_info(input_path, operation, create_dir=True, only_dir=True)
                    save_dirs[operation] = data_dir
                else:
                    chan_list = params[operation+'_params']['channels']
                    filter_list = params['general']['chans_imaged']
                    chans = select_chans(chan_list, filter_list, operation)
                    for chan in chans:
                        data_dir = get_info(input_path, operation, channel=chan, create_dir=True, only_dir=True)
                        save_dirs[operation] = data_dir.parent
    return save_dirs


def get_channels(params: Dict[str, Union[str, dict]]) -> List[str]:
    """
    Retrieve a list of channels from preprocessing parameters.

    Parameters:
    - params (Dict[str, Union[str, dict]]): Preprocessing parameters including operations and channels.

    Returns:
    - List[str]: List of unique channels to be processed.
    """
    channels = []
    if 'operations' in params.keys():
        operation_list = list(params['operations'].keys())
        for operation in operation_list:
            channels.extend(params[f"{operation}_params"]["channels"])
    # only keep unique values
    channels = list(set(channels))
    return channels


def load_stitched_images(input_path, chan: str, image: str):
    """
    Load and return stitched images from the specified directory.

    Parameters:
    - input_path: The path to the directory containing images.
    - chan (str): Channel name.
    - image (str): Image name.

    Returns:
    - The loaded image as a numpy array, or False if not found.
    """
    im_fn = input_path.joinpath('stitched', chan, image + '_stitched.tif')
    if im_fn.exists():
        image = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)  # 0 for grayscale mode
    else:
        show_info(f'WARNING - no stitched images of name {image}_stitched.tif found in {im_fn}!'
                  f'Do padding on images if _stitched.tif suffix is missing.'
                  f'DMC-BrainMap requires single channel 16-bit tif images for preprocessing.')
        image = False
    return image


def downsample_and_adjust_contrast(image: np.ndarray, params: Dict[str, Union[str, list]], scale_key: str, contrast_key: str) -> np.ndarray:
    """
    Downsample and adjust contrast for an image based on given parameters.

    Parameters:
    - image (np.ndarray): The image to be downsampled and contrast adjusted.
    - params (Dict[str, Union[str, list]]): Parameters for scaling and contrast adjustment.
    - scale_key (str): Key to retrieve scaling factor from params.
    - contrast_key (str): Key to retrieve contrast limits from params.

    Returns:
    - np.ndarray: The downsampled and contrast adjusted image.
    """
    if params[scale_key] > 1:
        scale_factor = params[scale_key]
        size_tuple = (math.floor(image.shape[1] / scale_factor), math.floor(image.shape[0] / scale_factor))
        image = cv2.resize(image, size_tuple)

    if params[contrast_key]:
        contrast_tuple = tuple(params[contrast_key])
        image = rescale_intensity(image, contrast_tuple)

    return image


def do_8bit(data: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit image to 8-bit, or return if it is already an 8-bit image.

    Parameters:
    - data (np.ndarray): The input image data.

    Returns:
    - np.ndarray: The image in 8-bit format.
    """
    if data.dtype == 'uint16':
        data = data.astype(int)
        data8bit = (data >> 8).astype('uint8')
        return data8bit
    elif data.dtype == 'uint8':
        return data
    else:
        raise TypeError("Input data for bit shift is not uint8 or uint16! got {} instead.".format(data.dtype))


def save_zstack(path, stack_dict: Dict[str, np.ndarray]) -> None:
    """
    Save a z-stack of images to a given path.

    Parameters:
    - path: The path to save the z-stack.
    - stack_dict (Dict[str, np.ndarray]): A dictionary containing the image stack data.
    """
    with tiff.TiffWriter(path) as tif:
        for value in stack_dict.values():
            tif.write(value)


def make_rgb(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, str], resolution_tuple) -> None:
    """
    Create an RGB image from a stack of different channel images.

    Parameters:
    - stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
    - params (Dict[str, Union[str, dict]]): Parameters for processing.
    - im (str): Image name.
    - save_dirs (Dict[str, str]): Save directories for processed images.
    - resolution_tuple: Tuple indicating the resolution.
    """
    rgb_list = ['cy3', 'green', 'dapi']  # channels for R(ed)G(reen)B(lue) images
    missing_channels = list(set(rgb_list) - set(stack_dict.keys()))
    default_dtype = "uint16"

    for chan in stack_dict.keys():
        stack_dict[chan] = downsample_and_adjust_contrast(stack_dict[chan], params['rgb_params'], 'downsampling', chan)

    image_size = stack_dict[next(iter(stack_dict))].shape  # get the shape of the images
    for missing_chan in missing_channels:
        stack_dict[missing_chan] = np.zeros(image_size, dtype=default_dtype)

    rgb_stack = np.dstack((stack_dict['cy3'], stack_dict['green'], stack_dict['dapi'])).astype(default_dtype)  # create a stack of all three channels
    rgb_stack_8bit = do_8bit(rgb_stack)  # convert to 8bit (RGB is 0-255)

    rgb_fn = im + '_RGB.tif'
    rgb_save_dir = save_dirs['rgb'].joinpath(rgb_fn)
    tiff.imwrite(str(rgb_save_dir), rgb_stack_8bit)


def make_single_channel(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, str], resolution_tuple) -> None:
    """
    Create single-channel images from a stack of channel images.

    Parameters:
    - stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
    - params (Dict[str, Union[str, dict]]): Parameters for processing.
    - im (str): Image name.
    - save_dirs (Dict[str, str]): Save directories for processed images.
    - resolution_tuple: Tuple indicating the resolution.
    """
    for chan in stack_dict.keys():
        single_channel_image = downsample_and_adjust_contrast(stack_dict[chan], params['single_channel_params'],
                                                              'downsampling', chan)
        single_fn = im + '_single.tif'
        single_save_dir = save_dirs['single_channel'].joinpath(chan, single_fn)
        tiff.imwrite(str(single_save_dir), single_channel_image)


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
        if params['sharpy_track_params']['contrast_adjustment']:
            contrast_tuple = tuple(params['sharpy_track_params'][chan])
            sharpy_image = rescale_intensity(stack_dict[chan], contrast_tuple)
        sharpy_image = cv2.resize(sharpy_image, resolution_tuple)
        sharpy_image = do_8bit(sharpy_image)
        ds_image_name = im + '_downsampled.tif'
        ds_image_path = save_dirs['sharpy_track'].joinpath(chan, ds_image_name)
        tiff.imwrite(str(ds_image_path), sharpy_image)


def make_stack(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, str], resolution_tuple) -> None:
    """
    Create a z-stack of images from a dictionary of channel images.

    Parameters:
    - stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
    - params (Dict[str, Union[str, dict]]): Parameters for processing.
    - im (str): Image name.
    - save_dirs (Dict[str, str]): Save directories for processed images.
    - resolution_tuple: Tuple indicating the resolution.
    """
    for chan in stack_dict.keys():
        stack_dict[chan] = downsample_and_adjust_contrast(stack_dict[chan], params['stack_params'],
                                                          'downsampling', chan)
    save_stack_name = im + '_stack.tif'
    save_stack_path = save_dirs['stack'].joinpath(save_stack_name)
    save_zstack(save_stack_path, stack_dict)


def make_binary(stack_dict: Dict[str, np.ndarray], params: Dict[str, Union[str, dict]], im: str, save_dirs: Dict[str, str], resolution_tuple) -> None:
    """
    Create binary images from a stack of channel images.

    Parameters:
    - stack_dict (Dict[str, np.ndarray]): Dictionary containing channel image stacks.
    - params (Dict[str, Union[str, dict]]): Parameters for processing.
    - im (str): Image name.
    - save_dirs (Dict[str, str]): Save directories for processed images.
    - resolution_tuple: Tuple indicating the resolution.
    """
    for chan in stack_dict.keys():
        if params['binary_params']['downsampling'] > 1:
            scale_factor = params['binary_params']['downsampling']
            size_tuple = (math.floor(stack_dict[chan].shape[1] / scale_factor),
                          math.floor(stack_dict[chan].shape[0] / scale_factor))
            image = cv2.resize(stack_dict[chan], size_tuple)
        if params['binary_params']['manual_threshold']:
            thresh = params['binary_params'][chan]
        else:
            thresh_method = params['binary_params']['thresh_method']
            module = importlib.import_module('skimage.filters')
            func = getattr(module, thresh_method)
            thresh = func(image)
        binary = do_8bit(image)
        binary[image < thresh] = 0
        binary[image >= thresh] = 255

        binary_image_name = im + '_binary.tif'
        binary_image_path = save_dirs['binary'].joinpath(chan, binary_image_name)
        tiff.imwrite(str(binary_image_path), binary)


def select_chans(chan_list: List[str], filter_list: List[str], operation: str) -> List[str]:
    """
    Select valid channels for a given operation.

    Parameters:
    - chan_list (List[str]): List of requested channels.
    - filter_list (List[str]): List of available channels.
    - operation (str): Name of the operation.

    Returns:
    - List[str]: List of selected channels.
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
    - im (str): Image name.
    - channels (List[str]): List of channels to be processed.
    - input_path: Path to the input directory containing images.
    - params (Dict[str, Union[str, dict]]): Parameters for preprocessing.
    - save_dirs (Dict[str, str]): Save directories for processed images.
    - resolution_tuple: Tuple indicating the resolution.
    """
    stack_dict = {}  # save all loaded channels as dict['chan'][array]
    for chan in channels:
        stack_dict[chan] = load_stitched_images(input_path, chan, im)
    # If no images found, skip processing
    if all(v is False for v in stack_dict.values()):
        return
    # Determine which filters are present
    chan_pres = [chan for chan in stack_dict.keys() if isinstance(stack_dict[chan], np.ndarray)]
    for operation, func in PROCESSING_STEPS.items():
        if operation in params['operations']:
            chans = select_chans(params[f'{operation}_params']['channels'], chan_pres, operation)
            # process data
            func({c: stack_dict[c] for c in chans}, params, im, save_dirs, resolution_tuple)
