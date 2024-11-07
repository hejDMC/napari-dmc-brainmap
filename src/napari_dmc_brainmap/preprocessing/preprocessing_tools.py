
import importlib
import cv2
import math
import numpy as np
from skimage.exposure import rescale_intensity
import tifffile as tiff
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils import get_info


# todo warning for overwriting
def create_dirs(params, input_path):
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

def get_channels(params):
    channels = []
    if 'operations' in params.keys():
        operation_list = list(params['operations'].keys())
        for operation in operation_list:
            channels.extend(params[f"{operation}_params"]["channels"])
    # only keep unique values
    channels = list(set(channels))
    return channels

def load_stitched_images(input_path, chan, image):
    #
    # load and return stitched image in R-made folder of same name
    im_fn = input_path.joinpath('stitched', chan, image + '_stitched.tif')
    if im_fn.exists():
        image = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)  # 0 for grayscale mode
    else:
        show_info(f'WARNING - no stitched images of name {image}_stitched.tif found in {im_fn}!\n '
                  f'Do padding on images if _stitched.tif suffix is missing.\n '
                  f'DMC-BrainMap requires single channel 16-bit tif images for preprocessing.')
        image = False
    return image

def downsample_and_adjust_contrast(image, params, scale_key, contrast_key):
    """
    Downsample and adjust contrast for an image.
    """
    if params[scale_key] > 1:
        scale_factor = params[scale_key]
        size_tuple = (math.floor(image.shape[1] / scale_factor), math.floor(image.shape[0] / scale_factor))
        image = cv2.resize(image, size_tuple)

    if params[contrast_key]:
        contrast_tuple = tuple(params[contrast_key])
        image = rescale_intensity(image, contrast_tuple)

    return image


def do_8bit(data):
    # only if input data is 16-bit
    if data.dtype == 'uint16':
        data = data.astype(int)
        data8bit = (data >> 8).astype('uint8')
        return data8bit
    elif data.dtype == 'uint8':
        return data
    else:
        raise TypeError("Input data for bit shift is not uint8 or uint16! got {} instead.".format(data.dtype))


def save_zstack(path, stack_dict):
    #
    with tiff.TiffWriter(path) as tif:
        for value in stack_dict.values():
            tif.write(value)


def make_rgb(stack_dict, params, im, save_dirs, resolution_tuple):
    #
    # check which channels are there, and add a zero array for the missing one
    rgb_list = ['cy3', 'green', 'dapi'] # channels for R(ed)G(reen)B(lue) images
    missing_channels = list(set(rgb_list) - set(stack_dict.keys()))
    default_dtype = "uint16"

    for chan in stack_dict.keys():
        stack_dict[chan] = downsample_and_adjust_contrast(stack_dict[chan], params['rgb_params'], 'downsampling', chan)
    # todo check if still needed?
    # dtype has to be either uint16 or uint8
    # dtype_list = []
    # dtype_chan = []
    # for k,v in stack_dict.items():
    #     dtype_list.append(v.dtype)
    #     dtype_chan.append(k)
    #     if v.dtype == default_dtype:
    #         pass
    #     elif v.dtype == 'uint8':
    #         print("INFO: Stitched Image for Channel: [{}] has data type of 'uint8', "
    #               "however 'uint16' is recommended. Please consider using 16-bit stitched image.\n"
    #               "Continue with 'uint8' data type".format(k))
    #         default_dtype = "uint8"
    #     else:
    #         raise TypeError("ERROR: Stitched Image for Channel: [{}] has data type of {}, "
    #                         "however 'uint16' (recommended) or 'uint8' is required. ".format(k, v.dtype))
    # # dtypes in stack_dict should be the same
    # dtype_same = True
    # for dtype,chan in zip(dtype_list,dtype_chan):
    #     if dtype != default_dtype:
    #         dtype_same = False
    #         print("ERROR: Handling Stitched Image as type: {}, but type : {} was found in Channel: [{}].".format(default_dtype, dtype, chan))
    # if not dtype_same:
    #     raise TypeError("ERROR: Data types of stitched images are not the same between channels! Please check the image data types.")

    image_size = stack_dict[next(iter(stack_dict))].shape  # get the shape of the images
    # add empty array for missing filters
    for missing_chan in missing_channels:
        stack_dict[missing_chan] = np.zeros(image_size, dtype=default_dtype)

    rgb_stack = np.dstack((stack_dict['cy3'], stack_dict['green'], stack_dict['dapi'])).astype(default_dtype) # create a stack of all three channels
    rgb_stack_8bit = do_8bit(rgb_stack)  # convert to 8bit (RGB is 0-255)

    rgb_fn = im + '_RGB.tif'
    rgb_save_dir = save_dirs['rgb'].joinpath(rgb_fn)
    tiff.imwrite(str(rgb_save_dir), rgb_stack_8bit)



def make_single_channel(stack_dict, params, im, save_dirs, resolution_tuple):
    for chan in stack_dict.keys():
        single_channel_image = downsample_and_adjust_contrast(stack_dict[chan], params['single_channel_params'],
                                                              'downsampling', chan)
        single_fn = im + '_single.tif'
        single_save_dir = save_dirs['single_channel'].joinpath(chan, single_fn)
        tiff.imwrite(str(single_save_dir), single_channel_image)


def make_sharpy_track(stack_dict, params, im, save_dirs, resolution_tuple):

    for chan in stack_dict.keys():
        if params['sharpy_track_params']['contrast_adjustment']:
            contrast_tuple = tuple(params['sharpy_track_params'][chan])
            sharpy_image = rescale_intensity(stack_dict[chan], contrast_tuple)
        sharpy_image = cv2.resize(sharpy_image, resolution_tuple)
        sharpy_image = do_8bit(sharpy_image)
        ds_image_name = im + '_downsampled.tif'
        ds_image_path = save_dirs['sharpy_track'].joinpath(chan, ds_image_name)
        tiff.imwrite(str(ds_image_path), sharpy_image)

# create a stack and process accordingly
def make_stack(stack_dict, params, im, save_dirs, resolution_tuple):

    for chan in stack_dict.keys():
        stack_dict[chan] = downsample_and_adjust_contrast(stack_dict[chan], params['stack_params'],
                                                          'downsampling', chan)
    save_stack_name = im + '_stack.tif'
    save_stack_path = save_dirs['stack'].joinpath(save_stack_name)
    save_zstack(save_stack_path, stack_dict)


#
def make_binary(stack_dict, params, im, save_dirs, resolution_tuple):
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
        # binary = data > thresh
        binary = do_8bit(image)
        binary[image < thresh] = 0
        binary[image >= thresh] = 255

        binary_image_name = im + '_binary.tif'
        binary_image_path = save_dirs['binary'].joinpath(chan, binary_image_name)
        tiff.imwrite(str(binary_image_path), binary)

def select_chans(chan_list, filter_list, operation):
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


def preprocess_images(im, channels, input_path, params, save_dirs, resolution_tuple):
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


