import importlib

import cv2
import math
import numpy as np

from skimage.exposure import rescale_intensity
import tifffile as tiff
from napari_dmc_brainmap.utils import get_info


# todo warning for overwriting
def create_dirs(params, input_path):
    save_dirs = {}
    for operation in params['operations']:
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





def load_stitched_images(input_path, filter, image):
    #
    # load and return stitched image in R-made folder of same name
    im_fn = input_path.joinpath('stitched', filter, image + '_stitched.tif')
    if im_fn.exists():
        image = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)  # 0 for grayscale mode
    else:
        print(f'WARNING - no stitched images of name {image}_stitched.tif found in {im_fn}!')
        print('Do padding on images if _stitched.tif suffix is missing.')
        print('DMC-BrainMap requires single channel 16-bit tif images for preprocessing.')
        image = False
    return image

def adjust_contrast(data, contrast_tuple):
    # input is individual image as np.darray[y,x]
    adjusted_image = rescale_intensity(data, contrast_tuple)
    return adjusted_image

def downsample_image(data, scale_factor=False, resolution_tuple=False):
    # downsampling of image; default is just downsampling the image
    if scale_factor:
        size_tuple = (math.floor(data.shape[1]/scale_factor), math.floor(data.shape[0]/scale_factor))
    # if no scale factor provided  downsample to sharpy-track size: max-[800-1140]
    else:
        size_tuple = resolution_tuple  # (1140, 800)
    data_resized = cv2.resize(data, size_tuple)
    # data_resized = data_resized.astype('uint16') # what is necessary for uint16 dtype?
    return data_resized


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


def make_rgb(stack_dict, params):
    #
    # check which filter are there, and add a zero array for the missing one
    rgb_list = ['cy3', 'green', 'dapi'] # filters for R(ed)G(reen)B(lue) images
    missing_filters = list(set(rgb_list) - set(stack_dict.keys()))
    default_dtype = "uint16"

    if params['rgb_params']['downsampling'] > 1:
        scale_factor = params['rgb_params']['downsampling']
        for f in stack_dict.keys():
            if f in rgb_list:
                stack_dict[f] = downsample_image(stack_dict[f], scale_factor)

    if params['rgb_params']['contrast_adjustment']:
        for f in stack_dict.keys(): # adjust contrast of all filters
            if f in rgb_list:
                contrast_tuple = tuple(params['rgb_params'][f])
                stack_dict[f] = adjust_contrast(stack_dict[f], contrast_tuple)
    # dtype has to be either uint16 or uint8
    dtype_list = []
    dtype_chan = []
    for k,v in stack_dict.items():
        dtype_list.append(v.dtype)
        dtype_chan.append(k)
        if v.dtype == default_dtype:
            pass
        elif v.dtype == 'uint8':
            print("INFO: Stitched Image for Channel: [{}] has data type of 'uint8', "
                  "however 'uint16' is recommended. Please consider using 16-bit stitched image.\n"
                  "Continue with 'uint8' data type".format(k))
            default_dtype = "uint8"
        else:
            raise TypeError("ERROR: Stitched Image for Channel: [{}] has data type of {}, "
                            "however 'uint16' (recommended) or 'uint8' is required. ".format(k, v.dtype))
    # dtypes in stack_dict should be the same
    dtype_same = True
    for dtype,chan in zip(dtype_list,dtype_chan):
        if dtype != default_dtype:
            dtype_same = False
            print("ERROR: Handling Stitched Image as type: {}, but type : {} was found in Channel: [{}].".format(default_dtype, dtype, chan))
    if not dtype_same:
        raise TypeError("ERROR: Data types of stitched images are not the same between channels! Please check the image data types.")

    image_size = stack_dict[next(iter(stack_dict))].shape  # get the shape of the images
    # add empty array for missing filters
    for missing_filter in missing_filters:
        stack_dict[missing_filter] = np.zeros(image_size, dtype=default_dtype)
    rgb_stack = np.dstack((stack_dict['cy3'], stack_dict['green'], stack_dict['dapi'])).astype(default_dtype) # create a stack of all three channels
    rgb_stack_8bit = do_8bit(rgb_stack)  # convert to 8bit (RGB is 0-255)
    return rgb_stack_8bit


def make_single_channel(data, chan, params):

    if params['single_channel_params']['downsampling'] > 1:
        scale_factor = params['single_channel_params']['downsampling']
        data = downsample_image(data, scale_factor)

    if params['single_channel_params']['contrast_adjustment']:
        contrast_tuple = tuple(params['single_channel_params'][chan])
        data = adjust_contrast(data, contrast_tuple)

    return data


def make_sharpy_track(data, chan, params, resolution_tuple):

    #
    if params['sharpy_track_params']['contrast_adjustment']:
        contrast_tuple = tuple(params['sharpy_track_params'][chan])
        data = adjust_contrast(data, contrast_tuple)
    #
    # and we always downsample
    data = downsample_image(data, scale_factor=False, resolution_tuple=resolution_tuple)
    # check dtype of data, if 16-bit: convert to 8-bit, if 8-bit: do nothing, else raise error
    if data.dtype == 'uint16':
        data = do_8bit(data)
    elif data.dtype == 'uint8':
        pass
    else:
        raise TypeError("WARNING: data type of stitched image is neither uint8 nor uint16!")
    return data

# create a stack and process accordingly
def make_stack(stack_dict, filter_list, params):
    #
    if params['stack_params']['downsampling'] > 1:
        scale_factor = params['stack_params']['downsampling']
    for f in filter_list:
        if params['stack_params']['downsampling']:
            stack_dict[f] = downsample_image(stack_dict[f], scale_factor)
        if params['stack_params']['contrast_adjustment']:
            contrast_tuple = tuple(params['stack_params'][f])
            stack_dict[f] = adjust_contrast(stack_dict[f], contrast_tuple)
    return stack_dict

#
def make_binary(data, chan, params):


    if params['binary_params']['downsampling'] > 1:
        scale_factor = params['binary_params']['downsampling']
        data = downsample_image(data, scale_factor)
    if params['binary_params']['thresh_bool']:
        thresh = params['binary_params'][chan]
    else:
        thresh_func = params['binary_params']['thresh_func']
        module = importlib.import_module('skimage.filters')
        func = getattr(module, thresh_func)
        thresh = func(data)
    #
    # binary = data > thresh
    binary = do_8bit(data)
    binary[data < thresh] = 0
    binary[data >= thresh] = 255

    return binary

def select_chans(chan_list, filter_list, operation):
    if chan_list == ['all']:
        chans = filter_list
    else:
        non_match = list(set(chan_list).difference(filter_list))
        for n in non_match:
            print("WARNING -- selected " + n + " channel for " + operation + " not found in imaged channels!")
        chans = list(set(chan_list) & set(filter_list))
    return chans

def preprocess_images(im, filter_list, input_path, params, save_dirs, resolution_tuple):
    print("started with " + str(im))
    stack_dict = {}  # save all loaded filters as dict['filter'][array]
    for f in filter_list:
        stack_dict[f] = load_stitched_images(input_path, f, im)  # todo change to not load all channels
    # todo this in one function, not all the same if clauses
    if all(v is False for v in stack_dict.values()):  # no images
        pass
    else:
        if any(v is False for v in stack_dict.values()):  # some channel missing
            filter_pres = [f for f in stack_dict.keys() if isinstance(stack_dict[f], np.ndarray)]
        else:
            filter_pres = filter_list
        if params['operations']['sharpy_track']:
            chans = select_chans(params['sharpy_track_params']['channels'], filter_pres, 'sharpy_track')
            for chan in chans:
                downsampled_image = make_sharpy_track(stack_dict[chan].copy(), chan, params, resolution_tuple)
                ds_image_name = im + '_downsampled.tif'
                ds_image_path = save_dirs['sharpy_track'].joinpath(chan, ds_image_name)
                tiff.imwrite(str(ds_image_path), downsampled_image)
                # use tifffile to write images, cv2.imwrite resulted in some errors

        if params['operations']['rgb']:
            chans = select_chans(params['rgb_params']['channels'], filter_pres, 'rgb')
            rgb_dict = dict((c, stack_dict[c]) for c in chans)
            rgb_stack = make_rgb(rgb_dict, params)
            rgb_fn = im + '_RGB.tif'
            rgb_save_dir = save_dirs['rgb'].joinpath(rgb_fn)
            tiff.imwrite(str(rgb_save_dir), rgb_stack)

        if params['operations']['single_channel']:
            chans = select_chans(params['single_channel_params']['channels'], filter_pres, 'single_channel')
            for chan in chans:
                single_channel_image = make_single_channel(stack_dict[chan].copy(), chan, params)
                single_fn = im + '_single.tif'
                single_save_dir = save_dirs['single_channel'].joinpath(chan, single_fn)
                cv2.imwrite(str(single_save_dir), single_channel_image)

        if params['operations']['stack']:
            if params['stack_params']['channels'] == ['all']:
                image_stack = make_stack(stack_dict.copy(), filter_pres, params)
            else:
                image_stack = make_stack(stack_dict.copy(), params['stack_params']['channels'], params)
            save_stack_name = im + '_stack.tif'
            save_stack_path = save_dirs['stack'].joinpath(save_stack_name)
            save_zstack(save_stack_path, image_stack)

        if params['operations']['binary']:
            chans = select_chans(params['binary_params']['channels'], filter_pres, 'binary_params')
            for chan in chans:
                binary_image = make_binary(stack_dict[chan].copy(), chan, params)
                binary_image_name = im + '_binary.tif'
                binary_image_path = save_dirs['binary'].joinpath(chan, binary_image_name)
                tiff.imwrite(str(binary_image_path), binary_image)



