import cv2
import math
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_yen
from tifffile import TiffWriter
from napari_dmc_brainmap.utils import get_info

def create_dirs(params, input_path):
    save_dirs = {}  # todo this as function?
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
    image = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)  # 0 for grayscale mode
    return image

def adjust_contrast(data, key, params):
    # input is individual image as np.darray[y,x]
    adjusted_image = rescale_intensity(data, tuple(params['contrast_params'][key]))
    return adjusted_image

def downsample_image(data, params, sharpy_track = False): #(data, downsampDict):
    # downsampling of image; default is just downsampling the image (either scale or fixed values)
    # sharp-track option goes for sharpy-track size: max-[800-1140]
    if sharpy_track:
        data_resized = cv2.resize(data, (1140, 800))
    # else use the scale factor
    else:
        scale_factor = params['downsample_params']['scale_factor']
        data_resized = cv2.resize(data, (math.floor(data.shape[1]/scale_factor), math.floor(data.shape[0]/scale_factor)))
    data_resized = data_resized.astype('uint16')  # todo necessary?
    return data_resized


def do_8bit(data):
    #
    data = data.astype(int)
    data8bit = (data >> 8).astype('uint8')
    return data8bit


def save_zstack(path, stack_dict):
    #
    with TiffWriter(path) as tif:
        for value in stack_dict.values():
            tif.write(value)


def make_rgb(stack_dict, params):
    #
    # check which filter are there, and add a zero array for the missing one
    rgb_list = ['cy3', 'green', 'dapi'] # filters for R(ed)G(reen)B(lue) images
    missing_filters = list(set(rgb_list) - set(stack_dict.keys()))
    #
    if params['rgb_params']['contrast_adjustment']:
        for filter in stack_dict.keys(): # adjust contrast of all filters
            stack_dict[filter] = adjust_contrast(stack_dict[filter], filter, params)
    if params['rgb_params']['downsampling']:
        for filter in stack_dict.keys():
            stack_dict[filter] = downsample_image(stack_dict[filter], params)
    image_size = stack_dict[next(iter(stack_dict))].shape  # get the shape of the images
    # add empty array for missing filters
    for missing_filter in missing_filters:
        stack_dict[missing_filter] = np.zeros(image_size)
    rgb_stack = np.dstack((stack_dict['cy3'], stack_dict['green'], stack_dict['dapi'])) # create a stack of all three channels
    rgb_stack_8bit = do_8bit(rgb_stack) # convert to 8bit (RGB is 0-255)
    return rgb_stack_8bit


def make_single_channel(stack_dict, chan, params):
    #
    data = stack_dict[chan]
    #
    if params['single_channel_params']['contrast_adjustment']:
        data = adjust_contrast(data, chan, params)
    #
    if params['single_channel_params']['downsampling']:
        data = downsample_image(data, params)
    return data


def make_sharpy_track(data, params):

    #
    # always adjust contrast for sharp-track
    data = adjust_contrast(data, 'sharpy_track_contrast', params)
    #
    # and we always downsample
    data = downsample_image(data, params, sharpy_track=True)
    data = do_8bit(data)
    return data

# create a stack and process accordingly
def make_stack(stack_dict, filter_list, params):
    #
    for filter in filter_list:
        if params['stack_params']['contrast_adjustment']:
            stack_dict[filter] = adjust_contrast(stack_dict[filter], filter, params)
        if params['stack_params']['downsampling']:
            stack_dict[filter] = downsample_image(stack_dict[filter], params)
    return stack_dict

#
def make_binary(stack_dict, chan, params):
    #
    data = stack_dict[chan]
    #
    # we always downsample for now (binary only used in sharp-track pipeline
    data = downsample_image(data, params)
    # thresh = threshold_yen(data)
    thresh = threshold_yen(data)
    #
    binary = data > thresh
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


# todo if not individual ds params per operations, do all in one go?
def preprocess_images(im, filter_list, input_path, params, save_dirs):
    print("started with " + str(im))
    stack_dict = {}  # save all loaded filters as dict['filter'][array]
    for f in filter_list:
        stack_dict[f] = load_stitched_images(input_path, f, im)
    if params['operations']['sharpy_track']:
        chans = select_chans(params['sharpy_track_params']['channels'], filter_list, 'sharpy_track')
        for chan in chans:
            downsampled_image = make_sharpy_track(stack_dict[chan].copy(), params)
            ds_image_name = im + '_downsampled.tif'
            ds_image_path = save_dirs['sharpy_track'].joinpath(chan, ds_image_name)
            print(str(ds_image_path))
            cv2.imwrite(str(ds_image_path), downsampled_image)
    # if params['operations']['rgb']:
    #     rgb_stack = make_rgb(stack_dict.copy(), params)  # todo do I need to copy dict?
    #     rgb_fn = im + '_RGB.tif'
    #     rgb_save_dir = save_dirs['rgb'].joinpath(rgb_fn)
    #     cv2.imwrite(str(rgb_save_dir), cv2.cvtColor(rgb_stack, cv2.COLOR_RGB2BGR))  # cv2 defaults to BGR, reverse this to write RGB image
    #
    # if params['operations']['single_channel']:
    #     for chan in params['single_channel_params']['channels']:
    #         single_channel_image = make_single_channel(stack_dict.copy(), chan, params)
    #         single_fn = im + '_single.tif'
    #         single_save_dir = save_dirs['single_channel'].joinpath(chan, single_fn)
    #         cv2.imwrite(str(single_save_dir), single_channel_image)
    #
    # if params['operations']['stack']:
    #     if params['stack_params']['channels'] == ['all']:
    #         image_stack = make_stack(stack_dict.copy(), filter_list, params)
    #     else:
    #         image_stack = make_stack(stack_dict.copy(), params['stack_params']['channels'], params)
    #     save_stack_name = im + '_stack.tif'
    #     save_stack_path = save_dirs['stack'].joinpath(save_stack_name)
    #     save_zstack(save_stack_path, image_stack)
    #
    # if params['operations']['sharpy_track']:
    #     for chan in params['sharpy_track_params']['channels']:
    #         downsampled_image = make_sharpy_track(stack_dict.copy(), chan, params)
    #         ds_image_name = im + '_downsampled.tif'
    #         ds_image_path = save_dirs['sharpy_track'].joinpath(chan, ds_image_name)
    #         cv2.imwrite(str(ds_image_path), downsampled_image)
    #
    # if params['operations']['binary']:
    #     for chan in params['binary_params']['channels']:
    #         binary_image = make_binary(stack_dict.copy(), chan, params)
    #         binary_image_name = im + '_binary.tif'
    #         binary_image_path = save_dirs['binary'].joinpath(chan, binary_image_name)
    #         cv2.imwrite(str(binary_image_path), binary_image)