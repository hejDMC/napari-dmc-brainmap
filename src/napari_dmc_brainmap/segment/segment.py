from napari import Viewer
from napari.qt.threading import thread_worker
from natsort import natsorted
import cv2
from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import json
import numpy as np
import pandas as pd
import random
import matplotlib.colors as mcolors
from pathlib import Path

from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_3d
from aicssegmentation.core.utils import hole_filling

from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops, regionprops_table
from napari.utils.notifications import show_info

from bg_atlasapi import BrainGlobeAtlas

from napari_dmc_brainmap.utils import get_bregma, coord_mm_transform, split_to_list, get_image_list, get_info, \
    load_params
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from napari_dmc_brainmap.segment.segment_tools import loadAnnotBool, angleSlice, get_cmap

# def cmap_cells():
#     # return default colormap for channel and color of cells
#     cmap = {
#         'dapi': 'yellow',
#         'green': 'magenta',
#         'n3': 'gray',
#         'cy3': 'cyan',
#         'cy5': 'lightblue'
#     }
#     return cmap
#
#
# def cmap_npx():
#     # return default colormap for channel and color of cells
#     cmap = {
#         '0': 'deepskyblue',
#         '1': 'orange',
#         '2': 'springgreen',
#         '3': 'darkgray',
#         '4': 'fuchsia',
#         '5': 'royalblue',
#         '6': 'gold',
#         '7': 'powderblue',
#         '8': 'lightsalmon',
#         '9': 'olive'
#     }
#
#     return cmap
#
#
# def cmap_injection():
#     # return default colormap for channel and color of cells
#     cmap = {
#         'dapi': 'gold',
#         'green': 'purple',
#         'n3': 'navy',
#         'cy3': 'darkorange',
#         'cy5': 'cornflowerblue'
#     }
#     return cmap
#
#
#
# def cmap_display():
#     cmap = {
#         'dapi': 'blue',
#         'green': 'green',
#         'n3': 'orange',
#         'cy3': 'red',
#         'cy5': 'pink'
#     }
#     return cmap


def default_save_dict():
    save_dict = {
        "image_idx": False,
        "seg_type": False,
        "chan_list": False,
        "n_probes": False
    }
    return save_dict


def get_path_to_im(input_path, image_idx, single_channel=False, chan=False, pre_seg=False):
    if single_channel:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channel=chan)
    else:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
    im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
        image_idx]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
    path_to_im = seg_im_dir.joinpath(im)
    if pre_seg:
        im_list = get_image_list(input_path)  # to return im base name for loading preseg
        im_name_candidates = [i for i in im_list if im.startswith(i)]
        if len(im_name_candidates) == 1:
            im_name = im_name_candidates[0]
        elif len(im_name_candidates) == 2:
            im_name = im_name_candidates[1]
        else:
            print("Can't identify image name, image candidates:")
            print(im_name_candidates)
        return im_name
    else:
        return path_to_im

################################### PRESEG ########################################################
def load_segmentation_image(image_path: Path, channel: str, single_channel: bool, structure_channel: int) -> np.ndarray:
    """
    Load the segmentation image for the specified channel.

    Parameters:
    image_path (Path): Path to the image file.
    channel (str): Channel identifier.
    single_channel (bool): Whether the image is single channel or RGB.
    structure_channel (int): Index of the structure channel in the RGB image.

    Returns:
    np.ndarray: Loaded image.
    """
    reader = AICSImage(str(image_path))
    img = reader.data.astype(np.float32)
    if single_channel:
        return img[0, 0, 0, :, :].copy()
    else:
        return img[0, 0, 0, :, :, structure_channel].copy()


# Method refactoring
@thread_worker
def do_presegmentation(input_path, params_dict, channels, single_channel, regi_bool, regi_chan, preseg_params,
                       start_end_im, mask_folder, output_folder, seg_type='cells'):
    """
    Perform pre-segmentation on the provided image data.

    Parameters:
    input_path (Path): The path to the input images.
    params_dict (dict): Dictionary containing various parameters for the process.
    channels (list): List of channels to segment.
    single_channel (bool): Flag to indicate if the image is a single channel.
    regi_bool (bool): Whether the registration was performed.
    regi_chan (str): Registration channel.
    preseg_params (dict): Pre-segmentation parameters.
    start_end_im (list): Start and end indices for image range to presegment.
    mask_folder (str): Folder name for masks.
    output_folder (str): Folder name for output.
    seg_type (str): Segmentation type.
    """
    xyz_dict = params_dict['atlas_info']['xyz_dict']
    atlas_id = params_dict['atlas_info']['atlas']
    regi_list = []
    if regi_bool:
        try:
            regi_data, annot_bool, z_idx, z_res, bregma = _load_registration_data(input_path, regi_chan, atlas_id, xyz_dict)
            regi_list = [regi_data, annot_bool, z_idx, z_res, bregma]
        except FileNotFoundError:
            show_info('NO REGISTRATION DATA FOUND')
            regi_bool = False


    for chan in channels:
        mask_dir, output_dir = _prepare_segmentation_folders(input_path, mask_folder, output_folder, chan, seg_type)

        seg_im_dir, seg_im_list, seg_im_suffix = _load_image_list(input_path, start_end_im, channels, single_channel)

        for im in seg_im_list:
            print(f'Processing image: {im}')
            image_path = seg_im_dir.joinpath(im)
            chan_dict = {
                'cy3': 0,
                'green': 1,
                'dapi': 2
            }
            struct_img0 = load_segmentation_image(image_path, chan, single_channel, structure_channel=chan_dict[chan])
            struct_img0 = np.array([struct_img0, struct_img0])  # Duplicate layer stack

            structure_img_smooth = preprocess_image(struct_img0, preseg_params)

            segmentation = segment_image(structure_img_smooth, preseg_params)
            if np.mean(segmentation[0]) != 0:
                mask_save_fn = mask_dir.joinpath(im[:-len(seg_im_suffix)] + '_masks.tiff')
                save_segmentation_to_tiff(segmentation, mask_save_fn)

                csv_save_name = output_dir.joinpath(im.split('.')[0] + '_' + seg_type + '.csv')
                save_segmentation_to_csv(segmentation, csv_save_name, regi_bool, regi_list, xyz_dict, im, seg_im_suffix)

    print("DONE with presegmentation")


# Supporting functions
def preprocess_image(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Pre-process the image using intensity normalization and Gaussian smoothing.

    Parameters:
    image (np.ndarray): The image to preprocess.
    params (dict): Pre-segmentation parameters.

    Returns:
    np.ndarray: The preprocessed image.
    """
    image = intensity_normalization(image, scaling_param=params["intensity_norm"])
    return image_smoothing_gaussian_slice_by_slice(image, sigma=params["gaussian_smoothing_sigma"])


def segment_image(image: np.ndarray, params: dict) -> np.ndarray:
    """
    Segment the given image based on dot_3d filter and other parameters.

    Parameters:
    image (np.ndarray): The preprocessed image to segment.
    params (dict): Pre-segmentation parameters.

    Returns:
    np.ndarray: Binary segmented image.
    """
    response = dot_3d(image, log_sigma=params["dot_3d_sigma"])
    bw = response > params["dot_3d_cutoff"]
    bw_filled = hole_filling(bw, params["hole_min_max"][0], params["hole_min_max"][1], True)
    seg = remove_small_objects(bw_filled, min_size=params["minArea"], connectivity=1)
    # output segmentation binary image
    seg = seg > 0
    seg = seg.astype(np.uint8)
    seg[seg > 0] = 255
    return seg

def save_segmentation_to_tiff(segmentation, mask_save_fn):
    writer = OmeTiffWriter()
    writer.save(segmentation[0], str(mask_save_fn))

def save_segmentation_to_csv(segmentation: np.ndarray, file_path: Path, regi_bool: bool, regi_list, xyz_dict, im, seg_im_suffix):

    # Additional CSV saving logic can be added here if needed.
    label_img = label(segmentation[0])
    props = regionprops_table(label_img, properties=['centroid'])

    # Example: centroid detection (if regi_bool is True)
    if regi_bool:
        drop_bool = exclude_segment_objects(segmentation, props, regi_list, xyz_dict, im, seg_im_suffix)
    else:
        drop_bool = False
    csv_to_save = pd.DataFrame(props)
    csv_to_save.rename(columns={"centroid-0": "Position Y", "centroid-1": "Position X"}, inplace=True)
    if regi_bool and drop_bool:
        csv_to_save = csv_to_save.iloc[np.where(np.array(drop_bool) == 0)[0], :].copy().reset_index(
                drop=True)
    csv_to_save.to_csv(file_path)

def exclude_segment_objects(segmentation, props, regi_list, xyz_dict, im, seg_im_suffix):
    regi_data, annot_bool, z_idx, z_res, bregma  = regi_list
    dim_rgb = segmentation[0].shape
    x_res = xyz_dict['x'][1]
    y_res = xyz_dict['y'][1]
    x_rgb = props["centroid-1"] / dim_rgb[1] * x_res
    y_rgb = props["centroid-0"] / dim_rgb[0] * y_res
    for k, v in regi_data['imgName'].items():
        if v.startswith(im[:-(len(seg_im_suffix)-1)]):
            regi_index = k
    try:
        # get transformation
        tform = fitGeoTrans(regi_data['sampleDots'][regi_index], regi_data['atlasDots'][regi_index])
        # slice annotation volume
        x_angle, y_angle, z = regi_data['atlasLocation'][regi_index]

        annot_slice = angleSlice(x_angle, y_angle, z, annot_bool, z_idx, z_res, bregma, xyz_dict)
        # mark invalid coordinates
        drop_bool = []
        for x, y in zip(x_rgb, y_rgb):
            x_atlas, y_atlas = mapPointTransform(x, y, tform)
            x_atlas, y_atlas = int(x_atlas), int(y_atlas)
            if (x_atlas < 0) | (y_atlas < 0) | (x_atlas >= xyz_dict['x'][1]) | (
                    y_atlas >= xyz_dict['y'][1]):
                drop_bool.append(1)
            else:
                if annot_slice[y_atlas, x_atlas] == 0:
                    drop_bool.append(1)
                else:
                    drop_bool.append(0)
    except KeyError:
        show_info(f"No registration data for {regi_data['imgName'][regi_index]}")
        drop_bool = False

    return drop_bool

def _load_registration_data(input_path, regi_chan, atlas_id, xyz_dict):
    """
    Load the registration data.

    Parameters:
    input_path (Path): The path to the input images.
    regi_chan (str): Registration channel.
    atlas_id (str): Atlas identifier.
    xyz_dict (dict): Atlas information.

    Returns:
    tuple: Registration data, annotation boolean volume, z index, z resolution, bregma coordinates.
    """
    regi_dir = get_info(input_path, 'sharpy_track', channel=regi_chan, only_dir=True)
    regi_fn = regi_dir.joinpath("registration.json")
    if regi_fn.is_file():
        with open(regi_fn, 'r') as f:
            regi_data = json.load(f)
        annot_bool = loadAnnotBool(atlas_id)
        atlas = BrainGlobeAtlas(atlas_id)
        z_idx = atlas.space.axes_description.index(xyz_dict['z'][0])
        z_res = xyz_dict["z"][2]
        bregma = get_bregma(atlas_id)
        return regi_data, annot_bool, z_idx, z_res, bregma
    else:
        raise FileNotFoundError


def _prepare_segmentation_folders(input_path, mask_folder, output_folder, chan, seg_type):
    """
    Prepare directories for segmentation.

    Parameters:
    input_path (Path): Path to the input images.
    mask_folder (str): Name of the folder for masks.
    output_folder (str): Name of the output folder.
    chan (str): Channel identifier.
    seg_type (str): Segmentation type.

    Returns:
    tuple: Paths to mask and output directories.
    """
    mask_dir = get_info(input_path, mask_folder, channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
    output_dir = get_info(input_path, output_folder, channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
    return mask_dir, output_dir


def _load_image_list(input_path, start_end_im, channels, single_channel):
    """
    Load the list of images to be segmented.

    Parameters:
    input_path (Path): Path to the input images.
    start_end_im (list): Start and end indices for image range.
    channels (list): List of channels to segment.
    single_channel (bool): Whether the images are single channel or RGB.

    Returns:
    tuple: Directory path, list of images, and image suffix.
    """
    if single_channel:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channel=channels[0])
    else:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
    if start_end_im:
        if len(start_end_im) == 2:
            seg_im_list = seg_im_list[start_end_im[0]:start_end_im[1] + 1]
    return seg_im_dir, seg_im_list, seg_im_suffix
####################################################################################################333

@thread_worker
def create_projection_preseg(input_path, params_dict, channels, regi_bool, regi_chan, binary_folder, output_folder):
    xyz_dict = params_dict['atlas_info']['xyz_dict']
    atlas_id = params_dict['atlas_info']['atlas']
    if regi_bool:
        regi_dir = get_info(input_path, 'sharpy_track', channel=regi_chan, only_dir=True)
        regi_fn = regi_dir.joinpath("registration.json")
        if regi_fn.is_file():
            with open(regi_fn, 'r') as f:
                regi_data = json.load(f)
            annot_bool = loadAnnotBool(atlas_id)
            atlas = BrainGlobeAtlas(atlas_id)
            z_idx = atlas.space.axes_description.index(xyz_dict['z'][0])
            z_res = xyz_dict["z"][2]
            bregma = get_bregma(atlas_id)
        else:
            print("NO REGISTRATION DATA FOUND")
            regi_bool = False
    print('running presegmentation of ...')

    for chan in channels:
        binary_dir, binary_images, binary_suffix = get_info(input_path, binary_folder, channel=chan)
        output_dir = get_info(input_path, output_folder, seg_type='projections', channel=chan, create_dir=True,
                              only_dir=True)
        # binary_images = natsorted([im.parts[-1] for im in binary_dir.glob('*.tif')])
        for im_name in binary_images:
            print(f'... {im_name}')
            path_to_im = binary_dir.joinpath(im_name)
            image = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
            idx = np.where(image == 255)
            if regi_bool:  # exclude idx outside of brain
                dim_binary = image.shape
                x_res = xyz_dict['x'][1]
                y_res = xyz_dict['y'][1]
                x_binary = idx[1] / dim_binary[1] * x_res
                y_binary = idx[0] / dim_binary[0] * y_res
                for k, v in regi_data['imgName'].items():
                    if v.startswith(im_name[:-len(binary_suffix)]):
                        regi_index = k
                # get transformation
                tform = fitGeoTrans(regi_data['sampleDots'][regi_index], regi_data['atlasDots'][regi_index])
                # slice annotation volume
                x_angle, y_angle, z = regi_data['atlasLocation'][regi_index]

                annot_slice = angleSlice(x_angle, y_angle, z, annot_bool, z_idx, z_res, bregma, xyz_dict)
                # mark invalid coordinates
                drop_bool = []
                for x, y in zip(x_binary, y_binary):
                    x_atlas, y_atlas = mapPointTransform(x, y, tform)
                    x_atlas, y_atlas = int(x_atlas), int(y_atlas)
                    if (x_atlas < 0) | (y_atlas < 0) | (x_atlas >= xyz_dict['x'][1]) | (y_atlas >= xyz_dict['y'][1]):
                        drop_bool.append(1)
                    else:
                        if annot_slice[y_atlas, x_atlas] == 0:
                            drop_bool.append(1)
                        else:
                            drop_bool.append(0)
                csv_to_save = pd.DataFrame({'Position Y': idx[0], 'Position X': idx[1]})
                csv_to_save = csv_to_save.iloc[np.where(np.array(drop_bool) == 0)[0], :].copy().reset_index(
                    drop=True)
            else:
                csv_to_save = pd.DataFrame({'Position Y': idx[0], 'Position X': idx[1]})
            csv_save_name = output_dir.joinpath(im_name.split('.')[0] + '_projections.csv')
            csv_to_save.to_csv(csv_save_name)
        print(f"Done with {chan}")
    print('Done with creating projections presegmentation files!')


@thread_worker
def get_center_coord(input_path, channels, mask_folder, output_folder, mask_type='cells'):
    for chan in channels:
        mask_dir = get_info(input_path, mask_folder, seg_type=mask_type, channel=chan, only_dir=True)
        output_dir = get_info(input_path, output_folder, seg_type=mask_type, channel=chan, create_dir=True,
                              only_dir=True)
        mask_images = natsorted([im.parts[-1] for im in mask_dir.glob('*.tiff')])
        for im_name in mask_images:
            path_to_im = mask_dir.joinpath(im_name)
            image = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
            label_img = label(image)  # identify individual segmented structures
            regions = regionprops(
                label_img)  # get there properties -> we want to have the centroid point as a "location" of the cell
            cent = np.zeros((np.size(regions), 2))
            for idx, props in enumerate(regions):
                cent[idx, 0] = props.centroid[0]  # y-coordinates
                cent[idx, 1] = props.centroid[1]  # x-coordinates
            # create csv file in folders to match imaris output
            csv_to_save = pd.DataFrame(cent)
            csv_to_save = csv_to_save.rename(columns={0: "Position Y", 1: "Position X"})
            csv_save_name = output_dir.joinpath(im_name.split('.')[0] + '_' + mask_type + '.csv')
            csv_to_save.to_csv(csv_save_name)
            #
            location_binary = np.zeros((image.shape))  # make new binary with centers of segmented cells only
            cent = (np.round(cent)).astype(int)
            for val in cent:
                location_binary[val[0], val[1]] = 255
            location_binary = location_binary.astype(int)
            location_binary = location_binary.astype('uint8')  # convert to 8-bit
            save_name = im_name.split('.')[0] + '_centroids.tif'  # get the name
            cv2.imwrite(str(mask_dir.joinpath(save_name)), location_binary)
            # progress_bar.update(100 / len(binary_images))
        print(f"Done with {chan}")


def initialize_segment_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit',
                              label='input path (animal_id): ',
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                      'folder containing segmentation results'),
              single_channel_bool=dict(widget_type='CheckBox',
                                       text='use single channel',
                                       value=False,
                                       tooltip='tick to use single channel images (not RGB), one can still select '
                                               'multiple channels'),
              seg_type=dict(widget_type='ComboBox',
                            label='segmentation type',
                            choices=['cells', 'injection_site', 'optic_fiber', 'neuropixels_probe', 'projections'],
                            value='cells',
                            tooltip='select to either segment cells, projections, optic fiber tracts, probe tracts (points) or injection sites (regions) '
                                    'IMPORTANT: before switching between types, load next image, delete all image layers '
                                    'and reload image of interest!'),
              n_probes=dict(widget_type='LineEdit',
                            label='number of fibers/probes',
                            value=1,
                            tooltip='number (int) of optic fibres and or probes used to segment, leave this value unchanged for '
                                    'segmenting cells/injection site/projections'),
              point_size=dict(widget_type='LineEdit',
                              label='point size',
                              value=5,
                              tooltip='enter the size of points for cells/projections/optic fibers/neuropixels probes'),
              channels=dict(widget_type='Select',
                            label='selected channels',
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels to be used for segmentation, '
                                    'to select multiple hold ctrl/shift'),
              contrast_dapi=dict(widget_type='LineEdit',
                                 label='set contrast limits for the dapi channel',
                                 value='0,100',
                                 tooltip='enter contrast limits: min,max (default values for 8-bit image)'),
              contrast_green=dict(widget_type='LineEdit',
                                  label='set contrast limits for the green channel',
                                  value='0,100',
                                  tooltip='enter contrast limits: min,max (default values for 8-bit image)'),
              contrast_n3=dict(widget_type='LineEdit',
                               label='set contrast limits for the n3 channel',
                               value='0,100',
                               tooltip='enter contrast limits: min,max (default values for 8-bit image)'),
              contrast_cy3=dict(widget_type='LineEdit',
                                label='set contrast limits for the cy3 channel',
                                value='0,100',
                                tooltip='enter contrast limits: min,max (default values for 8-bit image)'),
              contrast_cy5=dict(widget_type='LineEdit',
                                label='set contrast limits for the cy5 channel',
                                value='0,100',
                                tooltip='enter contrast limits: min,max (default values for 8-bit image)'),
              image_idx=dict(widget_type='LineEdit',
                             label='image to be loaded',
                             value=0,
                             tooltip='index (int) of image to be loaded and segmented next'),
              call_button=False)
    def segment_widget(
            viewer: Viewer,
            input_path,  # posix path
            seg_type,
            n_probes,
            point_size,
            channels,
            contrast_dapi,
            contrast_green,
            contrast_n3,
            contrast_cy3,
            contrast_cy5,
            image_idx,
            single_channel_bool):
        pass

    return segment_widget


def initialize_loadpreseg_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              load_bool=dict(widget_type='CheckBox',
                             label='load presegmented data',
                             value=False,
                             tooltip='tick to load presegmented data for manual curation'),
              pre_seg_folder=dict(widget_type='LineEdit',
                                  label='folder name with presegmented data',
                                  value='presegmentation',
                                  tooltip='folder needs to contain sub-folders with channel names. WARNING: if the channel is called '
                                          '*segmentation*, manual curation will override existing data. '
                                          'Presegmented data needs to be .csv file and column names specifying *Position X* and '
                                          '*Position Y* for coordinates. For loading neuropixels/optic fiber data specify the number of probes correctly.'),
              call_button=False,
              scrollable=True)
    def load_preseg_widget(
            viewer: Viewer,
            load_bool,
            pre_seg_folder
    ):
        pass

    return load_preseg_widget


def initialize_dopreseg_widget():
    @magicgui(layout='vertical',
              single_channel_bool=dict(widget_type='CheckBox',
                                       text='use single channel',
                                       value=False,
                                       tooltip='tick to use single channel images (not RGB), one can still select '
                                               'multiple channels'),
              regi_bool=dict(widget_type='CheckBox',
                             text='registration done?',
                             value=True,
                             tooltip='tick to indicate if brain was registered (it is advised to register '
                                     'the brain first to exclude presegmentation artefacts outside of the '
                                     'brain'),
              regi_chan=dict(widget_type='ComboBox',
                             label='registration channel',
                             choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                             value='green',
                             tooltip='select the registration channel (images need to be in sharpy_track folder)'),
              seg_type=dict(widget_type='ComboBox',
                            label='segmentation type',
                            choices=['cells'], value='cells',
                            tooltip='select segmentation type to load'),  # todo other than cells?
              intensity_norm=dict(widget_type='LineEdit', label='intensity normalization', value='0.5,17.5',
                                  tooltip='intensity normalization parameter for rab5a model from aics-segmentation;'
                                          'https://github.com/AllenInstitute/aics-segmentation'),
              gaussian_smoothing_sigma=dict(widget_type='LineEdit', label='gauss. smooth. sigma', value='1',
                                            tooltip='gaussian smoothing sigma parameter for rab5a model from aics-segmentation;'
                                                    'https://github.com/AllenInstitute/aics-segmentation'),
              # gaussian_smoothing_truncate_range=dict(widget_type='LineEdit', label='gauss. smooth. trunc. range',
              #                                       value='',
              #                                       tooltip='gaussian smoothing truncate range parameter for rab5a model from aics-segmentation; https://github.com/AllenInstitute/aics-segmentation'),
              dot_3d_sigma=dict(widget_type='LineEdit', label='dot 3d sigma',
                                value='1',
                                tooltip='dot 3d sigma parameter for rab5a model from aics-segmentation; https://github.com/AllenInstitute/aics-segmentation'),
              dot_3d_cutoff=dict(widget_type='LineEdit', label='dot 3d cutoff',
                                 value='0.03',
                                 tooltip='dot 3d cutoff parameter for rab5a model from aics-segmentation; https://github.com/AllenInstitute/aics-segmentation'),
              hole_min_max=dict(widget_type='LineEdit', label='hole min/max',
                                value='0,81',
                                tooltip='hole min/max parameters (COMMA SEPARATED) for rab5a model from aics-segmentation; https://github.com/AllenInstitute/aics-segmentation'),
              minArea=dict(widget_type='LineEdit', label='min. area',
                           value='3',
                           tooltip='min area parameter for rab5a model from aics-segmentation; https://github.com/AllenInstitute/aics-segmentation'),
              start_end_im=dict(widget_type='LineEdit', label='image range to presegment', value='',
                                tooltip='if you only want to segment a subset of images enter COMMA SEPARATED indices '
                                        'of the first and last image to presegment, e.g. 0,10'),
              mask_folder=dict(widget_type='LineEdit',
                               label='masks folder',
                               value='segmentation_masks',
                               tooltip='name of output folder for storing segmentation masks'),

              output_folder=dict(widget_type='LineEdit',
                                 label='output folder',
                                 value='presegmentation',
                                 tooltip='name of output folder for storing the presegmentation results'),
              call_button=False,
              scrollable=True)
    def do_preseg_widget(
            viewer: Viewer,
            single_channel_bool,
            regi_bool,
            regi_chan,
            seg_type,
            intensity_norm,
            gaussian_smoothing_sigma,
            # gaussian_smoothing_truncate_range,
            dot_3d_sigma,
            dot_3d_cutoff,
            hole_min_max,
            minArea,
            start_end_im,
            mask_folder,
            output_folder):
        pass

    return do_preseg_widget


def initialize_projectionpreseg_widget():
    @magicgui(layout='vertical',
              regi_bool=dict(widget_type='CheckBox',
                             text='registration done?',
                             value=True,
                             tooltip='tick to indicate if brain was registered and segmentation artefacts outside of '
                                     'the brain will be excluded'),
              regi_chan=dict(widget_type='ComboBox',
                             label='registration channel',
                             choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                             value='green',
                             tooltip='select the registration channel (images need to be in sharpy_track folder)'),
              binary_folder=dict(widget_type='LineEdit',
                                 label='folder name with presegmented projections',
                                 value='binary',
                                 tooltip='folder needs to contain subfolders with channel names and .tif binary images '
                                         'of segmented of projections'),
              output_folder=dict(widget_type='LineEdit',
                                 label='output folder',
                                 value='presegmentation',
                                 tooltip='name of output folder for storing presegmentation data of projections (to be loaded)'),
              call_button=False,
              scrollable=True)
    def create_projection_preseg(
            viewer: Viewer,
            regi_bool,
            regi_chan,
            binary_folder,
            output_folder):
        pass

    return create_projection_preseg


def initialize_findcentroids_widget():
    @magicgui(layout='vertical',
              mask_folder=dict(widget_type='LineEdit',
                               label='folder name with presegmented data',
                               value='segmentation_masks',
                               tooltip='folder needs to contain subfolders with channel names and .tif images with segmented '
                                       'of cells'),
              mask_type=dict(widget_type='ComboBox',
                             label='segmentation type',
                             choices=['cells'],
                             value='cells',
                             tooltip='select segmentation type to load'),  # todo other than cells?
              output_folder=dict(widget_type='LineEdit',
                                 label='output folder',
                                 value='presegmentation',
                                 tooltip='name of output folder for storing centroids of segmentation masks'),
              call_button=False,
              scrollable=True)
    def find_centroids_widget(
            viewer: Viewer,
            mask_folder,
            mask_type,
            output_folder):
        pass

    return find_centroids_widget


class SegmentWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.segment = initialize_segment_widget()
        self.segment.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.save_dict = default_save_dict()

        self._collapse_load_preseg = QCollapsible('Load presegmented data: expand for more', self)
        self.load_preseg = initialize_loadpreseg_widget()
        self.load_preseg.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_load_preseg.addWidget(self.load_preseg.root_native_widget)

        self._collapse_do_preseg = QCollapsible('Create presegmentation data for cells: expand for more', self)
        self.do_preseg = initialize_dopreseg_widget()
        self.do_preseg.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_do_preseg.addWidget(self.do_preseg.root_native_widget)
        btn_do_preseg = QPushButton("run presegmentation and store data")
        btn_do_preseg.clicked.connect(self._do_presegmentation)
        self._collapse_do_preseg.addWidget(btn_do_preseg)

        self._collapse_projections = QCollapsible('Create presegmentation data for projections: expand for more', self)
        self.projections = initialize_projectionpreseg_widget()
        self.projections.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_projections.addWidget(self.projections.root_native_widget)
        btn_projections = QPushButton("create presegmentation of projections data")
        btn_projections.clicked.connect(self._create_projection_preseg)
        self._collapse_projections.addWidget(btn_projections)

        self._collapse_center = QCollapsible('Find centroids for presegmented data (masks): expand for more', self)
        self.center = initialize_findcentroids_widget()
        self.center.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_center.addWidget(self.center.root_native_widget)
        btn_center = QPushButton("get center coordinates for presegmented data")
        btn_center.clicked.connect(self._get_center_coord)
        self._collapse_center.addWidget(btn_center)

        btn = QPushButton("save data and load next image")
        btn.clicked.connect(self._save_and_load)

        self.layout().addWidget(self.segment.native)
        self.layout().addWidget(self._collapse_load_preseg)
        self.layout().addWidget(self._collapse_do_preseg)
        self.layout().addWidget(self._collapse_projections)
        self.layout().addWidget(self._collapse_center)
        self.layout().addWidget(btn)

    def _update_save_dict(self, image_idx, seg_type, n_probes):
        # get image idx and segmentation type for saving segmentation data
        self.save_dict['image_idx'] = image_idx
        self.save_dict['seg_type'] = seg_type
        self.save_dict['n_probes'] = n_probes
        return self.save_dict

    def _get_contrast_dict(self, widget):

        return {
            "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
            "green": [int(i) for i in widget.contrast_green.value.split(',')],
            "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
            "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
            "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
        }

    def _save_and_load(self):

        input_path = self.segment.input_path.value
        # check if user provided a valid input_path
        if not input_path.is_dir():
            raise IOError("Input path is not a valid directory \n"
                          "Please make sure this exists: {}".format(input_path))
        image_idx = int(self.segment.image_idx.value)
        seg_type = self.segment.seg_type.value
        channels = self.segment.channels.value
        n_probes = int(self.segment.n_probes.value)
        single_channel = self.segment.single_channel_bool.value
        contrast_dict = self._get_contrast_dict(self.segment)

        if len(self.viewer.layers) == 0:  # no open images, set save_dict to defaults
            self.save_dict = default_save_dict()
        if type(self.save_dict['image_idx']) == int:  # todo there must be a better way :-D (for image_idx = 0)
            self._save_data(input_path, channels, single_channel)
        del (self.viewer.layers[:])  # remove open layers

        try:
            self._load_next(input_path, seg_type, channels, image_idx, n_probes, single_channel, contrast_dict)

        except IndexError:
            show_info("Index out of range, check that index matches image count in target folder")

    def _load_next(self, input_path, seg_type, channels, image_idx, n_probes, single_channel, contrast_dict):
        self.save_dict = self._update_save_dict(image_idx, seg_type, n_probes)
        if single_channel:
            for chan in channels:
                path_to_im = get_path_to_im(input_path, image_idx, single_channel=single_channel, chan=chan)
                self._load_single(path_to_im, chan, contrast_dict)
        else:
            path_to_im = get_path_to_im(input_path, image_idx)
            self._load_rgb(path_to_im, channels, contrast_dict)
        self._create_seg_objects(input_path, seg_type, channels, n_probes, image_idx)

        show_info("loaded " + path_to_im.parts[-1] + " (cnt=" + str(image_idx) + ")")
        image_idx += 1
        self.segment.image_idx.value = image_idx
        # change_index(image_idx)

    def _load_rgb(self, path_to_im, channels, contrast_dict):
        im_loaded = cv2.imread(str(path_to_im))  # loads RGB as BGR
        if 'cy3' in channels:
            self.viewer.add_image(im_loaded[:, :, 2], name='cy3 channel', colormap='red', opacity=1.0)
            self.viewer.layers['cy3 channel'].contrast_limits = contrast_dict['cy3']
        if 'green' in channels:
            self.viewer.add_image(im_loaded[:, :, 1], name='green channel', colormap='green', opacity=0.5)
            self.viewer.layers['green channel'].contrast_limits = contrast_dict['green']
        if 'dapi' in channels:
            self.viewer.add_image(im_loaded[:, :, 0], name='dapi channel', colormap='blue', opacity=0.5)
            self.viewer.layers['dapi channel'].contrast_limits = contrast_dict['dapi']

    def _load_single(self, path_to_im, chan, contrast_dict):

        cmap_disp = get_cmap('display')
        im_loaded = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
        self.viewer.add_image(im_loaded, name=chan + ' channel', colormap=cmap_disp[chan], opacity=0.5)
        self.viewer.layers[chan + ' channel'].contrast_limits = contrast_dict[chan]

    def _load_preseg_object(self, input_path, chan, image_idx, seg_type):
        pre_seg_folder = self.load_preseg.pre_seg_folder.value
        pre_seg_dir, pre_seg_list, pre_seg_suffix = get_info(input_path, pre_seg_folder, seg_type=seg_type,
                                                             channel=chan)
        im_name = get_path_to_im(input_path, image_idx, pre_seg=True)  # name of image that will be loaded
        fn_to_load = [d for d in pre_seg_list if d.startswith(im_name + '_')]
        if len(fn_to_load) > 0:
            pre_seg_data_dir = pre_seg_dir.joinpath(fn_to_load[0])
            df = pd.read_csv(pre_seg_data_dir)  # load dataframe
            try:
                pre_seg_data = df[['Position Y', 'Position X']].to_numpy()
            except KeyError:
                print("csv file missing columns (Position Y/X), no presegmented data loaded")
                pre_seg_data = []
        else:
            pre_seg_data = []

        return pre_seg_data

    def _create_seg_objects(self, input_path, seg_type, channels, n_probes, image_idx):
        if seg_type == 'injection_site':
            cmap_dict = get_cmap('injection')
            if self.load_preseg.load_bool.value:
                for chan in channels:
                    pre_seg_data = self._load_preseg_object(input_path, chan, image_idx, seg_type)
                    self.viewer.add_shapes(pre_seg_data, name=chan, shape_type='polygon', face_color=cmap_dict[chan],
                                           opacity=0.4)
            else:
                for chan in channels:
                    self.viewer.add_shapes(name=chan, face_color=cmap_dict[chan], opacity=0.4)
        elif seg_type in ['cells', 'projections']:
            cmap_dict = get_cmap('cells')
            if self.load_preseg.load_bool.value:
                for chan in channels:
                    pre_seg_data = self._load_preseg_object(input_path, chan, image_idx, seg_type)
                    self.viewer.add_points(pre_seg_data, size=int(self.segment.point_size.value), name=chan,
                                           face_color=cmap_dict[chan])
            else:
                for chan in channels:
                    self.viewer.add_points(size=int(self.segment.point_size.value), name=chan,
                                           face_color=cmap_dict[chan])
        else:
            cmap_dict = get_cmap('npx')
            for i in range(n_probes):
                if i < 10:
                    p_color = cmap_dict[str(i)]
                else:
                    p_color = random.choice(list(mcolors.CSS4_COLORS.keys()))
                p_id = seg_type + '_' + str(i)
                print(p_id)
                if self.load_preseg.load_bool.value:
                    pre_seg_data = self._load_preseg_object(input_path, p_id, image_idx, seg_type)
                    self.viewer.add_points(pre_seg_data, size=int(self.segment.point_size.value), name=p_id,
                                           face_color=p_color)
                else:
                    self.viewer.add_points(size=int(self.segment.point_size.value), name=p_id, face_color=p_color)

    def _save_data(self, input_path, channels, single_channel):
        # points data in [y, x] format
        save_idx = self.save_dict['image_idx']
        seg_type_save = self.save_dict['seg_type']
        if single_channel:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channels[0])
        else:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        path_to_im = seg_im_dir.joinpath(seg_im_list[save_idx])
        im_name_str = path_to_im.with_suffix('').parts[-1]
        if seg_type_save not in ['cells', 'injection_site', 'projections']:
            channels = [seg_type_save + '_' + str(i) for i in range(self.save_dict['n_probes'])]
        for chan in channels:
            try:

                segment_dir = get_info(input_path, 'segmentation', channel=chan, seg_type=seg_type_save,
                                       create_dir=True,
                                       only_dir=True)
                if seg_type_save == 'injection_site':
                    data = pd.DataFrame()
                    for i in range(len(self.viewer.layers[chan].data)):
                        data_temp = pd.DataFrame(self.viewer.layers[chan].data[i], columns=['Position Y', 'Position X'])
                        data_temp['idx_shape'] = [i] * len(data_temp)
                        data = pd.concat((data, data_temp))
                else:
                    data = pd.DataFrame(self.viewer.layers[chan].data, columns=['Position Y', 'Position X'])
                save_name = segment_dir.joinpath(im_name_str + '_' + seg_type_save + '.csv')
                if len(self.viewer.layers[chan].data) > 0:  # only create results file when data is present
                    data.to_csv(save_name)
                else:
                    if self.load_preseg.load_bool.value:  # this only in case preseg and seg are same folder, if you delete all cells delete existing file
                        if self.load_preseg.pre_seg_folder.value == 'segmentation':
                            if save_name.exists():
                                save_name.unlink()

            except KeyError:
                pass
        # else:
        #     for i in range(self.save_dict['n_probes']):
        #         p_id = seg_type_save + '_' + str(i)
        #         if len(self.viewer.layers[p_id].data) > 0:
        #             segment_dir = get_info(input_path, 'segmentation', channel=p_id, seg_type=seg_type_save,
        #                                    create_dir=True, only_dir=True)
        #             coords = pd.DataFrame(self.viewer.layers[p_id].data, columns=['Position Y', 'Position X'])
        #             save_name = segment_dir.joinpath(im_name_str + '_' + seg_type_save + '.csv')
        #             coords.to_csv(save_name)

    def _get_center_coord(self):
        input_path = self.segment.input_path.value
        channels = self.segment.channels.value
        mask_folder = self.center.mask_folder.value
        mask_type = self.center.mask_type.value
        output_folder = self.center.output_folder.value
        center_worker = get_center_coord(input_path, channels, mask_folder, output_folder, mask_type=mask_type)
        center_worker.start()

    def _do_presegmentation(self):
        input_path = self.segment.input_path.value
        params_dict = load_params(input_path)
        channels = self.segment.channels.value
        single_channel = self.do_preseg.single_channel_bool.value
        regi_bool = self.do_preseg.regi_bool.value
        regi_chan = self.do_preseg.regi_chan.value
        seg_type = self.do_preseg.seg_type.value
        preseg_params = {
            "intensity_norm": split_to_list(self.do_preseg.intensity_norm.value, out_format='float'),
            "gaussian_smoothing_sigma": int(self.do_preseg.gaussian_smoothing_sigma.value),
            # "gaussian_smoothing_truncate_range": int(self.do_preseg.gaussian_smoothing_truncate_range.value),
            "dot_3d_sigma": int(self.do_preseg.dot_3d_sigma.value),
            "dot_3d_cutoff": float(self.do_preseg.dot_3d_cutoff.value),
            "hole_min_max": split_to_list(self.do_preseg.hole_min_max.value, out_format='int'),
            "minArea": int(self.do_preseg.minArea.value)
        }
        start_end_im = split_to_list(self.do_preseg.start_end_im.value, out_format='int')
        mask_folder = self.do_preseg.mask_folder.value
        output_folder = self.do_preseg.output_folder.value
        do_preseg_worker = do_presegmentation(input_path, params_dict, channels, single_channel, regi_bool, regi_chan,
                                              preseg_params, start_end_im, mask_folder, output_folder,
                                              seg_type=seg_type)
        do_preseg_worker.start()

    def _create_projection_preseg(self):
        input_path = self.segment.input_path.value
        params_dict = load_params(input_path)
        channels = self.segment.channels.value
        regi_bool = self.projections.regi_bool.value
        regi_chan = self.projections.regi_chan.value
        binary_folder = self.projections.binary_folder.value
        output_folder = self.projections.output_folder.value
        projection_worker = create_projection_preseg(input_path, params_dict, channels, regi_bool, regi_chan,
                                                     binary_folder, output_folder)
        projection_worker.start()