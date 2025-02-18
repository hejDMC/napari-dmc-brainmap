from pathlib import Path
import json
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_3d
from aicssegmentation.core.utils import hole_filling
from skimage.measure import label, regionprops, regionprops_table
from skimage.morphology import remove_small_objects

from bg_atlasapi import BrainGlobeAtlas

from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils import get_bregma, coord_mm_transform, split_to_list, get_image_list, get_info, \
    load_params
from napari_dmc_brainmap.segment.processing.atlas_utils import loadAnnotBool, angleSlice

def get_total_images_centroids(input_path, channels, mask_folder, mask_type):
    n_images = 0
    for chan in channels:
        mask_dir = get_info(input_path, mask_folder, seg_type=mask_type, channel=chan, only_dir=True)
        n_images += len([im.parts[-1] for im in mask_dir.glob('*.tiff')])
    return n_images

def get_total_images(input_path, start_end_im, channels, single_channel):
    n_images = 0
    for chan in channels:
        _, seg_im_list, _ = load_image_list(input_path, start_end_im, chan, single_channel)
        n_images += len(seg_im_list)
    return n_images

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
    if single_channel == 'single_channel':
        return img[0, 0, 0, :, :].copy()
    else:
        return img[0, 0, 0, :, :, structure_channel].copy()


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

def transform_segmentation(regi_index, xyz_dict, x_im, y_im, regi_list):
    regi_data, annot_bool, z_idx, z_res, bregma = regi_list
    try:
        # get transformation
        tform = fitGeoTrans(regi_data['sampleDots'][regi_index], regi_data['atlasDots'][regi_index])
        # slice annotation volume
        x_angle, y_angle, z = regi_data['atlasLocation'][regi_index]

        annot_slice = angleSlice(x_angle, y_angle, z, annot_bool, z_idx, z_res, bregma, xyz_dict)
        # mark invalid coordinates
        drop_bool = []
        for x, y in zip(x_im, y_im):
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
    except Exception:
        show_info(f"No registration data for {regi_data['imgName'][regi_index]}")
        drop_bool = False
    return drop_bool

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
    drop_bool = transform_segmentation(regi_index, xyz_dict, x_rgb, y_rgb, regi_list)

    return drop_bool

def load_registration_data(input_path, regi_chan, atlas_id, xyz_dict):
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


def prepare_segmentation_folders(input_path, seg_folder, output_folder, chan, seg_type='cells'):
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
    output_dir = get_info(input_path, output_folder, channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
    if seg_type == 'projections':
        binary_dir, binary_images, binary_suffix = get_info(input_path, seg_folder, seg_type=seg_type,
                                                            channel=chan)
        return binary_dir, binary_images, binary_suffix, output_dir # todo check if all this is necessary
    else:
        mask_dir = get_info(input_path, seg_folder, channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
        return mask_dir, output_dir

def load_image_list(input_path, start_end_im, chan, single_channel):
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
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, single_channel, channel=chan)
    else:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
    if start_end_im:
        if len(start_end_im) == 2:
            seg_im_list = seg_im_list[start_end_im[0]:start_end_im[1] + 1]
    return seg_im_dir, seg_im_list, seg_im_suffix