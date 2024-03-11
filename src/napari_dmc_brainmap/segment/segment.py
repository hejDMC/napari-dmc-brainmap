from napari import Viewer
from napari.qt.threading import thread_worker
from natsort import natsorted
import cv2
from napari_dmc_brainmap.utils import get_info, load_params
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

from bg_atlasapi import config, BrainGlobeAtlas

from napari_dmc_brainmap.utils import get_bregma, coord_mm_transform, split_to_list
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform



# todo put these function in segment_tools.py file (to be created)
def calculateImageGrid(x_res, y_res): # one time calculation
    y = np.arange(y_res)
    x = np.arange(x_res)
    grid_x, grid_y = np.meshgrid(x, y)
    r_grid_x = grid_x.ravel()
    r_grid_y = grid_y.ravel()
    grid = np.stack([grid_y, grid_x], axis=2)
    return grid, r_grid_x, r_grid_y

def loadAnnotBool(atlas):
    brainglobe_dir = config.get_brainglobe_dir()
    atlas_name_general = f"{atlas}_v*"
    atlas_names_local = list(brainglobe_dir.glob(atlas_name_general))[
        0]  # glob returns generator object, need to exhaust it in list, then take out
    annot_bool_dir = brainglobe_dir.joinpath(atlas_names_local, 'annot_bool.npy')
    # for any atlas else, in this case test with zebrafish atlas
    print('checking for annot_bool volume...')
    if annot_bool_dir.exists():  # when directory has 8-bit template volume, load it
        print('loading annot_bool volume...')
        annot_bool = np.load(annot_bool_dir)

    else:  # when saved template not found
        # check if template volume from brainglobe is already 8-bit
        print('... local version not found, loading annotation volume...')
        annot = BrainGlobeAtlas(atlas).annotation

        print('... creating annot_bool version...')

        annot_bool = np.where(annot>0, 255, 0)  # 0, outside brain, 255 inside brain
        np.save(annot_bool_dir, annot_bool)

    return annot_bool

def angleSlice(x_angle, y_angle, z, annot_bool, z_idx, z_res, bregma, xyz_dict):
    # calculate from ml and dv angle, the plane of current slice
    x_shift = int(np.tan(np.deg2rad(x_angle)) * (xyz_dict['x'][1] / 2))
    y_shift = int(np.tan(np.deg2rad(y_angle)) * (xyz_dict['y'][1] / 2))
    # pick up slice
    z_coord = coord_mm_transform([z], [bregma[z_idx]],
                                 [z_res], mm_to_coord=True)

    center = np.array([z_coord, (xyz_dict['y'][1] / 2), (xyz_dict['x'][1] / 2)])
    c_right = np.array([z_coord+x_shift, (xyz_dict['y'][1] / 2), (xyz_dict['x'][1] - 1)])
    c_top = np.array([z_coord-y_shift, 0, (xyz_dict['x'][1] / 2)])
    # calculate plane normal vector
    vec_1 = c_right-center
    vec_2 = c_top-center
    vec_n = np.cross(vec_1,vec_2)
    # calculate ap matrix
    grid,r_grid_x,r_grid_y = calculateImageGrid(xyz_dict['x'][1], xyz_dict['y'][1])
    ap_mat = (-vec_n[1]*(grid[:,:,0]-center[1])-vec_n[2]*(grid[:,:,1]-center[2]))/vec_n[0] + center[0]
    ap_flat = ap_mat.astype(int).ravel()
    # within volume check
    outside_vol = np.argwhere((ap_flat<0)|(ap_flat>(xyz_dict['z'][1]-1))) # outside of volume index
    if outside_vol.size == 0: # if outside empty, inside of volume
        # index volume with ap_mat and grid
        slice = annot_bool[ap_mat.astype(int).ravel(),r_grid_y,r_grid_x].reshape(xyz_dict['y'][1],xyz_dict['x'][1])
    else: # if not empty, show black image
        slice = np.zeros((xyz_dict['y'][1], xyz_dict['x'][1]),dtype=np.uint8)
    return slice

def cmap_cells():
    # return default colormap for channel and color of cells
    cmap = {
        'dapi': 'yellow',
        'green': 'magenta',
        'n3': 'gray',
        'cy3': 'cyan',
        'cy5': 'lightblue'
    }
    return cmap

def cmap_injection():
    # return default colormap for channel and color of cells
    cmap = {
        'dapi': 'gold',
        'green': 'purple',
        'n3': 'navy',
        'cy3': 'darkorange',
        'cy5': 'cornflowerblue'
    }
    return cmap



def cmap_display():
    cmap = {
        'dapi': 'blue',
        'green': 'green',
        'n3': 'orange',
        'cy3': 'red',
        'cy5': 'pink'
    }
    return cmap


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
        return im
    else:
        return path_to_im


@thread_worker
def do_presegmentation(input_path, params_dict, channels, single_channel, regi_bool, regi_chan, preseg_params,
                                              start_end_im, mask_folder, output_folder, seg_type='cells'):

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

    if not single_channel:
        chan_dict = {
            'cy3': 0,
            'green': 1,
            'dapi': 2
        }
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        seg_im_list = natsorted(seg_im_list)
        if start_end_im:
            if len(start_end_im) == 2:
                seg_im_list = seg_im_list[start_end_im[0]:start_end_im[1]]

    for chan in channels:
        mask_dir = get_info(input_path, mask_folder, channel=chan, seg_type=seg_type,
                            create_dir=True, only_dir=True)
        output_dir = get_info(input_path, output_folder, channel=chan, seg_type=seg_type,
                            create_dir=True, only_dir=True)
        print('... channel ' + chan)
        if single_channel:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channel=chan)
            seg_im_list = natsorted(seg_im_list)
            if start_end_im:
                if len(start_end_im) == 2:
                    seg_im_list = seg_im_list[start_end_im[0]:start_end_im[1]]
        for im in seg_im_list:
            print('... ' + im)
            im_fn = seg_im_dir.joinpath(im)
            reader = AICSImage(str(im_fn))
            img = reader.data.astype(np.float32)  # input image is a single RGB image
            if single_channel:
                struct_img0 = img[0, 0, 0, :, :].copy()
            else:
                structure_channel = chan_dict[chan] # 0:cy3, 1:green, 2:dapi for RGB
                struct_img0 = img[0, 0, 0, :, :, structure_channel].copy()
            struct_img0 = np.array([struct_img0, struct_img0])  # make duplicate layer stack
            # preprocessing
            intensity_scaling_param = preseg_params["intensity_norm"] # default[1000], [0.5,17.5] works better
            gaussian_smoothing_sigma = preseg_params["gaussian_smoothing_sigma"]
            struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
            structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
            # core
            response = dot_3d(structure_img_smooth, log_sigma=preseg_params["dot_3d_sigma"])
            bw = response > preseg_params["dot_3d_cutoff"]
            # postprocessing
            bw_filled = hole_filling(bw, preseg_params["hole_min_max"][0], preseg_params["hole_min_max"][1], True)
            seg = remove_small_objects(bw_filled, min_size=preseg_params["minArea"], connectivity=1) # min_size=3 a lot of small objects detected # ,min size 6 is still too small
            # output segmentation binary image
            seg = seg>0
            seg = seg.astype(np.uint8)
            seg[seg>0]=255
            if np.mean(seg[0]) == 0:
                pass
            else:
                # write binary image to file
                writer = OmeTiffWriter()
                mask_save_fn = mask_dir.joinpath(im[:-len(seg_im_suffix)] + '_masks.tiff')
                writer.save(seg[0], str(mask_save_fn)) # save only one layer binary image
                # centroid detection
                label_img = label(seg[0])
                props = regionprops_table(label_img, properties=['centroid'])
                if regi_bool:
                    dim_rgb = seg[0].shape
                    x_res = xyz_dict['x'][1]
                    y_res = xyz_dict['y'][1]
                    x_rgb = props["centroid-1"] / dim_rgb[1] * x_res
                    y_rgb = props["centroid-0"] / dim_rgb[0] * y_res
                    for k, v in regi_data['imgName'].items():
                        if v.startswith(im[:-len(seg_im_suffix)]):
                            regi_index = k
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
                        if (x_atlas < 0) | (y_atlas < 0) | (x_atlas >= xyz_dict['x'][1]) | (y_atlas >= xyz_dict['y'][1]):
                            drop_bool.append(1)
                        else:
                            if annot_slice[y_atlas, x_atlas] == 0:
                                drop_bool.append(1)
                            else:
                                drop_bool.append(0)
                    csv_to_save = pd.DataFrame(props)
                    csv_to_save.rename(columns={"centroid-0": "Position Y", "centroid-1": "Position X"}, inplace=True)
                    csv_to_save = csv_to_save.iloc[np.where(np.array(drop_bool) == 0)[0], :].copy().reset_index(
                        drop=True)
                else:
                    csv_to_save = pd.DataFrame(props)
                    csv_to_save.rename(columns={"centroid-0": "Position Y", "centroid-1": "Position X"}, inplace=True)

                csv_save_name = output_dir.joinpath(im.split('.')[0] + '_' + seg_type + '.csv')
                csv_to_save.to_csv(csv_save_name)
    print("DONE with presegmentation")
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
            label_img = label(image) # identify individual segmented structures
            regions = regionprops(label_img) # get there properties -> we want to have the centroid point as a "location" of the cell
            cent = np.zeros((np.size(regions), 2))
            for idx, props in enumerate(regions):
                cent[idx, 0] = props.centroid[0] # y-coordinates
                cent[idx, 1] = props.centroid[1] # x-coordinates
            # create csv file in folders to match imaris output
            csv_to_save = pd.DataFrame(cent)
            csv_to_save = csv_to_save.rename(columns={0: "Position Y", 1: "Position X"})
            csv_save_name = output_dir.joinpath(im_name.split('.')[0] + '_' + mask_type + '.csv')
            csv_to_save.to_csv(csv_save_name)
            #
            location_binary = np.zeros((image.shape)) # make new binary with centers of segmented cells only
            cent = (np.round(cent)).astype(int)
            for val in cent:
                location_binary[val[0], val[1]] = 255
            location_binary = location_binary.astype(int)
            location_binary = location_binary.astype('uint8') # convert to 8-bit
            save_name = im_name.split('.')[0] + '_centroids.tif'  # get the name
            cv2.imwrite(str(mask_dir.joinpath(save_name)), location_binary)
            # progress_bar.update(100 / len(binary_images))
        print("Done with " + chan)



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
                            choices=['cells', 'injection_side', 'optic_fiber', 'neuropixels_probe'], 
                            value='cells',
                            tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'
                                'IMPORTANT: before switching between types, load next image, delete all image layers'
                                'and reload image of interest!'),
              n_probes=dict(widget_type='LineEdit', 
                            label='number of fibers/probes', 
                            value=1,
                            tooltip='number (int) of optic fibres and or probes used to segment, ignore this value for '
                                'segmenting cells/areas/'),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels to be selected for cell segmentation, '
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
                             label='load pre-segmented data', 
                             value=False,
                             tooltip='tick to load pre-segmented data for manual curation'),
              pre_seg_folder=dict(widget_type='LineEdit', 
                                  label='folder name with pre-segmented data', 
                                  value='presegmentation',
                                  tooltip='folder needs to contain sub-folders with channel names. WARNING: if the channel is called'
                                '*segmentation* (default), manual curation will override existing data. '
                                'Pre-segmented data needs to be .csv file and column names specifying *Position X* and '
                                '*Position Y* for coordinates'),
              seg_type=dict(widget_type='ComboBox', 
                            label='segmentation type',
                            choices=['cells'], value='cells',
                            tooltip='select segmentation type to load'),  # todo other than cells?
              call_button=False)

    def load_preseg_widget(
        viewer: Viewer,
        load_bool,
        pre_seg_folder,
        seg_type):
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
                                       tooltip='tick to indicate if brain was registered (it is advised to register'
                                               'the brain first to exclude presegmentation artefacts outside of the '
                                               'brain'),
              regi_chan=dict(widget_type='ComboBox',
                                     label='registration channel',
                                     choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                                     value='green',
                                     tooltip='select the registration channel (images need to be in sharpy track folder)'),
              seg_type=dict(widget_type='ComboBox',
                            label='segmentation type',
                            choices=['cells'], value='cells',
                            tooltip='select segmentation type to load'), # todo other than cells?
              intensity_norm=dict(widget_type='LineEdit', label='intensity normalization', value='0.5,17.5',
                                  tooltip='intensity normalization parameter for rab5a model from aics-segmentation;'
                                          'https://github.com/AllenInstitute/aics-segmentation'),
              gaussian_smoothing_sigma=dict(widget_type='LineEdit', label='gauss. smooth. sigma', value='1',
                                            tooltip='gaussian smoothing sigma parameter for rab5a model from aics-segmentation;'
                                                    'https://github.com/AllenInstitute/aics-segmentation'),
              #gaussian_smoothing_truncate_range=dict(widget_type='LineEdit', label='gauss. smooth. trunc. range',
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
                                 label='masks folder)',
                                 value='segmentation_masks',
                                 tooltip='name of output folder for storing segmentation masks'),

              output_folder=dict(widget_type='LineEdit',
                                 label='output folder',
                                 value='presegmentation',
                                 tooltip='name of output folder for storing the presegmentation results'),
              call_button=False)
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


def initialize_findcentroids_widget():
    @magicgui(layout='vertical',
              mask_folder=dict(widget_type='LineEdit', 
                               label='folder name with pre-segmented data', 
                               value='segmentation_masks',
                               tooltip='folder needs to contain sub-folders with channel names and .tif images with segmented '
                                'of cells.'),
              mask_type=dict(widget_type='ComboBox', 
                             label='segmentation type',
                             choices=['cells'], 
                             value='cells',
                             tooltip='select segmentation type to load'),  # todo other than cells?
              output_folder=dict(widget_type='LineEdit', 
                                 label='output folder', 
                                 value='presegmentation',
                                 tooltip='name of output folder for storing centroids of segmentation masks'),
              call_button=False)
    
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
        self.save_dict = default_save_dict()

        self._collapse_load_preseg = QCollapsible('Load pre-segmented data: expand for more', self)
        self.load_preseg = initialize_loadpreseg_widget()
        self._collapse_load_preseg.addWidget(self.load_preseg.native)

        self._collapse_do_preseg = QCollapsible('Do presegmentation of data: expand for more', self)
        self.do_preseg = initialize_dopreseg_widget()
        self._collapse_do_preseg.addWidget(self.do_preseg.native)
        btn_do_preseg = QPushButton("run presegmentation and store data")
        btn_do_preseg.clicked.connect(self._do_presegmentation)
        self._collapse_do_preseg.addWidget(btn_do_preseg)

        self._collapse_center = QCollapsible('Find centroids for pre-segmented data (masks): expand for more', self)
        self.center = initialize_findcentroids_widget()
        self._collapse_center.addWidget(self.center.native)
        btn_center = QPushButton("get center coordinates for pre-segmented data")
        btn_center.clicked.connect(self._get_center_coord)
        self._collapse_center.addWidget(btn_center)

        btn = QPushButton("save data and load next image")
        btn.clicked.connect(self._save_and_load)

        self.layout().addWidget(self.segment.native)
        self.layout().addWidget(self._collapse_load_preseg)
        self.layout().addWidget(self._collapse_do_preseg)
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
            self.viewer.add_image(im_loaded[:, :, 0], name='dapi channel')
            self.viewer.layers['dapi channel'].contrast_limits = contrast_dict['dapi']

    def _load_single(self, path_to_im, chan, contrast_dict):

        cmap_disp = cmap_display()
        im_loaded = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
        self.viewer.add_image(im_loaded, name=chan + ' channel', colormap=cmap_disp[chan], opacity=0.5)
        self.viewer.layers[chan + ' channel'].contrast_limits = contrast_dict[chan]

    def _load_preseg_object(self, input_path, chan, image_idx):

        pre_seg_folder = self.load_preseg.pre_seg_folder.value
        pre_seg_type = self.load_preseg.seg_type.value
        pre_seg_dir, pre_seg_list, pre_seg_suffix = get_info(input_path, pre_seg_folder, seg_type=pre_seg_type, channel=chan)
        im_name = get_path_to_im(input_path, image_idx, pre_seg=True)  # name of image that will be loaded
        fn_to_load = [d for d in pre_seg_list if d.startswith(im_name.split('.')[0])]
        if fn_to_load:
            df = pd.read_csv(pre_seg_dir.joinpath(fn_to_load[0]))  # load dataframe
            try:
                pre_seg_data = df[['Position Y', 'Position X']].to_numpy()
            except KeyError:
                print("csv file missing columns (Position Y/X), no pre-segmented data loaded")
                pre_seg_data = []
        else:
            pre_seg_data = []

        return pre_seg_data


    def _create_seg_objects(self, input_path, seg_type, channels, n_probes, image_idx):
        if seg_type == 'injection_side':
            cmap_dict = cmap_injection()
            for chan in channels:
                self.viewer.add_shapes(name=chan, face_color=cmap_dict[chan], opacity=0.4)
        elif seg_type == 'cells':
            cmap_dict = cmap_cells()
            if self.load_preseg.load_bool.value:  # loading presegmented cells
                for chan in channels:
                    pre_seg_data = self._load_preseg_object(input_path, chan, image_idx)
                    self.viewer.add_points(pre_seg_data, size=5, name=chan, face_color=cmap_dict[chan])
            else:
                for chan in channels:
                    self.viewer.add_points(size=5, name=chan, face_color=cmap_dict[chan])
        else:
            # todo keep colors constant
            for i in range(n_probes):
                p_color = random.choice(list(mcolors.CSS4_COLORS.keys()))
                p_id = seg_type + '_' + str(i)
                self.viewer.add_points(size=20, name=p_id, face_color=p_color)

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
        if seg_type_save not in ['cells', 'injection_side']:
            channels = [seg_type_save + '_' + str(i) for i in range(self.save_dict['n_probes'])]
        for chan in channels:
            try:
                if len(self.viewer.layers[chan].data) > 0:
                    segment_dir = get_info(input_path, 'segmentation', channel=chan, seg_type=seg_type_save,
                                            create_dir=True,
                                            only_dir=True)
                    if seg_type_save == 'injection_side':
                        data = pd.DataFrame()
                        for i in range(len(self.viewer.layers[chan].data)):
                            data_temp = pd.DataFrame(self.viewer.layers[chan].data[i], columns=['Position Y', 'Position X'])
                            data_temp['idx_shape'] = [i] * len(data_temp)
                            data = pd.concat((data, data_temp))
                    else:
                        data = pd.DataFrame(self.viewer.layers[chan].data, columns=['Position Y', 'Position X'])
                    save_name = segment_dir.joinpath(im_name_str + '_' + seg_type_save + '.csv')
                    data.to_csv(save_name)
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
                                              preseg_params, start_end_im, mask_folder, output_folder, seg_type=seg_type)
        do_preseg_worker.start()

