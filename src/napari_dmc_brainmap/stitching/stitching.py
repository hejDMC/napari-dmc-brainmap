"""
DMC-BrainMap widget for stitching .tif and .czi files

2024 - FJ, XC
"""

# import modules
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from natsort import natsorted
import json
import tifffile as tiff
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from napari_dmc_brainmap.stitching.stitching_tools import stitch_stack, stitch_folder
from napari_dmc_brainmap.utils import get_info, get_animal_id, update_params_dict, clean_params_dict

from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from natsort import natsorted
import json
import tifffile as tiff
import numpy as np

from napari_dmc_brainmap.stitching.stitching_tools import stitch_stack, stitch_folder
from napari_dmc_brainmap.utils import get_info, get_animal_id, update_params_dict, clean_params_dict


@thread_worker
def do_stitching(input_path, filter_list, params_dict, stitch_tiles, direct_sharpy_track):
    """

    :param input_path: PosixPath of dir to animal_id containing all data
    :param filter_list: list of channels to stitch
    :param params_dict: dict loaded from params.json
    :param stitch_tiles: bool, whether to stitch individual tiles, if False use DMC-FluoImager data as input
    :param direct_sharpy_track: bool, whether to directly create data for SHARPy-track
    :return:
    """

    animal_id = get_animal_id(input_path)
    resolution = tuple(params_dict['atlas_info']['resolution'])  # resolution of atlas used for registration, important for padding of stitched images
    # get obj sub-dirs
    data_dir = input_path.joinpath('raw')
    objs = natsorted([o.parts[-1] for o in data_dir.iterdir() if o.is_dir()])
    if not objs:
        print('no object slides under raw-data folder!')
        return

    # iterate objs and chans
    for obj in objs:
        in_obj = data_dir.joinpath(obj)
        for f in filter_list:
            stitch_dir = get_info(input_path, 'stitched', channel=f, create_dir=True, only_dir=True)
            if stitch_tiles:
                process_stitch_folder(input_path, in_obj, f, stitch_dir, animal_id, obj, params_dict, resolution,
                                      direct_sharpy_track)
            else:
                process_stitch_stack(input_path, in_obj, f, stitch_dir, animal_id, obj, params_dict, resolution,
                                     direct_sharpy_track)
    return animal_id

def process_stitch_folder(input_path, in_obj, f, stitch_dir, animal_id, obj, params_dict, resolution, direct_sharpy_track):
    in_chan = in_obj.joinpath(f)
    section_list = natsorted([s.parts[-1] for s in in_chan.iterdir() if s.is_dir()])
    section_list_new = [f"{animal_id}_{obj}_{str(k + 1)}" for k, ss in
                        enumerate(section_list)]
    [in_chan.joinpath(old).rename(in_chan.joinpath(new)) for old, new in
     zip(section_list, section_list_new)]
    section_dirs = natsorted([s for s in in_chan.iterdir() if s.is_dir()])
    for section in section_dirs:

        stitched_path = stitch_dir.joinpath(f'{section.parts[-1]}_stitched.tif')
        if direct_sharpy_track:
            sharpy_chans = params_dict['sharpy_track_params']['channels']
            if f in sharpy_chans:
                sharpy_dir = get_info(input_path, 'sharpy_track', channel=f, create_dir=True, only_dir=True)
                sharpy_im_dir = sharpy_dir.joinpath(f'{section.parts[-1]}_downsampled.tif')
                stitch_folder(section, 205, stitched_path, params_dict, f, sharpy_im_dir, resolution=resolution)
            else:
                stitch_folder(section, 205, stitched_path, params_dict, f, resolution=resolution)
        else:
            stitch_folder(section, 205, stitched_path, params_dict, f, resolution=resolution)

def process_stitch_stack(input_path, in_obj, f, stitch_dir, animal_id, obj, params_dict, resolution, direct_sharpy_track):
    in_chan = in_obj.joinpath(f'{obj}_{f}_1')
    # load tile stack name
    stack = natsorted([im.parts[-1] for im in in_chan.glob('*.tif')])
    whole_stack = load_tile_stack(in_chan, stack)

    # load tile location meta data from meta folder
    meta_json_where = in_obj.joinpath(f'{obj}_meta_1', 'regions_pos.json')
    with open(meta_json_where, 'r') as data:
        img_meta = json.load(data)

    # get number of regions on this objective slide
    region_n = len(img_meta)
    # iterate regions
    for rn in range(region_n):
        pos_list = img_meta['region_' + str(rn)]
        stitched_path = stitch_dir.joinpath(f'{animal_id}_{obj}_{str(rn + 1)}_stitched.tif')
        if direct_sharpy_track:
            sharpy_chans = params_dict['sharpy_track_params']['channels']
            if f in sharpy_chans:
                sharpy_dir = get_info(input_path, 'sharpy_track', channel=f, create_dir=True, only_dir=True)
                sharpy_im_dir = sharpy_dir.joinpath(f'{animal_id}_{obj}_{str(rn + 1)}_downsampled.tif')
                stitch_stack(pos_list, whole_stack, 205, stitched_path, params_dict, f,
                             resolution=resolution, downsampled_path=sharpy_im_dir)
            else:
                stitch_stack(pos_list, whole_stack, 205, stitched_path, params_dict, f,
                             resolution=resolution)
        else:
            stitch_stack(pos_list, whole_stack, 205, stitched_path, params_dict, f,
                         resolution=resolution)
        # remove stitched tiles from whole_stack
        whole_stack = np.delete(whole_stack, [np.arange(len(pos_list))], axis=0)

def load_tile_stack(in_chan, stack):
    # get number of tiles from NDTiff.index file
    tif_meta = tiff.read_ndtiff_index(in_chan.joinpath("NDTiff.index"))
    page_count = 0
    for _ in tif_meta:
        page_count += 1
    # initiate empty numpy array
    whole_stack = np.zeros((page_count, 2048, 2048), dtype=np.uint16)
    page_count = 0
    for stk in stack:
        with tiff.TiffFile(in_chan.joinpath(stk)) as tif:  # read multipaged tif
            for page in tif.pages:  # iterate over pages
                image = page.asarray()  # convert to array
                try:
                    whole_stack[page_count, :, :] = image
                except ValueError:
                    print("Tile:{} data corrupted. Setting tile pixels value to 0".format(page_count))
                page_count += 1
    return whole_stack


def initialize_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. raw data, images, segmentation results, NOT '
                                    'folder containing images'),
              stitch_tiles=dict(widget_type='CheckBox', 
                                text='stitching image tiles', 
                                value=False,
                                tooltip='option to stitch images from tiles acquired by micro-manager (ticked) or to stitch images acquired by DMC-FluoImager (not ticked)'),
              channels=dict(widget_type='Select', 
                            label='imaged channels', 
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select the imaged channels, '
                                'to select multiple hold ctrl/shift'),
              sharpy_bool=dict(widget_type='CheckBox', 
                               text='get images for registration (SHARPy-track)',
                               value=True,
                               tooltip='option to create downsampled images [1140x800 px] for brain registration using SHARPy-track'),
              sharpy_chan=dict(widget_type='Select', 
                               label='selected channels', 
                               value='green',
                               choices=['all', 'dapi', 'green', 'n3', 'cy3', 'cy5'],
                               tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
              contrast_bool=dict(widget_type='CheckBox', 
                                 text='perform contrast adjustment on images for registration',
                                 value=True,
                                 tooltip='option to adjust contrast on images, see option details below'),
              contrast_dapi=dict(widget_type='LineEdit', 
                                 label='set contrast limits for the dapi channel',
                                 value='50,1000', 
                                 tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_green=dict(widget_type='LineEdit', 
                                  label='set contrast limits for the green channel',
                                  value='50,300', 
                                  tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_n3=dict(widget_type='LineEdit', 
                               label='set contrast limits for the n3 channel',
                               value='50,500', 
                               tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy3=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy3 channel',
                                value='50,500', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy5=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy5 channel',
                                value='50,500', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              call_button=False)

    def stitching_widget(
        viewer: Viewer,
        input_path,  # posix path
        stitch_tiles,
        channels,
        sharpy_bool,
        sharpy_chan,
        contrast_bool,
        contrast_dapi,
        contrast_green,
        contrast_n3,
        contrast_cy3,
        contrast_cy5):
        pass
    return stitching_widget



class StitchingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.stitching = initialize_widget()
        btn = QPushButton("stitch images")
        btn.clicked.connect(self._do_stitching)

        self.layout().addWidget(self.stitching.native)
        self.layout().addWidget(btn)
    def _get_info(self, widget):

            return {
                "channels": widget.sharpy_chan.value,
                "contrast_adjustment": widget.contrast_bool.value,
                "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
                "green": [int(i) for i in widget.contrast_green.value.split(',')],
                "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
                "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
                "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
            }

    def _get_stitching_params(self):
        params_dict = {
            "general":
                {
                    "animal_id": get_animal_id(self.stitching.input_path.value),
                    "chans_imaged": self.stitching.channels.value
                },
            "operations":
                {
                    "sharpy_track": self.stitching.sharpy_bool.value,
                },
            "sharpy_track_params": self._get_info(self.stitching),

        }
        return params_dict

    def show_success_message(self, animal_id):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Stitching finished for {animal_id}!")
        msg_box.setWindowTitle("Stitching successful!")
        msg_box.exec_()

    def _do_stitching(self):
        input_path = self.stitching.input_path.value
        # check if user provided a valid input_path
        if not input_path.is_dir() or str(input_path) == '.':
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(
                f"Input path is not a valid directory. Please make sure this exists: >> '{str(input_path)}' <<")
            msg_box.setWindowTitle("Invalid Path Error")
            msg_box.exec_()  # Show the message box
            return
        stitch_tiles = self.stitching.stitch_tiles.value
        params_dict = self._get_stitching_params()
        direct_sharpy_track = params_dict['operations']['sharpy_track']
        params_dict = clean_params_dict(params_dict, "operations")  # remove empty keys
        params_dict = update_params_dict(input_path, params_dict)  # update params.json file, add info on stitching
        filter_list = params_dict['general']['chans_imaged']
        stitching_worker = do_stitching(input_path, filter_list, params_dict, stitch_tiles, direct_sharpy_track)
        stitching_worker.start()
        stitching_worker.returned.connect(self.show_success_message)
