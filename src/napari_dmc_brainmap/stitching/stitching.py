"""
DMC-BrainMap widget for stitching .tif files.

2024 - FJ, XC
"""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox, QProgressBar
from napari import Viewer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from natsort import natsorted
import json
import tifffile as tiff
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from napari_dmc_brainmap.stitching.stitching_tools import stitch_stack, stitch_folder
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.params_utils import clean_params_dict, update_params_dict
from napari_dmc_brainmap.utils.gui_utils import check_input_path


@thread_worker(progress={'total': 100})
def do_stitching(input_path: Path,
                 filter_list: List[str],
                 params_dict: Dict,
                 stitch_tiles: bool,
                 direct_sharpy_track: bool) -> str:
    """
    Perform stitching operation using input path and parameters provided.

    Parameters:
        input_path (Path): Path to directory containing data for stitching.
        filter_list (List[str]): List of channels to stitch.
        params_dict (Dict): Dictionary containing stitching parameters.
        stitch_tiles (bool): Whether to stitch individual tiles or use DMC-FluoImager data.
        direct_sharpy_track (bool): Whether to create data for SHARPy-track.

    Yields:
        int: Progress value for the stitching process.

    Returns:
        str: The animal ID for which stitching was performed.
    """
    animal_id = get_animal_id(input_path)
    resolution = tuple(params_dict['atlas_info']['resolution'])
    data_dir = input_path.joinpath('raw')
    objs = natsorted([o.parts[-1] for o in data_dir.iterdir() if o.is_dir()])

    if not objs:
        show_info('No object slides under raw-data folder!')
        return

    progress_value = 0
    progress_step = 100 / (len(objs) * len(filter_list))

    for obj in objs:
        in_obj = data_dir.joinpath(obj)
        for f in filter_list:
            stitch_dir = get_info(input_path, 'stitched', channel=f, create_dir=True, only_dir=True)
            if stitch_tiles:
                process_stitch_folder(input_path, in_obj, f, stitch_dir, animal_id, obj, params_dict, resolution, direct_sharpy_track)
            else:
                process_stitch_stack(input_path, in_obj, f, stitch_dir, animal_id, obj, params_dict, resolution, direct_sharpy_track)
            progress_value += progress_step
            yield int(progress_value)

    yield 100
    return animal_id


def process_stitch_folder(input_path: Path,
                          in_obj: Path,
                          f: str,
                          stitch_dir: Path,
                          animal_id: str,
                          obj: str,
                          params_dict: Dict,
                          resolution: Tuple[int, int],
                          direct_sharpy_track: bool,
                          overlap: int = 205) -> None:
    """
    Process stitching for a folder of tiles.

    Parameters:
        input_path (Path): Base path to animal data.
        in_obj (Path): Input path for object data.
        f (str): Channel to process.
        stitch_dir (Path): Directory to save stitched data.
        animal_id (str): Animal ID.
        obj (str): Object name.
        params_dict (Dict): Parameters for stitching.
        resolution (Tuple[int, int]): Resolution of the atlas used for registration.
        direct_sharpy_track (bool): Whether to create SHARPy-track data directly.
        overlap (int, optional): Overlap for stitching tiles. Defaults to 205.
    """
    in_chan = in_obj.joinpath(f)
    section_list = natsorted([s.parts[-1] for s in in_chan.iterdir() if s.is_dir()])
    section_list_new = [f"{animal_id}_{obj}_{str(k + 1)}" for k, ss in enumerate(section_list)]
    [in_chan.joinpath(old).rename(in_chan.joinpath(new)) for old, new in zip(section_list, section_list_new)]
    section_dirs = natsorted([s for s in in_chan.iterdir() if s.is_dir()])

    for section in section_dirs:
        stitched_path = stitch_dir.joinpath(f'{section.parts[-1]}_stitched.tif')
        if direct_sharpy_track:
            sharpy_chans = params_dict['sharpy_track_params']['channels']
            if f in sharpy_chans:
                sharpy_dir = get_info(input_path, 'sharpy_track', channel=f, create_dir=True, only_dir=True)
                sharpy_im_dir = sharpy_dir.joinpath(f'{section.parts[-1]}_downsampled.tif')
                stitch_folder(section, overlap, stitched_path, params_dict, f, sharpy_im_dir, resolution=resolution)
            else:
                stitch_folder(section, overlap, stitched_path, params_dict, f, resolution=resolution)
        else:
            stitch_folder(section, overlap, stitched_path, params_dict, f, resolution=resolution)


def process_stitch_stack(input_path: Path,
                         in_obj: Path,
                         f: str,
                         stitch_dir: Path,
                         animal_id: str,
                         obj: str,
                         params_dict: Dict,
                         resolution: Tuple[int, int],
                         direct_sharpy_track: bool,
                         overlap: int = 205) -> None:
    """
    Process stitching for a stack of tiles.

    Parameters:
        input_path (Path): Base path to animal data.
        in_obj (Path): Input path for object data.
        f (str): Channel to process.
        stitch_dir (Path): Directory to save stitched data.
        animal_id (str): Animal ID.
        obj (str): Object name.
        params_dict (Dict): Parameters for stitching.
        resolution (Tuple[int, int]): Resolution of the atlas used for registration.
        direct_sharpy_track (bool): Whether to create SHARPy-track data directly.
        overlap (int, optional): Overlap for stitching tiles. Defaults to 205.
    """
    in_chan = in_obj.joinpath(f'{obj}_{f}_1')
    stack = natsorted([im.parts[-1] for im in in_chan.glob('*.tif')])
    whole_stack = load_tile_stack(in_chan, stack)

    meta_json_where = in_obj.joinpath(f'{obj}_meta_1', 'regions_pos.json')
    with open(meta_json_where, 'r') as data:
        img_meta = json.load(data)

    region_n = len(img_meta)
    for rn in range(region_n):
        pos_list = img_meta['region_' + str(rn)]
        stitched_path = stitch_dir.joinpath(f'{animal_id}_{obj}_{str(rn + 1)}_stitched.tif')
        if direct_sharpy_track:
            sharpy_chans = params_dict['sharpy_track_params']['channels']
            if f in sharpy_chans:
                sharpy_dir = get_info(input_path, 'sharpy_track', channel=f, create_dir=True, only_dir=True)
                sharpy_im_dir = sharpy_dir.joinpath(f'{animal_id}_{obj}_{str(rn + 1)}_downsampled.tif')
                stitch_stack(pos_list, whole_stack, overlap, stitched_path, params_dict, f, resolution=resolution, downsampled_path=sharpy_im_dir)
            else:
                stitch_stack(pos_list, whole_stack, overlap, stitched_path, params_dict, f, resolution=resolution)
        else:
            stitch_stack(pos_list, whole_stack, overlap, stitched_path, params_dict, f, resolution=resolution)
        whole_stack = np.delete(whole_stack, [np.arange(len(pos_list))], axis=0)


def load_tile_stack(in_chan: Path, stack: List[str], c_size: int = 2048) -> np.ndarray:
    """
    Load a stack of tiles from the specified input channel.

    Parameters:
        in_chan (Path): Input path containing tiles.
        stack (List[str]): List of tile file names.
        c_size (int, optional): Size of the tiles. Default is 2048.

    Returns:
    np.ndarray: Loaded stack of images as a numpy array.
    """
    tif_meta = tiff.read_ndtiff_index(in_chan.joinpath("NDTiff.index"))
    page_count = 0
    for _ in tif_meta:
        page_count += 1

    whole_stack = np.zeros((page_count, c_size, c_size), dtype=np.uint16)
    page_count = 0
    for stk in stack:
        with tiff.TiffFile(in_chan.joinpath(stk)) as tif:
            for page in tif.pages:
                image = page.asarray()
                try:
                    whole_stack[page_count, :, :] = image
                except ValueError:
                    show_info("Tile:{} data corrupted. Setting tile pixels value to 0".format(page_count))
                page_count += 1
    return whole_stack


def initialize_widget() -> FunctionGui:
    """
    Initialize the magicgui widget for stitching configuration.

    Returns:
    FunctionGui: Initialized magicgui widget.
    """
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
        input_path: Path,
        stitch_tiles: bool,
        channels: List[str],
        sharpy_bool: bool,
        sharpy_chan: str,
        contrast_bool: bool,
        contrast_dapi: str,
        contrast_green: str,
        contrast_n3: str,
        contrast_cy3: str,
        contrast_cy5: str) -> None:
        """
        Function to handle stitching widget parameters.

        Parameters:
            viewer (Viewer): Napari viewer instance.
            input_path (Path): Input path for stitching.
            stitch_tiles (bool): Whether to stitch image tiles.
            channels (List[str]): Channels to stitch.
            sharpy_bool (bool): Whether to create downsampled images for registration.
            sharpy_chan (str): Channels for SHARPy processing.
            contrast_bool (bool): Whether to adjust contrast.
            contrast_dapi (str): Contrast limits for DAPI channel.
            contrast_green (str): Contrast limits for Green channel.
            contrast_n3 (str): Contrast limits for N3 channel.
            contrast_cy3 (str): Contrast limits for Cy3 channel.
            contrast_cy5 (str): Contrast limits for Cy5 channel.
        """
        pass

    return stitching_widget


class StitchingWidget(QWidget):
    """
    QWidget for configuring and initiating the stitching process.
    """
    progress_signal = Signal(int)
    """Signal emitted to update the progress bar with an integer value."""

    def __init__(self, napari_viewer: Viewer) -> None:
        """
        Initialize the StitchingWidget instance.

        Parameters:
            napari_viewer (Viewer): The Napari viewer instance where the widget is added.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.stitching = initialize_widget()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.btn = QPushButton("Stitch Images")
        self.btn.clicked.connect(self._do_stitching)

        self.layout().addWidget(self.stitching.native)
        self.layout().addWidget(self.btn)
        self.layout().addWidget(self.progress_bar)
        self.progress_signal.connect(self.progress_bar.setValue)

    def _get_info(self, widget: FunctionGui) -> Dict:
        """
        Retrieve stitching configuration from the widget.

        Parameters:
            widget (FunctionGui): The stitching widget instance.

        Returns:
            Dict: A dictionary containing contrast settings and SHARPy-track parameters.
        """
        return {
            "channels": widget.sharpy_chan.value,
            "contrast_adjustment": widget.contrast_bool.value,
            "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
            "green": [int(i) for i in widget.contrast_green.value.split(',')],
            "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
            "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
            "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')],
        }

    def _get_stitching_params(self) -> Dict:
        """
        Generate the stitching parameters from user input.

        Returns:
            Dict: A dictionary containing stitching parameters.
        """
        params_dict = {
            "general": {
                "animal_id": get_animal_id(self.stitching.input_path.value),
                "chans_imaged": self.stitching.channels.value
            },
            "operations": {
                "sharpy_track": self.stitching.sharpy_bool.value
            },
            "sharpy_track_params": self._get_info(self.stitching)
        }
        return params_dict

    def _show_success_message(self, animal_id: str) -> None:
        """
        Display a success message upon completion of stitching.

        Parameters:
            animal_id (str): The animal ID for which stitching was performed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Stitching finished for {animal_id}!")
        msg_box.setWindowTitle("Stitching Successful!")
        msg_box.exec_()
        self.btn.setText("Stitch Images")  # Reset button text after completion
        self.progress_signal.emit(0)

    def _update_progress(self, value: int) -> None:
        """
        Update the progress bar.

        Parameters:
            value (int): Progress value to set.
        """
        self.progress_signal.emit(value)

    def _do_stitching(self) -> None:
        """
        Initiate the stitching process based on user-configured parameters.
        """
        input_path = self.stitching.input_path.value
        if not check_input_path(input_path):
            return

        stitch_tiles = self.stitching.stitch_tiles.value
        params_dict = self._get_stitching_params()
        direct_sharpy_track = params_dict["operations"]["sharpy_track"]
        params_dict = clean_params_dict(params_dict, "operations")  # Remove empty keys
        params_dict = update_params_dict(input_path, params_dict)  # Update params.json with stitching info
        filter_list = params_dict["general"]["chans_imaged"]

        stitching_worker = do_stitching(input_path, filter_list, params_dict, stitch_tiles, direct_sharpy_track)
        stitching_worker.yielded.connect(self._update_progress)
        stitching_worker.started.connect(lambda: self.btn.setText("Stitching Images..."))  # Update button text
        stitching_worker.returned.connect(self._show_success_message)
        stitching_worker.start()
