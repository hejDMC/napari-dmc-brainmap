from qtpy.QtCore import Signal
from pathlib import Path
from napari import Viewer
from napari.qt.threading import thread_worker
from natsort import natsorted
import cv2
from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox, QProgressBar
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import numpy as np
import pandas as pd
import random
import matplotlib.colors as mcolors
import time
from napari.utils.notifications import show_info
from typing import Dict, Union, List
from napari_dmc_brainmap.utils.path_utils import get_image_list, get_info
from napari_dmc_brainmap.utils.general_utils import split_to_list
from napari_dmc_brainmap.utils.params_utils import load_params
from napari_dmc_brainmap.utils.gui_utils import ProgressBar, check_input_path
from napari_dmc_brainmap.segment.processing.presegmentation_tools import ProjectionSegmenter, CellsSegmenter
from napari_dmc_brainmap.segment.processing.centroid_finder import CentroidFinder



def get_cmap(name: str) -> Dict[str, str]:
    """
    Retrieve colormap definitions for specific segmentation types.

    Parameters:
        name (str): Name of the colormap category.

    Returns:
        Dict[str, str]: Mapping of segmentation types to colors.
    """
    cmaps = {
        'cells': {
            'dapi': 'yellow',
            'green': 'magenta',
            'n3': 'gray',
            'cy3': 'cyan',
            'cy5': 'lightblue'
        },
        'npx': {
            '0': 'deepskyblue',
            '1': 'orange',
            '2': 'springgreen',
            '3': 'darkgray',
            '4': 'fuchsia',
            '5': 'royalblue',
            '6': 'gold',
            '7': 'powderblue',
            '8': 'lightsalmon',
            '9': 'olive'
        },
        'injection': {
            'dapi': 'gold',
            'green': 'purple',
            'n3': 'navy',
            'cy3': 'darkorange',
            'cy5': 'cornflowerblue'
        },
        'display': {
            'dapi': 'blue',
            'green': 'green',
            'n3': 'orange',
            'cy3': 'red',
            'cy5': 'pink'
        }
    }
    return cmaps[name]
def get_path_to_im(
    input_path: Path, image_idx: int, single_channel: bool = False, chan: str = None, pre_seg: bool = False
) -> Union[Path, str]:
    """
    Get the path to the image to be segmented.

    Parameters:
        input_path (Path): Path to the base input directory.
        image_idx (int): Index of the image to retrieve.
        single_channel (bool, optional): Whether to use single-channel images. Defaults to False.
        chan (str, optional): The specific channel to retrieve. Defaults to None.
        pre_seg (bool, optional): Whether to retrieve the image name for pre-segmentation. Defaults to False.

    Returns:
        Union[Path, str]: The path to the image or the name of the image.
    """
    if single_channel:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channel=chan)
    else:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
    im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
        image_idx
    ]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
    path_to_im = seg_im_dir.joinpath(im)
    if pre_seg:
        if single_channel:
            im_list = get_image_list(input_path, chan, folder_id='single_channel')  # to return im base name for loading preseg
        else:
            im_list = get_image_list(input_path, chan, folder_id='rgb')
        im_name_candidates = [i for i in im_list if im.startswith(i)]
        if len(im_name_candidates) == 1:
            im_name = im_name_candidates[0]
        elif len(im_name_candidates) == 2:
            im_name = im_name_candidates[1]
        else:
            show_info("Can't identify image name, image candidates:")
            show_info(im_name_candidates)
        return im_name
    else:
        return path_to_im

@thread_worker
def create_cells_preseg(cells_segmenter: CellsSegmenter, progress_signal_cells: Signal) -> str:
    """
    Perform pre-segmentation of cells.

    Parameters:
        cells_segmenter (CellsSegmenter): The cells segmentation instance.
        progress_signal_cells (Signal): Progress signal for updating the progress bar.

    Returns:
        str: Status of the operation.
    """
    def progress_callback(progress):
        progress_signal_cells.emit(progress)
    cells_segmenter.process_images(progress_callback=progress_callback)
    return "preseg_cells"

@thread_worker
def create_projection_preseg(projection_segmenter: ProjectionSegmenter, progress_signal_projections: Signal) -> str:
    """
    Perform pre-segmentation of projections.

    Parameters:
        projection_segmenter (ProjectionSegmenter): The projections segmentation instance.
        progress_signal_projections (Signal): Progress signal for updating the progress bar.

    Returns:
        str: Status of the operation.
    """
    def progress_callback(progress):
        progress_signal_projections.emit(progress)

    projection_segmenter.process_images(progress_callback=progress_callback)
    return "preseg_proj"

@thread_worker
def get_centroid_coord(centroid_finder: CentroidFinder, progress_signal_centroid: Signal) -> str:
    """
    Perform centroid finding on pre-segmented data.

    Parameters:
        centroid_finder (CentroidFinder): The centroid finder instance.
        progress_signal_centroid (Signal): Progress signal for updating the progress bar.

    Returns:
        str: Status of the operation.
    """

    def progress_callback(progress):
        progress_signal_centroid.emit(progress)

    centroid_finder.process_all_masks(progress_callback=progress_callback)

    return "find_centroids"

def initialize_segment_widget() -> FunctionGui:
    """
    Initialize the segment widget for selecting input path and segmentation parameters.

    Returns:
        FunctionGui: The initialized widget for segment configuration.
    """
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
    """
    Initialize the widget for loading pre-segmentation data.

    Returns:
        FunctionGui: The initialized widget for loading pre-segmentation data.
    """
    @magicgui(layout='vertical',
              load_bool=dict(widget_type='CheckBox',
                             label='load presegmented data',
                             value=False,
                             tooltip='tick to load presegmented data for manual curation'),
              pre_seg_folder=dict(widget_type='LineEdit',
                                  label='folder name with presegmented data',
                                  value='presegmentation',
                                  tooltip='folder needs to contain sub-folders with channel names. WARNING: if the channel is called '
                                          '*segmentation* (as for loading old data), manual curation will override existing data. Copy the folder if you want to keep data. '
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


def initialize_presegcells_widget():
    """
    Initialize the widget for configuring cells pre-segmentation.

    Returns:
        FunctionGui: The initialized widget for cells pre-segmentation.
    """
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
    def create_preseg_cells(
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

    return create_preseg_cells


def initialize_presegproj_widget():
    """
    Initialize the widget for configuring projections pre-segmentation.

    Returns:
        FunctionGui: The initialized widget for projections pre-segmentation.
    """
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
              start_end_im=dict(widget_type='LineEdit', label='image range to presegment', value='',
                                tooltip='if you only want to segment a subset of images enter COMMA SEPARATED indices '
                                        'of the first and last image to presegment, e.g. 0,10'),
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
    def create_preseg_projections(
            viewer: Viewer,
            regi_bool,
            regi_chan,
            start_end_im,
            binary_folder,
            output_folder):
        pass

    return create_preseg_projections


def initialize_findcentroids_widget():
    """
    Initialize the widget for configuring centroid detection of pre-segmentation data.

    Returns:
        FunctionGui: The initialized widget for centroid detection.
    """
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
    """
    QWidget for configuring and running segmentation processes for cells and projections.
    """
    progress_signal_cells = Signal(int)
    """
    """
    progress_signal_projections = Signal(int)
    """
    """
    progress_signal_centroid = Signal(int)
    """
    """

    def __init__(self, napari_viewer: Viewer) -> None:
        """
        Initialize the SegmentWidget.

        Parameters:
            napari_viewer (Viewer): The Napari viewer instance where segmentation results will be visualized.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.segment = initialize_segment_widget()
        self.segment.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self.save_dict = self._default_save_dict()

        self._collapse_load_preseg = QCollapsible('Load old/presegmented data: expand for more', self)
        self.load_preseg = initialize_loadpreseg_widget()
        self.load_preseg.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_load_preseg.addWidget(self.load_preseg.root_native_widget)

        self._collapse_cells = QCollapsible('Create presegmentation data for cells: expand for more', self)
        self.preseg_cells = initialize_presegcells_widget()
        self.preseg_cells.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_cells.addWidget(self.preseg_cells.root_native_widget)
        self.btn_cells = QPushButton("run presegmentation and store data")
        self.btn_cells.clicked.connect(self._create_cells_preseg)
        self._collapse_cells.addWidget(self.btn_cells)
        self.progress_bar_cells = ProgressBar(self)
        self._collapse_cells.addWidget(self.progress_bar_cells)
        self.progress_signal_cells.connect(self.progress_bar_cells.set_value)

        self._collapse_projections = QCollapsible('Create presegmentation data for projections: expand for more', self)
        self.preseg_projections = initialize_presegproj_widget()
        self.preseg_projections.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_projections.addWidget(self.preseg_projections.root_native_widget)
        self.btn_projections = QPushButton("create presegmentation of projections data")
        self.btn_projections.clicked.connect(self._create_projection_preseg)#(self._create_projection_preseg)
        self._collapse_projections.addWidget(self.btn_projections)
        self.progress_bar_proj = ProgressBar(self)
        self._collapse_projections.addWidget(self.progress_bar_proj)
        self.progress_signal_projections.connect(self.progress_bar_proj.set_value)

        self._collapse_centroid = QCollapsible('Find centroids for presegmented data (masks): expand for more', self)
        self.centroid = initialize_findcentroids_widget()
        self.centroid.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_centroid.addWidget(self.centroid.root_native_widget)
        self.btn_find_centroids = QPushButton("get center coordinates for presegmented data")
        self.btn_find_centroids.clicked.connect(self._get_centroid_coord)
        self._collapse_centroid.addWidget(self.btn_find_centroids)
        self.progress_bar_centroid = ProgressBar(self)
        self._collapse_centroid.addWidget(self.progress_bar_centroid)
        self.progress_signal_centroid.connect(self.progress_bar_centroid.set_value)

        self.btn = QPushButton("save data and load next image")
        self.btn.clicked.connect(self._save_and_load)

        self.layout().addWidget(self.segment.native)
        self.layout().addWidget(self._collapse_load_preseg)
        self.layout().addWidget(self._collapse_cells)
        self.layout().addWidget(self._collapse_projections)
        self.layout().addWidget(self._collapse_centroid)
        self.layout().addWidget(self.btn)

    def _default_save_dict(self) -> Dict[str, Union[bool, int]]:
        """
        Create a default save dictionary for storing segmentation data.

        Returns:
            Dict[str, Union[bool, int]]: A dictionary with default values for image index, segmentation type, and probe count.
        """
        save_dict = {
            "image_idx": False,
            "seg_type": False,
            "n_probes": False
        }
        return save_dict

    def _update_save_dict(self, image_idx: Union[bool, int], seg_type: Union[bool, str], n_probes: Union[bool, int]) -> Dict[str, Union[int, str]]:
        """
        Update the save dictionary with the current image index, segmentation type, and number of probes.

        Parameters:
            image_idx (int): The index of the image being segmented.
            seg_type (str): The type of segmentation being performed.
            n_probes (int): The number of probes.

        Returns:
            Dict[str, Union[int, str]]: The updated save dictionary.
        """
        # get image idx and segmentation type for saving segmentation data
        self.save_dict['image_idx'] = image_idx
        self.save_dict['seg_type'] = seg_type
        self.save_dict['n_probes'] = n_probes
        return self.save_dict

    def _get_contrast_dict(self, widget: FunctionGui) -> Dict[str, List[int]]:
        """
        Retrieve contrast settings for each channel from the widget.

        Parameters:
            widget (FunctionGui): The widget containing contrast settings.

        Returns:
            Dict[str, List[int]]: A dictionary of contrast limits for each channel.
        """

        return {
            "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
            "green": [int(i) for i in widget.contrast_green.value.split(',')],
            "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
            "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
            "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
        }

    def _save_and_load(self) -> None:
        """
        Save the current segmentation data and load the next image for segmentation.

        Raises:
            IndexError: If the image index exceeds the number of images in the target folder.
        """

        input_path = self.segment.input_path.value
        # check if user provided a valid input_path
        if not check_input_path(input_path):
            return
        image_idx = int(self.segment.image_idx.value)
        seg_type = self.segment.seg_type.value
        channels = self.segment.channels.value
        n_probes = int(self.segment.n_probes.value)
        single_channel = self.segment.single_channel_bool.value
        contrast_dict = self._get_contrast_dict(self.segment)

        if len(self.viewer.layers) == 0:  # no open images, set save_dict to defaults
            self.save_dict = self._default_save_dict()
        if type(self.save_dict['image_idx']) == int:  # todo there must be a better way :-D (for image_idx = 0)
            self._save_data(input_path, channels, single_channel)
        del (self.viewer.layers[:])  # remove open layers

        try:
            self._load_next(input_path, seg_type, channels, image_idx, n_probes, single_channel, contrast_dict)

        except IndexError:
            show_info("Index out of range, check that index matches image count in target folder")

    def _load_next(self, input_path: Path, seg_type: str, channels: List[str], image_idx: int, n_probes: int,
                   single_channel: bool, contrast_dict: Dict[str, List[int]]) -> None:
        """
        Load the next image for segmentation and initialize segmentation objects.

        Parameters:
            input_path (Path): Path to the input directory.
            seg_type (str): The type of segmentation being performed.
            channels (List[str]): List of channels to process.
            image_idx (int): The index of the image to load.
            n_probes (int): The number of probes.
            single_channel (bool): Whether to process single-channel images.
            contrast_dict (Dict[str, List[int]]): Contrast settings for each channel.
        """
        self.save_dict = self._update_save_dict(image_idx, seg_type, n_probes)
        if single_channel:
            for chan in channels:
                path_to_im = get_path_to_im(input_path, image_idx, single_channel=single_channel, chan=chan)
                self._load_single(path_to_im, chan, contrast_dict)
        else:
            path_to_im = get_path_to_im(input_path, image_idx)
            self._load_rgb(path_to_im, channels, contrast_dict)
        self._create_seg_objects(input_path, seg_type, channels, n_probes, image_idx, single_channel)

        show_info(f"loaded {path_to_im.parts[-1]} (cnt={str(image_idx)})")
        image_idx += 1
        self.segment.image_idx.value = image_idx
        # change_index(image_idx)

    def _load_rgb(self, path_to_im: Path, channels: List[str], contrast_dict: Dict[str, List[int]]) -> None:
        """
        Load and display RGB image data.

        Parameters:
            path_to_im (Path): Path to the RGB image.
            channels (List[str]): List of channels to process.
            contrast_dict (Dict[str, List[int]]): Contrast settings for each channel.
        """
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

    def _load_single(self, path_to_im: Path, chan: str, contrast_dict: Dict[str, List[int]]) -> None:
        """
        Load and display single-channel image data.

        Parameters:
            path_to_im (Path): Path to the single-channel image.
            chan (str): The channel being processed.
            contrast_dict (Dict[str, List[int]]): Contrast settings for the channel.
        """

        cmap_disp = get_cmap('display')
        im_loaded = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
        self.viewer.add_image(im_loaded, name=f'{chan} channel', colormap=cmap_disp[chan], opacity=0.5)
        self.viewer.layers[f'{chan} channel'].contrast_limits = contrast_dict[chan]

    def _load_preseg_object(self, input_path: Path, chan: str, image_idx: int, seg_type: str, single_channel: bool) -> np.ndarray:
        """
        Load pre-segmented data for a specific image and channel.

        Parameters:
            input_path (Path): Path to the input directory.
            chan (str): The channel being processed.
            image_idx (int): The index of the image being segmented.
            seg_type (str): The type of segmentation.
            single_channel (bool): Whether to process single-channel images.

        Returns:
            np.ndarray: Array of pre-segmented data.
        """
        pre_seg_folder = self.load_preseg.pre_seg_folder.value
        pre_seg_dir, pre_seg_list, pre_seg_suffix = get_info(input_path, pre_seg_folder, seg_type=seg_type,
                                                             channel=chan)
        if single_channel:
            im_name = get_path_to_im(input_path, image_idx, single_channel=single_channel, chan=chan, pre_seg=True)
        else:
            im_name = get_path_to_im(input_path, image_idx, pre_seg=True)  # name of image that will be loaded
        fn_to_load = [d for d in pre_seg_list if d.startswith(f'{im_name}_')]
        if len(fn_to_load) > 0:
            pre_seg_data_dir = pre_seg_dir.joinpath(fn_to_load[0])
            df = pd.read_csv(pre_seg_data_dir)  # load dataframe
            try:
                if seg_type == 'injection_site':
                    if not 'idx_shape' in df.columns:
                        pre_seg_data = df[['Position Y', 'Position X']].to_numpy()
                    else:
                        pre_seg_data = []
                        for idx in df['idx_shape'].unique():
                            pre_seg_data.append(df[df['idx_shape'] == idx][['Position Y', 'Position X']].to_numpy())
                else:
                    pre_seg_data = df[['Position Y', 'Position X']].to_numpy()
            except KeyError:
                show_info("csv file missing columns (Position Y/X), no presegmented data loaded")
                pre_seg_data = []
        else:
            pre_seg_data = []
        return pre_seg_data

    def _create_seg_objects(self, input_path: Path, seg_type: str, channels: List[str], n_probes: int,
                            image_idx: int, single_channel: bool) -> None:
        """
        Create segmentation objects in the viewer for the specified image and segmentation type.

        Parameters:
            input_path (Path): Path to the input directory.
            seg_type (str): The type of segmentation being performed.
            channels (List[str]): List of channels to process.
            n_probes (int): The number of probes.
            image_idx (int): The index of the image being segmented.
            single_channel (bool): Whether to process single-channel images.
        """
        if seg_type == 'injection_site':
            cmap_dict = get_cmap('injection')
            if self.load_preseg.load_bool.value:
                for chan in channels:
                    pre_seg_data = self._load_preseg_object(input_path, chan, image_idx, seg_type, single_channel)
                    if type(pre_seg_data) is list:
                        for inj_poly in pre_seg_data:
                            self.viewer.add_shapes(inj_poly, name=chan, shape_type='polygon', face_color=cmap_dict[chan],
                                                   opacity=0.4)
                    else:
                        self.viewer.add_shapes(pre_seg_data, name=chan, shape_type='polygon', face_color=cmap_dict[chan],
                                           opacity=0.4)
            else:
                for chan in channels:
                    self.viewer.add_shapes(name=chan, face_color=cmap_dict[chan], opacity=0.4)
        elif seg_type in ['cells', 'projections']:
            cmap_dict = get_cmap('cells')
            if self.load_preseg.load_bool.value:
                for chan in channels:
                    pre_seg_data = self._load_preseg_object(input_path, chan, image_idx, seg_type, single_channel)
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
                p_id = f'{seg_type}_{str(i)}'
                if self.load_preseg.load_bool.value:
                    pre_seg_data = self._load_preseg_object(input_path, p_id, image_idx, seg_type, single_channel)
                    self.viewer.add_points(pre_seg_data, size=int(self.segment.point_size.value), name=p_id,
                                           face_color=p_color)
                else:
                    self.viewer.add_points(size=int(self.segment.point_size.value), name=p_id, face_color=p_color)

    def _save_data(self, input_path: Path, channels: List[str], single_channel: bool) -> None:
        """
        Save segmentation data for the current image.

        Parameters:
            input_path (Path): Path to the input directory.
            channels (List[str]): List of channels being segmented.
            single_channel (bool): Whether to process single-channel images.
        """
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

    def _create_cells_preseg(self) -> None:
        """
        Run cells pre-segmentation process.
        """
        input_path = self.segment.input_path.value
        # check if user provided a valid input_path
        if not check_input_path(input_path):
            return
        params_dict = load_params(input_path)

        general_params = {
            "xyz_dict": params_dict['atlas_info']['xyz_dict'],
            "atlas_id": params_dict['atlas_info']['atlas'],
            "channels": self.segment.channels.value,
            "regi_bool": self.preseg_cells.regi_bool.value, # todo change widgets here
            "regi_chan": self.preseg_cells.regi_chan.value,
            "seg_type": 'cells',
            "start_end_im": split_to_list(self.preseg_cells.start_end_im.value, out_format='int'),
            "output_folder": self.preseg_cells.output_folder.value
        }

        cells_params = {
            "single_channel": self.preseg_cells.single_channel_bool.value,
            "mask_folder": self.preseg_cells.mask_folder.value

        }

        preseg_params = {
            "intensity_norm": split_to_list(self.preseg_cells.intensity_norm.value, out_format='float'),
            "gaussian_smoothing_sigma": int(self.preseg_cells.gaussian_smoothing_sigma.value),
            # "gaussian_smoothing_truncate_range": int(self.preseg_cells.gaussian_smoothing_truncate_range.value),
            "dot_3d_sigma": int(self.preseg_cells.dot_3d_sigma.value),
            "dot_3d_cutoff": float(self.preseg_cells.dot_3d_cutoff.value),
            "hole_min_max": split_to_list(self.preseg_cells.hole_min_max.value, out_format='int'),
            "minArea": int(self.preseg_cells.minArea.value)
        }

        cells_segmenter = CellsSegmenter(
            input_path=input_path,
            general_params=general_params,
            cells_params=cells_params,
            preseg_params=preseg_params
        )
        cell_worker = create_cells_preseg(cells_segmenter, self.progress_signal_cells)
        #     center_worker.yielded.connect(self._update_progress)
        cell_worker.started.connect(
            lambda: self.btn_cells.setText("Presegmenting images..."))  # Change button text when stitching starts
        cell_worker.returned.connect(self._show_success_message)
        cell_worker.start()

    def _create_projection_preseg(self) -> None:
        """
        Run projections pre-segmentation process.
        """
        input_path = self.segment.input_path.value
        # check if user provided a valid input_path
        if not check_input_path(input_path):
            return
        params_dict = load_params(input_path)

        general_params = {
            "xyz_dict": params_dict['atlas_info']['xyz_dict'],
            "atlas_id": params_dict['atlas_info']['atlas'],
            "channels": self.segment.channels.value,
            "regi_bool": self.preseg_projections.regi_bool.value,
            "regi_chan": self.preseg_projections.regi_chan.value,
            "seg_type": 'projections',
            "start_end_im": split_to_list(self.preseg_projections.start_end_im.value, out_format='int'),
            "output_folder": self.preseg_projections.output_folder.value
        }
        projection_params = {
            "binary_folder": self.preseg_projections.binary_folder.value,

        }

        projection_segmenter = ProjectionSegmenter(
            input_path=input_path,
            general_params=general_params,
            projection_params=projection_params
        )
        projection_worker = create_projection_preseg(projection_segmenter, self.progress_signal_projections)
        #     center_worker.yielded.connect(self._update_progress)
        projection_worker.started.connect(
            lambda: self.btn_projections.setText("Presegmenting images..."))  # Change button text when stitching starts
        projection_worker.returned.connect(self._show_success_message)
        projection_worker.start()

    def _get_centroid_coord(self) -> None:
        """
        Run centroid finding process on pre-segmented data.
        """
        input_path = self.segment.input_path.value
        if not check_input_path(input_path):
            return
        channels = self.segment.channels.value
        mask_folder = self.centroid.mask_folder.value
        mask_type = self.centroid.mask_type.value
        output_folder = self.centroid.output_folder.value

        # Instantiate the CentroidFinder with user-provided parameters
        centroid_finder = CentroidFinder(
            input_path=input_path,
            mask_folder=mask_folder,
            output_folder=output_folder,
            channels=channels,
            mask_type=mask_type
        )
        centroid_worker = get_centroid_coord(centroid_finder, self.progress_signal_centroid)
        #     center_worker.yielded.connect(self._update_progress)
        centroid_worker.started.connect(
            lambda: self.btn_find_centroids.setText("Calculating centroids..."))  # Change button text when stitching starts
        centroid_worker.returned.connect(self._show_success_message)
        centroid_worker.start()

    def _show_success_message(self, operation: str) -> None:
        """
        Display a message box with the operation status.

        Parameters:
            operation (str): ID of the performed operation.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        if operation == 'preseg_cells':
            msg = 'Presegmentation finished!'
            self.btn_cells.setText("run presegmentation and store data")  # Reset button text after process completion
            self.progress_signal_cells.emit(0)
        elif operation == 'preseg_proj':
            msg = 'Presegmentation finished!'
            self.btn_projections.setText("create presegmentation of projections data")  # Reset button text after process completion
            self.progress_signal_projections.emit(0)
        else:
            msg = 'Centroid analysis completed!'
            self.btn_find_centroids.setText(
                "get center coordinates for presegmented data")  # Reset button text after process completion
            self.progress_signal_centroid.emit(0)

        msg_box.setText(msg)
        msg_box.setWindowTitle("Operation successful!")
        msg_box.exec_()

