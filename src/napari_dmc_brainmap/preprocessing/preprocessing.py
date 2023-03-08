"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from napari import Viewer
from napari.layers import Image, Shapes
from napari.qt.threading import thread_worker
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from napari_dmc_brainmap.utils import get_animal_id, get_im_list
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QFileDialog, QLineEdit
from superqt import QCollapsible
from magicgui import magicgui
from napari_dmc_brainmap.preprocessing.preprocessing_tools import preprocess_images, create_dirs


# todo for alignment https://forum.image.sc/t/widgets-alignment-in-the-plugin-when-nested-magic-class-and-magicgui-are-used/62929/12




@magicgui(
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                      tooltip='directory of folder containing subfolders with e.g. images, segmentation results, '
                              'NOT folder containing segmentation results'),
    chans_imaged=dict(widget_type='Select', label='imaged channels', choices=['dapi', 'green', 'cy3', 'cy5'],
                      value=['green','cy3'],
                      tooltip='select all channels imaged, to select multiple hold shift'),  # todo fix height of widget
    call_button=False
)
def header_widget(
        self,
        input_path,
        chans_imaged
):
    return header_widget

# todo this as class
@magicgui(
    rgb_button=dict(widget_type='CheckBox', text='create RGB images', value=False,
                    tooltip='option to create RGB images, tick to create RGB images'),
    rgb_channels=dict(widget_type='Select', label='imaged channels', value='all',
                      choices=['all', 'dapi', 'green', 'cy3'],
                      tooltip='select all channels imaged, to select multiple hold shift'),  # todo fix height of widget
    rgb_ds=dict(widget_type='CheckBox', text='perform downsampling on RGB images', value=True,
                tooltip='option to downsample images, see option details below'),
    rgb_contrast=dict(widget_type='CheckBox', text='perform contrast adjustment on RGB images', value=True,
                      tooltip='option to adjust contrast on images, see option details below'),
    call_button=False
)
def do_rgb(
        self,
        rgb_button,
        rgb_channels,
        rgb_ds,
        rgb_contrast

):
    return do_rgb

@magicgui(
    ds_params=dict(widget_type='SpinBox', label='enter downsampling factor', value=3,
                   tooltip='enter scale factor for downsampling'),
    contrast_dapi=dict(widget_type='LineEdit', label='set contrast limits for the dapi channel',
                       value='50,2000', tooltip='enter contrast limits (default values for 16-bit image)'),
    contrast_green=dict(widget_type='LineEdit', label='set contrast limits for the green channel',
                        value='50,1000', tooltip='enter contrast limits (default values for 16-bit image)'),
    contrast_cy3=dict(widget_type='LineEdit', label='set contrast limits for the cy3 channel',
                      value='50,2000', tooltip='enter contrast limits (default values for 16-bit image)'),
    contrast_cy5=dict(widget_type='LineEdit', label='set contrast limits for the cy5 channel',
                      value='50,1000', tooltip='enter contrast limits (default values for 16-bit image)'),
    num_cores=dict(widget_type='SpinBox', label='enter the number of parallel processes', value=1, min=1,
                   tooltip='XXX'),  # todo write this better
    call_button=False
)
def footer_widget(
        self,
        ds_params,
        contrast_dapi,
        contrast_green,
        contrast_cy3,
        contrast_cy5,
        num_cores
):
    return footer_widget

@thread_worker
def do_preprocessing(num_cores, input_path, filter_list, img_list, params_dict, save_dirs):

    if num_cores > multiprocessing.cpu_count():
        print("maximum available cores = " + str(multiprocessing.cpu_count()))
    if save_dirs:
        Parallel(n_jobs=num_cores)(
            delayed(preprocess_images)(im, filter_list, input_path, params_dict, save_dirs) for im in tqdm(img_list))
        print("DONE!")
    else:
        print("No preprocessing operations selected, expand the respective windows and tick check box")
class PreprocessingWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        header = header_widget

        #self.input_path = QLineEdit()
            #QFileDialog(self, 'input path (animal_id):')#, mode='d', tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                                      #'folder containing segmentation results')
        # input_path = dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
        #                                       tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
        #                                             'folder containing segmentation results')
        # QCollapsible creates a collapse container for inner widgets
        self._collapse2 = QCollapsible('Create RGB: expand for more', self)
        # magicgui doesn't need to be used as a decorator, you can call it
        # repeatedly to create new widgets:
        rgb_widget = do_rgb
        # if you're mixing Qt and magicgui, you need to use the "native"
        # attribute in magicgui to access the QWidget
        self._collapse2.addWidget(rgb_widget.native)

        self._collapse3 = QCollapsible('Class: expand for more', self)

        footer = footer_widget



        btn = QPushButton("Do the preprocessing!")
        btn.clicked.connect(self._do_preprocessing)
        self.layout().addWidget(header.native)
        self.layout().addWidget(self._collapse2)
        self.layout().addWidget(self._collapse3)
        self.layout().addWidget(footer.native)
        self.layout().addWidget(btn)

    def _get_preprocessing_params(self):
        params_dict = {
            "general":
                {
                    "animal_id": get_animal_id(header_widget.input_path.value),
                    "chans_imaged": header_widget.chans_imaged.value
                },
            "operations":
                {
                    "rgb": do_rgb.rgb_button.value,
                    "single_channel": False,
                    "stack": False,
                    "sharpy_track": False,
                    "binary": False
                },
            "rgb_params":
                {
                    "channels": do_rgb.rgb_channels.value,
                    "downsampling": do_rgb.rgb_ds.value,
                    "contrast_adjustment": do_rgb.rgb_contrast.value
                },
            "downsample_params":
                {  # todo rigid downsampling option
                    "scale_factor": footer_widget.ds_params.value
                },
            "contrast_params":
                {
                    "dapi": [int(i) for i in footer_widget.contrast_dapi.value.split(',')],
                    "green": [int(i) for i in footer_widget.contrast_green.value.split(',')],
                    "cy3": [int(i) for i in footer_widget.contrast_cy3.value.split(',')],
                    "cy5": [int(i) for i in footer_widget.contrast_cy5.value.split(',')],
                }
        }
        return params_dict

    def _do_preprocessing(self):
        input_path = header_widget.input_path.value
        params_dict = self._get_preprocessing_params()
        save_dirs = create_dirs(params_dict, input_path)
        filter_list = params_dict['general']['chans_imaged']
        img_list = get_im_list(input_path)
        num_cores = footer_widget.num_cores.value
        preprocessing_worker = do_preprocessing(num_cores, input_path, filter_list, img_list, params_dict, save_dirs)
        preprocessing_worker.start()




