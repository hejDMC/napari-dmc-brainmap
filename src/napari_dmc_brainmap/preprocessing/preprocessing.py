"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from napari.qt.threading import thread_worker
import json
from tqdm import tqdm
from joblib import Parallel, delayed
from napari_dmc_brainmap.utils import get_animal_id, get_im_list, update_params_dict, clean_params_dict
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QFileDialog, QLineEdit
from superqt import QCollapsible
from magicgui import magicgui
from napari_dmc_brainmap.preprocessing.preprocessing_tools import preprocess_images, create_dirs

@magicgui(
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                      tooltip='directory of folder containing subfolders with stitched images, '
                              'NOT folder containing stitched images itself'),
    chans_imaged=dict(widget_type='Select', label='imaged channels', choices=['dapi', 'green', 'cy3', 'cy5'],
                      value=['green', 'cy3'],
                      tooltip='select all channels imaged, to select multiple hold ctrl/shift'),
    call_button=False
)
def header_widget(
        self,
        input_path,
        chans_imaged
):
    return header_widget

# todo: all these function as class
@magicgui(
    button=dict(widget_type='CheckBox', text='create RGB images', value=False,
                    tooltip='option to create RGB images, tick to create RGB images'),
    channels=dict(widget_type='Select', label='selected channels', value='all',
                      choices=['all', 'dapi', 'green', 'cy3'],
                      tooltip='select channels to create RGB image, to select multiple hold ctrl/shift'),
    ds_params=dict(widget_type='SpinBox', label='enter downsampling factor', value=3,
                   tooltip='enter scale factor for downsampling'),
    contrast_bool=dict(widget_type='CheckBox', text='perform contrast adjustment on RGB images', value=True,
                      tooltip='option to adjust contrast on images, see option details below'),
    contrast_dapi=dict(widget_type='LineEdit', label='set contrast limits for the dapi channel',
                       value='50,2000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_green=dict(widget_type='LineEdit', label='set contrast limits for the green channel',
                        value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy3=dict(widget_type='LineEdit', label='set contrast limits for the cy3 channel',
                      value='50,2000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    call_button=False
)
def do_rgb(
        self,
        button,
        channels,
        ds_params,
        contrast_bool,
        contrast_dapi,
        contrast_green,
        contrast_cy3
):
    return do_rgb


@magicgui(
    button=dict(widget_type='CheckBox', text='process single channels', value=False,
                    tooltip='option to process single channels individually'),
    channels=dict(widget_type='Select', label='selected channels', value='all',
                  choices=['all', 'dapi', 'green', 'cy3', 'cy5'],
                  tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
    ds_params=dict(widget_type='SpinBox', label='enter downsampling factor', value=3,
                   tooltip='enter scale factor for downsampling'),
    contrast_bool=dict(widget_type='CheckBox', text='perform contrast adjustment on single images', value=True,
                       tooltip='option to adjust contrast on images, see option details below'),
    contrast_dapi=dict(widget_type='LineEdit', label='set contrast limits for the dapi channel',
                       value='50,2000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_green=dict(widget_type='LineEdit', label='set contrast limits for the green channel',
                        value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy3=dict(widget_type='LineEdit', label='set contrast limits for the cy3 channel',
                      value='50,2000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy5=dict(widget_type='LineEdit', label='set contrast limits for the cy5 channel',
                      value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    call_button=False
)
def do_single_channel(
        self,
        button,
        channels,
        ds_params,
        contrast_bool,
        contrast_dapi,
        contrast_green,
        contrast_cy3,
        contrast_cy5
):
    return do_single_channel

@magicgui(
    button=dict(widget_type='CheckBox', text='create image stack', value=False,
                tooltip='option to create image stacks'),
    channels=dict(widget_type='Select', label='selected channels', value='all',
                  choices=['all', 'dapi', 'green', 'cy3', 'cy5'],
                  tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
    ds_params=dict(widget_type='SpinBox', label='enter downsampling factor', value=3,
                   tooltip='enter scale factor for downsampling'),
    contrast_bool=dict(widget_type='CheckBox', text='perform contrast adjustment on image stacks', value=True,
                       tooltip='option to adjust contrast on images, see option details below'),
    contrast_dapi=dict(widget_type='LineEdit', label='set contrast limits for the dapi channel',
                       value='50,2000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_green=dict(widget_type='LineEdit', label='set contrast limits for the green channel',
                        value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy3=dict(widget_type='LineEdit', label='set contrast limits for the cy3 channel',
                      value='50,2000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy5=dict(widget_type='LineEdit', label='set contrast limits for the cy5 channel',
                      value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    call_button=False
)
def do_stack(
        self,
        button,
        channels,
        ds_params,
        contrast_bool,
        contrast_dapi,
        contrast_green,
        contrast_cy3,
        contrast_cy5
):
    return do_stack

@magicgui(
    button=dict(widget_type='CheckBox', text='create downsampled images for sharpy-track', value=False,
                tooltip='option to create downsampled images [1140x800 px] for brain registration using sharpy-track'),
    channels=dict(widget_type='Select', label='selected channels', value='all',
                  choices=['all', 'dapi', 'green', 'cy3', 'cy5'],
                  tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
    ds_params=dict(widget_type='CheckBox', text='perform downsampling on images', value=True,
                 tooltip='downsample image to resolution in sharpy track'),
    contrast_bool=dict(widget_type='CheckBox', text='perform contrast adjustment on images for registration', value=True,
                       tooltip='option to adjust contrast on images, see option details below'),
    contrast_dapi=dict(widget_type='LineEdit', label='set contrast limits for the dapi channel',
                       value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_green=dict(widget_type='LineEdit', label='set contrast limits for the green channel',
                        value='50,300', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy3=dict(widget_type='LineEdit', label='set contrast limits for the cy3 channel',
                      value='50,500', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy5=dict(widget_type='LineEdit', label='set contrast limits for the cy5 channel',
                      value='50,500', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    call_button=False
)
def do_sharpy(
        self,
        button,
        channels,
        ds_params,
        contrast_bool,
        contrast_dapi,
        contrast_green,
        contrast_cy3,
        contrast_cy5

):
    return do_sharpy

@magicgui(
    button=dict(widget_type='CheckBox', text='create binary images', value=False,
                tooltip='option to create binary images (yen-thresholding)'),  # todo add option for other methods and entering of threshold
    channels=dict(widget_type='Select', label='selected channels', value='all',
                  choices=['all', 'dapi', 'green', 'cy3', 'cy5'],
                  tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
    ds_params=dict(widget_type='SpinBox', label='enter downsampling factor', value=3,
                   tooltip='enter scale factor for downsampling'),
    thresh_bool=dict(widget_type='CheckBox', text='manually set threshold', value=False,
                       tooltip='option to use manually set thresholds, otherwise threshold will be determined automatically'),
    thresh_dapi=dict(widget_type='LineEdit', label='set threshold for the dapi channel',
                       value='4000', tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
    thresh_green=dict(widget_type='LineEdit', label='set threshold for the green channel',
                        value='1000', tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
    thresh_cy3=dict(widget_type='LineEdit', label='set threshold for the cy3 channel',
                      value='2000', tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
    thresh_cy5=dict(widget_type='LineEdit', label='set threshold for the cy5 channel',
                      value='2000', tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
    call_button=False
)
def do_binary(
        self,
        button,
        channels,
        ds_params,
        thresh_bool,
        thresh_dapi,
        thresh_green,
        thresh_cy3,
        thresh_cy5
):
    return do_binary

@magicgui(
    num_cores=dict(widget_type='SpinBox', label='enter the number of parallel processes', value=1, min=1,
                   tooltip='select number of parallel processes for image processing'),
    call_button=False
)
def footer_widget(
        self,
        num_cores
):
    return footer_widget

@thread_worker
def do_preprocessing(num_cores, input_path, filter_list, img_list, params_dict, save_dirs):

    # if num_cores > multiprocessing.cpu_count():
    #     print("maximum available cores = " + str(multiprocessing.cpu_count()))
    #     num_cores = multiprocessing.cpu_count()
    if any(params_dict['operations'].values()):
        if num_cores > 1:
            print("parallel processing not implemented yet")
        Parallel(n_jobs=num_cores)(delayed(preprocess_images)
                                   (im, filter_list, input_path, params_dict, save_dirs) for im in tqdm(img_list))
        params_dict = clean_params_dict(params_dict, "operations")
        params_fn = input_path.joinpath('params.json')
        params_dict = update_params_dict(input_path, params_dict)
        with open(params_fn, 'w') as fn:
            json.dump(params_dict, fn, indent=4)
        print("DONE!")
    else:
        print("No preprocessing operations selected, expand the respective windows and tick check box")


class PreprocessingWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        header = header_widget
        self._collapse_rgb = QCollapsible('Create RGB: expand for more', self)
        rgb_widget = do_rgb
        self._collapse_rgb.addWidget(rgb_widget.native)

        self._collapse_single = QCollapsible('Processed single channels: expand for more', self)
        single_channel_widget = do_single_channel
        self._collapse_single.addWidget(single_channel_widget.native)

        self._collapse_stack = QCollapsible('Create image stacks: expand for more', self)
        stack_widget = do_stack
        self._collapse_stack.addWidget(stack_widget.native)

        self._collapse_sharpy = QCollapsible('Create sharpy_track images: expand for more', self)
        sharpy_widget = do_sharpy
        self._collapse_sharpy.addWidget(sharpy_widget.native)

        self._collapse_binary = QCollapsible('Create binary images: expand for more', self)
        binary_widget = do_binary
        self._collapse_binary.addWidget(binary_widget.native)

        footer = footer_widget

        btn = QPushButton("Do the preprocessing!")
        btn.clicked.connect(self._do_preprocessing)
        self.layout().addWidget(header.native)
        self.layout().addWidget(self._collapse_rgb)
        self.layout().addWidget(self._collapse_sharpy)
        self.layout().addWidget(self._collapse_single)
        self.layout().addWidget(self._collapse_stack)
        self.layout().addWidget(self._collapse_binary)
        self.layout().addWidget(footer.native)
        self.layout().addWidget(btn)

    def _get_info(self, widget, rgb=False):
        if rgb:
            return {
                "channels": widget.channels.value,
                "downsampling": widget.ds_params.value,
                "contrast_adjustment": widget.contrast_bool.value,
                "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
                "green": [int(i) for i in widget.contrast_green.value.split(',')],
                "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')]
            }
        else:
            return {
                "channels": widget.channels.value,
                "downsampling": widget.ds_params.value,
                "contrast_adjustment": widget.contrast_bool.value,
                "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
                "green": [int(i) for i in widget.contrast_green.value.split(',')],
                "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
                "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
            }

    def _get_preprocessing_params(self):
        params_dict = {
            "general":
                {
                    "animal_id": get_animal_id(header_widget.input_path.value),
                    "chans_imaged": header_widget.chans_imaged.value
                },
            "operations":
                {
                    "rgb": do_rgb.button.value,
                    "single_channel": do_single_channel.button.value,
                    "stack": do_stack.button.value,
                    "sharpy_track": do_sharpy.button.value,
                    "binary": do_binary.button.value
                },
            "rgb_params": self._get_info(do_rgb, rgb=True),
            "single_channel_params": self._get_info(do_single_channel),
            "stack_params": self._get_info(do_stack),
            "sharpy_track_params": self._get_info(do_sharpy),
            "binary_params":
                {
                    "channels": do_binary.channels.value,
                    "downsampling": do_binary.ds_params.value,
                    "thresh": int(do_binary.thresh_bool.value),
                    "dapi": int(do_binary.thresh_dapi.value),
                    "green": int(do_binary.thresh_green.value),
                    "cy3": int(do_binary.thresh_cy3.value),
                    "cy5": int(do_binary.thresh_cy5.value)
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





