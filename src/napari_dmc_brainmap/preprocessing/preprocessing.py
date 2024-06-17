"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from napari.qt.threading import thread_worker
from tqdm import tqdm
from joblib import Parallel, delayed
from napari_dmc_brainmap.utils import get_animal_id, get_im_list, update_params_dict, clean_params_dict, load_params, \
    get_threshold_dropdown
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from superqt import QCollapsible
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from napari_dmc_brainmap.preprocessing.preprocessing_tools import preprocess_images, create_dirs



def initialize_header_widget() -> FunctionGui:
    @magicgui(input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with stitched images, '
                                'NOT folder containing stitched images'),
              chans_imaged=dict(widget_type='Select', 
                                label='imaged channels', 
                                choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                                value=['green', 'cy3'],
                                tooltip='select all channels imaged, to select multiple hold ctrl/shift'),
              call_button=False)

    def header_widget(
            self,
            input_path,
            chans_imaged):
        pass
    return header_widget


def initialize_dorgb_widget() -> FunctionGui:
    # todo: all these function as class
    @magicgui(button=dict(widget_type='CheckBox', 
                          text='create RGB images', 
                          value=False,
                          tooltip='tick to create RGB images'),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value='all',
                            choices=['all', 'dapi', 'green', 'cy3'],
                            tooltip='select channels to create RGB image, to select multiple hold ctrl/shift'),
              ds_params=dict(widget_type='SpinBox', 
                             label='enter downsampling factor', 
                             value=3,
                             tooltip='enter scale factor for downsampling'),
              contrast_bool=dict(widget_type='CheckBox', 
                                 text='perform contrast adjustment on RGB images', 
                                 value=True,
                                 tooltip='option to adjust contrast on images, see option details below'),
              contrast_dapi=dict(widget_type='LineEdit', 
                                 label='set contrast limits for the dapi channel',
                                 value='50,2000', 
                                 tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_green=dict(widget_type='LineEdit', 
                                  label='set contrast limits for the green channel',
                                  value='50,1000', 
                                  tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy3=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy3 channel',
                                value='50,2000', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              call_button=False,
              scrollable=True)
    
    def do_rgb(
            self,
            button,
            channels,
            ds_params,
            contrast_bool,
            contrast_dapi,
            contrast_green,
            contrast_cy3):
        pass
    return do_rgb


def initialize_dosinglechannel_widget() -> FunctionGui:
    @magicgui(button=dict(widget_type='CheckBox', 
                          text='process single channels', 
                          value=False,
                          tooltip='tick to process single channels individually'),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value='all',
                            choices=['all', 'dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
              ds_params=dict(widget_type='SpinBox', 
                             label='enter downsampling factor', 
                             value=3,
                             tooltip='enter scale factor for downsampling'),
              contrast_bool=dict(widget_type='CheckBox', 
                                 text='perform contrast adjustment on single images', 
                                 value=True,
                                 tooltip='option to adjust contrast on images, see option details below'),
              contrast_dapi=dict(widget_type='LineEdit', 
                                 label='set contrast limits for the dapi channel',
                                 value='50,2000',
                                 tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_green=dict(widget_type='LineEdit', 
                                  label='set contrast limits for the green channel',
                                  value='50,1000', 
                                  tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_n3=dict(widget_type='LineEdit', 
                               label='set contrast limits for the n3 channel',
                               value='50,2000', 
                               tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy3=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy3 channel',
                                value='50,2000', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy5=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy5 channel',
                                value='50,1000', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              call_button=False,
              scrollable=True)
    
    def do_single_channel(
            self,
            button,
            channels,
            ds_params,
            contrast_bool,
            contrast_dapi,
            contrast_green,
            contrast_n3,
            contrast_cy3,
            contrast_cy5):
        pass
    return do_single_channel


def initialize_dostack_widget() -> FunctionGui:
    @magicgui(button=dict(widget_type='CheckBox', 
                          text='create image stack', 
                          value=False,
                          tooltip='tick to create image stacks'),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value='all',
                            choices=['all', 'dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
              ds_params=dict(widget_type='SpinBox', 
                             label='enter downsampling factor', 
                             value=3,
                             tooltip='enter scale factor for downsampling'),
              contrast_bool=dict(widget_type='CheckBox', 
                                 text='perform contrast adjustment on image stacks', 
                                 value=True,
                                 tooltip='option to adjust contrast on images, see option details below'),
              contrast_dapi=dict(widget_type='LineEdit', 
                                 label='set contrast limits for the dapi channel',
                                 value='50,2000', 
                                 tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_green=dict(widget_type='LineEdit', 
                                  label='set contrast limits for the green channel',
                                  value='50,1000', 
                                  tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_n3=dict(widget_type='LineEdit', 
                               label='set contrast limits for the n3 channel',
                               value='50,2000', 
                               tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy3=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy3 channel',
                                value='50,2000', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy5=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy5 channel',
                                value='50,1000', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              call_button=False,
              scrollable=True)
    
    def do_stack(
            self,
            button,
            channels,
            ds_params,
            contrast_bool,
            contrast_dapi,
            contrast_green,
            contrast_n3,
            contrast_cy3,
            contrast_cy5):
        pass
    return do_stack



def initialize_dosharpy_widget() -> FunctionGui:
    @magicgui(button=dict(widget_type='CheckBox', 
                          text='create downsampled images for SHARPy-track',
                          value=False,
                          tooltip='tick to create downsampled images [1140x800 px] for brain registration using SHARPy-track'),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value='all',
                            choices=['all', 'dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels for SHARPy-track, to select multiple hold ctrl/shift'),
              ds_params=dict(widget_type='CheckBox', 
                             text='perform downsampling on images', 
                             value=True,
                             tooltip='downsample image to resolution in SHARPy track'),
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
                               value='50,2000', 
                               tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy3=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy3 channel',
                                value='50,500', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              contrast_cy5=dict(widget_type='LineEdit', 
                                label='set contrast limits for the cy5 channel',
                                value='50,500', 
                                tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
              call_button=False,
              scrollable=True)
    
    def do_sharpy(
            self,
            button,
            channels,
            ds_params,
            contrast_bool,
            contrast_dapi,
            contrast_green,
            contrast_n3,
            contrast_cy3,
            contrast_cy5):
        pass
    return do_sharpy


def initialize_dobinary_widget() -> FunctionGui:
    @magicgui(button=dict(widget_type='CheckBox', 
                          text='create binary images', 
                          value=False,
                          tooltip='tick to create binary images'),
              thresh_func=dict(label='thresholding method',
                         tooltip='select a method to compute the threshold value'
                                 '(from https://scikit-image.org/docs/stable/api/skimage.filters.html#module-skimage.filters'),
              thresh_bool=dict(widget_type='CheckBox',
                               text='manually set threshold',
                               value=False,
                               tooltip='option to use manually set thresholds, otherwise threshold will be determined automatically'),
              channels=dict(widget_type='Select',
                            label='selected channels', 
                            value='all',
                            choices=['all', 'dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
              ds_params=dict(widget_type='SpinBox', 
                             label='enter downsampling factor', 
                             value=3,
                             tooltip='enter scale factor for downsampling'),
              thresh_dapi=dict(widget_type='LineEdit', 
                               label='set threshold for the dapi channel',
                               value='4000', 
                               tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
              thresh_green=dict(widget_type='LineEdit', 
                                label='set threshold for the green channel',
                                value='1000', 
                                tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
              thresh_n3=dict(widget_type='LineEdit', 
                             label='set threshold for the n3 channel',
                             value='2000', 
                             tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
              thresh_cy3=dict(widget_type='LineEdit', 
                              label='set threshold for the cy3 channel',
                              value='2000', 
                              tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
              thresh_cy5=dict(widget_type='LineEdit', 
                              label='set threshold for the cy5 channel',
                              value='2000', 
                              tooltip='enter threshold for creating binary image (default values for 16-bit image)'),
              call_button=False,
              scrollable=True)
    
    def do_binary(
            self,
            button,
            thresh_func: get_threshold_dropdown(),
            thresh_bool,
            channels,
            ds_params,
            thresh_dapi,
            thresh_green,
            thresh_n3,
            thresh_cy3,
            thresh_cy5):
        pass
    return do_binary


def initialize_footer_widget() -> FunctionGui:
    @magicgui(num_cores=dict(widget_type='SpinBox', 
                             label='enter the number of parallel processes', 
                             value=1, 
                             min=1,
                             tooltip='select number of parallel processes for image processing - NOT IMPLEMENTED'),
              call_button=False)
    
    def footer_widget(
            self,
            num_cores):
        pass
    return footer_widget


@thread_worker
def do_preprocessing(num_cores, input_path, filter_list, img_list, preprocessing_params, resolution, save_dirs):

    # if num_cores > multiprocessing.cpu_count():
    #     print("maximum available cores = " + str(multiprocessing.cpu_count()))
    #     num_cores = multiprocessing.cpu_count()

    if preprocessing_params['operations']['sharpy_track']:
        resolution_tuple = tuple(resolution)
    else:
        resolution_tuple = False

    if any(preprocessing_params['operations'].values()):
        if num_cores > 1:
            print("parallel processing not implemented yet")
        Parallel(n_jobs=num_cores)(delayed(preprocess_images)
                                   (im, filter_list, input_path, preprocessing_params, save_dirs, resolution_tuple) for im in tqdm(img_list))
        preprocessing_params = clean_params_dict(preprocessing_params, "operations")
        update_params_dict(input_path, preprocessing_params)
        print("DONE!")
    else:
        print("No preprocessing operations selected, expand the respective windows and tick check box")


class PreprocessingWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.header = initialize_header_widget()
        self.header.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_rgb = QCollapsible('Create RGB: expand for more', self)
        self.rgb_widget = initialize_dorgb_widget()
        self.rgb_widget.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_rgb.addWidget(self.rgb_widget.root_native_widget)

        self._collapse_single = QCollapsible('Processed single channels: expand for more', self)
        self.single_channel_widget = initialize_dosinglechannel_widget()
        self.single_channel_widget.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_single.addWidget(self.single_channel_widget.root_native_widget)

        self._collapse_stack = QCollapsible('Create image stacks: expand for more', self)
        self.stack_widget = initialize_dostack_widget()
        self.stack_widget.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_stack.addWidget(self.stack_widget.root_native_widget)

        self._collapse_sharpy = QCollapsible('Create SHARPy-track images: expand for more', self)
        self.sharpy_widget = initialize_dosharpy_widget()
        self.sharpy_widget.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_sharpy.addWidget(self.sharpy_widget.root_native_widget)

        self._collapse_binary = QCollapsible('Create binary images: expand for more', self)
        self.binary_widget = initialize_dobinary_widget()
        self.binary_widget.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_binary.addWidget(self.binary_widget.root_native_widget)

        self.footer = initialize_footer_widget()
        self.footer.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        btn = QPushButton("Do the preprocessing!")
        btn.clicked.connect(self._do_preprocessing)
        self.layout().addWidget(self.header.native)
        self.layout().addWidget(self._collapse_rgb)
        self.layout().addWidget(self._collapse_sharpy)
        self.layout().addWidget(self._collapse_single)
        self.layout().addWidget(self._collapse_stack)
        self.layout().addWidget(self._collapse_binary)
        self.layout().addWidget(self.footer.native)
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
                "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
                "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
                "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
            }

    def _get_preprocessing_params(self):
        params_dict = {
            "general":
                {
                    "animal_id": get_animal_id(self.header.input_path.value),
                    "chans_imaged": self.header.chans_imaged.value
                },
            "operations":
                {
                    "rgb": self.rgb_widget.button.value,
                    "single_channel": self.single_channel_widget.button.value,
                    "stack": self.stack_widget.button.value,
                    "sharpy_track": self.sharpy_widget.button.value,
                    "binary": self.binary_widget.button.value
                },
            "rgb_params": self._get_info(self.rgb_widget, rgb=True),
            "single_channel_params": self._get_info(self.single_channel_widget),
            "stack_params": self._get_info(self.stack_widget),
            "sharpy_track_params": self._get_info(self.sharpy_widget),
            "binary_params":
                {
                    "channels": self.binary_widget.channels.value,
                    "downsampling": self.binary_widget.ds_params.value,
                    "thresh_func": self.binary_widget.thresh_func.value.value,
                    "thresh_bool": int(self.binary_widget.thresh_bool.value),
                    "dapi": int(self.binary_widget.thresh_dapi.value),
                    "green": int(self.binary_widget.thresh_green.value),
                    "n3": int(self.binary_widget.thresh_n3.value),
                    "cy3": int(self.binary_widget.thresh_cy3.value),
                    "cy5": int(self.binary_widget.thresh_cy5.value)
                }
        }
        return params_dict

    def _do_preprocessing(self):
        input_path = self.header.input_path.value
        # check if user provided a valid input_path
        if not input_path.is_dir():
            raise IOError("Input path is not a valid directory \n"
                          "Please make sure this exists: {}".format(input_path))
        preprocessing_params = self._get_preprocessing_params()
        save_dirs = create_dirs(preprocessing_params, input_path)
        filter_list = preprocessing_params['general']['chans_imaged']
        img_list = get_im_list(input_path)
        num_cores = self.footer.num_cores.value
        params_dict = load_params(input_path)
        resolution = params_dict['atlas_info']['resolution']
        preprocessing_worker = do_preprocessing(num_cores, input_path, filter_list, img_list, preprocessing_params,
                                                resolution, save_dirs)
        preprocessing_worker.start()





