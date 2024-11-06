"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from napari.qt.threading import thread_worker
from tqdm import tqdm
from joblib import Parallel, delayed
from napari_dmc_brainmap.utils import get_animal_id, get_image_list, update_params_dict, clean_params_dict, load_params, \
    get_threshold_dropdown
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from superqt import QCollapsible
from magicgui import magicgui, widgets
from magicgui.widgets import FunctionGui
from napari_dmc_brainmap.preprocessing.preprocessing_tools import preprocess_images, create_dirs



def create_general_widget(widget_type: str, channels: list, downsampling_default: int = 3, contrast_limits=None) -> magicgui:
    """
    Create a generalized MagicGUI widget for image processing.

    Parameters:
    - widget_type (str): The type of widget being created (e.g., 'RGB', 'Single Channel', etc.)
    - channels (list): List of available channels to select.
    - downsampling_default (int): Default value for the downsampling factor.
    - contrast_limits (dict): Default contrast limit values for each channel.

    Returns:
    - FunctionGui: The created MagicGUI widget.
    """
    if widget_type != 'Binary':
        contrast_limits = contrast_limits or {
            'dapi': '50,2000',
            'green': '50,1000',
            'cy3': '50,2000',
            'n3': '50,2000',
            'cy5': '50,1000'
        }

        # Create the base widget
        container = widgets.Container(widgets=[
            widgets.CheckBox(value=False, label=f'Process {widget_type}', tooltip=f'Tick to process {widget_type} images'),
            widgets.Select(choices=['all'] + channels, value='all', label='Select channels', tooltip='Select channels to process'),
            widgets.SpinBox(value=downsampling_default, min=1, label='Downsampling Factor', tooltip='Enter scale factor for downsampling'),
            widgets.CheckBox(value=True, label=f'Adjust Contrast for {widget_type}', tooltip=f'Option to adjust contrast for {widget_type} images')
        ],
            labels=True
        )
        if widget_type == 'SHARPy':
            container.pop(-2)
        # Add contrast widgets for each channel
        for channel in channels:
            container.append(widgets.LineEdit(value=contrast_limits[channel], label=f'Set contrast limits for {channel}', tooltip=f'Enter contrast limits: min,max for {channel}'))
    else:
        contrast_limits = contrast_limits or {
            'dapi': '4000',
            'green': '1000',
            'cy3': '2000',
            'n3': '2000',
            'cy5': '2000'
        }

        # Create the base widget
        container = widgets.Container(widgets=[
            widgets.CheckBox(value=False, label=f'Process {widget_type}',
                             tooltip=f'Tick to process {widget_type} images'),
            widgets.Select(choices=['all'] + channels, value='all', label='Select channels',
                           tooltip='Select channels to process'),
            widgets.SpinBox(value=downsampling_default, min=1, label='Downsampling Factor',
                            tooltip='Enter scale factor for downsampling'),
            widgets.ComboBox(choices=get_threshold_dropdown(), label='Thresholding method',
                             tooltip='select a method to compute the threshold value (from:'
                                     ' https://scikit-image.org/docs/stable/api/skimage.filters.html#module-skimage.filters'),
            widgets.CheckBox(value=False, label=f'Manually set threshold for {widget_type}',
                             tooltip=f'Option to manually set threshold for {widget_type} images '
                                     f'(if not ticked, thresholding method will be used)')
        ],
            labels=True
        )
        # Add contrast widgets for each channel
        for channel in channels:
            container.append(
                widgets.LineEdit(value=contrast_limits[channel], label=f'Set threshold for {channel}',
                                 tooltip=f'Enter threshold for {channel}'))
    return container

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

@thread_worker
def do_preprocessing(num_cores, input_path, filter_list, img_list, preprocessing_params, resolution, save_dirs):

    # if num_cores > multiprocessing.cpu_count():
    #     print("maximum available cores = " + str(multiprocessing.cpu_count()))
    #     num_cores = multiprocessing.cpu_count()

    # if preprocessing_params['operations']['sharpy_track']:
    #     resolution_tuple = tuple(resolution)
    # else:
    #     resolution_tuple = False

    if "operations" in preprocessing_params.keys():
        if 'sharpy_track' in preprocessing_params['operations'].keys():
            resolution_tuple = tuple(resolution)
        else:
            resolution_tuple = False
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

        # Add generalized widgets for different operations
        self.rgb_widget = create_general_widget('RGB', ['dapi', 'green', 'cy3'])
        self.sharpy_widget = create_general_widget('SHARPy', ['dapi', 'green', 'n3', 'cy3', 'cy5'], contrast_limits={
            'dapi': '50,1000',
            'green': '50,300',
            'cy3': '50,2000',
            'n3': '50,500',
            'cy5': '50,500'
        })
        self.single_channel_widget = create_general_widget('Single Channel', ['dapi', 'green', 'n3', 'cy3', 'cy5'])
        self.stack_widget = create_general_widget('Stack', ['dapi', 'green', 'n3', 'cy3', 'cy5'])
        self.binary_widget = create_general_widget('Binary', ['dapi', 'green', 'n3', 'cy3', 'cy5'])

        # Preprocessing button
        self.btn = QPushButton("Do the preprocessing!")
        self.btn.clicked.connect(self._do_preprocessing)

        self.layout().addWidget(self.header.native)
        self._add_gui_section('Create RGB: expand for more', self.rgb_widget)
        self._add_gui_section('Create SHARPy-track images: expand for more', self.sharpy_widget)
        self._add_gui_section('Processed single channels: expand for more', self.single_channel_widget)
        self._add_gui_section('Create image stacks: expand for more', self.stack_widget)
        self._add_gui_section('Create binary images: expand for more', self.binary_widget)
        self.layout().addWidget(self.btn)


    def _add_gui_section(self, name, widget):
        collapsible = QCollapsible(name, self)
        collapsible.addWidget(widget.native)
        self.layout().addWidget(collapsible)


    # def _get_info(self, widget, rgb=False):
    #     if rgb:
    #         return {
    #             "channels": widget.channels.value,
    #             "downsampling": widget.ds_params.value,
    #             "contrast_adjustment": widget.contrast_bool.value,
    #             "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
    #             "green": [int(i) for i in widget.contrast_green.value.split(',')],
    #             "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')]
    #         }
    #     else:
    #         return {
    #             "channels": widget.channels.value,
    #             "downsampling": widget.ds_params.value,
    #             "contrast_adjustment": widget.contrast_bool.value,
    #             "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
    #             "green": [int(i) for i in widget.contrast_green.value.split(',')],
    #             "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
    #             "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
    #             "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
    #         }
    # def _get_widget_info(self, widget, item):
    #     if item == 'rgb':
    #         chan_list = ['dapi', 'green', 'cy3']
    #     else:
    #         chan_list = ['dapi', 'green', 'n3', 'cy3', 'cy5']
    #
    #     if 'all' in widget[1].value:
    #         imaged_chan_list = self.header.chans_imaged.value
    #     else:
    #         imaged_chan_list = widget[1].value
    #         # make sure selected channels for preprocessing where imaged
    #         imaged_chan_list = [i for i in imaged_chan_list if i in self.header.chans_imaged.value]
    #     if item != 'binary':
    #         if item == 'sharpy_track':
    #             return {
    #                 "channels": imaged_chan_list,
    #                 "contrast_adjustment": widget[2].value,
    #                 **{channel: [int(i) for i in widget[3 + idx].value.split(',')]
    #                    for idx, channel in enumerate(chan_list) if channel in imaged_chan_list}
    #             }
    #         else:
    #             return {
    #                 "channels": imaged_chan_list,
    #                 "downsampling": widget[2].value,
    #                 "contrast_adjustment": widget[3].value,
    #                 **{channel: [int(i) for i in widget[4 + idx].value.split(',')]
    #                    for idx, channel in enumerate(chan_list) if channel in imaged_chan_list}
    #             }
    #     else:
    #         if widget[4].value: # manual thresholds
    #             return {
    #                 "channels": imaged_chan_list,
    #                 "downsampling": widget[2].value,
    #                 "manual_threshold": widget[4].value,
    #                 **{channel: [int(i) for i in widget[4 + idx].value.split(',')]
    #                    for idx, channel in enumerate(chan_list) if channel in imaged_chan_list}
    #             }
    #         else:
    #             return {
    #                 "channels": imaged_chan_list,
    #                 "downsampling": widget[2].value,
    #                 "manual_threshold": widget[4].value,
    #                 "thresh_method": widget[3].value.value
    #             }
    def _get_widget_info(self, widget, item):
        chan_list = ['dapi', 'green', 'cy3'] if item == 'rgb' else ['dapi', 'green', 'n3', 'cy3', 'cy5']

        imaged_chan_list = (widget[1].value if 'all' not in widget[1].value
                            else self.header.chans_imaged.value)
        imaged_chan_list = [i for i in imaged_chan_list if i in self.header.chans_imaged.value]

        base_info = {"channels": imaged_chan_list, "downsampling": widget[2].value}

        if item == 'sharpy_track':
            base_info["contrast_adjustment"] = widget[2].value
        elif item != 'binary':
            base_info["contrast_adjustment"] = widget[3].value

        if item == 'binary':
            if widget[4].value:  # manual thresholds
                base_info.update({"manual_threshold": widget[4].value})
                base_info.update({channel: [int(i) for i in widget[4 + idx].value.split(',')] for idx, channel in
                                  enumerate(chan_list) if channel in imaged_chan_list})
            else:
                base_info.update({"manual_threshold": widget[4].value, "thresh_method": widget[3].value.value})
        else:
            base_info.update({
                channel: [int(i) for i in widget[(3 if item == 'sharpy_track' else 4) + idx].value.split(',')]
                for idx, channel in enumerate(chan_list) if channel in imaged_chan_list
            })

        return base_info

    def _get_preprocessing_params(self):
        op_widg_dict = {
            "rgb": self.rgb_widget,
            "sharpy_track": self.sharpy_widget,
            "single_channel": self.single_channel_widget,
            "stack": self.stack_widget,
            "binary": self.binary_widget
        }
        params_dict = {
            "general":
                {
                    "animal_id": get_animal_id(self.header.input_path.value),
                    "chans_imaged": self.header.chans_imaged.value
                },
        }
        for op, widget in op_widg_dict.items():
            if widget[0].value:
                params_dict["operations"] = {op: widget[0].value}
                params_dict[f"{op}_params"] = self._get_widget_info(widget, op)
        # "operations":
        #     {
        #
        #     },
        # "rgb_params": self._get_widget_info(self.rgb_widget, rgb=True),
        # "single_channel_params": self._get_widget_info(self.single_channel_widget),
        # "stack_params": self._get_widget_info(self.stack_widget),
        # "sharpy_track_params": self._get_widget_info(self.sharpy_widget),
        # "binary_params": self._get_widget_info(self.binary_widget, binary=True)


        return params_dict

    def _do_preprocessing(self):
        input_path = self.header.input_path.value
        # check if user provided a valid input_path
        if not input_path.is_dir() or str(input_path) == '.':  # todo check on other OS
            # raise IOError("Input path is not a valid directory \n"
            #               "Please make sure this exists: {}".format(input_path))
            print(f"Input path is not a valid directory \n Please make sure this exists: >> '{str(input_path)}' <<")
            return
        preprocessing_params = self._get_preprocessing_params()
        save_dirs = create_dirs(preprocessing_params, input_path)
        filter_list = preprocessing_params['general']['chans_imaged']
        img_list = get_image_list(input_path)
        num_cores = 1
        params_dict = load_params(input_path)
        resolution = params_dict['atlas_info']['resolution']

        preprocessing_worker = do_preprocessing(num_cores, input_path, filter_list, img_list, preprocessing_params,
                                                resolution, save_dirs)
        preprocessing_worker.start()





