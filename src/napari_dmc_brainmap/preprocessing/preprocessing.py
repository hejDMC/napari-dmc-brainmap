"""
DMC-BrainMap widget for preprocessing of .tif files.

2024 - FJ
"""

import os
import platform
from pathlib import Path
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox, QProgressBar
from superqt import QCollapsible
from joblib import Parallel, delayed
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.path_utils import get_image_list, get_info
from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.params_utils import load_params, clean_params_dict, update_params_dict
from napari_dmc_brainmap.utils.dropdown_utils import get_threshold_dropdown
from magicgui import magicgui, widgets
from magicgui.widgets import FunctionGui
from napari_dmc_brainmap.preprocessing.preprocessing_tools import preprocess_images, create_dirs, get_channels, \
    chunk_list
from napari_dmc_brainmap.utils.gui_utils import check_input_path
from typing import List, Dict, Tuple, Union, Optional


@thread_worker(progress={"total": 100})
def do_preprocessing(
    input_path: Path,
    channels: List[str],
    img_list: List[str],
    preprocessing_params: Dict[str, Union[str, dict]],
    resolution: Tuple[int, int],
    save_dirs: Dict[str, str]
) -> str:
    """
    Perform preprocessing on a list of images in a multithreaded manner.

    Parameters:
        input_path (Path): Path to the input directory containing images.
        channels (List[str]): List of channels to process.
        img_list (List[str]): List of image file names to process.
        preprocessing_params (Dict[str, Union[str, dict]]): Parameters for preprocessing operations.
        resolution (Tuple[int, int]): Tuple containing resolution information for preprocessing.
        save_dirs (Dict[str, str]): Dictionary containing paths to save preprocessed images.

    Yields:
        int: Progress of the preprocessing operation in percentage.

    Returns:
        str: Animal ID for which preprocessing was performed.
    """
    if "operations" in preprocessing_params.keys():
        resolution_tuple = tuple(resolution) if 'sharpy_track' in preprocessing_params['operations'] else False
        num_cores = os.cpu_count()
        # overwrite parallelization to 1 if detects Darwin OS
        if platform.system() == 'Darwin':
            num_cores = 1
        chunk_img_list = chunk_list(img_list, chunk_size=num_cores)
        progress_value = 0
        progress_step = 100 / len(chunk_img_list)

        for chunk in chunk_img_list:
            Parallel(n_jobs=num_cores)(
                delayed(preprocess_images)(
                    im, channels, input_path, preprocessing_params, save_dirs, resolution_tuple
                ) for im in chunk
            )
            progress_value += progress_step
            yield int(progress_value)

        preprocessing_params = clean_params_dict(preprocessing_params, "operations")
        update_params_dict(input_path, preprocessing_params)
    else:
        show_info("No preprocessing operations selected. Expand the respective windows and tick the checkbox.")

    yield 100
    return get_animal_id(input_path)


def create_general_widget(
    widget_type: str,
    channels: List[str],
    downsampling_default: int = 3,
    contrast_limits: Optional[Dict[str, str]] = None
) -> widgets.Container:
    """
    Create a generalized MagicGUI widget for image processing.

    Parameters:
        widget_type (str): The type of widget being created (e.g., 'RGB', 'Single Channel').
        channels (List[str]): List of available channels to select.
        downsampling_default (int): Default value for the downsampling factor.
        contrast_limits (Optional[Dict[str, str]]): Default contrast limit values for each channel.

    Returns:
        widgets.Container: The created MagicGUI widget container.
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

    # Modify for SHARPy or Binary widget
    # if widget_type == 'SHARPy':
    #     container.pop(-2)
    if widget_type == 'Binary':
        container.append(
            widgets.ComboBox(choices=get_threshold_dropdown(), label='Thresholding Method',
                             tooltip='Select a thresholding method (see skimage.filters).')
        )
        # Add contrast widgets for each channel
        for channel in channels:
            container.append(
                widgets.LineEdit(value=contrast_limits[channel], label=f'Set threshold for {channel}',
                                 tooltip=f'Enter threshold for {channel}'))
    return container


def initialize_header_widget() -> FunctionGui:
    """
    Initialize a header widget for selecting the input path and imaged channels.

    Returns:
        FunctionGui: The initialized header widget.
    """
    @magicgui(
        input_path=dict(widget_type='FileEdit',
                        label='Input Path (Animal ID):',
                        mode='d',
                        tooltip='Directory containing subfolders with stitched images.'),
        chans_imaged=dict(widget_type='Select',
                          label='Imaged Channels',
                          choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                          value=['green', 'cy3'],
                          tooltip='Select all imaged channels. Hold Ctrl/Shift for multiple selections.'),
        call_button=False
    )
    def header_widget(input_path: Path, chans_imaged: List[str]) -> None:
        """
        Header widget for selecting input path and imaged channels.

        Parameters:
            input_path (Path): Path to the input directory.
            chans_imaged (List[str]): List of imaged channels.
        """
        pass

    return header_widget


class PreprocessingWidget(QWidget):
    """
    QWidget for configuring and performing preprocessing operations.
    """
    progress_signal = Signal(int)
    """Signal emitted to update the progress bar with an integer value."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the PreprocessingWidget.

        Parameters:
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.setLayout(QVBoxLayout())

        # Header widget
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

        # Add preprocessing button
        self.btn = QPushButton("Do the Preprocessing!")
        self.btn.clicked.connect(self._do_preprocessing)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        # Add widgets to layout
        self.layout().addWidget(self.header.native)
        self._add_gui_section('Create RGB: expand for more', self.rgb_widget)
        self._add_gui_section('Create SHARPy-track images: expand for more', self.sharpy_widget)
        self._add_gui_section('Process Single Channels: expand for more', self.single_channel_widget)
        self._add_gui_section('Create Image Stacks: expand for more', self.stack_widget)
        self._add_gui_section('Create Binary Images: expand for more', self.binary_widget)
        self.layout().addWidget(self.btn)
        self.layout().addWidget(self.progress_bar)
        self.progress_signal.connect(self.progress_bar.setValue)

    def _add_gui_section(self, name: str, widget: FunctionGui) -> None:
        """
        Add a collapsible GUI section to the layout.

        Parameters:
            name (str): The name of the collapsible section.
            widget (FunctionGui): The widget to add within the collapsible section.
        """
        collapsible = QCollapsible(name, self)
        collapsible.addWidget(widget.native)
        self.layout().addWidget(collapsible)

    def _get_widget_info(self, widget: FunctionGui, item: str) -> Dict[str, Union[List[int], int, str]]:
        """
        Retrieve information from a given widget based on the type of item.

        Parameters:
            widget (FunctionGui): The widget to retrieve information from.
            item (str): Type of operation (e.g., 'rgb', 'sharpy_track').

        Returns:
            Dict[str, Union[List[int], int, str]]: Information extracted from the widget.
        """
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

    def _get_preprocessing_params(self) -> Dict[str, Union[str, Dict[str, Union[str, List[int], int]]]]:
        """
        Retrieve preprocessing parameters based on user selections.

        Returns:
        - Dict[str, Union[str, Dict[str, Union[str, List[int], int]]]]: Dictionary of preprocessing parameters.
        """
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
        k = 0
        for op, widget in op_widg_dict.items():
            if widget[0].value:
                if k < 1:
                    params_dict["operations"] = {}
                    k += 1
                params_dict["operations"][op] = widget[0].value
                params_dict[f"{op}_params"] = self._get_widget_info(widget, op)
        return params_dict

    def _check_preprocessing_success(self) -> List[str]:
        """
        Check if preprocessing was successful for the given animal ID.

        Returns:
            List[str]: Return list of directories containing missing files.
        """
        input_path = self.header.input_path.value
        params_dict = load_params(input_path)
        missing_files = []
        for op, op_bool in params_dict["operations"].items():
            if op_bool:
                if op == "rgb":
                    _, op_data_list, _ = get_info(input_path, op)
                    if not op_data_list:
                        missing_files.append(f"{op}")

                else:
                    for chan in params_dict[f"{op}_params"]["channels"]:
                        _, op_data_list, _ = get_info(input_path, op, chan)
                        if not op_data_list:
                            missing_files.append(f"{op}_{chan}")

        return missing_files


    def _show_success_message(self, animal_id: str) -> None:
        """
        Display a success message after preprocessing is complete.

        Parameters:
            animal_id (str): The Animal ID for which preprocessing was performed.
        """
        missing_files = self._check_preprocessing_success()
        if len(missing_files) == 0:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(f"Preprocessing finished successfully for {animal_id}!")
            msg_box.setWindowTitle("Preprocessing Complete")
            msg_box.exec_()
        else:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText(f"Preprocessing finished, but the following files are missing: {', '.join(missing_files)}")
            # msg_box.setText(f"Preprocessing failed for {animal_id}:\n".join(missing_files))
            msg_box.setWindowTitle("Preprocessing Error")
            msg_box.exec_()

        self.btn.setText("Do the Preprocessing!")  # Reset button text
        self.progress_signal.emit(0)

    def _update_progress(self, value: int) -> None:
        """
        Update the progress bar with the current progress value.

        Parameters:
            value (int): Progress value to set.
        """
        self.progress_signal.emit(value)

    def _do_preprocessing(self) -> None:
        """
        Execute the preprocessing of images based on user input.
        """
        input_path = self.header.input_path.value

        # Validate input path
        if not check_input_path(input_path):
            return

        # Retrieve preprocessing parameters
        preprocessing_params = self._get_preprocessing_params()
        save_dirs = create_dirs(preprocessing_params, input_path)
        channels = get_channels(preprocessing_params)
        for chan in channels:
            img_list = get_image_list(input_path, chan)
        params_dict = load_params(input_path)
        resolution = params_dict['atlas_info']['resolution']

        # Start the preprocessing worker
        preprocessing_worker = do_preprocessing(input_path, channels, img_list, preprocessing_params, resolution, save_dirs)
        preprocessing_worker.yielded.connect(self._update_progress)
        preprocessing_worker.started.connect(lambda: self.btn.setText("Preprocessing Images..."))
        preprocessing_worker.returned.connect(self._show_success_message)
        preprocessing_worker.start()
