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
from napari_dmc_brainmap.preprocessing.preprocessing_tools import (
    AUTOMATIC_RGB_PROFILES,
    chunk_list,
    create_dirs,
    estimate_automatic_rgb_ranges,
    get_channels,
    preprocess_images,
    resolve_automatic_rgb_params,
)
from napari_dmc_brainmap.utils.gui_utils import check_input_path
from typing import Callable, Dict, List, Optional, Tuple, Union


RGB_GUI_CHANNELS = ('cy3', 'green', 'dapi')
RGB_PROFILE_CHOICES = [
    (config['label'], profile)
    for profile, config in AUTOMATIC_RGB_PROFILES.items()
]
RGB_DEFAULT_PROFILES = {
    'cy3': 'preserve_strong_signal_probe',
    'green': 'balanced_cells',
    'dapi': 'balanced_cells',
}


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
        if 'rgb' in preprocessing_params['operations']:
            resolve_automatic_rgb_params(
                input_path, img_list, preprocessing_params
            )
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


@thread_worker
def estimate_rgb_ranges_worker(
    input_path: Path,
    image_names: List[str],
    source_channels: List[str],
    target_profiles: Dict[str, str],
    calibration_key: Tuple,
    progress_callback: Callable[[object], None],
) -> Tuple[Tuple, Dict[str, object]]:
    """Estimate RGB ranges without blocking the napari user interface."""
    def report_progress(report: Dict[str, object]) -> None:
        progress_callback((calibration_key, report))

    calibration = estimate_automatic_rgb_ranges(
        input_path,
        image_names,
        source_channels,
        target_profiles,
        progress_callback=report_progress,
    )
    return calibration_key, calibration


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


def create_rgb_widget() -> widgets.Container:
    """Create RGB-specific controls with manual and automatic contrast modes."""
    contrast_limits = {
        'dapi': '50,2000',
        'green': '50,1000',
        'cy3': '50,2000',
    }
    container = widgets.Container(
        widgets=[
            widgets.CheckBox(
                name='process_rgb',
                value=False,
                label='Process RGB',
                tooltip='Tick to create RGB images.',
            ),
            widgets.SpinBox(
                name='downsampling',
                value=3,
                min=1,
                label='Downsampling Factor',
                tooltip='Enter scale factor for downsampling.',
            ),
            widgets.ComboBox(
                name='contrast_mode',
                choices=[
                    ('Manual', 'manual'),
                    ('Automatic', 'automatic'),
                    ('None', 'none'),
                ],
                value='manual',
                label='Contrast mode',
            ),
            widgets.Label(
                name='automatic_method_note',
                value=(
                    'Black padding is ignored; each section contributes '
                    'equally.'
                ),
                label='',
            ),
        ],
        labels=True,
    )

    for channel in RGB_GUI_CHANNELS:
        selected = channel in ('cy3', 'green')
        row = widgets.Container(
            name=f'{channel}_row',
            layout='horizontal',
            labels=True,
            widgets=[
                widgets.CheckBox(
                    name=f'{channel}_enabled',
                    value=selected,
                    label=channel.upper() if channel == 'dapi' else channel.capitalize(),
                ),
                widgets.LineEdit(
                    name=f'{channel}_manual_range',
                    value=contrast_limits[channel],
                    label='Range',
                    tooltip=f'Enter manual min,max limits for {channel}.',
                ),
                widgets.ComboBox(
                    name=f'{channel}_profile',
                    choices=RGB_PROFILE_CHOICES,
                    value=RGB_DEFAULT_PROFILES[channel],
                    label='Profile',
                    tooltip=(
                        'Balanced (cells) clips both histogram tails. '
                        'Preserve strong signal (probe) protects rare bright '
                        'sections.'
                    ),
                ),
                widgets.LineEdit(
                    name=f'{channel}_estimated_range',
                    value='Not estimated',
                    label='Estimated range',
                    enabled=False,
                ),
                widgets.LineEdit(
                    name=f'{channel}_clipped',
                    value='—',
                    label='Clipped low / high',
                    enabled=False,
                ),
            ],
        )
        container.append(row)

    container.append(
        widgets.PushButton(
            name='estimate_ranges',
            text='Estimate ranges',
        )
    )
    container.append(
        widgets.ProgressBar(
            name='estimation_progress',
            min=0,
            max=1,
            value=0,
            label='Estimate progress',
        )
    )
    update_rgb_widget_state(container)
    return container


def update_rgb_widget_state(widget: widgets.Container) -> None:
    """Show and enable RGB controls that apply to the current mode."""
    mode = widget['contrast_mode'].value
    widget['automatic_method_note'].visible = mode == 'automatic'
    any_selected = False
    for channel in RGB_GUI_CHANNELS:
        row = widget[f'{channel}_row']
        enabled_widget = row[f'{channel}_enabled']
        selected = enabled_widget.enabled and enabled_widget.value
        any_selected |= selected
        manual_range = row[f'{channel}_manual_range']
        profile = row[f'{channel}_profile']
        estimated_range = row[f'{channel}_estimated_range']
        clipped = row[f'{channel}_clipped']

        manual_range.visible = mode == 'manual'
        manual_range.enabled = mode == 'manual' and selected
        profile.visible = mode == 'automatic'
        profile.enabled = mode == 'automatic' and selected
        estimated_range.visible = mode == 'automatic'
        clipped.visible = mode == 'automatic'

        if not selected:
            if estimated_range.value != 'Not selected':
                estimated_range._automatic_value = estimated_range.value
                clipped._automatic_value = clipped.value
            estimated_range.value = 'Not selected'
            clipped.value = '—'
        elif estimated_range.value == 'Not selected':
            estimated_range.value = getattr(
                estimated_range, '_automatic_value', 'Not estimated'
            )
            clipped.value = getattr(clipped, '_automatic_value', '—')

    estimate_button = widget['estimate_ranges']
    estimate_button.visible = mode == 'automatic'
    estimate_button.enabled = mode == 'automatic' and any_selected
    widget['estimation_progress'].visible = mode == 'automatic'


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
    rgb_estimation_progress_signal = Signal(object)
    """Signal carrying automatic RGB calibration progress to the GUI."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the PreprocessingWidget.

        Parameters:
            parent (Optional[QWidget]): Parent widget.
        """
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self._rgb_calibration = None
        self._rgb_calibration_key = None
        self._rgb_imaged_channels = set()
        self._rgb_estimation_worker = None

        # Header widget
        self.header = initialize_header_widget()
        self.header.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        # Add generalized widgets for different operations
        self.rgb_widget = create_rgb_widget()
        self.sharpy_widget = create_general_widget('SHARPy', ['dapi', 'green', 'n3', 'cy3', 'cy5'], contrast_limits={
            'dapi': '50,1000',
            'green': '50,300',
            'cy3': '50,500',
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
        self.rgb_estimation_progress_signal.connect(
            self._update_rgb_estimation_progress
        )

        self.header.input_path.changed.connect(self._rgb_settings_changed)
        self.header.chans_imaged.changed.connect(self._sync_rgb_channels)
        self.rgb_widget['contrast_mode'].changed.connect(
            self._rgb_settings_changed
        )
        self.rgb_widget['estimate_ranges'].clicked.connect(
            self._estimate_rgb_ranges
        )
        for channel in RGB_GUI_CHANNELS:
            row = self.rgb_widget[f'{channel}_row']
            row[f'{channel}_enabled'].changed.connect(
                self._rgb_settings_changed
            )
            row[f'{channel}_profile'].changed.connect(
                self._rgb_settings_changed
            )
        self._sync_rgb_channels()

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

    def _selected_rgb_profiles(self) -> Dict[str, str]:
        """Return automatic profiles for enabled and imaged RGB channels."""
        profiles = {}
        for channel in RGB_GUI_CHANNELS:
            row = self.rgb_widget[f'{channel}_row']
            enabled_widget = row[f'{channel}_enabled']
            if enabled_widget.enabled and enabled_widget.value:
                profiles[channel] = row[f'{channel}_profile'].value
        return profiles

    def _current_rgb_calibration_key(self) -> Tuple:
        """Return the GUI state that determines automatic calibration."""
        return (
            str(self.header.input_path.value),
            tuple(self.header.chans_imaged.value),
            tuple(self._selected_rgb_profiles().items()),
        )

    def _clear_rgb_calibration_display(self) -> None:
        """Clear automatic results after an input or profile change."""
        for channel in RGB_GUI_CHANNELS:
            row = self.rgb_widget[f'{channel}_row']
            estimated_range = row[f'{channel}_estimated_range']
            clipped = row[f'{channel}_clipped']
            estimated_range._automatic_value = 'Not estimated'
            clipped._automatic_value = '—'
            if row[f'{channel}_enabled'].value:
                estimated_range.value = 'Not estimated'
                clipped.value = '—'

        self.rgb_widget['estimation_progress'].value = 0

    def _rgb_settings_changed(self, *_args) -> None:
        """Invalidate cached calibration and refresh conditional controls."""
        self._rgb_calibration = None
        self._rgb_calibration_key = None
        self._clear_rgb_calibration_display()
        update_rgb_widget_state(self.rgb_widget)

    def _sync_rgb_channels(self, *_args) -> None:
        """Synchronize RGB checkboxes with channels declared as imaged."""
        imaged_channels = set(self.header.chans_imaged.value)
        newly_imaged = imaged_channels - self._rgb_imaged_channels
        for channel in RGB_GUI_CHANNELS:
            enabled_widget = self.rgb_widget[f'{channel}_row'][
                f'{channel}_enabled'
            ]
            is_imaged = channel in imaged_channels
            enabled_widget.enabled = is_imaged
            if not is_imaged:
                enabled_widget.value = False
            elif channel in newly_imaged:
                enabled_widget.value = True
        self._rgb_imaged_channels = imaged_channels
        self._rgb_settings_changed()

    def _estimate_rgb_ranges(self, *_args) -> None:
        """Start an asynchronous preview of automatic RGB ranges."""
        input_path = self.header.input_path.value
        if not check_input_path(input_path):
            return
        profiles = self._selected_rgb_profiles()
        if not profiles:
            show_info("Select at least one RGB channel to estimate ranges.")
            return
        source_channels = list(self.header.chans_imaged.value)
        image_names = get_image_list(input_path, next(iter(profiles)))
        calibration_key = self._current_rgb_calibration_key()

        estimate_button = self.rgb_widget['estimate_ranges']
        estimate_button.text = 'Estimating ranges...'
        estimate_button.enabled = False
        progress = self.rgb_widget['estimation_progress']
        progress.min = 0
        progress.max = max(len(image_names), 1)
        progress.value = 0
        print(
            '[Automatic RGB contrast] Starting calibration for '
            f'{len(image_names)} sections. '
            f"Channels: {', '.join(profiles)}",
            flush=True,
        )
        worker = estimate_rgb_ranges_worker(
            input_path,
            image_names,
            source_channels,
            profiles,
            calibration_key,
            self.rgb_estimation_progress_signal.emit,
        )
        worker.returned.connect(self._apply_rgb_calibration)
        worker.errored.connect(self._show_rgb_calibration_error)
        self._rgb_estimation_worker = worker
        worker.start()

    def _update_rgb_estimation_progress(
        self, result: Tuple[Tuple, Dict[str, object]]
    ) -> None:
        """Append one section's histogram cutoffs to the live report."""
        calibration_key, report = result
        if calibration_key != self._current_rgb_calibration_key():
            return

        progress = self.rgb_widget['estimation_progress']
        event = report['event']
        if event == 'section':
            section_index = report['section_index']
            total_sections = report['total_sections']
            progress.max = max(total_sections, 1)
            progress.value = section_index
            self.rgb_widget['estimate_ranges'].text = (
                f'Estimating {section_index}/{total_sections}...'
            )
            if report['status'] == 'skipped_all_black':
                line = (
                    f"[{section_index}/{total_sections}] "
                    f"{report['image_name']}: skipped (all black)"
                )
            else:
                cutoffs = ', '.join(
                    f'{channel}={limits[0]}–{limits[1]}'
                    for channel, limits in report[
                        'channel_cutoffs'
                    ].items()
                )
                padding_percent = report['padding_fraction'] * 100
                line = (
                    f"[{section_index}/{total_sections}] "
                    f"{report['image_name']}: {cutoffs}; "
                    f'padding excluded={padding_percent:.2f}%'
                )
        else:
            ranges = ', '.join(
                f'{channel}={limits[0]}–{limits[1]}'
                for channel, limits in report['automatic_ranges'].items()
            )
            line = f'Complete. Dataset ranges: {ranges}'

        print(f'[Automatic RGB contrast] {line}', flush=True)

    def _apply_rgb_calibration(
        self, result: Tuple[Tuple, Dict[str, object]]
    ) -> None:
        """Display automatic ranges if the relevant GUI state is unchanged."""
        calibration_key, calibration = result
        if calibration_key == self._current_rgb_calibration_key():
            self._rgb_calibration_key = calibration_key
            self._rgb_calibration = calibration
            for channel, limits in calibration['automatic_ranges'].items():
                row = self.rgb_widget[f'{channel}_row']
                range_text = f'{limits[0]} – {limits[1]}'
                clipped_values = calibration[
                    'automatic_clipped_percent'
                ][channel]
                clipped_text = (
                    f"{clipped_values['low']:.3f}% / "
                    f"{clipped_values['high']:.3f}%"
                )
                row[f'{channel}_estimated_range']._automatic_value = (
                    range_text
                )
                row[f'{channel}_clipped']._automatic_value = clipped_text
                row[f'{channel}_estimated_range'].value = range_text
                row[f'{channel}_clipped'].value = clipped_text
        estimate_button = self.rgb_widget['estimate_ranges']
        estimate_button.text = 'Estimate ranges'
        update_rgb_widget_state(self.rgb_widget)

    def _show_rgb_calibration_error(self, error: Exception) -> None:
        """Report automatic calibration errors and restore the button."""
        show_info(f"Could not estimate automatic RGB ranges: {error}")
        print(f'[Automatic RGB contrast] Error: {error}', flush=True)
        estimate_button = self.rgb_widget['estimate_ranges']
        estimate_button.text = 'Estimate ranges'
        update_rgb_widget_state(self.rgb_widget)

    def _get_rgb_widget_info(self) -> Dict[str, object]:
        """Collect RGB mode, channel, profile, and range parameters."""
        profiles = self._selected_rgb_profiles()
        channels = list(profiles)
        mode = self.rgb_widget['contrast_mode'].value
        rgb_info = {
            'channels': channels,
            'downsampling': self.rgb_widget['downsampling'].value,
            'contrast_mode': mode,
            'contrast_adjustment': mode != 'none',
        }

        if mode == 'manual':
            for channel in channels:
                value = self.rgb_widget[f'{channel}_row'][
                    f'{channel}_manual_range'
                ].value
                rgb_info[channel] = [int(item) for item in value.split(',')]
        elif mode == 'automatic':
            rgb_info['automatic_profiles'] = profiles
            for channel in channels:
                rgb_info[channel] = []
            if (
                self._rgb_calibration is not None
                and self._rgb_calibration_key
                == self._current_rgb_calibration_key()
            ):
                rgb_info.update(self._rgb_calibration)
                for channel, limits in self._rgb_calibration[
                    'automatic_ranges'
                ].items():
                    rgb_info[channel] = list(limits)
        else:
            for channel in channels:
                rgb_info[channel] = []
        return rgb_info

    def _get_widget_info(self, widget: FunctionGui, item: str) -> Dict[str, Union[List[int], int, str]]:
        """
        Retrieve information from a given widget based on the type of item.

        Parameters:
            widget (FunctionGui): The widget to retrieve information from.
            item (str): Type of operation (e.g., 'rgb', 'sharpy_track').

        Returns:
            Dict[str, Union[List[int], int, str]]: Information extracted from the widget.
        """
        if item == 'rgb':
            return self._get_rgb_widget_info()

        chan_list = ['dapi', 'green', 'n3', 'cy3', 'cy5']

        imaged_chan_list = (widget[1].value if 'all' not in widget[1].value
                            else self.header.chans_imaged.value)
        imaged_chan_list = [i for i in imaged_chan_list if i in self.header.chans_imaged.value]

        base_info = {"channels": imaged_chan_list, "downsampling": widget[2].value}

        if item != 'binary':
            contrast_widget_index = 2 if item == 'sharpy_track' else 3
            base_info["contrast_adjustment"] = widget[
                contrast_widget_index
            ].value

        if item == 'binary':
            if widget[4].value:  # manual thresholds
                base_info.update({"manual_threshold": widget[4].value})
                base_info.update({channel: [int(i) for i in widget[4 + idx].value.split(',')] for idx, channel in
                                  enumerate(chan_list) if channel in imaged_chan_list})
            else:
                base_info.update({"manual_threshold": widget[4].value, "thresh_method": widget[3].value.value})
        elif base_info["contrast_adjustment"]:
            contrast_start_index = 3 if item == 'sharpy_track' else 4
            base_info.update({
                channel: [
                    int(i)
                    for i in widget[
                        contrast_start_index + idx
                    ].value.split(',')
                ]
                for idx, channel in enumerate(chan_list)
                if channel in imaged_chan_list
            })
        else:
            base_info.update({
                channel: []
                for channel in chan_list
                if channel in imaged_chan_list
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
        if (
            preprocessing_params.get('operations', {}).get('rgb')
            and not preprocessing_params['rgb_params']['channels']
        ):
            show_info("Select at least one channel for RGB preprocessing.")
            return
        save_dirs = create_dirs(preprocessing_params, input_path)
        channels = get_channels(preprocessing_params)
        if not channels:
            show_info("Select at least one channel for preprocessing.")
            return
        img_list = get_image_list(input_path, channels[0])
        params_dict = load_params(input_path)
        resolution = params_dict['atlas_info']['resolution']

        # Start the preprocessing worker
        preprocessing_worker = do_preprocessing(input_path, channels, img_list, preprocessing_params, resolution, save_dirs)
        preprocessing_worker.yielded.connect(self._update_progress)
        preprocessing_worker.started.connect(lambda: self.btn.setText("Preprocessing Images..."))
        preprocessing_worker.returned.connect(self._show_success_message)
        preprocessing_worker.start()
