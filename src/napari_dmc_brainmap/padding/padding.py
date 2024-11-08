"""
DMC-BrainMap widget for padding .tif files to match atlas resolution

2024 - FJ
"""
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox, QProgressBar
from napari import Viewer
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import tifffile
from natsort import natsorted
import pandas as pd
from pathlib import Path
from napari_dmc_brainmap.stitching.stitching_tools import padding_for_atlas
from napari_dmc_brainmap.utils import load_params, get_info, get_animal_id
from typing import List, Dict, Tuple, Generator


@thread_worker(progress={'total': 100})
def do_padding(input_path: Path, channels: List[str], pad_folder: str, resolution: Tuple[int, int]) -> Generator[int, None, str]:
    """
    Pad .tif images to match the atlas resolution.

    Parameters:
    input_path (Path): Path to the input directory containing subfolders for images.
    channels (List[str]): List of channels to process.
    pad_folder (str): Name of the folder containing images to be padded.
    resolution (Tuple[int, int]): The desired resolution for padding.

    Yields:
    int: Progress value during padding.

    Returns:
    str: The animal ID of the processed images.
    """
    if pad_folder == "confocal":
        raise NotImplementedError("'confocal' is a keyword reserved for CZI file format,"
                                  "\nPlease rename the folder something else (such as 'to_pad'), if these are tif files to pad."
                                  "\nFor CZI files, please goto 'Stitch czi images' function."
                                  "\nExiting padding function!")

    animal_id = get_animal_id(input_path)
    progress_value = 0
    image_count = count_images(input_path, pad_folder, channels)
    progress_step = 100 / image_count

    for chan in channels:
        tif_files = list(input_path.joinpath(pad_folder, chan).glob("*.tif"))
        if not tif_files[0].name.endswith("_stitched.tif"):
            rename_image_files(tif_files, input_path, pad_folder, chan)
            save_image_names_csv(tif_files, input_path, pad_folder, chan)

        pad_dir, pad_im_list, pad_suffix = get_info(input_path, pad_folder, channel=chan)
        for im in pad_im_list:
            im_fn = pad_dir.joinpath(im)
            try:
                im_array = tifffile.imread(str(im_fn))
            except Exception as e:
                show_info(f"Failed to read {im_fn}: {e}")
                continue

            im_padded = padding_for_atlas(im_array, resolution)
            try:
                tifffile.imwrite(str(im_fn), im_padded)
            except Exception as e:
                show_info(f"Failed to write {im_fn}: {e}")
                continue

            progress_value += progress_step
            yield int(progress_value)

    yield 100
    return animal_id


def count_images(input_path: Path, pad_folder: str, channels: List[str]) -> int:
    """
    Count the number of images in the specified folder and channels.

    Parameters:
    input_path (Path): Path to the input directory.
    pad_folder (str): Name of the folder containing images to be padded.
    channels (List[str]): List of channels to count images for.

    Returns:
    int: The total count of images in the specified channels.
    """
    image_count = 0
    for chan in channels:
        image_count += len(list(input_path.joinpath(pad_folder, chan).glob("*.tif")))
    if image_count == 0:
        image_count = 1
    return image_count


def rename_image_files(tif_files: List[Path], input_path: Path, pad_folder: str, chan: str) -> None:
    """
    Rename image files to add '_stitched' suffix if missing.

    Parameters:
    tif_files (List[Path]): List of tif files to rename.
    input_path (Path): Path to the input directory.
    pad_folder (str): Name of the folder containing images to be renamed.
    chan (str): Channel name for which images are being renamed.
    """
    for im in tif_files:
        im_old = input_path.joinpath(pad_folder, chan, im.name)
        im_new = input_path.joinpath(pad_folder, chan, f"{im.stem}_stitched.tif")
        im_old.rename(im_new)


def save_image_names_csv(tif_files: List[Path], input_path: Path) -> None:
    """
    Save the list of image names to 'image_names.csv' if it does not already exist.

    Parameters:
    tif_files (List[Path]): List of tif files whose names are to be saved.
    input_path (Path): Path to the input directory.
    """
    image_names_csv = input_path.joinpath("image_names.csv")
    if not image_names_csv.exists():
        image_list = natsorted([tif.name.split("_stitched.tif")[0] for tif in tif_files])
        image_list_store = pd.DataFrame(image_list)
        image_list_store.to_csv(image_names_csv, index=False)


def initialize_widget() -> FunctionGui:
    """
    Initialize the MagicGUI widget for padding.

    Returns:
    FunctionGui: The initialized MagicGUI widget.
    """
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit',
                              label='input path (animal_id): ',
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                      'folder containing images'),
              pad_folder=dict(widget_type='LineEdit',
                              label='folder name images to be padded: ',
                              value='stitched',
                              tooltip='name of folder containing the stitched images to be padded '
                                      '(animal_id/>pad_folder</chan1)'),
              channels=dict(widget_type='Select',
                            label='imaged channels',
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select the imaged channels, to select multiple hold ctrl/shift'),
              call_button=False)
    def padding_widget(viewer: Viewer, input_path: Path, channels: List[str], pad_folder: str):
        pass

    return padding_widget


class PaddingWidget(QWidget):
    progress_signal = Signal(int)
    """Signal emitted to update the progress bar with an integer value."""
    def __init__(self, napari_viewer: Viewer):
        """
        Initialize the padding widget.

        Parameters:
        napari_viewer (Viewer): The napari viewer instance.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.padding = initialize_widget()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.btn = QPushButton("Do the padding (WARNING - overriding existing files!)")
        self.btn.clicked.connect(self._do_padding)

        self.layout().addWidget(self.padding.native)
        self.layout().addWidget(self.btn)
        self.layout().addWidget(self.progress_bar)
        self.progress_signal.connect(self.progress_bar.setValue)

    def _show_success_message(self, animal_id: str) -> None:
        """
        Display a success message after padding is complete.

        Parameters:
        animal_id (str): The animal ID for which padding was performed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Padding finished for {animal_id}!")
        msg_box.setWindowTitle("Padding successful!")
        msg_box.exec_()
        self.btn.setText("Do the padding (WARNING - overriding existing files!)")  # Reset button text after process completion
        self.progress_signal.emit(0)

    def _update_progress(self, value: int) -> None:
        """
        Update the progress bar with the given value.

        Parameters:
        value (int): Progress value to set.
        """
        self.progress_signal.emit(value)

    def _do_padding(self) -> None:
        """
        Execute the padding operation with confirmation from the user.
        """
        reply = QMessageBox.question(self, 'Warning',
                                     "This will override existing files. Do you want to continue?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            input_path = self.padding.input_path.value
            channels = self.padding.channels.value
            pad_folder = self.padding.pad_folder.value
            params_dict = load_params(input_path)
            resolution = params_dict['atlas_info']['resolution']  # [x,y]
            padding_worker = do_padding(input_path, channels, pad_folder, resolution)
            padding_worker.yielded.connect(self._update_progress)
            padding_worker.started.connect(
                lambda: self.btn.setText("Padding images..."))  # Change button text when padding starts
            padding_worker.returned.connect(self._show_success_message)
            padding_worker.start()
