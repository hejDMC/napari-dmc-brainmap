"""
DMC-BrainMap widget for padding .tif files to match atlas resolution.

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
from napari_dmc_brainmap.utils.path_utils import get_info, get_image_list
from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.params_utils import load_params
from napari_dmc_brainmap.utils.gui_utils import check_input_path
from typing import List, Tuple, Generator


@thread_worker(progress={"total": 100})
def do_padding(input_path: Path,
               channels: List[str],
               pad_folder: str,
               resolution: Tuple[int, int]) -> Generator[int, None, str]:
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
        raise NotImplementedError(
            "'confocal' is reserved for CZI file format. "
            "Rename the folder for .tif files (e.g., 'to_pad'). "
            "For CZI files, use the 'Stitch czi images' function."
        )

    animal_id = get_animal_id(input_path)
    progress_value = 0
    image_count = count_images(input_path, pad_folder, channels)
    progress_step = 100 / image_count

    for chan in channels:
        tif_files = list(input_path.joinpath(pad_folder, chan).glob("*.tif"))
        try:
            if not tif_files[0].name.endswith("_stitched.tif"):
                rename_image_files(tif_files, input_path, pad_folder, chan)
                get_image_list(input_path, chan, folder_id=pad_folder)
                # save_image_names_csv(tif_files, input_path)

            pad_dir, pad_im_list, _ = get_info(input_path, pad_folder, channel=chan)
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
        except IndexError:
            show_info(f"No images found for channel {chan}. Skipping padding...")

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
    image_count = sum(len(list(input_path.joinpath(pad_folder, chan).glob("*.tif"))) for chan in channels)
    return max(image_count, 1)


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
    print(tif_files)
    if not image_names_csv.exists():
        image_list = natsorted([tif.name.split("_stitched.tif")[0] for tif in tif_files])
        print(image_list)
        pd.DataFrame(image_list).to_csv(image_names_csv, index=False)

def initialize_widget() -> FunctionGui:
    """
    Initialize the MagicGUI widget for padding configuration.

    Returns:
        FunctionGui: The initialized MagicGUI widget.
    """
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit',
                              label='Input Path (Animal ID):',
                              mode='d',
                              tooltip='Directory containing subfolders with images or segmentation results.'),
              pad_folder=dict(widget_type='LineEdit',
                              label='Folder Name for Images to Pad:',
                              value='stitched',
                              tooltip='Name of the folder containing the stitched images to be padded. '
                                      '(e.g., animal_id/pad_folder/chan1)'),
              channels=dict(widget_type='Select',
                            label='Imaged Channels:',
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='Select the imaged channels. Use Ctrl/Shift for multiple selections.'),
              call_button=False)
    def padding_widget(viewer: Viewer, input_path: Path, channels: List[str], pad_folder: str):
        """
        Padding configuration widget.

        Parameters:
            viewer (Viewer): The Napari viewer instance.
            input_path (Path): Path to the folder containing experimental data (animal ID level).
            channels (List[str]): List of imaged channels.
            pad_folder (str): Name of the folder containing images to be padded.
        """
        pass

    return padding_widget


class PaddingWidget(QWidget):
    """
    QWidget for configuring and initiating the padding process for images.
    """
    progress_signal = Signal(int)
    """Signal emitted to update the progress bar with an integer value."""

    def __init__(self, napari_viewer: Viewer) -> None:
        """
        Initialize the PaddingWidget instance.

        Parameters:
            napari_viewer (Viewer): The Napari viewer instance.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.padding = initialize_widget()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)

        self.btn = QPushButton("Do the Padding (WARNING: Overwrites Existing Files!)")
        self.btn.clicked.connect(self._do_padding)

        self.layout().addWidget(self.padding.native)
        self.layout().addWidget(self.btn)
        self.layout().addWidget(self.progress_bar)
        self.progress_signal.connect(self.progress_bar.setValue)

    def _show_success_message(self, animal_id: str) -> None:
        """
        Display a success message after the padding process is complete.

        Parameters:
            animal_id (str): The Animal ID for which padding was performed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Padding finished successfully for {animal_id}!")
        msg_box.setWindowTitle("Padding Successful")
        msg_box.exec_()
        self.btn.setText("Do the Padding (WARNING: Overwrites Existing Files!)")
        self.progress_signal.emit(0)

    def _update_progress(self, value: int) -> None:
        """
        Update the progress bar with the given value.

        Parameters:
            value (int): The progress value to set.
        """
        self.progress_signal.emit(value)

    def _do_padding(self) -> None:
        """
        Execute the padding operation with user confirmation.
        """
        reply = QMessageBox.question(
            self, "Warning",
            "This operation will overwrite existing files. Do you want to continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            input_path = self.padding.input_path.value
            if not check_input_path(input_path):
                return

            channels = self.padding.channels.value
            pad_folder = self.padding.pad_folder.value
            params_dict = load_params(input_path)
            resolution = params_dict['atlas_info']['resolution']  # [x, y]

            padding_worker = do_padding(input_path, channels, pad_folder, resolution)
            padding_worker.yielded.connect(self._update_progress)
            padding_worker.started.connect(lambda: self.btn.setText("Padding Images..."))
            padding_worker.returned.connect(self._show_success_message)
            padding_worker.start()
