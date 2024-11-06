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
from napari_dmc_brainmap.stitching.stitching_tools import padding_for_atlas
from napari_dmc_brainmap.utils import load_params, get_info, get_animal_id
from typing import List, Dict, Tuple

@thread_worker(progress={'total': 100})
def do_padding(input_path, channels, pad_folder, resolution):
    # check if pad_folder name is "confocal"
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
        # Cache the list of tif files to avoid multiple `.glob()` calls
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


def count_images(input_path, pad_folder, channels):
    image_count = 0
    for chan in channels:
        image_count += len(list(input_path.joinpath(pad_folder, chan).glob("*.tif")))
    if image_count == 0:
        image_count = 1
    return image_count

def rename_image_files(tif_files, input_path, pad_folder, chan):
    """
    Rename image files to add '_stitched' suffix if missing.
    """
    for im in tif_files:
        im_old = input_path.joinpath(pad_folder, chan, im.name)
        im_new = input_path.joinpath(pad_folder, chan, f"{im.stem}_stitched.tif")
        # print(f"Renaming ===> {str(im_old)} \nto {str(im_new)} <===")
        im_old.rename(im_new)


def save_image_names_csv(tif_files, input_path):
    """
    Save the list of image names to 'image_names.csv' if it does not already exist.
    """
    image_names_csv = input_path.joinpath("image_names.csv")
    if not image_names_csv.exists():
        image_list = natsorted([tif.name.split("_stitched.tif")[0] for tif in tif_files])
        image_list_store = pd.DataFrame(image_list)
        image_list_store.to_csv(image_names_csv, index=False)
        # print(f"Image names saved to {image_names_csv}")


def initialize_widget() -> FunctionGui:
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
    def padding_widget(viewer: Viewer, input_path, channels, pad_folder):
        pass

    return padding_widget


class PaddingWidget(QWidget):
    progress_signal = Signal(int)
    def __init__(self, napari_viewer):
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
        Display a success message after stitching is complete.

        Parameters:
        animal_id (str): The animal ID for which stitching was performed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Padding finished for {animal_id}!")
        msg_box.setWindowTitle("Padding successful!")
        msg_box.exec_()
        self.btn.setText("Do the padding (WARNING - overriding existing files!)")  # Reset button text after process completion

    def _update_progress(self, value: int) -> None:
        """
        Update the progress bar with the given value.

        Parameters:
        value (int): Progress value to set.
        """
        self.progress_signal.emit(value)

    def _do_padding(self):
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
                lambda: self.btn.setText("Padding images..."))  # Change button text when stitching starts
            padding_worker.returned.connect(self._show_success_message)
            padding_worker.start()
