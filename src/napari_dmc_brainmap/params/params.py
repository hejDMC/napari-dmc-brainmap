"""
DMC-BrainMap widget for creating params.json file

params.json file is used to keep track of experimental parameters of animal
  as well as to keep a history of preprocessing operations performed

2024 - FJ, XC
"""

# import modules
import json
import sys

from napari_dmc_brainmap.utils import get_animal_id, update_params_dict, clean_params_dict, get_atlas_dropdown, get_xyz
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from bg_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info

def initialize_widget() -> FunctionGui:
    """
    Initialize the params widget for creating experimental parameter files.

    :return: A FunctionGui widget with user input fields for experimental parameters.
    """

    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit',
                              label='input path (animal_id): ',
                              mode='d',
                              tooltip='Directory of folder containing subfolders with e.g. raw data, images, segmentation results, NOT '
                                      'folder containing images'),
              inj_side=dict(widget_type='ComboBox',
                            label='injection site',
                            choices=['', 'left', 'right'],
                            value='',
                            tooltip='Select the injection hemisphere (if applicable)'),
              geno=dict(widget_type='LineEdit',
                        label='genotype',
                        tooltip='Enter (if applicable) the genotype of the animal (be CONSISTENT in your naming across animals)'),
              group=dict(widget_type='LineEdit',
                         label='experimental group',
                         tooltip='Enter (if applicable) the experimental group of the animal '
                                 '(be CONSISTENT in your naming across animals)'),
              chans_imaged=dict(widget_type='Select',
                                label='imaged channels',
                                choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                                value=['green', 'cy3'],
                                tooltip='Select all channels imaged, to select multiple hold ctrl/shift'),
              section_orient=dict(widget_type='ComboBox', label='orientation of sectioning',
                                  choices=['coronal', 'sagittal', 'horizontal'], value='coronal',
                                  tooltip='Select the orientation you sliced the brain in'),
              atlas=dict(label='reference atlas',
                         tooltip='Select the reference atlas used for registration '
                                 '(from https://github.com/brainglobe/bg-atlasapi/ and '
                                 'https://github.com/brainglobe/brainreg-segment )'),
              call_button=False)
    def params_widget(
            input_path,  # posix path
            inj_side,
            geno,
            group,
            chans_imaged,
            section_orient,
            atlas: get_atlas_dropdown()):
        """
        Create the params_widget for collecting experimental parameter inputs.

        :param input_path: Path to the folder containing experimental data.
        :param inj_side: Injection hemisphere ('left', 'right', or '').
        :param geno: Genotype of the animal.
        :param group: Experimental group of the animal.
        :param chans_imaged: Channels imaged during the experiment.
        :param section_orient: Orientation of sectioning (coronal, sagittal, horizontal).
        :param atlas: Reference atlas used for registration.
        """
        pass

    return params_widget


class ParamsWidget(QWidget):
    """
    QWidget for creating params.json file for an experiment.
    """

    def __init__(self, napari_viewer):
        """
        Initialize the ParamsWidget.

        :param napari_viewer: The napari viewer instance where the widget is added.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.params = initialize_widget()
        btn = QPushButton("create params.json file")
        btn.clicked.connect(self._create_params_file)

        self.layout().addWidget(self.params.native)
        self.layout().addWidget(btn)

    def _create_params_file(self) -> None:
        """
        Create the params.json file based on user input and save it to the specified location.
        """
        input_path = self.params.input_path.value

        # check if user provided a valid input_path
        if not input_path.is_dir() or str(input_path) == '.':
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(
                f"Input path is not a valid directory. Please make sure this exists: >> '{str(input_path)}' <<")
            msg_box.setWindowTitle("Invalid Path Error")
            msg_box.exec_()  # Show the message box
            return

        animal_id = get_animal_id(input_path)
        injection_site = self.params.inj_side.value
        genotype = self.params.geno.value
        group = self.params.group.value
        chans_imaged = self.params.chans_imaged.value
        atlas_name = self.params.atlas.value.value
        orientation = self.params.section_orient.value
        try:
            show_info(f'check existence of local version of {atlas_name} atlas ...')
            show_info(f'loading reference atlas {atlas_name} ...')
            atlas = BrainGlobeAtlas(atlas_name)
            xyz_dict = get_xyz(atlas, orientation)
            resolution_tuple = (xyz_dict['x'][1], xyz_dict['y'][1])

            # basic structure of params.json dictionary
            params_dict = {
                "general": {
                    "animal_id": animal_id,
                    "injection_site": injection_site,
                    "genotype": genotype,
                    "group": group,
                    "chans_imaged": chans_imaged
                },
                "atlas_info": {
                    "atlas": atlas_name,
                    "orientation": orientation,
                    "resolution": resolution_tuple,
                    'xyz_dict': xyz_dict
                }
            }
            params_dict = clean_params_dict(params_dict,
                                            "general")  # remove empty keys, e.g. when no genotype specified
            params_fn = input_path.joinpath('params.json')
            params_dict = update_params_dict(input_path, params_dict, create=True)
            with open(params_fn, 'w') as fn:
                json.dump(params_dict, fn, indent=4)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(f"params.json file for {animal_id} created successfully!")
            msg_box.setWindowTitle("Success")
            msg_box.exec_()
        except Exception as e:
            # If any error occurs during processing, show an error message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(f"An error occurred: {str(e)}")
            msg_box.setWindowTitle("Processing Error")
            msg_box.exec_()
