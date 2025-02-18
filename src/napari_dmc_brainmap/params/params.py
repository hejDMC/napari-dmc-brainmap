"""
DMC-BrainMap widget for creating params.json file.

params.json is used to track experimental parameters for animals
and maintain a history of preprocessing operations performed.

2024 - FJ, XC
"""

# Import modules
import json
from pathlib import Path
from typing import Optional

from napari_dmc_brainmap.utils.params_utils import clean_params_dict, update_params_dict
from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.dropdown_utils import get_atlas_dropdown
from napari_dmc_brainmap.utils.atlas_utils import get_xyz
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from bg_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.gui_utils import check_input_path


def initialize_widget() -> FunctionGui:
    """
    Initialize the params widget for creating experimental parameter files.

    Returns:
        FunctionGui: A widget for user input fields to collect experimental parameters.
    """

    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit',
                              label='Input Path (animal_id): ',
                              mode='d',
                              tooltip='Directory containing subfolders with raw data, images, or segmentation results.'),
              inj_side=dict(widget_type='ComboBox',
                            label='Injection Site',
                            choices=['', 'left', 'right'],
                            value='',
                            tooltip='Select the injection hemisphere (if applicable).'),
              geno=dict(widget_type='LineEdit',
                        label='Genotype',
                        tooltip='Enter the genotype of the animal (if applicable).'),
              group=dict(widget_type='LineEdit',
                         label='Experimental Group',
                         tooltip='Enter the experimental group of the animal (if applicable).'),
              chans_imaged=dict(widget_type='Select',
                                label='Imaged Channels',
                                choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                                value=['green', 'cy3'],
                                tooltip='Select all imaged channels. Use ctrl/shift for multiple selections.'),
              section_orient=dict(widget_type='ComboBox',
                                  label='Orientation of Sectioning',
                                  choices=['coronal', 'sagittal', 'horizontal'],
                                  value='coronal',
                                  tooltip='Select the brain slicing orientation.'),
              atlas=dict(label='Reference Atlas',
                         tooltip='Select the reference atlas for registration.'),
              call_button=False)
    def params_widget(input_path: Path,
                      inj_side: str,
                      geno: str,
                      group: str,
                      chans_imaged: list,
                      section_orient: str,
                      atlas: get_atlas_dropdown()):
        """
        Create the params widget for collecting experimental parameter inputs.

        Parameters:
            input_path (Path): Path to the folder containing experimental data.
            inj_side (str): Injection hemisphere ('left', 'right', or '').
            geno (str): Genotype of the animal.
            group (str): Experimental group of the animal.
            chans_imaged (list): Channels imaged during the experiment.
            section_orient (str): Orientation of sectioning (coronal, sagittal, horizontal).
            atlas (Enum): Reference atlas used for registration.
        """
        pass

    return params_widget


class ParamsWidget(QWidget):
    """
    QWidget for creating the params.json file for an experiment.
    """

    def __init__(self, napari_viewer):
        """
        Initialize the ParamsWidget.

        Parameters:
            napari_viewer: The napari viewer instance where the widget will be added.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.params = initialize_widget()
        btn = QPushButton("Create params.json File")
        btn.clicked.connect(self._create_params_file)

        self.layout().addWidget(self.params.native)
        self.layout().addWidget(btn)

    def _create_params_file(self) -> None:
        """
        Create the params.json file based on user input and save it to the specified location.
        """
        input_path: Optional[Path] = self.params.input_path.value

        # Check if user provided a valid input_path
        if not check_input_path(input_path):
            return

        animal_id = get_animal_id(input_path)
        injection_site = self.params.inj_side.value
        genotype = self.params.geno.value
        group = self.params.group.value
        chans_imaged = self.params.chans_imaged.value
        atlas_name = self.params.atlas.value.value
        orientation = self.params.section_orient.value

        try:
            show_info(f'Checking existence of local version of {atlas_name} atlas...')
            show_info(f'Loading reference atlas {atlas_name}...')
            atlas = BrainGlobeAtlas(atlas_name)
            xyz_dict = get_xyz(atlas, orientation)
            resolution_tuple = (xyz_dict['x'][1], xyz_dict['y'][1])

            # Basic structure of params.json dictionary
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
            # Clean params dictionary to remove empty keys
            params_dict = clean_params_dict(params_dict, "general")

            params_fn = input_path / 'params.json'
            params_dict = update_params_dict(input_path, params_dict, create=True)

            with open(params_fn, 'w') as fn:
                json.dump(params_dict, fn, indent=4)

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setText(f"params.json file for {animal_id} created successfully!")
            msg_box.setWindowTitle("Success")
            msg_box.exec_()
        except Exception as e:
            # Show an error message box if any error occurs
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setText(f"An error occurred: {str(e)}")
            msg_box.setWindowTitle("Processing Error")
            msg_box.exec_()
