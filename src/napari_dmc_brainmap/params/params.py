import json
from napari_dmc_brainmap.utils import get_animal_id, update_params_dict, clean_params_dict, get_atlas_dropdown, get_xyz
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from bg_atlasapi import BrainGlobeAtlas


def initialize_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                        'folder containing segmentation results'),
              inj_side=dict(widget_type='ComboBox', 
                            label='injection side',
                            choices=['','left', 'right'], 
                            value='',
                            tooltip='select the injection hemisphere (if applicable)'),
              geno=dict(widget_type='LineEdit', 
                        label='genotype',
                        tooltip='enter (if applicable) the genotype of the animal (be CONSISTENT in your naming across animals)'),
              group=dict(widget_type='LineEdit', 
                         label='experimental group',
                         tooltip='enter (if applicable) the experimental group of the animal '
                            '(be CONSISTENT in your naming across animals)'),
              chans_imaged=dict(widget_type='Select', 
                                label='imaged channels', 
                                choices=['dapi', 'green', 'cy3', 'cy5'],
                                value=['green', 'cy3'],
                                tooltip='select all channels imaged, to select multiple hold ctrl/shift'),
              section_orient=dict(widget_type='ComboBox', label='orientation of sectioning',
                                  choices=['coronal', 'sagittal', 'horizontal'], value='coronal',
                                  tooltip="select the how you sliced the brain"),
              atlas=dict(label='reference atlas',
                         tooltip='select the reference atlas using for registration '
                            '(from https://github.com/brainglobe/bg-atlasapi/ and '
                            'https://github.com/brainglobe/brainreg-segment'),
              call_button=False)
    
    def params_widget(
        input_path,  # posix path
        inj_side,
        geno,
        group,
        chans_imaged,
        section_orient,
        atlas: get_atlas_dropdown()):
        pass
    return params_widget



class ParamsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.params = initialize_widget()
        btn = QPushButton("create params.json file")
        btn.clicked.connect(self._create_params_file)

        self.layout().addWidget(self.params.native)
        self.layout().addWidget(btn)


    def _create_params_file(self):
        input_path = self.params.input_path.value
        animal_id = get_animal_id(input_path)
        injection_side = self.params.inj_side.value
        genotype = self.params.geno.value
        group = self.params.group.value
        chans_imaged = self.params.chans_imaged.value
        atlas_name = self.params.atlas.value.value
        orientation = self.params.section_orient.value
        print('check existence of local version of ' + atlas_name + ' atlas ...')
        print('loading reference atlas ' + atlas_name + ' ...')
        atlas = BrainGlobeAtlas(atlas_name)
        xyz_dict = get_xyz(atlas, orientation)
        resolution_tuple = (xyz_dict['x'][1], xyz_dict['y'][1])

        params_dict = {
            "general": {
                "animal_id": animal_id,
                "injection_side": injection_side,
                "genotype": genotype,
                "group": group,
                "chans_imaged": chans_imaged
            },
            "atlas_info": {
                "atlas":  atlas_name,
                "orientation": orientation,
                "resolution": resolution_tuple,
                'xyz_dict': xyz_dict

            }
        }
        params_dict = clean_params_dict(params_dict, "general")
        params_fn = input_path.joinpath('params.json')
        params_dict = update_params_dict(input_path, params_dict)
        with open(params_fn, 'w') as fn:
            json.dump(params_dict, fn, indent=4)
        print('params.json file for ' + animal_id + ' created !')

