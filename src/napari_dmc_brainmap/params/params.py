import json
from napari_dmc_brainmap.utils import get_animal_id, update_params_dict, clean_params_dict
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui


@magicgui(
    # todo sate that only RGB images for now, think about different image formats
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
    inj_side=dict(widget_type='ComboBox', label='injection side',
                  choices=['','left', 'right', 'both'], value='',
                  tooltip='select the injection hemisphere (if applicable)'),
    geno=dict(widget_type='LineEdit', label='genotype',
              tooltip='enter the genotype of the animal (be CONSISTENT in your naming across animals)'),
    # todo function to check for genos?
    chans_imaged=dict(widget_type='Select', label='imaged channels', choices=['dapi', 'green', 'cy3', 'cy5'],
                      value=['green', 'cy3'],
                      tooltip='select all channels imaged, to select multiple hold ctrl/shift'),
    call_button=False
)
def params_widget(
    input_path,  # posix path
    inj_side,
    geno,
    chans_imaged
) -> None:

    return params_widget



class ParamsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        params = params_widget
        btn = QPushButton("create params.json file")
        btn.clicked.connect(self._create_params_file)

        self.layout().addWidget(params.native)
        self.layout().addWidget(btn)


    def _create_params_file(self):
        input_path = params_widget.input_path.value
        animal_id = get_animal_id(input_path)
        injection_side = params_widget.inj_side.value
        genotype = params_widget.geno.value
        chans_imaged = params_widget.chans_imaged.value
        params_dict = {
            "general": {
                "animal_id": animal_id,
                "injection_side": injection_side,
                "genotype": genotype,
                "chans_imaged": chans_imaged
            }
        }
        params_dict = clean_params_dict(params_dict, "general")
        params_fn = input_path.joinpath('params.json')
        params_dict = update_params_dict(input_path, params_dict)
        with open(params_fn, 'w') as fn:
            json.dump(params_dict, fn, indent=4)

