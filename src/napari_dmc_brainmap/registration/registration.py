from magicgui import magicgui
import json
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari_dmc_brainmap.utils import get_info
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer

@magicgui(
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                      tooltip='directory of folder containing subfolders with stitched images, '
                              'NOT folder containing stitched images itself'),
    regi_chan=dict(widget_type='ComboBox', label='registration channel',
                    choices=['dapi', 'green', 'n3', 'cy3', 'cy5'], value='green',
                    tooltip="select the registration channel (images need to be in sharpy track folder)"),

    call_button=False
)
def header_widget(
        self,
        input_path,
        regi_chan
):
    return header_widget



class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        header = header_widget
        btn = QPushButton("start registration GUI")
        btn.clicked.connect(self._start_sharpy_track)

        self.layout().addWidget(header.native)
        self.layout().addWidget(btn)

    def create_regi_dict(self):
        input_path = header_widget.input_path.value
        regi_chan = header_widget.regi_chan.value
        regi_dir = get_info(input_path, 'sharpy_track', channel=regi_chan, only_dir=True)
        params_fn = input_path.joinpath('params.json')
        with open(params_fn) as fn:
            params_dict = json.load(fn)


        regi_dict = {
            'input_path': input_path,
            'regi_dir': regi_dir,
            'atlas': params_dict['sharpy_track_params']['atlas'],
            'xyz_dict': params_dict['sharpy_track_params']['xyz_dict']
        }

        return regi_dict

    def _start_sharpy_track(self):
        # todo think about solution to check and load atlas data

        regi_dict = self.create_regi_dict()

        reg_viewer = RegistrationViewer(self.viewer, regi_dict)
        reg_viewer.show()
