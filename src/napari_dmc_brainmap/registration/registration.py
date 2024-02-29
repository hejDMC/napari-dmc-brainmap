from magicgui import magicgui
from magicgui.widgets import FunctionGui
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari_dmc_brainmap.utils import create_regi_dict
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer


def initialize_widget() -> FunctionGui:
    @magicgui(input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with stitched images, '
                                'NOT folder containing stitched images itself'),
              regi_chan=dict(widget_type='ComboBox', 
                             label='registration channel',
                             choices=['dapi', 'green', 'n3', 'cy3', 'cy5'], 
                             value='green',
                             tooltip="select the registration channel (images need to be in sharpy track folder)"),
              call_button=False)

    def header_widget(
            self,
            input_path,
            regi_chan):
        pass
    return header_widget



class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.header = initialize_widget()
        btn = QPushButton("start registration GUI")
        btn.clicked.connect(self._start_sharpy_track)

        self.layout().addWidget(self.header.native)
        self.layout().addWidget(btn)


    def _start_sharpy_track(self):
        # todo think about solution to check and load atlas data
        input_path = self.header.input_path.value
        regi_chan = self.header.regi_chan.value

        regi_dict = create_regi_dict(input_path, regi_chan)

        self.reg_viewer = RegistrationViewer(self, regi_dict)
        self.reg_viewer.show()


    def del_regviewer_instance(self): # temporary fix for memory leak, maybe not complete, get back to this in the future
        self.reg_viewer.widget.viewerLeft.scene.changed.disconnect()
        self.reg_viewer.widget.viewerRight.scene.changed.disconnect()

        if self.reg_viewer.helperAct.isEnabled():
            pass
        else: # if registration helper is opened, close it too
            self.reg_viewer.helperPage.close()
    
        del self.reg_viewer.regViewerWidget
        del self.reg_viewer.app
        del self.reg_viewer.regi_dict
        del self.reg_viewer.widget
        del self.reg_viewer.status
        del self.reg_viewer.atlasModel
        del self.reg_viewer
        
