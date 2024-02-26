
from PyQt5.QtWidgets import QApplication
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer
from napari_dmc_brainmap.utils import create_regi_dict
import sys
from pathlib import Path

app = QApplication(sys.argv)

class NapariProxy():
    def __init__(self,proxy) -> None:
        self.viewer = proxy
    
    def del_regviewer_instance(self):
        pass

napari_proxy = NapariProxy(app)

regi_dict = create_regi_dict(Path("C:\\Users\\xiao\\histology_data\\DP-411"),
                             "green")
# regi_dict = create_regi_dict(Path("C:\\Users\\xiao\\histology_data\\ZF-000"),
#                              "green")
# regi_dict = create_regi_dict(Path("C:\\Users\\xiao\\histology_data\\790311"),
#                              "green")

reg_viewer = RegistrationViewer(napari_proxy, regi_dict)
reg_viewer.show()

sys.exit(app.exec_())

