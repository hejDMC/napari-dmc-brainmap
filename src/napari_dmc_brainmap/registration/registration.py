from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QVBoxLayout, QFileDialog, QLineEdit
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        btn = QPushButton("start registration GUI")
        btn.clicked.connect(self._start_sharpy_track)

        self.layout().addWidget(btn)

    def _start_sharpy_track(self):
        # todo probe_track in sharpy_track
        # todo think about solution to check and load atlas data
        reg_viewer = RegistrationViewer(self.viewer)
        reg_viewer.show()
