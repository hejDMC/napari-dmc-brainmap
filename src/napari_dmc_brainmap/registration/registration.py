from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
from pkg_resources import resource_filename
from pathlib import Path
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer
import sys
from PyQt5.QtWidgets import QApplication
# todo clean import

def registration_widget():
    # todo probe_track in sharpy_track
    from napari.qt.threading import thread_worker

    # @thread_worker
    # def run():
    #     app = QApplication(sys.argv)
    #     RegViewer = RegistrationViewer(app)
    #     RegViewer.show()
    #     sys.exit(app.exec_())

    # todo think about solution to check and load atlas data
    @magicgui(
        layout='vertical',
        call_button='start registration GUI'
    )
    def widget(
            viewer: Viewer
    ) -> None:
        # app = QApplication(sys.argv)
        # RegViewer = RegistrationViewer(app)
        RegViewer = RegistrationViewer(viewer)
        RegViewer.show()
        # viewer.window.add_dock_widget(RegViewer)
        # viewer.window.qt_viewer(RegViewer)
        # sys.exit(app.exec_())
    return widget
