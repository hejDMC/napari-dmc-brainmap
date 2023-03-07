from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
from pkg_resources import resource_filename
from pathlib import Path
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer
import sys
from PyQt5.QtWidgets import QApplication


def registration_widget():
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
        input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                        tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
        call_button='start registration GUI'
    )

    def widget(
            viewer: Viewer,
            input_path
    )-> None:
        # if not hasattr(widget, 'segment_layers'):
        #     widget.segment_layers = []
        # pp = '/home/felix/Academia/DMC-lab/Projects/dmc-brainmap/napari-dmc-brainmap/src/napari_dmc_brainmap/registration/sharpy_track'
        # import os
        # os.system("python " + pp)
        # app = QApplication(sys.argv)
        # RegViewer = RegistrationViewer(app)
        RegViewer = RegistrationViewer(viewer)
        RegViewer.show()
        # viewer.window.add_dock_widget(RegViewer)
        # viewer.window.qt_viewer(RegViewer)
        # RegViewer.show()
        # sys.exit(app.exec_())
    return widget
