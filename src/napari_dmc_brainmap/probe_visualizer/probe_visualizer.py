from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui

from napari_dmc_brainmap.probe_visualizer.probe_vis.probe_vis.view.ProbeVisualizer import ProbeVisualizer

@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                            'folder containing segmentation results'),
    save_path=dict(widget_type='FileEdit', label='save path: ', mode='d',
                   value='',
                   tooltip='select a folder for saving plots, default will save in *input path*'),
    probe_insert=dict(widget_type='LineEdit', label='insertion depth of probe', value='4000',
                    tooltip='specifiy the depth of neuropixels probe in brain'),
    call_button=False
)
def probe_visualizer(
    input_path,  # posix path
    save_path,
    animal_list,
    channels,

) -> None:

    return probe_visualizer


class ProbeVisualizerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        p_vis = probe_visualizer

        btn_calc_probe = QPushButton("calculate probe tract")
        btn_calc_probe.clicked.connect(self._calculate_probe_tract)

        btn_probe_vis = QPushButton("start probe visualizer")
        btn_probe_vis.clicked.connect(self._start_probe_visualizer)


        self.layout().addWidget(p_vis.native)
        self.layout().addWidget(btn_calc_probe)
        self.layout().addWidget(btn_probe_vis)


    def _calculate_probe_tract(self):
        input_path = probe_visualizer.input_path.value
        if not probe_visualizer.save_path.value:
            save_path = input_path
        else:
            save_path = probe_visualizer.save_path.value
        probe_insert = int(probe_visualizer.probe_insert.value)


    def _start_probe_visualizer(self):

        probe_vis = ProbeVisualizer(self.viewer)
        probe_vis.show()
