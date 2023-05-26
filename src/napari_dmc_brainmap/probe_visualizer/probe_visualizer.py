from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from napari.qt.threading import thread_worker

import numpy as np
import pandas as pd
from skspatial.objects import Line, Points # scikit-spatial package: https://scikit-spatial.readthedocs.io/en/stable/
from napari_dmc_brainmap.probe_visualizer.probe_vis.probe_vis.view.ProbeVisualizer import ProbeVisualizer
from napari_dmc_brainmap.probe_visualizer.probe_visualizer_tools import get_primary_axis, get_voxelized_coord
from napari_dmc_brainmap.utils import get_info


def get_linefit3d(probe_df):

    points = Points(probe_df[['zpixel', 'ypixel', 'xpixel']].values)  # AP, DV,  ML
    line = Line.best_fit(points)  # fit 3D line

    linefit = pd.DataFrame()
    linefit['point'] = line.point  # add point coordinates to dataframe
    linefit['direction'] = line.direction  # add direction vector coordinates to dataframe

    ax_primary = get_primary_axis(line.direction)  # get primary axis
    voxel_line = np.array(get_voxelized_coord(ax_primary, line)).T  # voxelize
    linevox = pd.DataFrame(voxel_line, columns=['zpixel', 'ypixel', 'xpixel'])  # add to dataframe

    return linefit, linevox


@thread_worker
def calculate_probe_tract(input_path):
    # get number of probes
    results_dir = get_info(input_path, 'results', seg_type='neuropixels_probe', only_dir=True)
    probes_list = [p.parts[-1] for p in results_dir.iterdir() if p.is_dir()]
    for probe in probes_list:

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
