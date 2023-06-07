from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from napari.qt.threading import thread_worker

import numpy as np
import pandas as pd
import json
from skspatial.objects import Line, Points # scikit-spatial package: https://scikit-spatial.readthedocs.io/en/stable/
from napari_dmc_brainmap.probe_visualizer.probe_vis.probe_vis.view.ProbeVisualizer import ProbeVisualizer
from napari_dmc_brainmap.probe_visualizer.probe_visualizer_tools import get_primary_axis, get_voxelized_coord, \
    get_certainty_list, check_probe_insert, save_probe_tract_fig
from napari_dmc_brainmap.visualization.visualization_tools import dummy_load_allen_annot, dummy_load_allen_structure_tree
from napari_dmc_brainmap.utils import get_info


def load_probe_data(results_dir, probe):

    data_dir = results_dir.joinpath(probe)
    data_fn = list(data_dir.glob('*csv'))[0]
    probe_df = pd.read_csv(data_fn)

    return probe_df


def get_linefit3d(probe_df):

    points = Points(probe_df[['zpixel', 'ypixel', 'xpixel']].values)  # AP, DV,  ML
    line = Line.best_fit(points)  # fit 3D line

    linefit = pd.DataFrame()
    linefit['point'] = line.point  # add point coordinates to dataframe
    linefit['direction'] = line.direction  # add direction vector coordinates to dataframe

    # ax_primary = get_primary_axis(line.direction)  # get primary axis  todo:function doesn't work?
    ax_primary = 1  # 0:AP, 1:DV, 2:ML
    voxel_line = np.array(get_voxelized_coord(ax_primary, line)).T  # voxelize
    linevox = pd.DataFrame(voxel_line, columns=['xpixel', 'ypixel', 'zpixel'])  # add to dataframe  # todo here I swapped z and x

    return linefit, linevox, ax_primary

def get_probe_tract(input_path, save_path, probe, probe_insert, linefit, linevox):
    # find brain surface
    annot = dummy_load_allen_annot()
    sphinxID_list = annot[linevox['zpixel'], linevox['ypixel'], linevox['xpixel']]
    sphinx_split = np.split(sphinxID_list, np.where(np.diff(sphinxID_list))[0] + 1)
    surface_index = len(sphinx_split[0])  # get index of first non-root structure
    surface_vox = linevox.iloc[surface_index, :].values  # get brain surface voxel coordinates

    probe_insert, direction_unit = check_probe_insert(probe_insert, linefit, surface_vox)

    # create probe information dataframe
    probe_tract = pd.DataFrame()
    # create electrode left and right channel columns
    probe_tract['Channel_L'] = np.arange(1, 384, 2)
    probe_tract['Channel_R'] = np.arange(2, 385, 2)
    probe_tract['Distance_To_Tip(um)'] = np.arange(0, 192 * 20, 20) + 185  # 185 = 175 + 10 (electrode center to bottom)
    probe_tract['Inside_Brain'] = probe_tract[
                                 'Distance_To_Tip(um)'] <= probe_insert  # change according to manipulator readout
    probe_tract[['Voxel_AP', 'Voxel_DV', 'Voxel_ML']] = pd.DataFrame(np.round(
        np.dot(
            np.expand_dims(
                (probe_insert - probe_tract['Distance_To_Tip(um)'].values) / 10, axis=1),
            # manipulator_readout - [distance_to_tip] = distance_from_surface
            np.expand_dims(direction_unit, axis=0))
        + surface_vox).astype(int))

    probe_tract['SphinxID'] = annot[probe_tract['Voxel_AP'], probe_tract['Voxel_DV'], probe_tract['Voxel_ML']]
    # read df_tree
    df_tree = dummy_load_allen_structure_tree()
    probe_tract['Acronym'] = df_tree.iloc[probe_tract['SphinxID'] - 1, :]['acronym'].values
    probe_tract['Name'] = df_tree.iloc[probe_tract['SphinxID'] - 1, :]['name'].values

    certainty_list = get_certainty_list(probe_tract, annot)
    probe_tract['Certainty'] = certainty_list
    # save_probe_tract_fig(input_path, probe, save_path, probe_tract)
    return probe_tract
    # df_out.to_csv('step4_output_probetrack.csv')  # save probe information csv

@thread_worker
def calculate_probe_tract(input_path, save_path, probe_insert):
    # get number of probes
    results_dir = get_info(input_path, 'results', seg_type='neuropixels_probe', only_dir=True)
    probes_list = [p.parts[-1] for p in results_dir.iterdir() if p.is_dir()]
    probes_dict = {}
    for probe in probes_list:
        probe_df = load_probe_data(results_dir, probe)
        linefit, linevox, ax_primary = get_linefit3d(probe_df)
        # if len(probe) == 800:
        #     ax_primary = 'DV'
        # elif len(probe) == 1320:
        #     ax_primary = 'AP'
        # else:
        #     ax_primary = 'ML'
        # 0: AP, 1: DV, 2: ML
        probe_tract = get_probe_tract(input_path, save_path, probe, probe_insert, linefit, linevox)
        probes_dict[probe] = {'axis': ax_primary}
        probes_dict[probe]['Voxel'] = probe_tract.to_numpy().tolist()


    save_fn = save_path.joinpath('neuropixels_probes_data.json')
    with open(save_fn, 'w') as f:
        json.dump(probes_dict, f)  # write multiple voxelized probes, file can be opened in probe visualizer

@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                            'folder containing segmentation results'),
    save_path=dict(widget_type='FileEdit', label='save path: ', mode='d',
                   value='',
                   tooltip='select a folder for saving plots, default will save in *input path*'),
    probe_insert=dict(widget_type='LineEdit', label='insertion depth of probe (um)', value='4000',
                    tooltip='specifiy the depth of neuropixels probe in brain in um'),
    # ax_primary=dict(widget_type='ComboBox', label='insertion axis',
    #               choices=['AP', 'DV', 'ML'], value='DV',
    #               tooltip=""),
    call_button=False
)
def probe_visualizer(
    input_path,  # posix path
    save_path,
    animal_list,
    probe_insert
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
        pt_worker = calculate_probe_tract(input_path, save_path, probe_insert)
        pt_worker.start()


    def _start_probe_visualizer(self):

        probe_vis = ProbeVisualizer(self.viewer)
        probe_vis.show()
