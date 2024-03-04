from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from napari.qt.threading import thread_worker

import numpy as np
import pandas as pd
import json
from natsort import natsorted
from skspatial.objects import Line, Points # scikit-spatial package: https://scikit-spatial.readthedocs.io/en/stable/
from napari_dmc_brainmap.probe_visualizer.probe_vis.probe_vis.view.ProbeVisualizer import ProbeVisualizer
from napari_dmc_brainmap.probe_visualizer.probe_visualizer_tools import get_primary_axis_idx, get_voxelized_coord, \
    estimate_confidence, check_probe_insert, save_probe_tract_fig
from napari_dmc_brainmap.utils import get_info, load_params, split_to_list

from bg_atlasapi import BrainGlobeAtlas


def load_probe_data(results_dir, probe, atlas):

    data_dir = results_dir.joinpath(probe)
    data_fn = list(data_dir.glob('*csv'))[0]
    probe_df = pd.read_csv(data_fn)
    name_dict = {
        'ap': 'ap_coords',
        'si': 'dv_coords',
        'rl': 'ml_coords'
    }
    abc_list = ['a_coord', 'b_coord', 'c_coord']
    # append columns (a,b,c) in case atlas ordering is different that ap, dv, ml (e.g. dv, ml, ap)
    for atlas_ax, abc_name in zip(atlas.space.axes_description, abc_list):
        probe_df[abc_name] = probe_df[name_dict[atlas_ax]].copy()
    print(probe_df)
    return probe_df


def get_linefit3d(probe_df, atlas):


    points = Points(probe_df[['a_coord', 'b_coord', 'c_coord']].values)
    line = Line.best_fit(points)  # fit 3D line
    linefit = pd.DataFrame()
    linefit['point'] = line.point  # add point coordinates to dataframe
    linefit['direction'] = line.direction  # add direction vector coordinates to dataframe
    primary_axis_idx = get_primary_axis_idx(line.direction)  # get primary axis index: 0:a, 1:b, 2:c
    voxel_line = np.array(get_voxelized_coord(primary_axis_idx, line, atlas)).T  # voxelize

    linevox = pd.DataFrame(voxel_line, columns=['a_coord', 'b_coord', 'c_coord'])  # add to dataframe
    return linefit, linevox, primary_axis_idx


def get_probe_tract(input_path, save_path, atlas, ax_primary, probe_df, probe, probe_insert, linefit, linevox):

    # find brain surface
    annot = atlas.annotation
    structure_id_list = annot[linevox['a_coord'], linevox['b_coord'], linevox['c_coord']]
    structure_split = np.split(structure_id_list, np.where(np.diff(structure_id_list))[0] + 1)
    surface_index = len(structure_split[0])  # get index of first non-root structure
    surface_vox = linevox.iloc[surface_index, :].values  # get brain surface voxel coordinates
    probe_insert, direction_unit = check_probe_insert(probe_df, probe_insert, linefit, surface_vox,
                                                      atlas.resolution[ax_primary],ax_primary)
    # create probe information dataframe
    probe_tract = pd.DataFrame()
    # create electrode left and right channel columns
    in_brain_chan = round(probe_insert/10)
    if in_brain_chan < 385:
        l_chan = 384
        r_chan = 385
    else:
        if in_brain_chan % 2 == 0:
            l_chan = in_brain_chan
            r_chan = l_chan + 1
        else:
            r_chan = in_brain_chan
            l_chan = r_chan - 1
    probe_tract['Channel_L'] = np.arange(1, l_chan, 2)
    probe_tract['Channel_R'] = np.arange(2, r_chan, 2)
    probe_tract['Distance_To_Tip(um)'] = np.arange(0, int(l_chan/2) * 20, 20) + 185  # 185 = 175 + 10 (electrode center to bottom)
    probe_tract['Recorded_Channels'] = [True] * int(384/2) + [False] * (int(l_chan/2)-int(384/2))
    probe_tract['Inside_Brain'] = probe_tract[
                                 'Distance_To_Tip(um)'] <= probe_insert  # change according to manipulator readout
    name_dict = {
        'ap': 'AP',
        'si': 'DV',
        'rl': 'ML'
    }
    col_names = []
    col_names.extend(['Voxel_' + name_dict[n] for n in atlas.space.axes_description])
    probe_tract[col_names] = pd.DataFrame(np.round(
        np.dot(
            np.expand_dims(
                (probe_insert - probe_tract['Distance_To_Tip(um)'].values) / 10, axis=1),
            # manipulator_readout - [distance_to_tip] = distance_from_surface
            np.expand_dims(direction_unit, axis=0))
        + surface_vox).astype(int))

    probe_tract['structure_id'] = annot[probe_tract[col_names[0]], probe_tract[col_names[1]], probe_tract[col_names[2]]]

    # read df_tree
    df_tree = atlas.structures

    probe_tract['Acronym'] = [df_tree.data[i]['acronym'] if i > 0 else 'root' for i in probe_tract['structure_id']]
    probe_tract['Name'] = [df_tree.data[i]['name'] if i > 0 else 'root' for i in probe_tract['structure_id']]

    #certainty_list = get_certainty_list(probe_tract, annot, col_names)
    probe_tract['Certainty'] = estimate_confidence(v_coords = probe_tract[[col_names[0],
                                                                           col_names[1],
                                                                           col_names[2]]],
                                                    atlas_resolution_um = 10,
                                                    annot = annot)




    save_probe_tract_fig(input_path, probe, save_path, probe_tract)
    return probe_tract, col_names


@thread_worker
def calculate_probe_tract(input_path, save_path, params_dict, probe_insert):
    # get number of probes
    results_dir = get_info(input_path, 'results', seg_type='neuropixels_probe', only_dir=True)
    probes_list = natsorted([p.parts[-1] for p in results_dir.iterdir() if p.is_dir()])
    probes_dict = {}
    ax_map = {'ap': 'AP', 'si': 'DV', 'rl': 'ML'}
    # if len(probe_insert) != len(probes_list):
    #     print("WARNING, different number of probes and probe insert lengths detected!")
    #     diff = len(probe_insert) - len(probes_list)
    #     if diff < 0:
    #         for d in range(diff):
    #             probe_insert.append(4000)
    print("loading reference atlas...")
    atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
    print("calculating probe tract for...")
    for i in range(len(probes_list) - len(probe_insert)):  # append false value if less probe insert values that probes found
        print("Warning -- less manipulator values than probes provides, estimation of probe track from clicked points "
              "is still experimental!")
        probe_insert.append(False)

    for probe, p_insert in zip(probes_list, probe_insert):
        print("... " + probe)
        probe_df = load_probe_data(results_dir, probe, atlas)

        linefit, linevox, ax_primary = get_linefit3d(probe_df, atlas)
        print(p_insert)
        probe_tract, col_names = get_probe_tract(input_path, save_path, atlas, ax_primary, probe_df, probe,
                                                 p_insert, linefit, linevox)
        # save probe tract data
        save_fn = save_path.joinpath(probe + '_data.csv')
        probe_tract.to_csv(save_fn)
        probes_dict[probe] = {'axis': ax_map[atlas.space.axes_description[ax_primary]]}
        probes_dict[probe]['Voxel'] = probe_tract[col_names].to_numpy().tolist()

    save_fn = save_path.joinpath('neuropixels_probes_data.json')
    with open(save_fn, 'w') as f:
        json.dump(probes_dict, f)  # write multiple voxelized probes, file can be opened in probe visualizer
    print("DONE!")



def initiate_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
              save_path=dict(widget_type='FileEdit', 
                             label='save path: ', 
                             mode='d',
                             value='',
                             tooltip='select a folder for saving plots, default will save in *input path*'),
              probe_insert=dict(widget_type='LineEdit', 
                                label='insertion depth of probe (um)', 
                                value='4000',
                                tooltip='specifiy the depth of neuropixels probe in brain in um'),
              call_button=False)
    
    def probe_visualizer(
        input_path,  # posix path
        save_path,
        animal_list,
        probe_insert):
        pass
    return probe_visualizer


class ProbeVisualizerWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.p_vis = initiate_widget()

        btn_calc_probe = QPushButton("calculate probe tract")
        btn_calc_probe.clicked.connect(self._calculate_probe_tract)

        btn_probe_vis = QPushButton("start probe visualizer")
        btn_probe_vis.clicked.connect(self._start_probe_visualizer)


        self.layout().addWidget(self.p_vis.native)
        self.layout().addWidget(btn_calc_probe)
        self.layout().addWidget(btn_probe_vis)


    def _calculate_probe_tract(self):
        input_path = self.p_vis.input_path.value
        if str(self.p_vis.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.p_vis.save_path.value

        probe_insert = split_to_list(self.p_vis.probe_insert.value, out_format='int')  # [int(i) for i in probe_visualizer.probe_insert.value.split(',')]
        if not probe_insert:
            probe_insert = [probe_insert]  # make list if not already
        params_dict = load_params(input_path)
        pt_worker = calculate_probe_tract(input_path, save_path, params_dict, probe_insert)
        pt_worker.start()


    def _start_probe_visualizer(self):
        input_path = self.p_vis.input_path.value
        params_dict = load_params(input_path)
        probe_vis = ProbeVisualizer(self.viewer, params_dict)
        probe_vis.show()
