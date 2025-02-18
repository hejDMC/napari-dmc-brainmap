# tract_calculation.py as a class
from natsort import natsorted
import json
import numpy as np
import pandas as pd
import distinctipy
import matplotlib.axes as mpl_axes
import matplotlib.pyplot as plt
from skspatial.objects import Line, Points
from typing import List, Tuple
from pathlib import Path

from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.general_utils import get_animal_id
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, coord_mm_transform
from napari_dmc_brainmap.results.results_helpers.slice_handle import SliceHandle

class TractCalculator:
    """
    Class for calculating probe tracts from segmentation data, including brain region certainty,
    voxelization, and visualization of probe paths.
    """
    ABC_LIST = ['a_coord', 'b_coord', 'c_coord']
    def __init__(self, s: SliceHandle, input_path: Path, seg_type: str, probe_insert: List[int]) -> None:
        """
        Initialize the TractCalculator.

        Parameters:
            s (SliceHandle): Instance of the SliceHandle object.
            input_path (Path): Path to the input directory.
            seg_type (str): Type of segmentation (e.g., "neuropixels_probe").
            probe_insert (List[int]): List of probe insertion depths.
        """
        self.atlas = s.atlas
        self.annot = self.atlas.annotation
        self.input_path = input_path
        self.seg_type = seg_type
        self.probe_insert = probe_insert
        self.results_dir = get_info(self.input_path, 'results', seg_type=self.seg_type, only_dir=True)

    def calculate_probe_tract(self) -> None:
        """
        Calculate and save probe tracts based on segmentation and atlas data.
        """
        probes_list = natsorted([p.parts[-1] for p in self.results_dir.iterdir() if p.is_dir()])
        probes_dict = {}
        ax_map = {'ap': 'AP', 'si': 'DV', 'rl': 'ML'}
        show_info("calculating probe tract for...")
        for i in range(len(probes_list) - len(self.probe_insert)):
            show_info(
                "Warning -- less manipulator values than probes provides, estimation of probe track from clicked points "
                "is still experimental!")
            self.probe_insert.append(False)

        for probe, p_insert in zip(probes_list, self.probe_insert):
            show_info(f"... {probe}")
            probe_df = self._load_probe_data(probe)

            linefit, linevox, primary_axis_idx = self._get_linefit3d(probe_df)
            probe_tract, _col_names = self._get_probe_tract(primary_axis_idx, probe_df, probe,
                                                            p_insert, linefit, linevox)
            # save probe tract data
            animal_id = get_animal_id(self.input_path)
            save_fn = self.results_dir.joinpath(probe, f'{animal_id}_{self.seg_type}.csv')  # override file
            probe_tract.to_csv(save_fn)
            probes_dict[probe] = {'axis': ax_map[self.atlas.space.axes_description[primary_axis_idx]]}
            probes_dict[probe]['Voxel'] = linevox[self.ABC_LIST].to_numpy().tolist()
        save_fn = self.results_dir.joinpath(f'{self.seg_type}_data.json')
        with open(save_fn, 'w') as f:
            json.dump(probes_dict, f)  # write multiple voxelized probes, file can be opened in probe vis_plots

    def _load_probe_data(self, probe: str) -> pd.DataFrame:
        """
        Load probe data from a CSV file.

        Parameters:
            probe (str): Name of the probe.

        Returns:
            pd.DataFrame: Dataframe containing probe data.
        """
        data_dir = self.results_dir.joinpath(probe)
        data_fn = list(data_dir.glob('*csv'))[0]
        probe_df = pd.read_csv(data_fn)
        name_dict = {'ap': 'ap_coords', 'si': 'dv_coords', 'rl': 'ml_coords'}
        for atlas_ax, abc_name in zip(self.atlas.space.axes_description, self.ABC_LIST):
            probe_df[abc_name] = probe_df[name_dict[atlas_ax]].copy()
        return probe_df



    def _get_primary_axis_idx(self, direction_vector: np.ndarray) -> int:
        """
        Determine the primary axis based on the largest component of the direction vector.

        Parameters:
            direction_vector (np.ndarray): Vector representing the direction of the probe.

        Returns:
            int: Index of the primary axis.
        """
        direction_comp = np.abs(direction_vector)
        direction_comp[1] += 1e-10  # 1st default, DV axis
        direction_comp[0] += 1e-11  # 2nd default, AP axis
        return direction_comp.argmax()  # select biggest component as primary axis

    def _get_voxelized_coord(self, primary_axis_idx: int, line_object: Line) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate voxelized coordinates along the probe line.

        Parameters:
            primary_axis_idx (int): Index of the primary axis.
            line_object (Line): Line object representing the probe's trajectory.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Voxel coordinates along the probe.
        """
        z = np.arange(self.atlas.shape[primary_axis_idx])
        lamb = (z - line_object.point[primary_axis_idx]) / line_object.direction[primary_axis_idx]

        y_idx, x_idx = self.atlas.space.index_pairs[primary_axis_idx]
        x = (line_object.point[x_idx] + lamb * line_object.direction[x_idx]).astype(int)
        y = (line_object.point[y_idx] + lamb * line_object.direction[y_idx]).astype(int)

        if primary_axis_idx == 0:
            a, b, c = z, y, x
        elif primary_axis_idx == 1:
            a, b, c = y, z, x
        else:
            a, b, c = y, x, z

        x[x >= self.atlas.shape[x_idx]] = self.atlas.shape[x_idx] - 1
        y[y >= self.atlas.shape[y_idx]] = self.atlas.shape[y_idx] - 1
        return a, b, c

    def _get_certainty_list(self, probe_tract: pd.DataFrame, col_names: List[str]) -> List[float]:
        """
        Calculate certainty scores for each voxel in the probe tract.

        Parameters:
            probe_tract (pd.DataFrame): Dataframe containing the probe tract.
            col_names (List[str]): List of column names for coordinates.

        Returns:
            List[float]: Certainty scores for each voxel.
        """

        # calculate certainty value
        check_size = 3  # [-3,-2,-1,0,1,2,3] # check neibouring (n*2+1)**3 voxels , only odd numbers here

        d_a, d_b, d_c = np.meshgrid(np.arange(-check_size, check_size + 1, 1),
                                    np.arange(-check_size, check_size + 1, 1),
                                    np.arange(-check_size, check_size + 1, 1))

        certainty_list = []

        for row in range(len(probe_tract)):
            nA = probe_tract[col_names[0]][row] + d_a.ravel()
            nB = probe_tract[col_names[1]][row] + d_b.ravel()
            nC = probe_tract[col_names[2]][row] + d_c.ravel()
            # handle outlier voxels
            outlierA = np.where((nA < 0) | (nA > (self.annot.shape[0] - 1)))[0].tolist()
            outlierB = np.where((nB < 0) | (nB > (self.annot.shape[1] - 1)))[0].tolist()
            outlierC = np.where((nC < 0) | (nC > (self.annot.shape[2] - 1)))[0].tolist()
            if len(outlierA) + len(outlierB) + len(outlierC) == 0:  # no outlier voxel
                voxel_reduce = 0
            else:  # has out of range voxels
                i_to_remove = np.unique(outlierA + outlierB + outlierC)  # get voxel to remove index
                nA = np.delete(nA, i_to_remove)  # remove
                nB = np.delete(nB, i_to_remove)  # from
                nC = np.delete(nC, i_to_remove)  # index lists
                voxel_reduce = len(i_to_remove)  # reduce demoninator at certainty

            structures_neighbor = self.annot[nA, nB, nC]  # get structure_ids of all neighboring voxels, except outliers
            structure_id = self.annot[probe_tract[col_names[0]][row],
                                 probe_tract[col_names[1]][row],
                                 probe_tract[col_names[2]][row]]  # center voxel
            uni, count = np.unique(structures_neighbor, return_counts=True)  # summarize neibouring structures
            try:
                certainty = dict(zip(uni, count))[structure_id] / (
                        (check_size * 2 + 1) ** 3 - voxel_reduce)  # calculate certainty score
                certainty_list.append(certainty)
            except KeyError:
                certainty_list.append(0)

        return certainty_list

    def _estimate_confidence(self, v_coords: pd.DataFrame, atlas_resolution_um: float) -> np.ndarray:
        """
        Estimate confidence values for probe locations based on neighboring voxels.

        Parameters:
            v_coords (pd.DataFrame): Voxel coordinates of the probe.
            atlas_resolution_um (float): Atlas resolution in micrometers.

        Returns:
            np.ndarray: Array of confidence values.
        """
        # calculate r<=10 sphere
        cube_10 = np.array(np.meshgrid(np.arange(-10, 11, 1),
                                       np.arange(-10, 11, 1),
                                       np.arange(-10, 11, 1))).T.reshape(9261, 3)
        d = np.sqrt((cube_10 ** 2).sum(axis=1))
        sphere_10 = cube_10[d <= 10]  # filter with r<=10, 4169 voxels

        confidence_list = []
        for _, row in v_coords.iterrows():
            c1, c2, c3 = row.values
            current_id = self.annot[c1, c2, c3]  # electrode structure_id
            # restrict view to r=10 voxels sphere space
            within_sphere = np.tile(np.array([c1, c2, c3]), (4169, 1)) + sphere_10
            sphere_struct = self.annot[within_sphere.T[0], within_sphere.T[1], within_sphere.T[2]]
            struct_else = (sphere_struct != current_id)
            if np.sum(struct_else) == 0:
                confidence_list.append(10 * atlas_resolution_um)
            else:
                confidence_list.append(np.sqrt((((within_sphere[struct_else] - np.tile(np.array([c1, c2, c3]), (
                np.sum(struct_else), 1)))) ** 2).sum(axis=1)).min() * atlas_resolution_um)
        confidence_list = np.array(confidence_list, dtype=np.uint8)
        return confidence_list

    def _check_probe_insert(
        self, probe_df: pd.DataFrame, probe_insert: int, linefit: pd.DataFrame, surface_vox: np.ndarray, primary_axis_idx: int
    ) -> Tuple[int, np.ndarray]:
        """
        Validate and adjust the probe insertion depth and direction.

        Parameters:
            probe_df (pd.DataFrame): Dataframe of probe data.
            probe_insert (int): Probe insertion depth.
            linefit (pd.DataFrame): Line fit data for the probe.
            surface_vox (np.ndarray): Voxel coordinates at the brain surface.
            primary_axis_idx (int): Index of the primary axis.

        Returns:
            Tuple[int, np.ndarray]: Adjusted probe insertion depth and direction vector.
        """

        resolution = self.atlas.resolution[primary_axis_idx]
        # get probe tip coordinate
        # get direction unit vector
        direction_vec = linefit['direction'].values  # line direction vector
        direction_unit = direction_vec / np.linalg.norm(direction_vec)  # scale direction vector to length 1
        # Rules to flip direction unit
        if primary_axis_idx == 1:  # DV axis is primary axis
            if direction_unit[1] < 0:
                direction_unit = -direction_unit
            else:  # future add more here
                pass
        else:
            pass

        # read probe depth from histology evidence  todo delete this?
        if not probe_insert:
            show_info('manipulator readout not provided, using histology.')
            scatter_vox = np.array(probe_df[self.ABC_LIST].values)

            # calculate scatter projection on fit line
            projection_online = np.matmul(
                np.expand_dims(np.dot(scatter_vox - linefit['point'].values, direction_unit), 1),
                np.expand_dims(direction_unit, 0)) + linefit['point'].values

            # calculate distance of projection from surface
            dist_to_surface = ((projection_online.T[0] - surface_vox[0]) ** 2 +
                               (projection_online.T[1] - surface_vox[1]) ** 2 +
                               (projection_online.T[2] - surface_vox[2]) ** 2) ** 0.5

            furthest_um = np.max(dist_to_surface) * resolution  # convert voxel to um, 10um/voxel
            probe_insert = int(furthest_um)
        else:
            pass

        return probe_insert, direction_unit



    def _get_linefit3d(self, probe_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Calculate a 3D line fit for the probe data.

        Parameters:
            probe_df (pd.DataFrame): Dataframe of probe data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, int]: Line fit data, voxelized line data, and primary axis index.
        """
        points = Points(probe_df[self.ABC_LIST].values)
        line = Line.best_fit(points)
        linefit = pd.DataFrame({'point': line.point, 'direction': line.direction})
        primary_axis_idx = self._get_primary_axis_idx(line.direction)
        voxel_line = np.array(self._get_voxelized_coord(primary_axis_idx, line)).T
        linevox = pd.DataFrame(voxel_line, columns=self.ABC_LIST)
        return linefit, linevox, primary_axis_idx

    def _get_probe_tract(
        self, primary_axis_idx: int, probe_df: pd.DataFrame, probe: str, probe_insert: int, linefit: pd.DataFrame, linevox: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Calculate the probe tract including coordinates and depth information.

        Parameters:
            primary_axis_idx (int): Index of the primary axis.
            probe_df (pd.DataFrame): Dataframe of probe data.
            probe (str): Name of the probe.
            probe_insert (int): Probe insertion depth.
            linefit (pd.DataFrame): Line fit data for the probe.
            linevox (pd.DataFrame): Voxelized line data.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Probe tract dataframe and column names for coordinates.
        """
        # find brain surface
        bregma = get_bregma(self.atlas.atlas_name)
        a, b, c = self.ABC_LIST
        structure_id_list = self.annot[linevox[a], linevox[b], linevox[c]]
        structure_split = np.split(structure_id_list, np.where(np.diff(structure_id_list))[0] + 1)
        surface_index = len(structure_split[0])
        surface_vox = linevox.iloc[surface_index, :].values
        probe_insert, direction_unit = self._check_probe_insert(probe_df, probe_insert, linefit, surface_vox, primary_axis_idx)
        probe_tract = pd.DataFrame()
        # todo: for now only bank 0
        bank = 0  # specify recording bank here, if other than 0
        if bank == 0:
            l_chan = np.arange(1, 384, 2)
            r_chan = np.arange(2, 385, 2)
            dtt_offset = 0
        elif bank == 1:
            l_chan = np.arange(385, 768, 2)
            r_chan = np.arange(386, 769, 2)
            dtt_offset = 3840
        elif bank == 2:
            l_chan = np.arange(769, 1152, 2)
            r_chan = np.arange(770, 1153, 2)
            dtt_offset = 7680
        else:
            show_info('Invalid bank number, using bank 0')
            l_chan = np.arange(1, 384, 2)
            r_chan = np.arange(2, 385, 2)
            dtt_offset = 0

        probe_tract['channel_l'] = l_chan
        probe_tract['channel_r'] = r_chan
        probe_tract['distance_to_tip(um)'] = np.arange(192) * 20 + 175 + dtt_offset  #
        probe_tract['depth(um)'] = probe_insert - probe_tract['distance_to_tip(um)']
        probe_tract['inside_brain'] = (probe_tract['depth(um)'] >= -5)
        name_dict = {
            'ap': 'ap',
            'si': 'dv',
            'rl': 'ml'
        }
        col_names = []
        col_names.extend([name_dict[n] + '_coords' for n in self.atlas.space.axes_description])
        probe_tract[col_names] = pd.DataFrame(np.round(
            np.dot(
                np.expand_dims(
                    (probe_insert - probe_tract['distance_to_tip(um)'].values) / 10, axis=1),
                # manipulator_readout - [distance_to_tip] = distance_from_surface
                np.expand_dims(direction_unit, axis=0))
            + surface_vox).astype(int))
        probe_tract[[name_dict[n] + '_mm' for n in self.atlas.space.axes_description]] = \
            probe_tract.apply(lambda row: pd.Series(coord_mm_transform([row[col_names[0]],
                                                                        row[col_names[1]],
                                                                        row[col_names[2]]],
                                                                       bregma, self.atlas.space.resolution)), axis=1)

        probe_tract['structure_id'] = self.annot[
            probe_tract[col_names[0]], probe_tract[col_names[1]], probe_tract[col_names[2]]]

        # read df_tree
        df_tree = self.atlas.structures

        probe_tract['acronym'] = [df_tree.data[i]['acronym'] if i > 0 else 'root' for i in probe_tract['structure_id']]
        probe_tract['name'] = [df_tree.data[i]['name'] if i > 0 else 'root' for i in probe_tract['structure_id']]

        # certainty_list = _get_certainty_list(probe_tract, annot, col_names)
        probe_tract['distance_to_nearest_structure(um)'] = self._estimate_confidence(v_coords=probe_tract[[col_names[0],
                                                                                                           col_names[1],
                                                                                                           col_names[2]]],
                                                                                     atlas_resolution_um=self.atlas.resolution[primary_axis_idx])
        if self.seg_type == "neuropixels_probe":
            self._save_probe_tract_fig(probe, probe_tract, bank)
        else:
            probe_tract['depth(um)'] += probe_tract['distance_to_tip(um)']
            probe_tract = probe_tract.drop(columns=['channel_l', 'channel_r', 'distance_to_tip(um)'])
        return probe_tract, col_names

    def _save_probe_tract_fig(self, probe: str, probe_tract: pd.DataFrame, bank: int = 0) -> None:
        """
        Generate and save a visualization of the probe tract.

        Parameters:
            probe (str): Name of the probe.
            probe_tract (pd.DataFrame): Dataframe containing the probe tract.
            bank (int, optional): Recording bank identifier. Defaults to 0.
        """
        animal_id = get_animal_id(self.input_path)

        # Prepare data for plotting
        df_plot = probe_tract.copy()
        unique_regions = df_plot['structure_id'].unique()
        colors = distinctipy.get_colors(len(unique_regions))
        reg_color_map = dict(zip(unique_regions, colors))

        region_splits = np.split(
            df_plot['structure_id'].values,
            np.where(np.diff(df_plot['structure_id'].values))[0] + 1
        )

        # Create the plot
        fig, ax = plt.subplots(figsize=(5, 10))
        y2_ticks, y2_labels = [], []
        chan_row_idx = 0

        for region in region_splits:
            # Get acronym and region-specific values
            acro = df_plot['acronym'].iloc[chan_row_idx]
            region_depths = df_plot['distance_to_tip(um)'].iloc[chan_row_idx: chan_row_idx + len(region)] + 15
            confidence_vals = df_plot['distance_to_nearest_structure(um)'].iloc[
                              chan_row_idx: chan_row_idx + len(region)]

            # Fill region with color
            ax.fill_betweenx(region_depths, confidence_vals, color=reg_color_map[region[0]], zorder=2)

            # Update y-axis ticks and labels
            mid_point = (region_depths.iloc[0] + region_depths.iloc[-1]) / 2
            y2_ticks.append(mid_point)
            y2_labels.append(acro)

            chan_row_idx += len(region)

        # Configure primary y-axis
        _configure_primary_y_axis(ax, df_plot)

        # Configure secondary y-axis
        ax2 = ax.twinx()
        ax2.set_yticks(y2_ticks)
        ax2.set_yticklabels(y2_labels, fontsize=10)
        _configure_secondary_y_axis(ax2)

        # Add title, labels, and grid lines
        ax.set_xlabel('Confidence (um)', fontsize=13)
        ax.set_title(f'Animal: {animal_id}', fontsize=15)
        # for ytick in y2_ticks:
        #     ax.axhline(y=ytick, linestyle='--', color='lightgray', linewidth=1, zorder=0)

        # Save the figure
        plt.tight_layout()
        save_path = self.results_dir.joinpath(f"{probe}.svg")
        fig.savefig(save_path, dpi=400)
        plt.close(fig)

def _configure_primary_y_axis(ax: mpl_axes.Axes, df_plot: pd.DataFrame) -> None:
    """
    Configures the primary y-axis for the plot.

    Parameters:
    ax (mpl_axes.Axes): The matplotlib Axes object to configure.
    df_plot (pd.DataFrame): A pandas DataFrame containing the data for the plot.
                            Must include columns 'depth(um)' and 'distance_to_tip(um)'.

    Behavior:
    - Sets y-ticks and y-tick labels based on the depth values in the DataFrame.
    - Customizes y-axis label, limits, and appearance by hiding certain spines and disabling tick lengths.
    """
    depth_max = np.ceil(df_plot['depth(um)'].max() / 1000) * 1000
    yticklabels = np.arange(0, depth_max + 1000, 1000).astype(int)

    yticks = (df_plot["distance_to_tip(um)"] + df_plot["depth(um)"]).iloc[0] - yticklabels + 15
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=10)
    ax.set_ylabel('Depth from dura (um)', fontsize=13)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 4015)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', length=0)

def _configure_secondary_y_axis(ax: mpl_axes.Axes) -> None:
    """
    Configures the secondary y-axis for the plot.

    Parameters:
    ax (mpl_axes.Axes): The matplotlib Axes object to configure.

    Behavior:
    - Sets x and y axis limits.
    - Hides the left, right, and top spines.
    - Disables y-axis tick lengths.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 4015)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='y', length=0)