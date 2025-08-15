import concurrent.futures
import json
import math
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui.widgets import FunctionGui
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from shapely.geometry import Polygon
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
from bg_atlasapi import BrainGlobeAtlas
from napari_dmc_brainmap.utils.color_manager import ColorManager
from napari_dmc_brainmap.utils.general_utils import split_to_list
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, get_orient_map, get_xyz
from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import get_unique_folder
from napari_dmc_brainmap.visualization.vis_plots.brainsection_plotter import BrainsectionPlotter
from napari.utils.notifications import show_info

class BrainsectionVisualization:
    """
    Class for visualizing brain sections and generating plots such as schematics, density maps,
    projections, and gene expression visualizations.
    """
    def __init__(
        self,
        input_path: Path,
        atlas: BrainGlobeAtlas,
        data_dict: Dict,
        animal_list: List[str],
        brainsec_widget: FunctionGui,
        save_path: Path,
        gene: str
    ) -> None:
        """
        Initialize the BrainsectionVisualization.

        Parameters:
            input_path (Path): Path to the input directory.
            atlas (BrainGlobeAtlas): BrainGlobeAtlas instance for reference.
            data_dict (Dict): Dictionary containing data to visualize.
            animal_list (List[str]): List of animal IDs.
            brainsec_widget (FunctionGui): Widget for configuring visualization parameters.
            save_path (Path): Directory to save plots and data.
            gene (str): Name of the gene to visualize, if applicable.
        """
        self.input_path = input_path
        self.atlas = atlas
        self.data_dict = data_dict
        self.animal_list = animal_list
        self.save_path = save_path
        self.plotting_params = self._get_brainsec_params(brainsec_widget, gene)
        self.color_manager = ColorManager()
        self.color_dict = self._initialize_color_dict()
        self.orient_mapping = get_orient_map(self.atlas, self.plotting_params)
        self.bregma = get_bregma(self.atlas.atlas_name)
        self.brainsection_plotter = self._initialize_brainsection_plotter()
        self.progress_step = 0
        self.progress_total = None

    def _initialize_color_dict(self) -> Dict:
        """
        Create a color dictionary using the ColorManager.

        Returns:
            Dict: Dictionary of colors for various visualization elements.
        """
        return self.color_manager.create_color_dict(
            self.input_path,
            self.animal_list,
            self.data_dict,
            self.plotting_params
        )

    def _initialize_brainsection_plotter(self) -> BrainsectionPlotter:
        """
        Initialize the BrainsectionPlotter instance for section plotting.

        Returns:
            BrainsectionPlotter: Instance of BrainsectionPlotter.
        """
        return BrainsectionPlotter(
            self.atlas,
            self.plotting_params,
            self.data_dict,
            self.color_manager,
            self.color_dict
        )

    def _get_brainsec_params(self, brainsec_widget: FunctionGui, gene: str) -> Dict:
        """
        Extract visualization parameters from the widget.

        Parameters:
            brainsec_widget (FunctionGui): Widget containing user configurations.
            gene (str): Name of the gene to visualize.

        Returns:
            Dict: Dictionary of visualization parameters.
        """
        plotting_params = {
            # "figsize": split_to_list(brainsec_widget.plot_size.value, out_format='int'),
            "section_orient": brainsec_widget.section_orient.value,
            "plot_item": brainsec_widget.plot_item.value,
            "hemisphere": brainsec_widget.hemisphere.value,
            "unilateral": brainsec_widget.unilateral.value,
            "brain_areas": split_to_list(brainsec_widget.brain_areas.value),
            "brain_areas_color": split_to_list(brainsec_widget.brain_areas_color.value),
            "color_brain_density": brainsec_widget.color_brain_density.value,
            "section_list": split_to_list(brainsec_widget.section_list.value, out_format='float'),
            "section_range": float(brainsec_widget.section_range.value),
            "groups": brainsec_widget.groups.value,
            "dot_size": int(brainsec_widget.dot_size.value),
            "color_cells_atlas": brainsec_widget.color_cells_atlas.value,
            "color_cells": split_to_list(brainsec_widget.color_cells.value),
            "show_cbar": brainsec_widget.show_cbar.value,
            "color_cells_density": split_to_list(brainsec_widget.cmap_cells.value),
            "bin_size_cells_density": int(brainsec_widget.bin_size_cells.value),
            "vmin_cells_density": int(brainsec_widget.vmin_cells.value),
            "vmax_cells_density": int(brainsec_widget.vmax_cells.value),
            "group_diff_cells_density_idx": self._check_diff_idx(brainsec_widget.group_diff_cells.value)[1],
            "group_diff_cells_density": self._check_diff_idx(brainsec_widget.group_diff_cells.value)[0], # brainsec_widget.group_diff_cells.value,
            "group_diff_items_cells_density": brainsec_widget.group_diff_items_cells.value.split('-'),
            "color_projections": split_to_list(brainsec_widget.cmap_projection.value),
            "bin_size_projections": int(brainsec_widget.bin_size_proj.value),
            "vmin_projections": int(brainsec_widget.vmin_proj.value),
            "vmax_projections": int(brainsec_widget.vmax_proj.value),
            "group_diff_projections_idx": self._check_diff_idx(brainsec_widget.group_diff_proj.value)[1],
            "group_diff_projections": self._check_diff_idx(brainsec_widget.group_diff_proj.value)[0],
            "group_diff_items_projections": brainsec_widget.group_diff_items_proj.value.split('-'),
            # "smooth_proj": brainsec_widget.smooth_proj.value,
            # "smooth_thresh_proj": float(brainsec_widget.smooth_thresh_proj.value),
            "color_injection_site": split_to_list(brainsec_widget.color_inj.value),
            "color_optic_fiber": split_to_list(brainsec_widget.color_optic.value),
            "color_neuropixels_probe": split_to_list(brainsec_widget.color_npx.value),
            "plot_gene": brainsec_widget.plot_gene.value,
            "color_genes": split_to_list(brainsec_widget.color_genes.value),
            "gene": gene,
            "color_brain_genes": brainsec_widget.color_brain_genes.value,
            "color_hcr": split_to_list(brainsec_widget.color_hcr.value),
            "color_swc": split_to_list(brainsec_widget.color_swc.value),
            "group_swc": brainsec_widget.group_swc.value,
            "save_name": brainsec_widget.save_name.value,
            "save_fig": brainsec_widget.save_fig.value,
        }
        return plotting_params

    def _check_diff_idx(self, diff_str):
        if 'index' in diff_str:
            item_key = diff_str.split(' ')[0]
            diff_bool = True
        else:
            item_key = diff_str
            diff_bool = False
        return [item_key, diff_bool]

    def _calculate_slice_indices(self, section: float) -> Tuple[List[int], int]:
        """
        Calculate slice indices for visualization based on section and range.

        Parameters:
            section (float): Section coordinate.

        Returns:
            Tuple[List[int], int]: Target z-coordinates and slice index.
        """
        target_z = [section + self.plotting_params["section_range"], section - self.plotting_params["section_range"]]
        target_z = [int(-(target / self.orient_mapping['z_plot'][2] - self.bregma[self.orient_mapping['z_plot'][1]]))
                    for target in target_z]
        slice_idx = int(-(section / self.orient_mapping['z_plot'][2] - self.bregma[self.orient_mapping['z_plot'][1]]))
        return target_z, slice_idx

    def _generate_brain_schematic(self, slice_idx: int) -> Optional[List]:
        """
        Generate a brain schematic plot for the given slice index.

        Parameters:
            slice_idx (int): Index of the slice to plot.

        Returns:
            Optional[List]: Annotated section data and color dictionary.
        """
        if self.plotting_params['color_brain_genes'] == 'voronoi':
            # Skip plotting if brain areas are colored according to clusters
            return None
        else:
            return self.brainsection_plotter.plot_brain_schematic(slice_idx, self.orient_mapping['z_plot'][1])


    def _get_section_filter_data(self, slice_idx: int, target_z: List[int]) -> Tuple[Dict[str, pd.DataFrame], Optional[List]]:
        """
        Filter data for the section and generate annotations.

        Parameters:
            slice_idx (int): Slice index.
            target_z (List[int]): Target z-coordinates for filtering.

        Returns:
            Tuple[Dict[str, pd.DataFrame], Optional[List]]: Filtered data dictionary and annotations.
        """

        annot_data = self._generate_brain_schematic(slice_idx)
        plot_dict = {}
        for item in self.data_dict:
            if item == 'swc':
                scw_filt_ids = []
                for n_id in self.data_dict[item]['neuron_id'].unique():
                    if self.data_dict[item][(self.data_dict[item]['type'] == 1) &
                                            (self.data_dict[item]['neuron_id'] == n_id)][self.orient_mapping['z_plot'][0]].between(target_z[0],target_z[1]).any():
                            scw_filt_ids.append(n_id)
                plot_dict[item] = self.data_dict[item][self.data_dict[item]['neuron_id'].isin(scw_filt_ids)]

            else:
                plot_dict[item] = self.data_dict[item][(self.data_dict[item][self.orient_mapping['z_plot'][0]] >= target_z[0])
                                              & (self.data_dict[item][self.orient_mapping['z_plot'][0]] <= target_z[1])]
            if item == 'genes' and self.plotting_params['color_brain_genes'] == 'voronoi':
                # calculate colors according to number of cluster_ids in brain regions
                    annot_data = self.brainsection_plotter.plot_brain_schematic_voronoi(plot_dict[item], slice_idx,
                                                                                        self.orient_mapping)
            rl_index = self.atlas.space.axes_description.index('rl')
            bregma_rl = self.bregma[rl_index]

            # Check unilateral condition and orientation
            if self.plotting_params['unilateral'] in ['left', 'right'] and self.orient_mapping['z_plot'][1] < 2:
                # Filter and adjust based on hemisphere
                if self.plotting_params['unilateral'] == 'left':
                    # Retain only left hemisphere values
                    plot_dict[item] = plot_dict[item][plot_dict[item]['ml_coords'] > bregma_rl]
                    # Adjust ML coordinates to make left hemisphere relative
                    plot_dict[item].loc[:, 'ml_coords'] -= bregma_rl
                else:  # plotting_params['unilateral'] == 'right'
                    # Retain only right hemisphere values
                    plot_dict[item] = plot_dict[item][plot_dict[item]['ml_coords'] < bregma_rl]

                # Reset index after filtering
                plot_dict[item] = plot_dict[item].reset_index(drop=True)

        return plot_dict, annot_data
    def _collect_section_data(self, section: float) -> Tuple[Optional[List], Dict[str, pd.DataFrame], int]:
        """
        Collect data and annotations for a given section.

        Parameters:
            section (float): Section coordinate.

        Returns:
            Tuple[Optional[List], Dict[str, pd.DataFrame], int]: Annotations, data dictionary, and slice index.
        """
        target_z, slice_idx = self._calculate_slice_indices(section)
        plot_dict, annot_data = self._get_section_filter_data(slice_idx, target_z)

        return (annot_data, plot_dict, slice_idx)

    def calculate_plot(self, progress_callback: Optional[callable] = None) -> List[
        Tuple[Optional[List], Dict[str, pd.DataFrame], int]]:
        """
        Calculate data and annotations for all sections.

        Parameters:
            progress_callback (Optional[callable]): Callback function for progress updates.

        Returns:
            List[Tuple[Optional[List], Dict[str, pd.DataFrame], int]]: Data and annotations for each section.
        """
        # density = self._check_color_brain_density()
        self.progress_total = len(self.plotting_params["section_list"])
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._collect_section_data, section): section
                       for section in self.plotting_params["section_list"]}

            # Process results as they complete
            results = []
            for future in concurrent.futures.as_completed(futures):
                self.progress_step += 1  # Increment processed count
                if progress_callback is not None:
                    progress_callback(int((self.progress_step / self.progress_total) * 100))
                results.append(future.result())
            # futures = []
            # for section in self.plotting_params["section_list"]:
            #     futures.append(
            #         executor.submit(self._collect_section_data, section, progress_callback))
            # results = [f.result() for f in concurrent.futures.as_completed(futures)]

        results.sort(key=lambda x: x[2])
        return results


    def _get_rows_cols(self) -> Tuple[int, int]:
        """
        Determine the number of rows and columns for the plot grid.

        Returns:
            Tuple[int, int]: Number of rows and columns.
        """
        n_sec = len(self.plotting_params["section_list"])
        n_cols = int(np.ceil(math.sqrt(n_sec)))
        if (n_cols ** 2 - n_sec) >= n_cols:
            n_rows = n_cols - 1
        else:
            n_rows = n_cols
        return n_rows, n_cols

    def do_plot(self, results: List[Tuple[Optional[List], Dict[str, pd.DataFrame], int]]) -> FigureCanvas:
        """
        Generate plots for the given results.

        Parameters:
            results (List[Tuple[Optional[List], Dict[str, pd.DataFrame], int]]): Data and annotations for plotting.

        Returns:
            FigureCanvas: Canvas containing the generated plots.
        """
        n_rows, n_cols = self._get_rows_cols()
        # mpl_widget = FigureCanvas(Figure(figsize=self.plotting_params['figsize']))
        xyz_dict = get_xyz(self.atlas, self.plotting_params['section_orient'])
        xlim = xyz_dict['x'][1]
        ylim = xyz_dict['y'][1]
        aspect_ratio = xlim/ylim
        figsize = (n_cols * 8 * aspect_ratio, n_rows * 8)
        mpl_widget = FigureCanvas(Figure(figsize=figsize))
        static_ax = mpl_widget.figure.subplots(n_rows, n_cols)
        if len(self.plotting_params["section_list"]) == 1:
            static_ax = np.array([static_ax])
        static_ax = static_ax.ravel()
        for s, (annot_data, plot_dict, slice_idx) in enumerate(results):
            if not plot_dict:
                plot_dict = {'dummy': None}
                show_info("no plotting item selected, plotting only contours of brain section")
            self._do_brainsection_plot(static_ax[s], annot_data)

            plot_functions = {
                'cells': self._plot_cells,
                'cells_density': self._plot_cells_density,
                'projections': self._plot_projections,
                'injection_site': self._plot_injection_site,
                'optic_fiber': self._plot_optic_or_probe,
                'neuropixels_probe': self._plot_optic_or_probe,
                'genes': self._plot_genes,
                'hcr': self._plot_hcr,
                'swc': self._plot_swc
            }

            for item, plot_data in plot_dict.items():
                plot_function = plot_functions.get(item)
                if plot_function:
                    plot_function(static_ax[s], plot_data, item, annot_data[0])


            # ylim, xlim = annot_data[0].shape
            # static_ax[s].set_aspect(ylim/xlim, adjustable='box')
            static_ax[s].set_xlim(0, xlim)
            static_ax[s].set_ylim(ylim, 0)
            static_ax[s].title.set_text(
                f"bregma - {round((-(slice_idx - self.bregma[self.orient_mapping['z_plot'][1]]) * self.orient_mapping['z_plot'][2]), 1)} mm")

            static_ax[s].axis('off')

        if self.plotting_params["save_fig"]:
            self._save_figure_and_data(mpl_widget, results)
        return mpl_widget

    def _do_brainsection_plot(self, ax: plt.Axes, annot_data: List) -> None:
        """
        Plot brain section contours and regions.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            annot_data (List): Annotation data including regions and colors.
        """
        annot_section, unique_ids, color_dict = annot_data
        for uid in unique_ids:
            # Create a binary mask for the current region ID in the original data
            mask = np.uint8(annot_section == uid)

            # Find contours for the current region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw each contour as a path in the SVG
            for contour in contours:
                # Convert contour points to a list of (x, y) tuples
                points = [(int(point[0][0]), int(point[0][1])) for point in contour]

                # Create an SVG path from the contour points
                if len(points) > 4 and uid != -1:  # Only add if contour has more than 1 point
                    poly = Polygon(points)
                    x, y = poly.exterior.xy  # Extract x and y coordinates of the polygon boundary
                    v = np.column_stack((x, y))
                    # ax.add_patch(plt.Polygon(v * np.array([1, -1])[None, :], fc=color_dict[uid], ec='k', alpha=1.))
                    # print(color_dict[uid])
                    ax.add_patch(plt.Polygon(v, fc=color_dict[uid], ec='gainsboro', lw=1, alpha=1.))

    def _plot_cells(self, ax: plt.Axes, plot_data: pd.DataFrame, item: Optional[str] = None, annot_section_plt: Optional[np.ndarray] = None) -> None:
        """
        Plot cells data.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Data to plot.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        color_atlas = self.plotting_params['color_cells_atlas']
        palette = (
            {s: tuple([c / 255 for c in self.atlas.structures[s]['rgb_triplet']])
             for s in plot_data.structure_id.unique()}
            if color_atlas else None
        )

        sns.scatterplot(
            ax=ax,
            x=self.orient_mapping['x_plot'],
            y=self.orient_mapping['y_plot'],
            data=plot_data,
            hue='structure_id' if color_atlas else (
                self.plotting_params["groups"] if not self.color_dict['cells']['single_color'] else None),
            palette=palette if color_atlas else (
                self.color_dict['cells']["cmap"] if not self.color_dict['cells']['single_color'] else None),
            color=self.color_dict['cells']["cmap"] if self.color_dict['cells']['single_color'] else None,
            s=self.plotting_params["dot_size"],
            legend=False
        )

    def _plot_hcr(self, ax: plt.Axes, plot_data: pd.DataFrame, item: Optional[str] = None, annot_section_plt: Optional[np.ndarray] = None) -> None:
        """
        Plot HCR data.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Data to plot.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        color_atlas = self.plotting_params['color_cells_atlas']
        palette = (
            {s: tuple([c / 255 for c in self.atlas.structures[s]['rgb_triplet']])
             for s in plot_data.structure_id.unique()}
            if color_atlas else None
        )

        sns.scatterplot(
            ax=ax,
            x=self.orient_mapping['x_plot'],
            y=self.orient_mapping['y_plot'],
            data=plot_data,
            hue='structure_id' if color_atlas else (
                'hcr' if not self.color_dict['hcr']['single_color'] else None),
            palette=palette if color_atlas else (
                self.color_dict['hcr']["cmap"] if not self.color_dict['hcr']['single_color'] else None),
            color=self.color_dict['hcr']["cmap"] if self.color_dict['hcr']['single_color'] else None,
            s=self.plotting_params["dot_size"]
        )

    def _plot_swc(self, ax: plt.Axes, plot_data: pd.DataFrame, item: Optional[str] = None, annot_section_plt: Optional[np.ndarray] = None) -> None:
        """
        Plot HCR data.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Data to plot.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        plot_group = self.plotting_params['group_swc']
        for n_id in plot_data['neuron_id'].unique():
            if self.color_dict['swc']['single_color']:
                color = self.color_dict['swc']['cmap']
            else:
                if not plot_group:
                    color = self.color_dict['swc']['cmap'].get(n_id, 'k')
                else:
                    color = self.color_dict['swc']['cmap'].get(plot_data[plot_data['neuron_id'] == n_id]['group_id'].unique()[0], 'k')
            swc = plot_data[plot_data['neuron_id'] == n_id]
            idx_of = {nid: i for i, nid in enumerate(swc["id"].values)}
            for _, row in swc.iterrows():
                pid = int(row["parent"])
                if pid == -1 or pid not in idx_of:
                    continue
                parent = swc.iloc[idx_of[pid]]

                x1, y1 = row[self.orient_mapping['x_plot']], row[self.orient_mapping['y_plot']]
                x0, y0 = parent[self.orient_mapping['x_plot']], parent[self.orient_mapping['y_plot']]
                # color = type_colors.get(int(row["type"]), "k")

                ax.plot([x0, x1], [y0, y1], color=color, lw=0.5)
        soma_df = plot_data[plot_data['type'] == 1]
        if not soma_df.empty:
            sns.scatterplot(
                ax=ax,
                x=self.orient_mapping['x_plot'],
                y=self.orient_mapping['y_plot'],
                data=soma_df,
                hue='group_id' if plot_group else (
                    'neuron_id' if not self.color_dict['swc']['single_color'] else None),
                palette=self.color_dict['swc']["cmap"] if not self.color_dict['swc']['single_color'] else None,
                color=self.color_dict['swc']["cmap"] if self.color_dict['swc']['single_color'] else None,
                s=self.plotting_params["dot_size"]
            )
        #     soma = swc[swc["type"] == 1]
        #     if not soma.empty:
        #         ax.scatter(
        #             soma[self.orient_mapping['x_plot']],
        #             soma[self.orient_mapping['y_plot']],
        #             s=15,
        #             color=color,
        #             label=n_id if not self.plotting_params['group_swc'] else
        #             plot_data[plot_data['neuron_id'] == n_id]['group_id'].unique()[0],
        #         )
        # if not self.color_dict['swc']['single_color']:
        #     ax.legend()
        # color_atlas = self.plotting_params['color_cells_atlas']
        # palette = (
        #     {s: tuple([c / 255 for c in self.atlas.structures[s]['rgb_triplet']])
        #      for s in plot_data.structure_id.unique()}
        #     if color_atlas else None
        # )
        #
        # sns.scatterplot(
        #     ax=ax,
        #     x=self.orient_mapping['x_plot'],
        #     y=self.orient_mapping['y_plot'],
        #     data=plot_data,
        #     hue='structure_id' if color_atlas else (
        #         'hcr' if not self.color_dict['hcr']['single_color'] else None),
        #     palette=palette if color_atlas else (
        #         self.color_dict['hcr']["cmap"] if not self.color_dict['hcr']['single_color'] else None),
        #     color=self.color_dict['hcr']["cmap"] if self.color_dict['hcr']['single_color'] else None,
        #     s=self.plotting_params["dot_size"]
        # )

    def _plot_cells_density(self, ax: plt.Axes, plot_data: pd.DataFrame, item: Optional[str] = None, annot_section_plt: Optional[np.ndarray] = None) -> None:
        """
        Plot cell density heatmap.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Data to plot.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        if plot_data.empty:
            return

        self._plot_heatmap(ax, plot_data, 'cells_density', annot_section_plt)

    def _plot_projections(self, ax: plt.Axes, plot_data: pd.DataFrame, item: Optional[str] = None, annot_section_plt: Optional[np.ndarray] = None) -> None:
        """
        Plot projection density heatmap.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Data to plot.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        if plot_data.empty:
            show_info("data empty")
            return

        self._plot_heatmap(ax, plot_data, 'projections', annot_section_plt)

    def _plot_heatmap(
            self,
            ax: plt.Axes,
            plot_data: pd.DataFrame,
            item_key: str,
            annot_section_plt: np.ndarray
    ) -> None:
        """
        Plot a heatmap for the given data and annotations.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Dataframe containing the data to visualize.
            item_key (str): Key for the plotting item.
            annot_section_plt (np.ndarray): Annotated section array.
        """
        bin_size = self.plotting_params[f'bin_size_{item_key}']
        x_dim, y_dim = annot_section_plt.shape[1], annot_section_plt.shape[0]
        x_bins, y_bins = np.arange(0, x_dim + bin_size, bin_size), np.arange(0, y_dim + bin_size, bin_size)
        heatmap_data, mask = (
            self.brainsection_plotter.calculate_heatmap(annot_section_plt, plot_data, self.orient_mapping, y_bins, x_bins, bin_size)
            if self.plotting_params[f'group_diff_{item_key}'] == ''
            else self.brainsection_plotter.calculate_heatmap_difference(
                annot_section_plt, plot_data, self.plotting_params, self.orient_mapping, y_bins, x_bins, bin_size,
                f'group_diff_{item_key}', f'group_diff_items_{item_key}'
            )
        )
        sns.heatmap(
            ax=ax,
            data=heatmap_data,
            mask=mask,
            cbar=self.plotting_params['show_cbar'],
            cbar_kws={'shrink': 0.5},
            cmap=self.color_dict[item_key]["cmap"],
            vmin=self.plotting_params[f'vmin_{item_key}'],
            vmax=self.plotting_params[f'vmax_{item_key}'],
            rasterized=True
        )

    def _plot_injection_site(
            self,
            ax: plt.Axes,
            plot_data: pd.DataFrame,
            item: Optional[str] = None,
            annot_section_plt: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot injection site data using kernel density estimation (KDE).

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Dataframe containing the data to visualize.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        sns.kdeplot(
            ax=ax,
            data=plot_data,
            x=self.orient_mapping['x_plot'],
            y=self.orient_mapping['y_plot'],
            fill=True,
            color=self.color_dict[item]["cmap"] if self.color_dict[item]['single_color'] else None,
            hue=self.plotting_params["groups"] if not self.color_dict[item]['single_color'] else None,
            palette=self.color_dict[item]["cmap"] if not self.color_dict[item]['single_color'] else None
        )

    def _plot_optic_or_probe(
            self,
            ax: plt.Axes,
            plot_data: pd.DataFrame,
            item: str,
            annot_section_plt: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot optic fiber or probe trajectory data.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Dataframe containing the data to visualize.
            item (str): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        if self.color_dict[item]["single_color"]:
            sns.regplot(
                ax=ax,
                x=self.orient_mapping['x_plot'],
                y=self.orient_mapping['y_plot'],
                data=plot_data,
                line_kws=dict(alpha=0.7, color=self.color_dict[item]["cmap"]),
                scatter=None,
                ci=None
            )
        else:
            for c in plot_data['channel'].unique():
                sns.regplot(
                    ax=ax,
                    x=self.orient_mapping['x_plot'],
                    y=self.orient_mapping['y_plot'],
                    data=plot_data[plot_data['channel'] == c],
                    line_kws=dict(alpha=0.7, color=self.color_dict[item]["cmap"][c]),
                    scatter=None,
                    ci=None
                )

    def _plot_genes(
            self,
            ax: plt.Axes,
            plot_data: pd.DataFrame,
            item: Optional[str] = None,
            annot_section_plt: Optional[np.ndarray] = None
    ) -> None:
        """
        Plot gene expression or cluster data.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            plot_data (pd.DataFrame): Dataframe containing the data to visualize.
            item (Optional[str]): Plot item key.
            annot_section_plt (Optional[np.ndarray]): Annotated section array.
        """
        if self.plotting_params['color_cells_atlas']:
            palette = {
                s: tuple([c / 255 for c in self.atlas.structures[s]['rgb_triplet']])
                for s in plot_data.structure_id.unique()
            }
            sns.scatterplot(
                ax=ax,
                x=self.orient_mapping['x_plot'],
                y=self.orient_mapping['y_plot'],
                data=plot_data,
                hue='structure_id',
                palette=palette,
                s=self.plotting_params["dot_size"],
                legend=False
            )
        else:
            hue_param = "cluster_id" if self.plotting_params["plot_gene"] == 'clusters' and not self.color_dict[item]["single_color"] else None
            color = self.color_dict[item]["cmap"] if self.color_dict[item]["single_color"] else None
            palette = self.color_dict[item]["cmap"] if hue_param else None

            if self.plotting_params["plot_gene"] == 'clusters':
                sns.scatterplot(ax=ax,
                                x=self.orient_mapping['x_plot'],
                                y=self.orient_mapping['y_plot'],
                                data=plot_data,
                                color=color,
                                hue=hue_param,
                                palette=palette,
                                s=self.plotting_params["dot_size"]
                                )
            else:
                im = ax.scatter(
                    x=plot_data[self.orient_mapping['x_plot']],
                    y=plot_data[self.orient_mapping['y_plot']],
                    c=plot_data['gene_expression_norm'],
                    cmap=color,
                    vmin=0,
                    vmax=1,
                    s=self.plotting_params["dot_size"]
                )
                ax.collections[0].set_clim(0, 1)
                if self.plotting_params['show_cbar']:
                    plt.colorbar(im)

    def _save_figure_and_data(self, mpl_widget: "FigureCanvas", results: List[Tuple[Optional[List], Dict[str, pd.DataFrame], int]]) -> None:
        """
        Save the generated plots and data to disk.

        Parameters:
            mpl_widget (FigureCanvas): Canvas containing the plots.
            results (List[Tuple[Optional[List], Dict[str, pd.DataFrame], int]]): Data and annotations used for plotting.
        """
        save_folder = self.save_path.joinpath(self.plotting_params["save_name"])
        save_folder = get_unique_folder(save_folder)
        save_folder.mkdir(exist_ok=True)
        for _, plot_dict, slice_idx in results:
            section = f"{round((-(slice_idx - self.bregma[self.orient_mapping['z_plot'][1]]) * self.orient_mapping['z_plot'][2]), 1)}mm"
            for item in plot_dict:
                data_fn = save_folder.joinpath(f'{self.plotting_params["save_name"]}_{section}_{item}.csv')
                plot_dict[item].to_csv(data_fn)
        fig_fn = save_folder.joinpath(f'{self.plotting_params["save_name"]}.svg')
        mpl_widget.figure.savefig(fig_fn)
        params_fn = save_folder.joinpath(f'{self.plotting_params["save_name"]}.json')
        with open(params_fn, 'w') as fn:
            json.dump(self.plotting_params, fn, indent=4)




