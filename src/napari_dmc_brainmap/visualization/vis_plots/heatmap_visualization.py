import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from magicgui.widgets import FunctionGui
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'
from matplotlib.patches import Rectangle
from napari_dmc_brainmap.utils.general_utils import split_strings_layers, split_to_list
from napari_dmc_brainmap.utils.atlas_utils import get_bregma
from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import resort_df, \
    get_unique_folder, get_descendants, get_ancestors
from napari_dmc_brainmap.utils.color_manager import ColorManager
from napari.utils.notifications import show_info
class HeatmapVisualization:
    """
    Class for generating heatmaps and visualizing data using customizable parameters.
    """
    def __init__(
        self,
        df_all: pd.DataFrame,
        df: pd.DataFrame,
        atlas,
        animal_list: list,
        tgt_list: list,
        gene: str,
        heatmap_widget: FunctionGui,
        save_path: Path
    ) -> None:
        """
        Initialize the HeatmapVisualization.

        Parameters:
            df (pd.DataFrame): Dataset for visualization.
            atlas: Brain atlas instance for reference.
            heatmap_widget (FunctionGui): Widget containing user configurations for heatmap generation.
            save_path (Path): Directory to save the visualizations.
            data_dict (Dict[str, pd.DataFrame]): Dictionary containing additional datasets for visualizations.
        """
        self.df_all = df_all
        self.df = df  # filtered already
        self.atlas = atlas
        self.animal_list = animal_list
        self.tgt_list = tgt_list
        self.save_path = save_path
        self.plotting_params = self._get_heatmap_params(heatmap_widget, gene)
        self.progress_step = 0
        self.progress_total = None
        self.color_manager = ColorManager()

    def _get_heatmap_params(self, heatmap_widget: FunctionGui, gene: str) -> Dict:
        """
        Extract heatmap parameters from the widget.

        Parameters:
            heatmap_widget (FunctionGui): Widget containing user configurations.
            gene (str): Gene name for visualization.

        Returns:
            Dict: Dictionary of heatmap parameters.
        """
        plotting_params = {
            "group_diff_idx": self._check_diff_idx(heatmap_widget.group_diff.value)[1],
            "group_diff": self._check_diff_idx(heatmap_widget.group_diff.value)[0],
            "group_diff_items": heatmap_widget.group_diff_items.value.split('-'),
            "gene": gene,
            "figsize": split_to_list(heatmap_widget.plot_size.value, out_format='int'),
            "ylabel": [heatmap_widget.ylabel.value, int(heatmap_widget.ylabel_size.value)],
            "tick_size": [int(heatmap_widget.xticklabel_size.value), int(heatmap_widget.yticklabel_size.value)],
            "subtitle_size": int(heatmap_widget.subtitle_size.value),
            "style": heatmap_widget.style.value,
            "color": heatmap_widget.color.value,
            "cmap": split_to_list(heatmap_widget.cmap.value),
            "cbar_label": heatmap_widget.cbar_label.value,
            "cmap_min_max": split_to_list(heatmap_widget.cmap_min_max.value, out_format='float'),
            "intervals": sorted(split_to_list(heatmap_widget.intervals.value, out_format='float')),
            # assure ascending order
            "interval_labels": self._get_interval_labels(split_to_list(heatmap_widget.intervals.value, out_format='float')),
            "descendants": heatmap_widget.descendants.value,
            "save_name": heatmap_widget.save_name.value,
            "save_fig": heatmap_widget.save_fig.value,
            "absolute_numbers": heatmap_widget.absolute_numbers.value
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

    def _get_interval_labels(self, intervals: List[float]) -> List[str]:
        """
        Generate interval labels for a given list of interval bounds.

        Parameters:
            intervals (List[float]): List of interval boundaries.

        Returns:
            List[str]: List of formatted interval labels in the form 'start to end'.
        """
        intervals = sorted(intervals)  # assure ascending order
        interval_labels = [f"{start} to {end}" for start, end in zip(intervals, intervals[1:])]
        return interval_labels

    def calculate_plot(self, progress_callback: Optional[callable] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Calculate data for plotting, including group differences if specified.

        Parameters:
            progress_callback (Optional[callable]): Callback function to report progress. Defaults to None.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Data ready for plotting, or the difference between groups if specified.
        """

        if self.df.empty:
            return self._create_empty_df()

        if self.plotting_params['group_diff'] == '':
            self.progress_total = len(self.plotting_params["interval_labels"])
            return self._calculate_plot(progress_callback=progress_callback)
        else:
            group_diff_items = self.plotting_params['group_diff_items']
            group_list = set(self.df[self.plotting_params['group_diff']].unique())
            if set(group_diff_items).issubset(group_list):
                diff_data = []
                self.progress_total = len(self.plotting_params["interval_labels"]) * 2
                for i_d in group_diff_items:
                    group_df = self.df[self.df[self.plotting_params['group_diff']] == i_d]
                    sub_data_to_plot = self._calculate_plot(df=group_df, progress_callback=progress_callback)
                    diff_data.append(sub_data_to_plot)
                if self.plotting_params['group_diff_idx']:
                    return (diff_data[0] - diff_data[1])/(diff_data[0] + diff_data[1])
                else:
                    return diff_data[0] - diff_data[1]

            else:
                show_info(f"selected items to calculate difference not found: {group_diff_items}  \n"
                      f"check if items exist, also check params file if items are stated \n"
                      f"--> plotting regular heatmap")
                self.progress_total = len(self.plotting_params["interval_labels"])
                return self._calculate_plot(progress_callback=progress_callback)

    def _create_empty_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame if the input dataset is empty.

        Returns:
            pd.DataFrame: Empty DataFrame.
        """
        dummy_df = pd.DataFrame(np.nan, index=self.plotting_params['interval_labels'], columns=self.tgt_list)
        return dummy_df

    def _calculate_plot(self, df=None, progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Perform calculations to prepare the data for plotting.

        Parameters:
            progress_callback (Optional[callable]): Callback function to report progress. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe containing the calculated values for plotting.
        """
        if df is None:
            df = self.df
            animal_list = self.animal_list
        else:
            animal_list = df['animal_id'].unique()
        df = self._extract_target_data(df)
        df_pivot = self._create_pivot_table(df)
        if self.plotting_params['gene']:
            df_plot = self._check_area_in_bin(df_pivot, progress_callback=progress_callback)
            return df_plot.transpose()
            # return df_pivot.transpose()

        df_plot = self._calculate_values(df_pivot, animal_list)
        df_plot /= len(animal_list)
        df_plot = self._check_area_in_bin(df_plot, progress_callback=progress_callback)
        return df_plot.transpose()

    def _extract_target_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract target data by binning the 'ap_mm' column into specified intervals.

        Parameters:
            df (pd.DataFrame): Dataframe containing the input data.

        Returns:
            pd.DataFrame: Dataframe with an additional 'bin' column indicating the interval for each row.
        """

        interval_labels = self.plotting_params["interval_labels"]
        intervals = self.plotting_params["intervals"]
        df['bin'] = pd.cut(df['ap_mm'], intervals, labels=interval_labels)

        return df

    def _create_pivot_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a pivot table from the dataframe based on specified parameters.

        Parameters:
            df (pd.DataFrame): Input dataframe to pivot.

        Returns:
            pd.DataFrame: Pivot table with missing brain structures added and index sorted.
        """
        index, columns, aggfunc = self._get_pivot_parameters()

        pivot_kwargs = {
            'index': index,
            'columns': columns,
            'aggfunc': aggfunc
        }
        df_pivot = df.pivot_table(values=self.plotting_params.get('gene') if self.plotting_params['gene'] else None,
                                  **pivot_kwargs).fillna(0)
        if self.plotting_params.get('gene', None):
            df_pivot = df_pivot.transpose()
        # Add "missing" brain structures that have no data
        missing_areas = list(set(self.tgt_list) - set(df_pivot.index))
        if missing_areas:
            missing_df = pd.DataFrame(0, index=missing_areas, columns=df_pivot.columns)
            df_pivot = pd.concat([df_pivot, missing_df])

        # Sort the index to match the target list
        df_pivot = resort_df(df_pivot, self.tgt_list, index_sort=True)

        return df_pivot

    def _get_pivot_parameters(self) -> Tuple[Union[str, List[str]], Union[str, List[str]], str]:
        """
        Determine the parameters for creating a pivot table based on plotting settings.

        Returns:
            Tuple[Union[str, List[str]], Union[str, List[str]], str]:
            A tuple containing index, columns, and aggregation function for the pivot table.
        """
        if self.plotting_params['gene']:
            index = 'bin'
            columns = 'acronym' if self.plotting_params["descendants"] else 'tgt_name'
            aggfunc = 'mean'
        else:
            if self.plotting_params["descendants"]:
                index = 'acronym'
            else:
                index = 'tgt_name'
            columns = ['animal_id', 'bin']
            aggfunc = 'count'

        return index, columns, aggfunc

    def _calculate_values(self, df_pivot: pd.DataFrame, animal_list: List) -> pd.DataFrame:
        """
        Calculate values for plotting based on absolute or percentage settings.

        Parameters:
            df_pivot (pd.DataFrame): Pivot table containing aggregated data.
            animal_list (List): List of animal IDs to include in the calculation.

        Returns:
            pd.DataFrame: Dataframe with calculated values for plotting.
        """
        absolute_numbers = self.plotting_params["absolute_numbers"]
        df_plot = pd.DataFrame(0, index=self.tgt_list, columns=self.plotting_params["interval_labels"])
        for animal_id in animal_list:
            if absolute_numbers == 'absolute':
                data = df_pivot['ap_coords'][animal_id]
            elif absolute_numbers == 'percentage_selection':
                data = (df_pivot['ap_coords'][animal_id] / df_pivot['ap_coords'][animal_id].sum().sum()) * 100
            else:  # percentage relative to all cells in animal
                data = (df_pivot['ap_coords'][animal_id] / len(self.df_all[self.df_all['animal_id'] == animal_id])) * 100
            df_plot += data.fillna(0)

        return df_plot

    def _check_area_in_bin(self, df: pd.DataFrame, progress_callback: Optional[callable] = None) -> pd.DataFrame:
        """
        Validate the existence of brain areas within specified bins for heatmap plotting.

        Parameters:
            df (pd.DataFrame): Dataframe containing brain areas and bins.
            progress_callback (Optional[callable]): Callback function to report progress. Defaults to None.

        Returns:
            pd.DataFrame: Updated dataframe with cells set to NaN for non-existent areas in each bin.
        """
        # for heatmap plotting, check if brain area exists in bin
        # show_info('checking for existence of brain area in bin ...')

        bregma = get_bregma(self.atlas.atlas_name)
        ap_idx = self.atlas.space.axes_description.index('ap')
        ap_res = self.atlas.resolution[ap_idx]/1000.0
        annot = self.atlas.annotation
        for bin_name in df.columns:
            self.progress_step += 1
            b_start, b_end = bin_name.split(' to ')

            # Calculate the coordinates of the bin start and end, adjusted by bregma
            b_start_coord = int(-(float(b_start) / ap_res - bregma[ap_idx]))
            b_end_coord = int(-(float(b_end) / ap_res - bregma[ap_idx]))

            # Ensure b0 is the smaller coordinate, and b1 is the larger coordinate
            b0, b1 = min(b_start_coord, b_end_coord), max(b_start_coord, b_end_coord)

            # Extract the relevant ids within the bin, depending on the AP axis
            if ap_idx == 0:
                ids_in_bin = np.unique(annot[b0:b1, :, :])
            elif ap_idx == 1:
                ids_in_bin = np.unique(annot[:, b0:b1, :])
            else:
                ids_in_bin = np.unique(annot[:, :, b0:b1])

            # Check if each area in the dataframe exists in the current bin
            area_list = df.index
            for area in area_list:
                area_descendants = get_descendants([area], self.atlas)
                area_ids = [self.atlas.structures[a]['id'] for a in area_descendants]

                # If no area_id is found in the ids_in_bin, set the corresponding cell to NaN
                if not any(area_id in ids_in_bin for area_id in area_ids):
                    df.at[area, bin_name] = np.nan
            if progress_callback is not None:
                progress_callback(int((self.progress_step / self.progress_total) * 100))

        return df

    def do_plot(self, tgt_data_to_plot: pd.DataFrame) -> FigureCanvas:
        """
        Generate and display heatmaps for the specified target data.

        Parameters:
            tgt_data_to_plot (pd.DataFrame): Dataframe containing the target data to plot.

        Returns:
            FigureCanvas: Matplotlib canvas containing the heatmap visualizations.
        """

        # Set seaborn style
        if self.plotting_params["descendants"]:
            tgt_list = get_ancestors(self.tgt_list, self.atlas)
        else:
            tgt_list = self.tgt_list
        sns.set(style=self.plotting_params["style"])
        mask_cbar = 'binary_r' if self.plotting_params["style"] != 'white' else 'binary'

        # Determine color range for heatmap
        if self.plotting_params["cmap_min_max"] == 'auto':
            vmax = tgt_data_to_plot.max().max() * 0.75
            vmin = tgt_data_to_plot.min().min() * 0.75 if self.plotting_params['group_diff'] != '' else -1
        else:
            vmin, vmax = self.plotting_params["cmap_min_max"]

        cmap = self.color_manager.create_custom_colormap(self.plotting_params["cmap"])
        # try:
        #     cmap = plt.get_cmap(self.plotting_params["cmap"])
        # except ValueError:
        #     cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['whitesmoke', self.plotting_params["cmap"][0]])

        mpl_widget = FigureCanvas(Figure(figsize=self.plotting_params['figsize']))
        static_ax = mpl_widget.figure.subplots(1, (len(tgt_list) + 1),
                                               gridspec_kw={'width_ratios': [1] * len(tgt_list) + [0.15]})


        # Plot each target heatmap
        for t, tgt in enumerate(tgt_list):
            is_last = (t + 1 == len(tgt_list))
            if self.plotting_params["descendants"]:
                tgt_col = get_descendants([tgt], self.atlas)
                i_start = tgt_data_to_plot.columns.get_loc(tgt_col[0])
                i_end = tgt_data_to_plot.columns.get_loc(tgt_col[-1])
                plot_data = tgt_data_to_plot.iloc[:, i_start:i_end + 1]
            else:
                i_col = tgt_data_to_plot.columns.get_loc(tgt)
                plot_data = tgt_data_to_plot.iloc[:, i_col:i_col + 1]

            self._plot_heatmap(static_ax[t], plot_data, cmap, vmin, vmax, static_ax[t + 1] if is_last else None, is_last)
            static_ax[t].set_title(tgt, fontsize=self.plotting_params['subtitle_size'])
            static_ax[t].set_ylabel('' if t > 0 else self.plotting_params["ylabel"][0],
                                    fontsize=self.plotting_params["ylabel"][1])
            if t > 0:
                static_ax[t].set_yticks([])
            if self.plotting_params["descendants"]:
                tl = [split_strings_layers(t, atlas_name=self.atlas.metadata['name'], return_str=True)[1] for t in tgt_col]
                # tl = [tgt]
                static_ax[t].set_xticks(np.arange(len(tl))+0.5)
                static_ax[t].set_xticklabels(tl, rotation=90, fontsize=self.plotting_params["tick_size"][0])
            tly = static_ax[t].get_yticklabels()
            static_ax[t].set_yticklabels(tly, rotation=0, fontsize=self.plotting_params["tick_size"][1])

            # static_ax[t].transAxes
            bbox = static_ax[t].get_position()
            patch = Rectangle((bbox.xmin, bbox.ymin), bbox.width, bbox.height,
                              linewidth=1, edgecolor='black', facecolor='none')
            mpl_widget.figure.add_artist(patch)

        # Save figure if required
        if self.plotting_params["save_fig"]:
            self._save_figure_and_data(mpl_widget, tgt_data_to_plot)

        return mpl_widget

    def _plot_heatmap(
            self,
            ax: plt.Axes,
            data: pd.DataFrame,
            cmap: str,
            vmin: float,
            vmax: float,
            cbar_ax: Optional[plt.Axes] = None,
            is_last: bool = False
    ) -> None:
        """
        Plot a single heatmap on the given axis.

        Parameters:
            ax (plt.Axes): Matplotlib axis to plot on.
            data (pd.DataFrame): Dataframe containing the heatmap values.
            cmap (str): Colormap for the heatmap.
            vmin (float): Minimum value for the color scale.
            vmax (float): Maximum value for the color scale.
            cbar_ax (Optional[plt.Axes]): Axis for the color bar. Defaults to None.
            is_last (bool): Whether this is the last plot in the grid. Defaults to False.
        """
        sns.heatmap(
            ax=ax,
            data=data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=1,
            cbar=is_last,
            cbar_ax=cbar_ax,
            cbar_kws={'label': self.plotting_params["cbar_label"]} if is_last else None
        )
        # ax.xaxis.tick_top()
        # ax.xaxis.set_label_position('top')
        ax.tick_params(left=False, bottom=False, top=False)
        ax.invert_yaxis()
        ax.spines['bottom'].set_color(self.plotting_params["color"])
        ax.spines['left'].set_color(self.plotting_params["color"])
        ax.xaxis.label.set_color(self.plotting_params["color"])
        ax.yaxis.label.set_color(self.plotting_params["color"])
        ax.tick_params(colors=self.plotting_params["color"])

    def _save_figure_and_data(self, mpl_widget: FigureCanvas, df: pd.DataFrame) -> None:
        """
        Save the generated figure and data to the specified directory.

        Parameters:
            mpl_widget (FigureCanvas): Canvas containing the generated figure.
            df (pd.DataFrame): Dataframe containing the plotted data.
        """
        save_folder = self.save_path.joinpath(self.plotting_params["save_name"])
        save_folder = get_unique_folder(save_folder)
        save_folder.mkdir(exist_ok=True)
        data_fn = save_folder.joinpath(f'{self.plotting_params["save_name"]}.csv')
        df.to_csv(data_fn)
        fig_fn = save_folder.joinpath(f'{self.plotting_params["save_name"]}.svg')
        mpl_widget.figure.savefig(fig_fn)
        params_fn = save_folder.joinpath(f'{self.plotting_params["save_name"]}.json')
        with open(params_fn, 'w') as fn:
            json.dump(self.plotting_params, fn, indent=4)