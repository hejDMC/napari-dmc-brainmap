import json
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Union
from magicgui.widgets import FunctionGui
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'
from napari_dmc_brainmap.utils.general_utils import split_to_list
from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import resort_df, \
    get_unique_folder
from napari_dmc_brainmap.utils.color_manager import ColorManager
from napari.utils.notifications import show_info

class BarplotVisualization:
    """
    Class for generating bar plots and visualizations based on segmentation data and user configurations.
    """
    def __init__(
        self,
        df_all: pd.DataFrame,
        df: pd.DataFrame,
        atlas: str,
        animal_list: List[str],
        tgt_list: List[str],
        save_path: Path,
        barplot_widget: FunctionGui,
        gene_list: List[str]
    ) -> None:
        """
        Initialize the BarplotVisualization class.

        Parameters:
            df_all (pd.DataFrame): Complete dataset containing segmentation information.
            df (pd.DataFrame): Filtered dataset for plotting.
            atlas (str): Atlas name for reference.
            animal_list (List[str]): List of animal IDs.
            tgt_list (List[str]): List of target regions.
            save_path (Path): Directory to save visualizations.
            barplot_widget (FunctionGui): Widget containing user configurations for the bar plot.
            gene_list (List[str]): List of genes to analyze, if applicable.
        """
        self.df_all = df_all
        self.df = df  # filtered
        self.atlas = atlas
        self.animal_list = animal_list
        self.tgt_list = tgt_list
        self.save_path = save_path
        self.plotting_params = self._get_barplot_params(barplot_widget, gene_list)
        self.color_manager = ColorManager()

    def _get_barplot_params(self, barplot_widget: FunctionGui, gene_list: List[str]) -> Dict:
        """
        Extract bar plot parameters from the widget.

        Parameters:
            barplot_widget (FunctionGui): Widget containing user configurations for the bar plot.
            gene_list (List[str]): List of genes to analyze.

        Returns:
            Dict: A dictionary of bar plot parameters.
        """
        if barplot_widget.plot_item.value == 'hcr':
            groups = 'hcr'
        else:
            groups = barplot_widget.groups.value

        self.plotting_params = {
            "gene_list": gene_list,
            "groups": groups,
            "horizontal": barplot_widget.orient.value,
            "figsize": split_to_list(barplot_widget.plot_size.value, out_format='int'),
            "xlabel": [barplot_widget.xlabel.value, int(barplot_widget.xlabel_size.value)],  # 0: label, 1: fontsize
            "ylabel": [barplot_widget.ylabel.value, int(barplot_widget.ylabel_size.value)],
            "tick_size": int(barplot_widget.tick_size.value),  # for now only y and x same size
            "rotate_xticks": int(barplot_widget.rotate_xticks.value),  # set to False of no rotation
            "title": [barplot_widget.title.value, int(barplot_widget.title_size.value)],
            "alphabetic": barplot_widget.alphabetic.value,
            "style": barplot_widget.style.value,
            "color": barplot_widget.color.value,
            "bar_palette": split_to_list(barplot_widget.tgt_colors.value),
            "scatter_palette": split_to_list(barplot_widget.scatter_palette.value),
            "scatter_hue": barplot_widget.scatter_hue.value,
            "scatter_size": int(barplot_widget.scatter_size.value),
            # "legend_hide": barplot_widget.legend_hide.value,
            "save_name": barplot_widget.save_name.value,
            "save_fig": barplot_widget.save_fig.value,
            "absolute_numbers": barplot_widget.absolute_numbers.value
        }
        return self.plotting_params

    def calculate_plot(self) -> pd.DataFrame:
        """
        Calculate data for bar plots based on the provided dataset and user parameters.

        Returns:
            pd.DataFrame: Data ready for plotting.
        """
        # Step 1: Transform the dataset
        if self.df.empty:
            return self._create_empty_df()

        df_transformed = self._transform_data(self.df)

        # Step 2: Handle missing areas
        df_adjusted = self._handle_missing_areas(df_transformed)

        if self.plotting_params["gene_list"]:
            return self._calculate_expression_data(df_adjusted)
        # Step 4: Calculate percentages
        df_to_plot = self._calculate_percentage_data(df_adjusted)
        return df_to_plot

    def _create_empty_df(self) -> pd.DataFrame:
        """
        Create an empty DataFrame if the input dataset is empty.

        Returns:
            pd.DataFrame: Empty DataFrame.
        """
        combinations = list(itertools.product(self.animal_list, self.tgt_list))
        dummy_df = pd.DataFrame(combinations, columns=['tgt_name', 'animal_id'])
        dummy_df['percent'] = 0
        return dummy_df

    def _transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset based on user-specified grouping.

        Parameters:
            df (pd.DataFrame): Dataset to transform.

        Returns:
            pd.DataFrame: Transformed dataset.
        """
        if self.plotting_params["groups"] in ["channel", "ipsi_contra", "hcr"]:
            return df.pivot_table(index='tgt_name', columns=['animal_id', self.plotting_params["groups"]],
                                  aggfunc='count').fillna(0)
        elif self.plotting_params["gene_list"]:
            return pd.DataFrame(df.groupby('tgt_name')[self.plotting_params["gene_list"]].mean().fillna(0))
        else:
            return df.pivot_table(index='tgt_name', columns=['animal_id'], aggfunc='count').fillna(0)

    def _handle_missing_areas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add missing regions to the dataset with zero values.

        Parameters:
            df (pd.DataFrame): Dataset to check for missing regions.

        Returns:
            pd.DataFrame: Dataset with missing regions filled.
        """
        missing_areas = list(set(self.tgt_list) - set(df.index))
        if missing_areas:
            missing_df = pd.DataFrame(0, index=missing_areas, columns=df.columns)
            df = pd.concat([df, missing_df])
        return df

    def _calculate_expression_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate expression data for bar plotting.

        Parameters:
            df (pd.DataFrame): Dataset to analyze.

        Returns:
            pd.DataFrame: Expression data.
        """
        df.reset_index(inplace=True)
        df['animal_id'] = self.animal_list[0]

        if len(self.plotting_params['gene_list']) > 1:
            df.rename(columns={'index': 'tgt_name'}, inplace=True)
            df = pd.melt(df, id_vars=['tgt_name', 'animal_id'], value_vars=self.plotting_params['gene_list'],
                         var_name='genes', value_name='percent')
        else:
            df.rename(columns={'index': 'tgt_name', self.plotting_params['gene_list'][0]: 'percent'}, inplace=True)

        return df


    def _calculate_percentage_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate percentage data for bar plotting.

        Parameters:
            df (pd.DataFrame): Dataset to analyze.

        Returns:
            pd.DataFrame: Percentage data for plotting.
        """
        df_to_plot = pd.DataFrame()

        for animal_id in self.animal_list:
            if self.plotting_params["absolute_numbers"] == 'absolute':
                dummy_df = pd.DataFrame(df['ap_mm'][animal_id])
            elif self.plotting_params["absolute_numbers"] == 'percentage_selection':
                sum_value = df['ap_mm'][animal_id].sum().sum() if self.plotting_params["groups"] in ["channel",
                                                                                                "ipsi_contra", "hcr"] else \
                    df['ap_mm'][animal_id].sum()
                dummy_df = pd.DataFrame((df['ap_mm'][animal_id] / sum_value) * 100)
            else:
                dummy_df = pd.DataFrame(
                    (df['ap_mm'][animal_id] / len(self.df_all[self.df_all['animal_id'] == animal_id])) * 100)
            if self.plotting_params["groups"] in ["channel", "ipsi_contra", "hcr"]:
                dummy_df = dummy_df.stack().reset_index()
                dummy_df.rename(columns={self.plotting_params["groups"]: "groups", 0: 'percent'}, inplace=True)
            else:
                dummy_df.rename(columns={animal_id: "percent"}, inplace=True)

            dummy_df['animal_id'] = animal_id
            if self.plotting_params["groups"] in ['group', 'genotype']:
                dummy_df['groups'] = self.df[self.df['animal_id'] == animal_id][self.plotting_params["groups"]].unique()[0]

            df_to_plot = pd.concat([df_to_plot, dummy_df])

        if self.plotting_params["groups"] not in ["channel", "ipsi_contra", "hcr"]:
            df_to_plot.index.name = 'tgt_name'
            df_to_plot.reset_index(inplace=True)
        if 'level_0' in df_to_plot.columns:
            df_to_plot.rename(columns={'level_0': 'tgt_name'}, inplace=True)

        return df_to_plot

    def do_bar_plot(self, tgt_data_to_plot: pd.DataFrame) -> FigureCanvas:
        """
        Generate a bar plot with the given data and parameters.

        Parameters:
            tgt_data_to_plot (pd.DataFrame): Data to plot.

        Returns:
            FigureCanvas: Bar plot visualization.
        """
        # Step 1: Set plot orientation and variables
        plot_orient, x_var, y_var = self._get_plot_orientation()

        # Step 3: Reorder DataFrame if needed
        if self.plotting_params["alphabetic"]:
            tgt_data_to_plot = resort_df(tgt_data_to_plot, self.tgt_list)

        # Step 4: Create Plot Canvas
        mpl_widget = FigureCanvas(Figure(figsize=self.plotting_params['figsize']))
        sns.set(style=self.plotting_params["style"])
        static_ax = mpl_widget.figure.subplots()
        # Step 6: Plot Data Using Seaborn
        self._plot_data(static_ax=static_ax, data=tgt_data_to_plot,
                  x_var=x_var, y_var=y_var, plot_orient=plot_orient)

        # Step 7: Customize Axes and Legends
        self._customize_axes(static_ax, plot_orient)
        if self.plotting_params["save_fig"]:
            self._save_figure_and_data(mpl_widget, tgt_data_to_plot)

        return mpl_widget

    def _get_plot_orientation(self) -> Tuple[str, str, str]:
        """
        Determine the orientation of the bar plot (horizontal or vertical).

        Returns:
            Tuple[str, str, str]: Plot orientation and variable mappings for x and y axes.
        """
        if self.plotting_params["horizontal"] == "horizontal":
            return 'h', "percent", "tgt_name"
        else:
            return 'v', "tgt_name", "percent"
    def _check_color_palette(self, cmap: Union[str, List[str]], bar: bool = True) -> Union[str, List[str]]:
        """
        Validate or adjust the color palette for bar or scatter plots.

        Parameters:
            cmap (Union[str, List[str]]): Color palette.
            bar (bool, optional): Whether the palette is for bar plots. Defaults to True.

        Returns:
            Union[str, List[str]]: Validated color palette.
        """
        if isinstance(cmap, str):
            try:
                plt.get_cmap(cmap)
            except ValueError:
                show_info(f'cmap does not exist, use default cmap instead')
                cmap = 'Blues' if bar else 'Greys'
        elif isinstance(cmap, list):
            cmap = [self.color_manager.check_color_name(c) for c in cmap]
        else:
            cmap = [self.color_manager.check_color_name(cmap)]
        return cmap


    def _plot_data(
        self, static_ax: plt.Axes, data: pd.DataFrame, x_var: str, y_var: str, plot_orient: str
    ) -> None:
        """
        Plot bar and optional scatter data.

        Parameters:
            static_ax (plt.Axes): Matplotlib axis to plot on.
            data (pd.DataFrame): Dataset to plot.
            x_var (str): X-axis variable.
            y_var (str): Y-axis variable.
            plot_orient (str): Plot orientation ('h' for horizontal, 'v' for vertical).
        """
        bar_palette = self._check_color_palette(self.plotting_params["bar_palette"])
        scatter_palette = self._check_color_palette(self.plotting_params["scatter_palette"], bar=False)
        if self.plotting_params["groups"] == '' and not self.plotting_params['gene_list']:
            sns.barplot(
                ax=static_ax,
                x=x_var,
                y=y_var,
                data=data,
                palette=bar_palette,
                capsize=.1,
                errorbar=None,
                orient=plot_orient)
            if self.plotting_params["scatter_hue"]:
                sns.swarmplot(
                    ax=static_ax,
                    x=x_var,
                    y=y_var,
                    hue='animal_id',
                    data=data,
                    palette=scatter_palette,
                    size=self.plotting_params["scatter_size"],
                    orient=plot_orient,
                    legend=False
                )
        elif self.plotting_params['gene_list']:
            self._plot_expression(static_ax, data, x_var, y_var, plot_orient)
        else:
            self._plot_group_data(static_ax, data, x_var, y_var, plot_orient)

    def _plot_expression(
        self, static_ax: plt.Axes, data: pd.DataFrame, x_var: str, y_var: str, plot_orient: str
    ) -> None:
        """
        Plot gene expression data.

        Parameters:
            static_ax (plt.Axes): Matplotlib axis to plot on.
            data (pd.DataFrame): Dataset to plot.
            x_var (str): X-axis variable.
            y_var (str): Y-axis variable.
            plot_orient (str): Plot orientation ('h' for horizontal, 'v' for vertical).
        """
        palette = self.color_manager.create_color_palette([], self.plotting_params, "bar_palette", df=data, hue_id='genes') if len(self.plotting_params['gene_list']) > 1 else self.plotting_params["bar_palette"]
        if len(self.plotting_params['gene_list']) == 1:
            palette = self._check_color_palette(palette.values())
        sns.barplot(
                ax=static_ax,
                x=x_var,
                y=y_var,
                data=data,
                hue='genes' if len(self.plotting_params['gene_list']) > 1 else None,
                palette=palette,
                capsize=.1,
                errorbar=None,
                orient=plot_orient
            )

    def _plot_group_data(
        self, static_ax: plt.Axes, data: pd.DataFrame, x_var: str, y_var: str, plot_orient: str
    ) -> None:
        """
        Plot grouped data with hue.

        Parameters:
            static_ax (plt.Axes): Matplotlib axis to plot on.
            data (pd.DataFrame): Dataset to plot.
            x_var (str): X-axis variable.
            y_var (str): Y-axis variable.
            plot_orient (str): Plot orientation ('h' for horizontal, 'v' for vertical).
        """
        hue = 'animal_id' if self.plotting_params["groups"] == 'animal_id' else 'groups'
        cmap = self.color_manager.create_color_palette([], self.plotting_params, "bar_palette", df=data, hue_id=hue)
        scatter_palette = self._check_color_palette(self.plotting_params["scatter_palette"], bar=False)
        sns.barplot(
            ax=static_ax,
            x=x_var,
            y=y_var,
            data=data,
            hue=hue,
            palette=cmap,
            capsize=.1,
            errorbar=None,
            orient=plot_orient
        )
        if self.plotting_params["scatter_hue"]:
            sns.swarmplot(
                ax=static_ax,
                x=x_var,
                y=y_var,
                hue=hue,
                data=data,
                palette=scatter_palette,
                size=self.plotting_params["scatter_size"],
                dodge=True,
                orient=plot_orient,
                legend=False
            )

    def _customize_axes(self, static_ax: plt.Axes, plot_orient: str) -> None:
        """
        Customize axis labels, ticks, and grid lines.

        Parameters:
            static_ax (plt.Axes): Matplotlib axis to customize.
            plot_orient (str): Plot orientation ('h' for horizontal, 'v' for vertical).
        """
        if plot_orient == 'v':
            static_ax.set_xlabel(self.plotting_params["xlabel"][0], fontsize=self.plotting_params["xlabel"][1])
            static_ax.set_ylabel(self.plotting_params["ylabel"][0], fontsize=self.plotting_params["ylabel"][1])
        else:
            static_ax.set_ylabel(self.plotting_params["xlabel"][0], fontsize=self.plotting_params["xlabel"][1])
            static_ax.set_xlabel(self.plotting_params["ylabel"][0], fontsize=self.plotting_params["ylabel"][1])
        static_ax.set_title(self.plotting_params["title"][0], fontsize=self.plotting_params["title"][1])
        static_ax.spines['top'].set_visible(False)
        static_ax.spines['right'].set_visible(False)

        for tick in static_ax.get_yticks():
            static_ax.axhline(y=tick, color='gray', linestyle='--', linewidth=0.7, alpha=0.6, zorder=0)

        if self.plotting_params["groups"]:
            self._customize_legend(static_ax)

        self._set_axis_colors(static_ax)
        if self.plotting_params["rotate_xticks"]:
            static_ax.set_xticklabels(static_ax.get_xticklabels(), rotation=self.plotting_params["rotate_xticks"])

    def _customize_legend(self, static_ax: plt.Axes) -> None:
        """
        Customize the legend properties.

        Parameters:
            static_ax (plt.Axes): Matplotlib axis containing the legend.
        """
        leg = static_ax.get_legend()
        leg.set_title(self.plotting_params['groups'])
        leg.get_title().set_color(self.plotting_params["color"])
        frame = leg.get_frame()
        frame.set_alpha(None)
        frame.set_facecolor((0, 0, 1, 0))
        for text in leg.get_texts():
            text.set_color(self.plotting_params["color"])

    def _set_axis_colors(self, static_ax: plt.Axes) -> None:
        """
        Set axis and tick colors.

        Parameters:
            static_ax (plt.Axes): Matplotlib axis to modify.
        """
        static_ax.spines['bottom'].set_color(self.plotting_params["color"])
        static_ax.spines['left'].set_color(self.plotting_params["color"])
        static_ax.xaxis.label.set_color(self.plotting_params["color"])
        static_ax.yaxis.label.set_color(self.plotting_params["color"])
        static_ax.tick_params(colors=self.plotting_params["color"], labelsize=self.plotting_params["tick_size"])

    def _save_figure_and_data(self, mpl_widget: FigureCanvas, df: pd.DataFrame) -> None:
        """
        Save the plot and data to the specified directory.

        Parameters:
            mpl_widget (FigureCanvas): Canvas containing the plot.
            df (pd.DataFrame): Data used for plotting.
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

