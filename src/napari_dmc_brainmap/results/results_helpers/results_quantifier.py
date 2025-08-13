
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Tuple
from bg_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.general_utils import split_strings_layers, get_animal_id
from napari_dmc_brainmap.utils.data_loader import DataLoader

class ResultsQuantifier:
    """
    Class for quantifying segmentation results, including gene expression analysis
    and creating summary datasets for visualizations.
    """
    def __init__(
        self,
        input_path: Path,
        atlas: BrainGlobeAtlas,
        channel: List[str],
        plotting_params: Dict,
        seg_type: str = "injection_site",
        expression: Union[bool, Tuple[Path, str]] = False,
        is_merge: bool = False
    ) -> None:
        """
        Initialize the ResultsQuantifier.

        Parameters:
            input_path (Path): Path to the input directory.
            atlas (BrainGlobeAtlas): Atlas instance for anatomical reference.
            channel (List[str]): List of segmentation channels.
            plotting_params (Dict): Parameters for plotting and visualizations.
            seg_type (str, optional): Type of segmentation ('injection_site' by default).
            expression (Union[bool, Tuple[Path, str]], optional): Gene expression data file and gene name.
            is_merge (bool, optional): Flag to indicate if results are from merged datasets.
        """
        self.input_path = input_path
        self.atlas = atlas
        self.channel = channel
        self.chan = None
        self.plotting_params = plotting_params
        self.seg_type = seg_type
        self.expression = expression
        self.is_merge = is_merge
        self.animal_id = get_animal_id(input_path)
        self.results_data = None

    def quantify(self, progress_callback: Union[None, callable] = None) -> List:
        """
        Quantify segmentation results for all specified channels.

        Parameters:
            progress_callback (Union[None, callable], optional): Callback function for updating progress.

        Returns:
            List: Data for plotting or further analysis.
        """
        plot_data = []
        completed_steps = 0
        total_steps = len(self.channel)

        for self.chan in self.channel:
            try:
                plot_data.append(self._quantify_channel())
                completed_steps += 1
                if progress_callback:
                    progress = int((completed_steps / total_steps) * 100)
                    progress_callback(progress)
            except Exception:
                show_info(f"Failed to quantify channel {self.chan}: check path integrity")
                continue
        return plot_data

    def _quantify_channel(self) -> Union[None, List]:
        """
        Perform quantification for a single channel.

        Returns:
            Union[None, List]: Quantification data or None if loading results failed.
        """
        if self.results_data is None:
            if not self._load_results_data():
                return None

        # acronym_parent = [split_strings_layers(s, atlas_name=self.atlas.metadata['name'])[0] for s in self.results_data['acronym']]
        # self.results_data['acronym_parent'] = acronym_parent
        self.results_data['acronym_parent'] = self.results_data['acronym'].apply(
            lambda s: split_strings_layers(s, atlas_name=self.atlas.metadata['name'])[0])
        quant_data = []
        if self.expression:
            quant_data.append(self._get_gene_expression_data())
        else:
            if self.is_merge:
                animal_key = 'animal_id_ind'
            else:
                animal_key = 'animal_id'
            animal_list = self.results_data[animal_key].unique()

            for animal_id in animal_list:
                quant_data.append(self._get_animal_data(animal_key, animal_id))

        quant_df = pd.concat(quant_data, axis=0)
        quant_df_pivot = quant_df.pivot(columns='acronym', values='quant_distribution', index='animal_id').fillna(0)
        self.results_dir = get_info(self.input_path, 'results', channel=self.chan, seg_type=self.seg_type,
                                    only_dir=True)
        if self.expression:
            gene = self.expression[1]
            save_fn = self.results_dir.joinpath(f'quantification_{self.seg_type}_{gene}.csv')
        else:
            save_fn = self.results_dir.joinpath(f'quantification_{self.seg_type}.csv')

        quant_df_pivot.to_csv(save_fn)
        if not self.expression:
            quant_df_pivot_raw = quant_df_pivot.loc[:, quant_df_pivot.columns != 'animal_id'] * len(self.results_data)
            save_fn_raw = self.results_dir.joinpath(f'quantification_raw_number_{self.seg_type}.csv')
            quant_df_pivot_raw.to_csv(save_fn_raw)
        p_data = [quant_df_pivot, self.chan, self.seg_type, self.results_data, self.expression, self.is_merge]
        # mpl_widget = self._plot_quant_data(quant_df_pivot)
        self.results_data = None
        return p_data


    def _load_results_data(self) -> bool:
        """
        Load results data from the input path.

        Returns:
            bool: True if results data was loaded successfully, False otherwise.
        """
        # self.results_dir = get_info(self.input_path, 'results', channel=self.chan, seg_type=self.seg_type, only_dir=True)
        # results_fn = self.results_dir.joinpath(f'{self.animal_id}_{self.seg_type}.csv')
        # if results_fn.exists():
        #     self.results_data = pd.read_csv(results_fn)
        #     self.results_data['animal_id'] = [self.animal_id] * len(self.results_data)
        #     if self.atlas.metadata['name'] == 'allen_mouse':
        #         self.results_data = clean_results_df(self.results_data, self.atlas)
        data_loader = DataLoader(self.input_path.parent, self.atlas, [self.animal_id], [self.chan], data_type=self.seg_type)
        self.results_data = data_loader.load_data()

        if self.results_data is None:
            return False
        return True

    def _get_gene_expression_data(self) -> pd.DataFrame:
        """
        Extract and quantify gene expression data.

        Returns:
            pd.DataFrame: Dataframe containing quantification of gene expression by region.
        """
        gene_expression_fn, gene = self.expression
        columns_to_load = ['spot_id', gene]
        show_info("Loading gene expression data...")
        gene_expression_df = pd.read_csv(gene_expression_fn, usecols=columns_to_load)
        gene_expression_df.rename(columns={gene: 'gene_expression'}, inplace=True)
        self.results_data = pd.merge(self.results_data, gene_expression_df, on='spot_id', how='left')
        self.results_data['gene_expression'] = self.results_data['gene_expression'].fillna(0)
        gene_data = self.results_data.groupby('acronym_parent')['gene_expression'].mean().reset_index()
        gene_data.rename(columns={"acronym_parent": "acronym", 'gene_expression': 'quant_distribution'}, inplace=True)
        gene_data['animal_id'] = self.animal_id
        return gene_data

    def _get_animal_data(self, animal_key: str, animal_id: str) -> pd.DataFrame:
        """
        Get segmentation quantification data for a single animal.

        Parameters:
            animal_key (str): Key identifying the animal in the results data.
            animal_id (str): ID of the animal.

        Returns:
            pd.DataFrame: Dataframe containing quantification data for the animal.
        """
        animal_data = self.results_data[self.results_data[animal_key] == animal_id].groupby(
            "acronym_parent").size().reset_index(name="quant_volume")
        animal_data.rename(columns={"acronym_parent": "acronym"}, inplace=True)
        animal_data['quant_distribution'] = animal_data['quant_volume'] / animal_data['quant_volume'].sum()
        animal_data['animal_id'] = animal_id

        return animal_data

    # def _plot_quant_data(self, df):
    #
    #     clrs = sns.color_palette(self.plotting_params["cmap"])
    #     mpl_widget = FigureCanvas(Figure(figsize=self.plotting_params["figsize"]))
    #
    #     plt_axis = self.plotting_params["plt_axis"]
    #     axis_dict = {
    #         'AP': ['ap_mm', 'antero-posterior coordinates [mm]'],
    #         'ML': ['ml_mm', 'medio-lateral coordinates [mm]'],
    #         'DV': ['dv_mm', 'dorso-ventral coordinates [mm]']
    #     }
    #
    #     static_ax = mpl_widget.figure.subplots(1, 2)
    #     df.iloc[0][df.iloc[0]<0] = 0
    #     if self.is_merge:
    #         df = pd.DataFrame(df.mean(axis=0)).transpose()
    #     static_ax[0].pie(df.iloc[0], labels=df.columns.to_list(), colors=clrs, autopct='%.0f%%', normalize=True)
    #     if self.expression:
    #         static_ax[0].title.set_text(f"quantification of {self.expression[1]} expression")
    #     else:
    #         static_ax[0].title.set_text(f"quantification of {self.seg_type} in {self.chan} channel")
    #     static_ax[0].axis('off')
    #     if len(plt_axis) == 1:
    #         if self.expression:
    #             sns.lineplot(ax=static_ax[1], data=self.results_data, x=axis_dict[plt_axis[0]][0], y='gene_expression',
    #                          color=clrs[-2])
    #         else:
    #             if self.is_merge:
    #                 sns.kdeplot(ax=static_ax[1], data=self.results_data, x=axis_dict[plt_axis[0]][0], hue='animal_id_ind',
    #                             palette=sns.light_palette(clrs[-2]), common_norm=False, fill=True, legend=False)
    #                 sns.kdeplot(ax=static_ax[1], data=self.results_data, x=axis_dict[plt_axis[0]][0], color=clrs[-2],
    #                         common_norm=False, fill=True, legend=False)
    #             else:
    #                 sns.kdeplot(ax=static_ax[1], data=self.results_data, x=axis_dict[plt_axis[0]][0], color=clrs[-2],
    #                             common_norm=False, fill=True, legend=False)
    #         static_ax[1].set_xlabel(axis_dict[plt_axis[0]][1])
    #     else:
    #         if self.expression:
    #             x_bins = 25
    #             y_bins = 15
    #             # results_data_binned = pd.DataFrame()
    #             self.results_data['x'] = pd.cut(self.results_data[axis_dict[plt_axis[0]][0]], bins=x_bins, labels=False)
    #             self.results_data['y']= pd.cut(self.results_data[axis_dict[plt_axis[1]][0]], bins=y_bins, labels=False)
    #             x_bin_labels = self.results_data.groupby('x')[axis_dict[plt_axis[0]][0]].mean()
    #             y_bin_labels = self.results_data.groupby('y')[axis_dict[plt_axis[1]][0]].mean()
    #             pivot_df = self.results_data.pivot_table(values='gene_expression', index='y', columns='x', aggfunc='mean', dropna=False)
    #             pivot_df.index = pivot_df.index.map(round(y_bin_labels, 2))
    #             pivot_df.columns = pivot_df.columns.map(round(x_bin_labels, 2))
    #             # pivot_df=pivot_df.fillna(0)
    #             sns.heatmap(ax=static_ax[1], data=pivot_df, cmap=self.plotting_params["cmap"], vmin=0, vmax=pivot_df.max().max()*1.5)
    #         else:
    #             if self.is_merge:
    #                 sns.kdeplot(ax=static_ax[1], data=self.results_data, x=axis_dict[plt_axis[0]][0],
    #                             y=axis_dict[plt_axis[1]][0], hue='animal_id_ind', palette=sns.light_palette(clrs[-2]),
    #                             common_norm=False, fill=True, legend=False)
    #                 # sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0],
    #                 #             y=axis_dict[plt_axis[1]][0],
    #                 #             color=clrs[-2], common_norm=False, fill=True, legend=False)
    #             else:
    #                 sns.kdeplot(ax=static_ax[1], data=self.results_data, x=axis_dict[plt_axis[0]][0], y=axis_dict[plt_axis[1]][0] ,
    #                         color=clrs[-2], common_norm=False, fill=True, legend=False)
    #         static_ax[1].set_xlabel(axis_dict[plt_axis[0]][1])
    #         static_ax[1].set_ylabel(axis_dict[plt_axis[1]][1])
    #     static_ax[1].spines['top'].set_visible(False)
    #     static_ax[1].spines['right'].set_visible(False)
    #     if self.expression:
    #         static_ax[1].title.set_text(f"kde plot of {self.expression[1]} expression")
    #         save_fn = self.results_dir.joinpath(f'quantification_{self.seg_type}_{self.expression[1]}.svg')
    #     else:
    #         static_ax[1].title.set_text(f"kde plot of {self.seg_type} in {self.chan} channel")
    #         save_fn = self.results_dir.joinpath(f'quantification_{self.seg_type}_{self.chan}.svg')
    #     if self.plotting_params["save_fig"]:
    #         mpl_widget.figure.savefig(save_fn)
    #
    #     return mpl_widget