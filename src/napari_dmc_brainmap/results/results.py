from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox
from superqt import QCollapsible
from napari import Viewer
from pathlib import Path
from typing import List, Tuple, Dict
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'

from napari.utils.notifications import show_info
from napari_dmc_brainmap.utils.color_manager import ColorManager
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.general_utils import split_to_list
from napari_dmc_brainmap.utils.params_utils import load_params
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, coord_mm_transform
from napari_dmc_brainmap.utils.gui_utils import ProgressBar, check_input_path
from napari_dmc_brainmap.utils.data_loader import DataLoader
from napari_dmc_brainmap.results.probe_vis.probe_vis.view.ProbeVisualizer import ProbeVisualizer
from napari_dmc_brainmap.results.results_helpers.results_creator import ResultsCreator
from napari_dmc_brainmap.results.results_helpers.results_quantifier import ResultsQuantifier
from bg_atlasapi import BrainGlobeAtlas
import json
import numpy as np


@thread_worker
def create_results_file(results_creator: ResultsCreator, progress_create_results: Signal) -> str:
    """
    Create a results file based on segmentation data.

    Parameters:
        results_creator (ResultsCreator): Instance to handle result file creation.
        progress_create_results (Signal): Signal to track progress.

    Returns:
        str: ID of the operation.
    """
    def progress_callback(progress):
        progress_create_results.emit(progress)
    results_creator.create_results(progress_callback=progress_callback)
    return "create_results"

@thread_worker
def quantify_results(results_quantifier: ResultsQuantifier, progress_quant_results: Signal) -> Tuple[str, List]:
    """
    Quantify segmentation results.

    Parameters:
        results_quantifier (ResultsQuantifier): Instance to handle quantification.
        progress_quant_results (Signal): Signal to track progress.

    Returns:
        Tuple[str, List]: ID of operation and plot data.
    """
    def progress_callback(progress):
        progress_quant_results.emit(progress)
    plot_data = results_quantifier.quantify(progress_callback=progress_callback)
    return ["quant_results", plot_data]

@thread_worker
def merge_datasets(
    input_path: Path,
    merge_id: str,
    channels: List[str],
    atlas: BrainGlobeAtlas,
    animal_list: List[str],
    seg_type: str,
    params_dict: Dict
) -> str:
    """
    Merge datasets from multiple animals into a single dataset.

    Parameters:
        input_path (Path): Path to the directory containing animal data.
        merge_id (str): Identifier for the merged dataset.
        channels (List[str]): Channels to include in the merge.
        atlas (BrainGlobeAtlas): Atlas instance for reference.
        animal_list (List[str]): List of animal IDs to merge.
        seg_type (str): Type of segmentation.
        params_dict (Dict): Parameters for merging.

    Returns:
        str: ID of operation.
    """
    merge_path = input_path.joinpath(merge_id)
    for chan in channels:
        data_loader = DataLoader(input_path, atlas, animal_list, [chan], data_type=seg_type)
        df = data_loader.load_data()
        # df = load_data(input_path, atlas, animal_list, [chan], data_type=seg_type)
        df.rename(columns={'animal_id': 'animal_id_ind'}, inplace=True)
        results_dir = get_info(merge_path, 'results', channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
        results_fn = results_dir.joinpath(merge_id + '_' + seg_type + '.csv')
        df.to_csv(results_fn)
    params_dict['animal_id'] = merge_id
    params_fn = merge_path.joinpath('params.json')
    with open(params_fn, 'w') as fn:
        json.dump(params_dict, fn, indent=4)
    return "merge_datasets"



def initialize_results_widget() -> FunctionGui:
    """
    Initialize the results widget for configuring segmentation result creation.

    Returns:
        FunctionGui: Configured widget for results creation.
    """
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
              seg_folder=dict(widget_type='LineEdit', 
                              label='folder name of segmentation images: ', 
                              value='rgb',
                              tooltip='name of folder containing the segmentation images, needs to be in same folder as '
                                    'folder containing the segmentation results  (i.e. animal_id folder)'),
              regi_chan=dict(widget_type='ComboBox', 
                             label='registration channel',
                             choices=['dapi', 'green', 'n3', 'cy3', 'cy5'], 
                             value='green',
                             tooltip='select the channel you registered to the brain atlas'),
              seg_type=dict(widget_type='ComboBox', 
                            label='segmentation type',
                            choices=['cells', 'injection_site', 'projections', 'optic_fiber', 'neuropixels_probe',
                                     'genes', 'single_cell', 'hcr'],
                            value='cells',
                            tooltip="select the segmentation type you want to create results from."),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels for results files, '
                            'to select multiple hold ctrl/shift'),
              include_all=dict(widget_type='CheckBox',
                            label='include segmented objects outside of brain?',
                            value=False,
                            tooltip='tick to include segmented objects that are outside of the brain'),
              probe_insert=dict(widget_type='LineEdit',
                                label='insertion depth of probe (um)',
                                value='4000',
                                tooltip='specifiy the depth of optic fibers/neuropixels probe in brain in um, if left '
                                        'empty insertion depth will be estimated based on segmentation (experimental)'),
              export=dict(widget_type='CheckBox', label='export to brainrender', value=False,
                          tooltip='export data to .npy formatted to be loaded into brainrender software '
                                  '(https://brainglobe.info/documentation/brainrender/). Currently, only implemented '
                                  'for cells.'),
              call_button=False)

    def results_widget(
            input_path,
            seg_folder,
            regi_chan,
            seg_type,
            channels,
            include_all,
            probe_insert,
            export):
        pass
    return results_widget


def initialize_quant_widget() -> FunctionGui:
    """
    Initialize the quantification widget for configuring result quantification.

    Returns:
        FunctionGui: Configured widget for result quantification.
    """
    @magicgui(layout='vertical',
              is_merge=dict(widget_type='CheckBox',
                            label='using merged data?',
                            value=False,
                            tooltip='tick when using data from merged animals '
                                    '(created with Create merged dataset widget below)'),
              save_fig=dict(widget_type='CheckBox', 
                            label='save figure?', 
                            value=False,
                            tooltip='tick to save figure'),
              expression=dict(widget_type='CheckBox',
                             label='quantify gene expression levels?',
                             value=False,
                             tooltip="Choose to visualize the expression levels of one target gene. This option requires "
                                     "the presence of a .csv file holding gene expression data, rows are gene "
                                     "expression, columns genes plus one column named 'spot_id' containing the spot ID"),
              gene_expression_file=dict(widget_type='FileEdit',
                                        label='gene expression data file: ',
                                        value='',
                                        mode='r',
                                        tooltip='file containing gene expression data by spot'),
              gene=dict(widget_type='LineEdit',
                        label='gene:',
                        value='Slc17a7',
                        tooltip='enter the name of the gene to visualize'),
              plot_size=dict(widget_type='LineEdit', 
                             label='enter plot size',
                             value='16,12',
                             tooltip='enter the COMMA SEPERATED size of the plot'),
              cmap=dict(widget_type='LineEdit', 
                        label='colormap',
                        value='Blues', 
                        tooltip='enter colormap to use for the pie chart'),
              kde_axis=dict(widget_type='ComboBox', 
                            label='select axis for density plots',
                            choices=['AP', 'ML', 'DV', 'AP/ML', 'AP/DV', 'DV/ML'],
                            value='AP',
                            tooltip='AP=antero-posterior, ML=medio-lateral, DV=dorso-ventral'),
              call_button=False)
    
    def quant_widget(
            is_merge,
            save_fig,
            expression,
            gene_expression_file,
            gene,
            plot_size,
            cmap,
            kde_axis):
        pass
    return quant_widget


def initialize_merge_widget() -> FunctionGui:
    """
    Initialize the merge widget for configuring dataset merging.

    Returns:
        FunctionGui: Configured widget for dataset merging.
    """
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit',
                              label='input path: ',
                              value='',
                              mode='d',
                              tooltip='directory of folder containing folders with animals'),
              animal_list=dict(widget_type='LineEdit',
                               label='list of animals',
                               value='animal1,animal2',
                               tooltip='enter the COMMA SEPERATED list of animals (no white spaces: animal1,animal2)'),
              merge_id=dict(widget_type='LineEdit',
                                        label='merge_id: ',
                                        value='merge_animal',
                                        tooltip='dummy animal_id that will store merged results for quantification, '
                                                'data will be stored as for normal animal_ids, '
                                                'i.e. results/seg_type/channel/*.csv'),
              channels=dict(widget_type='Select',
                            label='select channels to plot',
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select the channels with segmented cells to be plotted, '
                                    'to select multiple hold ctrl/shift'),
              seg_type=dict(widget_type='ComboBox',
                            label='segmentation type',
                            choices=['cells', 'injection_site', 'projections', 'optic_fiber', 'neuropixels_probe',
                                     'genes'],
                            value='cells',
                            tooltip="select the segmentation type you want to create results from."),
              call_button=False)
    def merge_widget(
            input_path,
            animal_list,
            merge_id,
            channels,
            seg_type):
        pass

    return merge_widget


def initialize_probevis_widget() -> FunctionGui:
    """
    Initialize the probe visualization widget for launching ProbeVisualizer.

    Returns:
        FunctionGui: Configured widget for ProbeVisualizer.
    """
    @magicgui(layout='vertical',
              call_button=False)
    def probe_visualizer():
        pass

    return probe_visualizer

class ResultsWidget(QWidget):
    """
    Widget for managing result creation, quantification, and dataset merging.
    """
    progress_create_results = Signal(int)
    """
    """
    progress_quant_results = Signal(int)
    """
    """
    def __init__(self, napari_viewer: Viewer) -> None:
        """
        Initialize the ResultsWidget.

        Parameters:
            napari_viewer (Viewer): Instance of Napari viewer.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.results = initialize_results_widget()
        self.btn_results = QPushButton("create results file")
        self.btn_results.clicked.connect(self._create_results_file)
        self.progress_bar_create_results = ProgressBar(self)
        self.progress_create_results.connect(self.progress_bar_create_results.set_value)

        self._collapse_quant = QCollapsible('Quantify results file: expand for more', self)
        self.quant = initialize_quant_widget()
        self._collapse_quant.addWidget(self.quant.native)
        self.btn_quant = QPushButton("quantify results file")
        self.btn_quant.clicked.connect(self._quantify_results)
        self._collapse_quant.addWidget(self.btn_quant)
        self.progress_bar_quant = ProgressBar(self)
        self._collapse_quant.addWidget(self.progress_bar_quant)
        self.progress_quant_results.connect(self.progress_bar_quant.set_value)

        self._collapse_merge = QCollapsible('Create merged datasets: expand for more', self)
        self.merge = initialize_merge_widget()
        self._collapse_merge.addWidget(self.merge.native)
        self.btn_merge = QPushButton("create merged datasets")
        self.btn_merge.clicked.connect(self._merge_datasets)
        self._collapse_merge.addWidget(self.btn_merge)

        self._collapse_probe_vis = QCollapsible('Launch ProbeViewer: expand for more', self)
        self.probe_vis = initialize_probevis_widget()
        self._collapse_probe_vis.addWidget(self.probe_vis.native)
        btn_probe_vis = QPushButton("start ProbeViewer")
        btn_probe_vis.clicked.connect(self._start_probe_visualizer)
        self._collapse_probe_vis.addWidget(btn_probe_vis)

        self.layout().addWidget(self.results.native)
        self.layout().addWidget(self.progress_bar_create_results)
        self.layout().addWidget(self.btn_results)
        self.layout().addWidget(self._collapse_quant)
        self.layout().addWidget(self._collapse_merge)
        self.layout().addWidget(self._collapse_probe_vis)

        self.atlas = None

    def _create_results_file(self) -> None:
        """
        Create results file from segmentation data.
        """
        input_path = self.results.input_path.value
        if not check_input_path(input_path):
            return
        seg_folder = self.results.seg_folder.value
        regi_chan = self.results.regi_chan.value
        seg_type = self.results.seg_type.value
        channels = self.results.channels.value
        params_dict = load_params(input_path)
        include_all = self.results.include_all.value
        export = self.results.export.value
        probe_insert = split_to_list(self.results.probe_insert.value, out_format='int')
        if not probe_insert:
            probe_insert = []

        results_creator = ResultsCreator(input_path, seg_type, channels, seg_folder, regi_chan, include_all, export,
                                         probe_insert)

        create_results_worker = create_results_file(results_creator, self.progress_create_results)
        create_results_worker.started.connect(
            lambda: self.btn_results.setText("Creating results file..."))
        create_results_worker.returned.connect(self._show_success_message)
        create_results_worker.start()


    def _quantify_results(self) -> None:
        """
        Quantify segmentation results.
        """
        input_path = self.results.input_path.value
        if not check_input_path(input_path):
            return
        params_dict = load_params(input_path)
        channels = self.results.channels.value
        seg_type = self.results.seg_type.value
        is_merge = self.quant.is_merge.value
        if self.quant.expression.value:
            try:
                gene_expression_fn = self.quant.gene_expression_file.value
            except IsADirectoryError:
                show_info(f'no gene expression .csv file found under: {str(gene_expression_fn)}')
            gene = self.quant.gene.value
            expression = [gene_expression_fn, gene]
        else:
            expression = False
        if not self.atlas:
            show_info("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
            show_info("...done!")
        plotting_params = {
            "cmap": self.quant.cmap.value,
            "figsize": [int(i) for i in self.quant.plot_size.value.split(',')],
            "plt_axis": self.quant.kde_axis.value.split('/'),
            "save_fig": self.quant.save_fig.value
        }
        results_quantifier = ResultsQuantifier(input_path, self.atlas, channels, plotting_params, seg_type=seg_type, expression=expression, is_merge=is_merge)
        worker_quantification = quantify_results(results_quantifier, self.progress_quant_results)
        worker_quantification.started.connect(
            lambda: self.btn_quant.setText("Quantifying data..."))
        worker_quantification.returned.connect(self._show_success_message)
        worker_quantification.returned.connect(self._plot_quant_data)
        worker_quantification.start()
        # for chan in channels:
        #     worker_quantification = quantify_results(input_path, atlas, chan, seg_type=seg_type, expression=expression, is_merge=is_merge)
        #     worker_quantification.returned.connect(self._plot_quant_data)
        #     worker_quantification.start()

    # def _plot_quant_data(self, in_data):
    #     plots = in_data[1]
    #     for p in plots:
    #         print(p)
    #         self.viewer.window.add_dock_widget(p, area='left').setFloating(True)

    def _plot_quant_data(self, in_data: Tuple) -> None:
        """
        Plot quantification results.

        Parameters:
            in_data (Tuple): Data to plot including DataFrame, channel, and parameters.
        """
        color_manager = ColorManager()
        plot_data = in_data[1]
        bregma = get_bregma(self.atlas.atlas_name)

        for data in [p_d for p_d in plot_data if p_d is not None]:
            df, chan, seg_type, results_data, expression, is_merge = data
            input_path = self.results.input_path.value
            if not check_input_path(input_path):
                return
            results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, only_dir=True)
            cmap = color_manager.create_custom_colormap(self.quant.cmap.value)

            figsize = [int(i) for i in self.quant.plot_size.value.split(',')]
            mpl_widget = FigureCanvas(Figure(figsize=figsize))

            plt_axis = self.quant.kde_axis.value.split('/')
            axis_dict = {
                'AP': ['ap_mm', 'antero-posterior coordinates [mm]', 'frontal'],
                'DV': ['dv_mm', 'dorso-ventral coordinates [mm]', 'horizontal'],
                'ML': ['ml_mm', 'medio-lateral coordinates [mm]', 'sagittal'],

            }

            static_ax = mpl_widget.figure.subplots(1, 2)
            df.iloc[0][df.iloc[0] < 0] = 0
            if is_merge:
                df = pd.DataFrame(df.mean(axis=0)).transpose()
            c_range = np.linspace(0.2, 1, len(df.columns))
            static_ax[0].pie(df.iloc[0], labels=df.columns.to_list(), colors=[cmap(i) for i in c_range], autopct='%.0f%%', normalize=True)
            if expression:
                static_ax[0].title.set_text(f"quantification of {expression[1]} expression")
            else:
                static_ax[0].title.set_text(f"quantification of {seg_type} in {chan} channel")
            static_ax[0].axis('off')
            if len(plt_axis) == 1:
                x_key = axis_dict[plt_axis[0]]
                if expression:
                    sns.lineplot(ax=static_ax[1], data=results_data, x=x_key[0], y='gene_expression',
                                 color=cmap(0.8))
                else:
                    if is_merge:
                        sns.kdeplot(ax=static_ax[1], data=results_data, x=x_key[0], hue='animal_id_ind',
                                    palette=sns.light_palette(cmap(0.8)), common_norm=False, fill=True, legend=False)
                        sns.kdeplot(ax=static_ax[1], data=results_data, x=x_key[0], color=cmap(0.8),
                                    common_norm=False, fill=True, legend=False)
                    else:
                        sns.kdeplot(ax=static_ax[1], data=results_data, x=x_key[0], color=cmap(0.8),
                                    common_norm=False, fill=True, legend=False)
                static_ax[1].set_xlabel(x_key[1])

                axis_idx = self.atlas.space.sections.index(x_key[2])
                axis_dim = self.atlas.shape[axis_idx]

                xmin = coord_mm_transform([axis_dim], [bregma[axis_idx]], [self.atlas.resolution[axis_idx]])
                xmax = coord_mm_transform([0], [bregma[axis_idx]], [self.atlas.resolution[axis_idx]])
                xticks = np.linspace(xmin, xmax, 5)

                static_ax[1].set_xlim([xmin, xmax])
                static_ax[1].set_xticks(xticks)

                _, ymax = static_ax[1].get_ylim()
                ymin = 0
                ymax = self._update_axis_max(ymax)
                yticks = np.linspace(ymin, ymax, 5)
                static_ax[1].set_ylim([ymin, ymax])
                static_ax[1].set_yticks(yticks)
                for y in yticks:
                    static_ax[1].axhline(y, color='gray', linestyle='dotted', linewidth=0.5, alpha=0.5)
            else:
                x_key = axis_dict[plt_axis[1]]
                y_key = axis_dict[plt_axis[0]]

                x_axis_idx = self.atlas.space.sections.index(x_key[2])
                x_axis_dim = self.atlas.shape[x_axis_idx]
                xmin = coord_mm_transform([x_axis_dim], [bregma[x_axis_idx]], [self.atlas.resolution[x_axis_idx]])
                xmax = coord_mm_transform([0], [bregma[x_axis_idx]], [self.atlas.resolution[x_axis_idx]])
                xticks = np.linspace(xmin, xmax, 5)

                y_axis_idx = self.atlas.space.sections.index(y_key[2])
                y_axis_dim = self.atlas.shape[y_axis_idx]
                ymin = coord_mm_transform([y_axis_dim], [bregma[y_axis_idx]], [self.atlas.resolution[y_axis_idx]])
                ymax = coord_mm_transform([0], [bregma[y_axis_idx]], [self.atlas.resolution[y_axis_idx]])
                yticks = np.linspace(ymin, ymax, 5)

                if expression:
                    x_bins = 15
                    y_bins = 25
                    # results_data_binned = pd.DataFrame()
                    results_data['x'] = pd.cut(results_data[x_key[0]], bins=x_bins, labels=False)
                    results_data['y'] = pd.cut(results_data[y_key[0]], bins=y_bins, labels=False)

                    x_bin_labels = results_data.groupby('x')[x_key[0]].mean()
                    y_bin_labels = results_data.groupby('y')[y_key[0]].mean()
                    pivot_df = results_data.pivot_table(values='gene_expression', index='y', columns='x', aggfunc='mean',
                                                        dropna=False)
                    pivot_df.index = pivot_df.index.map(round(y_bin_labels, 2))
                    pivot_df.columns = pivot_df.columns.map(round(x_bin_labels, 2))
                    if y_key[0] == 'ap_mm':
                        pivot_df = pivot_df.sort_index(ascending=False)
                    # pivot_df=pivot_df.fillna(0)
                    sns.heatmap(ax=static_ax[1], data=pivot_df, cmap=cmap, vmin=0,
                                vmax=pivot_df.max().max() * 1.5)
                    static_ax[1].tick_params(axis='y', rotation=360)


                else:
                    if is_merge:
                        sns.kdeplot(ax=static_ax[1], data=results_data, x=x_key[0],
                                    y=y_key[0], hue='animal_id_ind', palette=sns.light_palette(cmap(0.8)),
                                    common_norm=False, fill=True, legend=False)
                        # sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0],
                        #             y=axis_dict[plt_axis[1]][0],
                        #             color=clrs[-2], common_norm=False, fill=True, legend=False)
                    else:
                        sns.kdeplot(ax=static_ax[1], data=results_data, x=x_key[0],
                                    y=y_key[0],
                                    color=cmap(0.8), common_norm=False, fill=True, legend=False)
                    static_ax[1].set_xlim([xmin, xmax])
                    static_ax[1].set_xticks(xticks)

                    static_ax[1].set_ylim([ymin, ymax])
                    static_ax[1].set_yticks(yticks)
                static_ax[1].set_xlabel(x_key[1])
                static_ax[1].set_ylabel(y_key[1])


                # for y in yticks:
                #     static_ax[1].axhline(y, color='gray', linestyle='dotted', linewidth=0.5, alpha=0.5)

            static_ax[1].spines['top'].set_visible(False)
            static_ax[1].spines['right'].set_visible(False)

            if expression:
                static_ax[1].title.set_text(f"kde plot of {expression[1]} expression")
                save_fn = results_dir.joinpath(f'quantification_{seg_type}_{expression[1]}.svg')
            else:
                static_ax[1].title.set_text(f"kde plot of {seg_type} in {chan} channel")
                save_fn = results_dir.joinpath(f'quantification_{seg_type}_{chan}.svg')
            if self.quant.save_fig.value:
                mpl_widget.figure.savefig(save_fn)
            self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

    def _update_axis_max(self, ax_max: float) -> float:
        """
        Update ax_max value by rounding

        Parameters:
            ax_max: float
        Return:
            ax_max: float
        """
        num_str = f"{ax_max:.16f}"  # Ensures precision for small numbers
        decimal_part = num_str.split(".")[1]  # Extract decimal part
        count_zeros = len(decimal_part) - len(decimal_part.lstrip("0"))
        factor = 10 ** (count_zeros + 1)
        return np.ceil(ax_max * factor) / factor

    def _merge_datasets(self) -> None:
        """
        Merge datasets from multiple animals into one.
        """
        input_path = self.merge.input_path.value
        if not check_input_path(input_path):
            return
        if self.merge.animal_list.value == ':':
            animal_list = [f.parts[-1] for f in input_path.iterdir() if f.is_dir()]
        else:
            animal_list = split_to_list(self.merge.animal_list.value)
        channels = self.merge.channels.value
        seg_type = self.merge.seg_type.value
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        if not self.atlas:
            show_info("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
            show_info("...done!")
        merge_id = self.merge.merge_id.value
        worker_merge = merge_datasets(input_path, merge_id, channels, self.atlas, animal_list, seg_type, params_dict)
        worker_merge.started.connect(
            lambda: self.btn_merge.setText("Merging datasets..."))
        worker_merge.returned.connect(self._show_success_message)
        worker_merge.start()



    def _start_probe_visualizer(self) -> None:
        """
        Launch the ProbeVisualizer tool.
        """
        input_path = self.results.input_path.value
        if not check_input_path(input_path):
            return
        params_dict = load_params(input_path)
        probe_vis = ProbeVisualizer(self.viewer, params_dict)
        probe_vis.show()

    def _show_success_message(self, operation: str) -> None:
        """
        Display a success message after an operation completes.

        Parameters:
            operation (str): Type of operation completed.
        """
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        if operation == 'create_results':
            msg = 'Results file created!'
            self.btn_results.setText("create results file")  # Reset button text after process completion
            self.progress_create_results.emit(0)
        elif 'quant_results' in operation:
            msg = 'Quantification finished!'
            self.btn_quant.setText("quantify results file")  # Reset button text after process completion
            self.progress_quant_results.emit(0)
        else:
            msg = 'Datasets merged!'
            self.btn_merge.setText("create merged datasets")  # Reset button text after process completion

        msg_box.setText(msg)
        msg_box.setWindowTitle("Operation successful!")
        msg_box.exec_()