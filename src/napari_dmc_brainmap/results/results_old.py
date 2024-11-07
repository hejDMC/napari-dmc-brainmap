import sys

from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from superqt import QCollapsible
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import numpy as np
import pandas as pd

from natsort import natsorted

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'

import seaborn as sns
from napari_dmc_brainmap.utils import get_animal_id, get_info, split_strings_layers, clean_results_df, load_params, \
    create_regi_dict, split_to_list
from napari_dmc_brainmap.visualization.visualization_tools import load_data
from napari_dmc_brainmap.results.results_tools import sliceHandle, transform_points_to_regi, \
    export_results_to_brainrender
from napari_dmc_brainmap.results.tract_calculation import load_probe_data, get_linefit3d, get_probe_tract
from napari_dmc_brainmap.results.probe_vis.probe_vis.view.ProbeVisualizer import ProbeVisualizer
from bg_atlasapi import BrainGlobeAtlas
import json


# def plot_quant_injection_site(df): #(input_path, c):
#
#     # results_dir = get_info(input_path, 'results', channel=c, seg_type='injection_site', only_dir=True)
#     # fn = results_dir.joinpath('quantification_injection_site.csv')
#     # df = pd.read_csv(fn)
#     # df = df.drop('animal_id', axis=1)
#     clrs = sns.color_palette(quant_inj_widget.cmap.value)
#     mpl_widget = FigureCanvas(Figure(figsize=([int(i) for i in quant_inj_widget.plot_size.value.split(',')])))
#     static_ax = mpl_widget.figure.subplots()
#     static_ax.pie(df.iloc[0], labels=df.columns.to_list(), colors=clrs, autopct='%.0f%%', normalize=True)
#     # static_ax.title.set_text('quantification of the injection site in ' + c + " channel")
#     static_ax.axis('off')
#     # if quant_inj_widget.save_fig.value:
#         # save_fn = results_dir.joinpath('quantification_injection_site.svg')
#         # mpl_widget.figure.savefig(save_fn)
#     return mpl_widget

def calculate_probe_tract(s, input_path, seg_type, params_dict, probe_insert):
    # get number of probes
    results_dir = get_info(input_path, 'results', seg_type=seg_type, only_dir=True)
    probes_list = natsorted([p.parts[-1] for p in results_dir.iterdir() if p.is_dir()])
    probes_dict = {}
    ax_map = {'ap': 'AP', 'si': 'DV', 'rl': 'ML'}
    # if len(probe_insert) != len(probes_list):
    #     print("WARNING, different number of probes and probe insert lengths detected!")
    #     diff = len(probe_insert) - len(probes_list)
    #     if diff < 0:
    #         for d in range(diff):
    #             probe_insert.append(4000)

    print("calculating probe tract for...")
    for i in range(
            len(probes_list) - len(probe_insert)):  # append false value if less probe insert values that probes found
        print("Warning -- less manipulator values than probes provides, estimation of probe track from clicked points "
              "is still experimental!")
        probe_insert.append(False)

    for probe, p_insert in zip(probes_list, probe_insert):
        print(f"... {probe}")
        probe_df = load_probe_data(results_dir, probe, s.atlas)

        linefit, linevox, ax_primary = get_linefit3d(probe_df, s.atlas)
        print(p_insert)
        probe_tract, _col_names = get_probe_tract(input_path, s.atlas, seg_type, ax_primary, probe_df, probe,
                                                  p_insert, linefit, linevox)
        # save probe tract data
        animal_id = get_animal_id(input_path)
        save_fn = results_dir.joinpath(probe, f'{animal_id}_{seg_type}.csv')  # override file
        probe_tract.to_csv(save_fn)
        probes_dict[probe] = {'axis': ax_map[s.atlas.space.axes_description[ax_primary]]}
        probes_dict[probe]['Voxel'] = linevox[["a_coord", "b_coord", "c_coord"]].to_numpy().tolist()

    save_fn = results_dir.joinpath(f'{seg_type}_data.json')
    with open(save_fn, 'w') as f:
        json.dump(probes_dict, f)  # write multiple voxelized probes, file can be opened in probe visualizer
    print("DONE!")


@thread_worker
def create_results_file(input_path, seg_type, channels, seg_folder, regi_chan, params_dict, include_all, export,
                        probe_insert):
    animal_id = get_animal_id(input_path)
    regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'sharpy_track', channel=regi_chan)
    with open(regi_dir.joinpath('registration.json')) as fn:
        regi_data = json.load(fn)

    regi_dict = create_regi_dict(input_path, regi_chan)
    s = sliceHandle(regi_dict)
    if seg_type in ["optic_fiber", "neuropixels_probe"]:
        seg_super_dir = get_info(input_path, 'segmentation', seg_type=seg_type, only_dir=True)
        channels = natsorted([f.parts[-1] for f in seg_super_dir.iterdir() if f.is_dir()])

    for chan in channels:
        data = pd.DataFrame()
        if seg_folder == 'rgb':
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, seg_folder)
        else:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, seg_folder, channel=chan)
        segment_dir, segment_list, segment_suffix = get_info(input_path, 'segmentation', channel=chan,
                                                             seg_type=seg_type)
        if len(segment_list) > 0:
            results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, create_dir=True,
                                   only_dir=True)
            for im in segment_list:
                try:
                    section_data = transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir,
                                                            seg_im_suffix, regi_data,
                                                            regi_dir, regi_suffix)
                    if not include_all:
                        section_data = section_data[section_data['structure_id'] != 0].reset_index(drop=True)
                    if section_data is None:
                        pass
                    else:
                        data = pd.concat((data, section_data))
                except KeyError:  # something wrong with registration data
                    print("Registration data for {} is not complete, skip.".format(im))
            fn = results_dir.joinpath(f'{animal_id}_{seg_type}.csv')
            data.to_csv(fn)
            if export and seg_type == 'cells':
                bg_data = export_results_to_brainrender(data, s.atlas)
                bg_fn = results_dir.joinpath(f'{animal_id}_{seg_type}.npy')
                np.save(bg_fn, bg_data)
                print(f"exported data to brainrender format in {str(bg_fn)}")
            print(f"done! data saved to: {str(fn)}")
        else:
            print(f"No segmentation images found in {str(segment_dir)}")
    print("DONE!")
    if seg_type in ["optic_fiber", "neuropixels_probe"]:
        print(f'..calculating {seg_type} trajectory for {chan} ...')
        calculate_probe_tract(s, input_path, seg_type, params_dict, probe_insert)


@thread_worker
def quantify_results(input_path, atlas, chan, seg_type='injection_site', expression=False, is_merge=False):
    # if seg_type not in ['injection_site', 'cells']:
    #     print("not implemented! please, select 'injection_site' as segmentation type")
    #     return

    animal_id = get_animal_id(input_path)
    results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
    results_fn = results_dir.joinpath(animal_id + '_' + seg_type + '.csv')
    if results_fn.exists():
        results_data = pd.read_csv(results_fn)  # load the data
        results_data['animal_id'] = [animal_id] * len(
            results_data)  # add the animal_id as a column for later identification
        # add the injection hemisphere stored in params.json file
    else:
        print(f'no results file under: {results_fn}')
        return
    if atlas.metadata['name'] == 'allen_mouse':
        results_data = clean_results_df(results_data, atlas)
    # step 1: get the absolute pixel count on area level (not layers)
    # add parent acronym to the injection data
    acronym_parent = [split_strings_layers(s, atlas_name=atlas.metadata['name'])[0] for s in results_data['acronym']]
    results_data['acronym_parent'] = acronym_parent
    # count pixels (injection site) for each cell, add 0 for empty regions
    quant_df = pd.DataFrame()
    if expression:
        gene_expression_fn, gene = expression
        columns_to_load = ['spot_id', gene]
        print("loading gene expression data...")
        gene_expression_df = pd.read_csv(gene_expression_fn, usecols=columns_to_load)
        gene_expression_df.rename(columns={gene: 'gene_expression'}, inplace=True)
        results_data = pd.merge(results_data, gene_expression_df, on='spot_id', how='left')
        results_data['gene_expression'] = results_data['gene_expression'].fillna(0)
        temp_data = results_data.groupby('acronym_parent')['gene_expression'].mean().reset_index()
        temp_data.rename(columns={"acronym_parent": "acronym", 'gene_expression': 'quant_distribution'}, inplace=True)
    else:
        if is_merge:
            animal_key = 'animal_id_ind'
            animal_list = results_data[animal_key].unique()
        else:
            animal_key = 'animal_id'
            animal_list = results_data[animal_key].unique()
        for animal_id in animal_list:
            temp_data = pd.DataFrame(results_data[results_data[animal_key] == animal_id]
                                     ["acronym_parent"].value_counts())
            temp_data = temp_data.reset_index()
            temp_data.rename(columns={"acronym_parent": "acronym", "count": "quant_volume"}, inplace=True)

            temp_data['quant_distribution'] = temp_data['quant_volume'] / temp_data[
                'quant_volume'].sum()

            temp_data['animal_id'] = animal_id
            quant_df = pd.concat((quant_df, temp_data), axis=0)

    quant_df_pivot = quant_df.pivot(columns='acronym', values='quant_distribution',
                                    index='animal_id').fillna(0)

    if expression:
        save_fn = results_dir.joinpath(f'quantification_{seg_type}_{gene}.csv')
        quant_df_pivot.to_csv(save_fn)
        # print(f"Quantification of {seg_type} for {gene} gene:")
    else:
        save_fn = results_dir.joinpath(f'quantification_{seg_type}_{chan}.csv')
        quant_df_pivot.to_csv(save_fn)
    #     print(f"Quantification of {seg_type} for {chan} channel:")
    return [quant_df_pivot, chan, seg_type, results_data, expression, is_merge]


def initialize_results_widget() -> FunctionGui:
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
                                     'genes'],
                            value='cells',
                            tooltip="select the segmentation type you want to create results from."),
              probe_insert=dict(widget_type='LineEdit',
                                label='insertion depth of probe (um)',
                                value='4000',
                                tooltip='specifiy the depth of optic fibers/neuropixels probe in brain in um, if left '
                                        'empty insertion depth will be estimated based on segmentation (experimental)'),
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
            probe_insert,
            channels,
            include_all,
            export):
        pass

    return results_widget


def initialize_quant_widget() -> FunctionGui:
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
                            choices=['AP', 'ML', 'DV', 'AP/ML', 'AP/DV', 'ML/DV'],
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
    @magicgui(layout='vertical',
              call_button=False)
    def probe_visualizer():
        pass

    return probe_visualizer


class ResultsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.results = initialize_results_widget()
        btn_results = QPushButton("create results file")
        btn_results.clicked.connect(self._create_results_file)

        self._collapse_quant = QCollapsible('Quantify results file: expand for more', self)
        self.quant = initialize_quant_widget()
        self._collapse_quant.addWidget(self.quant.native)
        btn_quant = QPushButton("quantify results file")
        btn_quant.clicked.connect(self._quantify_results)
        self._collapse_quant.addWidget(btn_quant)

        self._collapse_merge = QCollapsible('Create merged datasets: expand for more', self)
        self.merge = initialize_merge_widget()
        self._collapse_merge.addWidget(self.merge.native)
        btn_merge = QPushButton("create merged datasets")
        btn_merge.clicked.connect(self._merge_datasets)
        self._collapse_merge.addWidget(btn_merge)

        self._collapse_probe_vis = QCollapsible('Launch ProbeViewer: expand for more', self)
        self.probe_vis = initialize_probevis_widget()
        self._collapse_probe_vis.addWidget(self.probe_vis.native)
        btn_probe_vis = QPushButton("start ProbeViewer")
        btn_probe_vis.clicked.connect(self._start_probe_visualizer)
        self._collapse_probe_vis.addWidget(btn_probe_vis)

        self.layout().addWidget(self.results.native)
        self.layout().addWidget(btn_results)
        self.layout().addWidget(self._collapse_quant)
        self.layout().addWidget(self._collapse_merge)
        self.layout().addWidget(self._collapse_probe_vis)

    def _create_results_file(self):
        input_path = self.results.input_path.value
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
        worker_results_file = create_results_file(input_path, seg_type, channels, seg_folder, regi_chan, params_dict,
                                                  include_all, export, probe_insert)
        worker_results_file.start()

    def _quantify_results(self):
        input_path = self.results.input_path.value
        params_dict = load_params(input_path)
        channels = self.results.channels.value
        seg_type = self.results.seg_type.value
        is_merge = self.quant.is_merge.value
        if self.quant.expression.value:
            try:
                gene_expression_fn = self.quant.gene_expression_file.value
            except IsADirectoryError:
                print(f'no gene expression .csv file found under: {str(gene_expression_fn)}')
            gene = self.quant.gene.value
            expression = [gene_expression_fn, gene]
        else:
            expression = False
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
        for chan in channels:
            worker_quantification = quantify_results(input_path, atlas, chan, seg_type=seg_type, expression=expression,
                                                     is_merge=is_merge)
            worker_quantification.returned.connect(self._plot_quant_data)
            worker_quantification.start()

    def _merge_datasets(self):
        input_path = self.merge.input_path.value
        animal_list = split_to_list(self.merge.animal_list.value)
        channels = self.merge.channels.value
        seg_type = self.merge.seg_type.value
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
        merge_id = self.merge.merge_id.value
        merge_path = input_path.joinpath(merge_id)
        for chan in channels:
            df = load_data(input_path, atlas, animal_list, [chan], data_type=seg_type)
            df.rename(columns={'animal_id': 'animal_id_ind'}, inplace=True)
            results_dir = get_info(merge_path, 'results', channel=chan, seg_type=seg_type, create_dir=True,
                                   only_dir=True)
            results_fn = results_dir.joinpath(merge_id + '_' + seg_type + '.csv')
            df.to_csv(results_fn)
        params_dict['animal_id'] = merge_id
        params_fn = merge_path.joinpath('params.json')
        with open(params_fn, 'w') as fn:
            json.dump(params_dict, fn, indent=4)

    def _plot_quant_data(self, in_data):
        df, chan, seg_type, results_data, expression, is_merge = in_data
        input_path = self.results.input_path.value
        results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, only_dir=True)
        clrs = sns.color_palette(self.quant.cmap.value)
        figsize = [int(i) for i in self.quant.plot_size.value.split(',')]
        mpl_widget = FigureCanvas(Figure(figsize=figsize))

        plt_axis = self.quant.kde_axis.value.split('/')
        axis_dict = {
            'AP': ['ap_mm', 'antero-posterior coordinates [mm]'],
            'ML': ['ml_mm', 'medio-lateral coordinates [mm]'],
            'DV': ['dv_mm', 'dorso-ventral coordinates [mm]']
        }

        static_ax = mpl_widget.figure.subplots(1, 2)
        df.iloc[0][df.iloc[0] < 0] = 0
        if is_merge:
            df = pd.DataFrame(df.mean(axis=0)).transpose()
        static_ax[0].pie(df.iloc[0], labels=df.columns.to_list(), colors=clrs, autopct='%.0f%%', normalize=True)
        if expression:
            static_ax[0].title.set_text(f"quantification of {expression[1]} expression")
        else:
            static_ax[0].title.set_text(f"quantification of {seg_type} in {chan} channel")
        static_ax[0].axis('off')
        if len(plt_axis) == 1:
            if expression:
                sns.lineplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0], y='gene_expression',
                             color=clrs[-2])
            else:
                if is_merge:
                    sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0], hue='animal_id_ind',
                                palette=sns.light_palette(clrs[-2]), common_norm=False, fill=True, legend=False)
                    sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0], color=clrs[-2],
                                common_norm=False, fill=True, legend=False)
                else:
                    sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0], color=clrs[-2],
                                common_norm=False, fill=True, legend=False)
            static_ax[1].set_xlabel(axis_dict[plt_axis[0]][1])
        else:
            if expression:
                x_bins = 25
                y_bins = 15
                # results_data_binned = pd.DataFrame()
                results_data['x'] = pd.cut(results_data[axis_dict[plt_axis[0]][0]], bins=x_bins, labels=False)
                results_data['y'] = pd.cut(results_data[axis_dict[plt_axis[1]][0]], bins=y_bins, labels=False)
                x_bin_labels = results_data.groupby('x')[axis_dict[plt_axis[0]][0]].mean()
                y_bin_labels = results_data.groupby('y')[axis_dict[plt_axis[1]][0]].mean()
                pivot_df = results_data.pivot_table(values='gene_expression', index='y', columns='x', aggfunc='mean',
                                                    dropna=False)
                pivot_df.index = pivot_df.index.map(round(y_bin_labels, 2))
                pivot_df.columns = pivot_df.columns.map(round(x_bin_labels, 2))
                # pivot_df=pivot_df.fillna(0)
                sns.heatmap(ax=static_ax[1], data=pivot_df, cmap=self.quant.cmap.value, vmin=0,
                            vmax=pivot_df.max().max() * 1.5)
            else:
                if is_merge:
                    sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0],
                                y=axis_dict[plt_axis[1]][0], hue='animal_id_ind', palette=sns.light_palette(clrs[-2]),
                                common_norm=False, fill=True, legend=False)
                    # sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0],
                    #             y=axis_dict[plt_axis[1]][0],
                    #             color=clrs[-2], common_norm=False, fill=True, legend=False)
                else:
                    sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0],
                                y=axis_dict[plt_axis[1]][0],
                                color=clrs[-2], common_norm=False, fill=True, legend=False)
            static_ax[1].set_xlabel(axis_dict[plt_axis[0]][1])
            static_ax[1].set_ylabel(axis_dict[plt_axis[1]][1])
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

    def _start_probe_visualizer(self):
        input_path = self.results.input_path.value
        params_dict = load_params(input_path)
        probe_vis = ProbeVisualizer(self.viewer, params_dict)
        probe_vis.show()