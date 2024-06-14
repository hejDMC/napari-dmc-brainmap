from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from superqt import QCollapsible
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import cv2
import numpy as np
import pandas as pd
from matplotlib import path
from natsort import natsorted
from sklearn.preprocessing import minmax_scale
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from napari_dmc_brainmap.utils import get_animal_id, get_info, split_strings_layers, clean_results_df, load_params, \
    create_regi_dict
from napari_dmc_brainmap.results.find_structure import sliceHandle
from bg_atlasapi import BrainGlobeAtlas
import json


def regi_points_polygon(x_scaled, y_scaled):

    poly_points = [(x_scaled[i], y_scaled[i]) for i in range(0, len(x_scaled))]
    polygon = path.Path(poly_points)
    x_min, x_max = x_scaled.min(), x_scaled.max()
    y_min, y_max = y_scaled.min(), y_scaled.max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, (x_max - x_min) + 1),
                         np.linspace(y_min, y_max, (y_max - y_min) + 1))
    canvas_points = [(np.ndarray.flatten(xx)[i], np.ndarray.flatten(yy)[i]) for i in
                     range(0, len(np.ndarray.flatten(xx)))]
    idx_in_polygon = polygon.contains_points(canvas_points)
    points_in_polygon = [c for c, i in zip(canvas_points, idx_in_polygon) if i]
    x_poly = [p[0] for p in points_in_polygon]
    y_poly = [p[1] for p in points_in_polygon]
    coords = np.stack([x_poly, y_poly], axis=1)
    return coords



def transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir, seg_im_suffix, regi_data, regi_dir, regi_suffix):
    # todo input differently?
    curr_im = im[:-len(segment_suffix)]
    img = cv2.imread(str(seg_im_dir.joinpath(curr_im + seg_im_suffix)))
    y_im, x_im, z_im = img.shape  # original resolution of image
    # correct for 0 indices
    y_im -= 1
    x_im -= 1
    img_regi = cv2.imread(str(regi_dir.joinpath(curr_im + regi_suffix)))
    y_low, x_low, z_low = img_regi.shape  # original resolution of image
    # correct for 0 indices
    y_low -= 1
    x_low -= 1

    segment_data = pd.read_csv(segment_dir.joinpath(im))
    y_pos = list(segment_data['Position Y'])
    x_pos = list(segment_data['Position X'])
    # append mix max values for rescaling
    y_pos.append(0)
    y_pos.append(y_im)
    x_pos.append(0)
    x_pos.append(x_im)
    y_scaled = np.ceil(minmax_scale(y_pos, feature_range=(0, y_low)))[:-2].astype(int)
    x_scaled = np.ceil(minmax_scale(x_pos, feature_range=(0, x_low)))[:-2].astype(int)
    if seg_type == 'injection_site':
        for n in segment_data['idx_shape'].unique():
            n_idx = segment_data.index[segment_data['idx_shape'] == n].tolist()
            curr_x = np.array([x_scaled[i] for i in n_idx])
            curr_y = np.array([y_scaled[i] for i in n_idx])
            curr_coords = regi_points_polygon(curr_x, curr_y)
            if n == 0:
                coords = curr_coords
            else:
                coords = np.concatenate((coords, curr_coords), axis=0)

    else:
        coords = np.stack([x_scaled, y_scaled], axis=1)

    # slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
    s.setImgFolder(regi_dir)
    # set which slice in there
    s.setSlice(curr_im + regi_suffix)
    section_data = s.getBrainArea(coords, (curr_im + regi_suffix))
    return section_data

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


@thread_worker
def create_results_file(input_path, seg_type, channels, seg_folder, regi_chan, params_dict):


    animal_id = get_animal_id(input_path)
    regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'sharpy_track', channel=regi_chan)
    with open(regi_dir.joinpath('registration.json')) as fn:
        regi_data = json.load(fn)

    regi_dict = create_regi_dict(input_path, regi_chan)
    s = sliceHandle(regi_dict)
    if seg_type == "optic_fiber" or seg_type == "neuropixels_probe":
        seg_super_dir = get_info(input_path, 'segmentation', seg_type=seg_type, only_dir=True)
        channels = natsorted([f.parts[-1] for f in seg_super_dir.iterdir() if f.is_dir()])

    for chan in channels:
        data = pd.DataFrame()
        if seg_folder == 'rgb':
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, seg_folder)
        else:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, seg_folder, channel=chan)
        segment_dir, segment_list, segment_suffix = get_info(input_path, 'segmentation', channel=chan, seg_type=seg_type)
        if len(segment_list) > 0:
            results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
            for im in segment_list:
                try:
                    section_data = transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir,
                                                                seg_im_suffix, regi_data,
                                                                regi_dir, regi_suffix)
                    if section_data is None:
                        pass
                    else:
                        data = pd.concat((data, section_data))
                except KeyError: # something wrong with registration data
                    print("Registration data for {} is not complete, skip.".format(im))
                
            fn = results_dir.joinpath(animal_id + '_' + seg_type + '.csv')
            data.to_csv(fn)
            print("done! data saved to: " + str(fn))
        else:
            print("No segmentation images found in " + str(segment_dir))
    print("DONE!")

@thread_worker
def quantify_injection_site(input_path, atlas, chan, seg_type='injection_site'):

    if seg_type not in ['injection_site', 'cells']:
        print("not implemented! please, select 'injection_site' as segmentation type")
        return

    animal_id = get_animal_id(input_path)
    results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
    results_fn = results_dir.joinpath(animal_id + '_' + seg_type + '.csv')
    if results_fn.exists():
        results_data = pd.read_csv(results_fn)  # load the data
        results_data['animal_id'] = [animal_id] * len(
                results_data)  # add the animal_id as a column for later identification
        # add the injection hemisphere stored in params.json file
    else:
        return
    results_data = clean_results_df(results_data, atlas)
    # step 1: get the absolute pixel count on area level (not layers)
    # add parent acronym to the injection data
    print(results_data['acronym'].unique())
    print(atlas.metadata['name'])
    acronym_parent = [split_strings_layers(s, atlas_name=atlas.metadata['name'])[0] for s in results_data['acronym']]
    results_data['acronym_parent'] = acronym_parent
    print(acronym_parent)
    # count pixels (injection site) for each cell, add 0 for empty regions
    quant_df = pd.DataFrame()
    temp_data = pd.DataFrame(results_data[results_data["animal_id"] == animal_id]
                                         ["acronym_parent"].value_counts())
    temp_data = temp_data.reset_index()
    temp_data = temp_data.rename(columns={"acronym_parent": "acronym", "count": "injection_volume"})

    temp_data['injection_distribution'] = temp_data['injection_volume'] / temp_data[
            'injection_volume'].sum()

    temp_data['animal_id'] = animal_id
    quant_df = pd.concat((quant_df, temp_data), axis=0)

    quant_df_pivot = quant_df.pivot(columns='acronym', values='injection_distribution',
                                    index='animal_id')

    save_fn = results_dir.joinpath('quantification_injection_site.csv')
    quant_df_pivot.to_csv(save_fn)
    print("Relative injection side for " + chan + " channel:")
    print(quant_df_pivot)
    return [quant_df_pivot, chan, seg_type, results_data]



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
                            choices=['cells', 'injection_site', 'projections', 'optic_fiber', 'neuropixels_probe'],
                            value='cells',
                            tooltip='select the segmentation type you want to create results from'),
              channels=dict(widget_type='Select', 
                            label='selected channels', 
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select channels for results files, '
                            'to select multiple hold ctrl/shift'),
              call_button=False)

    def results_widget(
            input_path,
            seg_folder,
            regi_chan,
            seg_type,
            channels):
        pass
    return results_widget


def initialize_quantinj_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox', 
                            label='save figure?', 
                            value=False,
                            tooltip='tick to save figure'),
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
    
    def quant_inj_widget(
            save_fig,
            plot_size,
            cmap,
            kde_axis):
        pass
    return quant_inj_widget


class ResultsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.results = initialize_results_widget()
        btn_results = QPushButton("create results file")
        btn_results.clicked.connect(self._create_results_file)

        self._collapse_quant = QCollapsible('Quantify injection site: expand for more', self)
        self.quant_inj = initialize_quantinj_widget()
        self._collapse_quant.addWidget(self.quant_inj.native)
        btn_quant_inj = QPushButton("quantify injection site")
        btn_quant_inj.clicked.connect(self._quantify_injection_site)
        self._collapse_quant.addWidget(btn_quant_inj)



        self.layout().addWidget(self.results.native)
        self.layout().addWidget(btn_results)
        self.layout().addWidget(self._collapse_quant)

    def _create_results_file(self):
        input_path = self.results.input_path.value
        seg_folder = self.results.seg_folder.value
        regi_chan = self.results.regi_chan.value
        seg_type = self.results.seg_type.value
        channels = self.results.channels.value
        params_dict = load_params(input_path)
        worker_results_file = create_results_file(input_path, seg_type, channels, seg_folder, regi_chan, params_dict)
        worker_results_file.start()


    def _quantify_injection_site(self):
        input_path = self.results.input_path.value
        channels = self.results.channels.value
        seg_type = self.results.seg_type.value
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas("allen_mouse_10um")
        for chan in channels:
            worker_quantification = quantify_injection_site(input_path, atlas, chan, seg_type=seg_type)
            worker_quantification.returned.connect(self._plot_quant_injection_site)
            worker_quantification.start()


    def _plot_quant_injection_site(self, in_data):
        df, chan, seg_type, results_data = in_data
        input_path = self.results.input_path.value
        results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, only_dir=True)
        clrs = sns.color_palette(self.quant_inj.cmap.value)
        mpl_widget = FigureCanvas(Figure(figsize=([int(i) for i in self.quant_inj.plot_size.value.split(',')])))

        plt_axis = self.quant_inj.kde_axis.value.split('/')
        axis_dict = {
            'AP': ['ap_mm', 'antero-posterior coordinates [mm]'],
            'ML': ['ml_mm', 'medio-lateral coordinates [mm]'],
            'DV': ['dv_mm', 'dorso-ventral coordinates [mm]']
        }

        static_ax = mpl_widget.figure.subplots(1, 2)
        static_ax[0].pie(df.iloc[0], labels=df.columns.to_list(), colors=clrs, autopct='%.0f%%', normalize=True)
        static_ax[0].title.set_text('quantification of ' + seg_type + ' in ' + chan + " channel")
        static_ax[0].axis('off')
        if len(plt_axis) == 1:
            sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0], color=clrs[-1],
                        common_norm=False, fill=True, legend=False)
            static_ax[1].set_xlabel(axis_dict[plt_axis[0]][1])
        else:
            sns.kdeplot(ax=static_ax[1], data=results_data, x=axis_dict[plt_axis[0]][0], y=axis_dict[plt_axis[1]][0] ,
                        color=clrs[-1], common_norm=False, fill=True, legend=False)
            static_ax[1].set_xlabel(axis_dict[plt_axis[0]][1])
            static_ax[1].set_ylabel(axis_dict[plt_axis[1]][1])
        static_ax[1].spines['top'].set_visible(False)
        static_ax[1].spines['right'].set_visible(False)
        static_ax[1].title.set_text('kde plot of ' + seg_type + ' in ' + chan + " channel")
        if self.quant_inj.save_fig.value:
            save_fn = results_dir.joinpath('quantification_injection_site.svg')
            mpl_widget.figure.savefig(save_fn)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

