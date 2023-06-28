from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from superqt import QCollapsible
from napari.qt.threading import thread_worker
from magicgui import magicgui
import cv2
import numpy as np
import pandas as pd
from matplotlib import path
from natsort import natsorted
from sklearn.preprocessing import minmax_scale
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from napari_dmc_brainmap.utils import get_animal_id, get_info, get_parent, split_strings_layers, clean_results_df
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.find_structure import sliceHandle
from napari_dmc_brainmap.visualization.visualization_tools import dummy_load_allen_structure_tree
import json
import time


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
    if seg_type == 'injection_side':
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

    slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
    s.setImgFolder(regi_dir)
    # set which slice in there
    s.setSlice(slice_idx)
    # s.visualizeMapping(coords)
    section_data = s.getBrainArea(coords, (curr_im + regi_suffix))
    return section_data

def plot_quant_injection_side(df): #(input_path, c):

    # results_dir = get_info(input_path, 'results', channel=c, seg_type='injection_side', only_dir=True)
    # fn = results_dir.joinpath('quantification_injection_side.csv')
    # df = pd.read_csv(fn)
    # df = df.drop('animal_id', axis=1)
    clrs = sns.color_palette(quant_inj_widget.cmap.value)
    mpl_widget = FigureCanvas(Figure(figsize=([int(i) for i in quant_inj_widget.plot_size.value.split(',')])))
    static_ax = mpl_widget.figure.subplots()
    static_ax.pie(df.iloc[0], labels=df.columns.to_list(), colors=clrs, autopct='%.0f%%', normalize=True)
    # static_ax.title.set_text('quantification of the injection side in ' + c + " channel")
    static_ax.axis('off')
    # if quant_inj_widget.save_fig.value:
        # save_fn = results_dir.joinpath('quantification_injection_side.svg')
        # mpl_widget.figure.savefig(save_fn)
    return mpl_widget


@thread_worker
def create_results_file(input_path, seg_type, channels, seg_folder, regi_chan):


    animal_id = get_animal_id(input_path)
    regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'sharpy_track', channel=regi_chan)
    with open(regi_dir.joinpath('registration.json')) as fn:
        regi_data = json.load(fn)

    s = sliceHandle(regi_dir.joinpath('registration.json'))
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
                section_data = transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir,
                                                            seg_im_suffix, regi_data,
                                                            regi_dir, regi_suffix)
                data = pd.concat((data, section_data))
            fn = results_dir.joinpath(animal_id + '_' + seg_type + '.csv')
            data.to_csv(fn)
            print("done! data saved to: " + str(fn))
        else:
            print("No segmentation images found in " + str(segment_dir))

@thread_worker
def quantify_injection_side(input_path, chan, seg_type='injection_side'):


    if not seg_type == 'injection_side':
        print("not implemented! please, select 'injection_side' as segmentation type")
        return

    st = dummy_load_allen_structure_tree()
    animal_id = get_animal_id(input_path)
    results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
    results_fn = results_dir.joinpath(animal_id + '_' + seg_type + '.csv')
    if results_fn.exists():
        results_data = pd.read_csv(results_fn)  # load the data
        results_data['sphinx_id'] -= 1  # correct for matlab indices starting at 1
        results_data['animal_id'] = [animal_id] * len(
                results_data)  # add the animal_id as a column for later identification
        # add the injection hemisphere stored in params.json file
        params_file = input_path.joinpath('params.json')  # directory of params.json file   # todo this as function
        with open(params_file) as fn:  # load the file
            params_data = json.load(fn)

    results_data = clean_results_df(results_data, st)
    # step 1: get the absolute pixel count on area level (not layers)
    # add parent acronym to the injection data
    acronym_parent = [split_strings_layers(s)[0] for s in results_data['acronym']]
    # acronym_parent = [get_parent(s, st) for s in results_data['acronym']]  # todo this is for layers, but very slow
    results_data['acronym_parent'] = acronym_parent
    # count pixels (injection side) for each cell, add 0 for empty regions
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

    save_fn = results_dir.joinpath('quantification_injection_side.csv')
    quant_df_pivot.to_csv(save_fn)
    print("Relative injection side for " + chan + " channel:")
    print(quant_df_pivot)
    return [quant_df_pivot, chan]



@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                            'folder containing segmentation results'),
    seg_folder=dict(widget_type='LineEdit', label='folder name of segmentation images: ', value='rgb',
                        tooltip='name of folder containing the segmentation images, needs to be in same folder as '
                                'folder containing the segmentation results  (i.e. animal_id folder)'),
    regi_chan=dict(widget_type='ComboBox', label='registration channel',
                  choices=['dapi', 'green', 'n3', 'cy3', 'cy5'], value='green',
                  tooltip='select the channel you registered to the brain atlas'),
    seg_type=dict(widget_type='ComboBox', label='segmentation type',
                  choices=['cells', 'injection_side', 'projections', 'optic_fiber', 'neuropixels_probe'], value='cells',
                  tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'),
    channels=dict(widget_type='Select', label='selected channels', value=['green', 'cy3'],
                  choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                  tooltip='select channels to be selected for cell segmentation, '
                          'to select multiple hold ctrl/shift'),
    call_button=False
)
def results_widget(
        input_path,
        seg_folder,
        regi_chan,
        seg_type,
        channels

):
    return results_widget

@magicgui(
    layout='vertical',
    save_fig=dict(widget_type='CheckBox', label='save figure?', value=False,
                       tooltip='tick to save figure'),
    plot_size=dict(widget_type='LineEdit', label='enter plot size',
                            value='8,6', tooltip='enter the COMMA SEPERATED size of the plot'),
    cmap=dict(widget_type='LineEdit', label='colormap',
              value='Blues', tooltip='enter colormap to use for the pie chart'),
    call_button=False
)
def quant_inj_widget(
        save_fig,
        plot_size,
        cmap
):
    return quant_inj_widget


class ResultsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        results = results_widget
        btn_results = QPushButton("create results file")
        btn_results.clicked.connect(self._create_results_file)

        self._collapse_quant = QCollapsible('Quantify injection side: expand for more', self)
        quant_inj = quant_inj_widget
        self._collapse_quant.addWidget(quant_inj.native)
        btn_quant_inj = QPushButton("quantify injection side")
        btn_quant_inj.clicked.connect(self._quantify_injection_side)
        self._collapse_quant.addWidget(btn_quant_inj)



        self.layout().addWidget(results.native)
        self.layout().addWidget(btn_results)
        self.layout().addWidget(self._collapse_quant)

    def _create_results_file(self):
        input_path = results_widget.input_path.value
        seg_folder = results_widget.seg_folder.value
        regi_chan = results_widget.regi_chan.value
        seg_type = results_widget.seg_type.value
        channels = results_widget.channels.value
        worker_results_file = create_results_file(input_path, seg_type, channels, seg_folder, regi_chan)
        worker_results_file.start()


    def _quantify_injection_side(self):
        input_path = results_widget.input_path.value
        channels = results_widget.channels.value
        for chan in channels:
            worker_quantification = quantify_injection_side(input_path, chan)
            worker_quantification.returned.connect(self._plot_quant_injection_side)
            worker_quantification.start()


    def _plot_quant_injection_side(self, in_data):
        df, chan = in_data
        input_path = results_widget.input_path.value
        results_dir = get_info(input_path, 'results', channel=chan, seg_type='injection_side', only_dir=True)
        clrs = sns.color_palette(quant_inj_widget.cmap.value)
        mpl_widget = FigureCanvas(Figure(figsize=([int(i) for i in quant_inj_widget.plot_size.value.split(',')])))
        static_ax = mpl_widget.figure.subplots()
        static_ax.pie(df.iloc[0], labels=df.columns.to_list(), colors=clrs, autopct='%.0f%%', normalize=True)
        static_ax.title.set_text('quantification of the injection side in ' + chan + " channel")
        static_ax.axis('off')
        if quant_inj_widget.save_fig.value:
            save_fn = results_dir.joinpath('quantification_injection_side.svg')
            mpl_widget.figure.savefig(save_fn)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

