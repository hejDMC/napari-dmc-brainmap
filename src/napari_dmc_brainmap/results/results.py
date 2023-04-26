from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari.qt.threading import thread_worker
from magicgui import magicgui
import cv2
import numpy as np
import pandas as pd
from matplotlib import path
from sklearn.preprocessing import minmax_scale
from napari_dmc_brainmap.utils import get_animal_id, get_info, split_strings_layers, clean_results_df
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.find_structure import sliceHandle
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

    elif seg_type == 'cells':
        coords = np.stack([x_scaled, y_scaled], axis=1)
    # todo areas
    else:
        print('invalid segmentation type')
    slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
    s.setImgFolder(regi_dir)
    # set which slice in there
    s.setSlice(slice_idx)
    # s.visualizeMapping(coords)
    section_data = s.getBrainArea(coords, (curr_im + regi_suffix))
    return section_data



@thread_worker
def create_results_file(input_path, seg_type, channels, regi_chan):


    animal_id = get_animal_id(input_path)
    seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
    regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'sharpy_track', channel=regi_chan)
    with open(regi_dir.joinpath('registration.json')) as fn:
        regi_data = json.load(fn)

    s = sliceHandle(regi_dir.joinpath('registration.json'))
    if seg_type == 'injection_side':
        data = pd.DataFrame()
        segment_dir, segment_list, segment_suffix = get_info(input_path, 'segmentation', seg_type=seg_type)
        results_dir = get_info(input_path, 'results', seg_type=seg_type, create_dir=True, only_dir=True)
        for im in segment_list:
            section_data = transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir, seg_im_suffix,
                                                    regi_data,
                                                    regi_dir, regi_suffix)
            data = pd.concat((data, section_data))
        fn = results_dir.joinpath(animal_id + '_injection.csv')
        data.to_csv(fn)
        print("done! data saved to: " + str(fn))
    elif seg_type == 'cells':
        for chan in channels:
            data = pd.DataFrame()
            segment_dir, segment_list, segment_suffix = get_info(input_path, 'segmentation', channel=chan, seg_type=seg_type)
            results_dir = get_info(input_path, 'results', channel=chan, seg_type=seg_type, create_dir=True, only_dir=True)
            for im in segment_list:
                section_data = transform_points_to_regi(s, im, seg_type, segment_dir, segment_suffix, seg_im_dir,
                                                        seg_im_suffix, regi_data,
                                                        regi_dir, regi_suffix)
                data = pd.concat((data, section_data))
            fn = results_dir.joinpath(animal_id + '_cells.csv')
            data.to_csv(fn)
            print("done! data saved to: " + str(fn))

@thread_worker
def quantify_injection_side(input_path, seg_type, regi_chan):
    #todo: specify level
    if seg_type == 'cells':
        print("not implemented! please, select 'injection_side' as segmentation type")
        return
    regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'sharpy_track', channel=regi_chan)
    s = sliceHandle(regi_dir.joinpath('registration.json'))
    st = s.df_tree
    animal_id = get_animal_id(input_path)
    results_dir = get_info(input_path, 'results', seg_type=seg_type, create_dir=True, only_dir=True)
    results_fn = results_dir.joinpath(animal_id + '_injection.csv')  # todo fix this to be seg_type name
    if results_fn.exists():
        results_data = pd.read_csv(results_fn)  # load the data
        results_data['sphinx_id'] -= 1  # correct for matlab indices starting at 1
        results_data['animal_id'] = [animal_id] * len(
                results_data)  # add the animal_id as a column for later identification
        # add the injection hemisphere stored in params.json file
        params_file = input_path.joinpath('params.json')  # directory of params.json file   # todo this as function
        with open(params_file) as fn:  # load the file
            params_data = json.load(fn)
        injection_side = params_data['general']['injection_side']  # add the injection_side as a column
        results_data['injection_side'] = [injection_side] * len(results_data)
        # add if the location of a cell is ipsi or contralateral to the injection side
        # injection_data = get_ipsi_contra(injection_data)
        # read the genotype
        genotype = params_data['general']['genotype']
        # if geno != genotype:
        #     print("WARNING: genotype doesn't match for " + animal_id)
        # and add column
        results_data['genotype'] = [genotype] * len(results_data)
        # injection_data_merged = pd.concat([injection_data_merged, injection_data])
    print("loaded " + animal_id)
    results_data = clean_results_df(results_data, st)
    # step 1: get the absolute pixel count on area level (not layers)
    # add parent acronym to the injection data
    acronym_parent = [split_strings_layers(s)[0] for s in results_data['acronym']]
    results_data['acronym_parent'] = acronym_parent

    # get list of all areas with cells (=tgt_list)
    tgt_list = results_data['acronym_parent'].unique().tolist()

    # count pixels (injection side) for each cell, add 0 for empty regions
    quant_df = pd.DataFrame()
    temp_data = pd.DataFrame(results_data[results_data["animal_id"] == animal_id]
                                         ["acronym_parent"].value_counts())
    temp_data = temp_data.reset_index()
    temp_data = temp_data.rename(columns={"index": "acronym", "acronym_parent": "injection_volume"})
    # missing_areas = pd.DataFrame(set(tgt_list).difference(temp_data['acronym'].to_list()),
    #                                 columns={'acronym'})
    # missing_areas['injection_volume'] = 0  # todo this crashes on windows if no missing areas? (if n=1, no missing areas anyhow)
    # temp_data = pd.concat((temp_data, missing_areas), axis=0)
    # temp_data = temp_data.reset_index(drop=True)
    temp_data['injection_distribution'] = temp_data['injection_volume'] / temp_data[
            'injection_volume'].sum()

    temp_data['animal_id'] = animal_id
    quant_df = pd.concat((quant_df, temp_data), axis=0)

    quant_df_pivot = quant_df.pivot(columns='acronym', values='injection_distribution',
                                    index='animal_id')

    save_fn = results_dir.joinpath('quantification_injection_side.csv')
    quant_df_pivot.to_csv(save_fn)
    print(quant_df_pivot)

@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                            'folder containing segmentation results'),
    regi_chan=dict(widget_type='ComboBox', label='registration channel',
                  choices=['dapi', 'green', 'n3', 'cy3', 'cy5'], value='green',
                  tooltip='select the channel you registered to the brain atlas'),
    seg_type=dict(widget_type='ComboBox', label='segmentation type',
                  choices=['cells', 'injection_side'], value='cells',
                  tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'),
    channels=dict(widget_type='Select', label='selected channels', value=['green', 'cy3'],
                  choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                  tooltip='select channels to be selected for cell segmentation, '
                          'to select multiple hold ctrl/shift'),
    call_button=False
)
def results_widget(
        input_path,
        regi_chan,
        seg_type,
        channels

):
    return results_widget

class ResultsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        results = results_widget
        btn_results = QPushButton("create results file")
        btn_results.clicked.connect(self._create_results_file)
        btn_quant_inj = QPushButton("quantify injection side")
        btn_quant_inj.clicked.connect(self._quantify_injection_side)
        self.layout().addWidget(results.native)
        self.layout().addWidget(btn_results)
        self.layout().addWidget(btn_quant_inj)

    def _create_results_file(self):
        input_path = results_widget.input_path.value
        regi_chan = results_widget.regi_chan.value
        seg_type = results_widget.seg_type.value
        channels = results_widget.channels.value
        worker_results_file = create_results_file(input_path, seg_type, channels, regi_chan)
        worker_results_file.start()


    def _quantify_injection_side(self):
        input_path = results_widget.input_path.value
        regi_chan = results_widget.regi_chan.value
        seg_type = results_widget.seg_type.value
        worker_quantification = quantify_injection_side(input_path, seg_type, regi_chan)
        worker_quantification.start()

