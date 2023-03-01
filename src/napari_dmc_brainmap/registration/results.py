from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
from natsort import natsorted
import cv2
import numpy as np
import pandas as pd
from matplotlib import path
from sklearn.preprocessing import minmax_scale
from napari_dmc_brainmap.registration.sharpy_track.model.find_structure import sliceHandle
import json

# todo path things for regi data

def results_widget():
    from napari.qt.threading import thread_worker

    def find_common_suffix(image_list, folder='unknown'):
        if len(image_list) > 1:
            for i in range(len(image_list[0])):
                if i > 0:
                    if image_list[0][-i] == image_list[1][-i]:
                        continue
                    else:
                        break
            common_suffix = image_list[0][-i + 1:]
            print("estimated common_suffix for " + folder + " folder: " + common_suffix)
        else:
            common_suffix = input("only one image in folder, manually enter suffix for " + folder + " folder")
        return common_suffix

    def get_animal_id(input_path):
        animal_id = input_path.parts[-1]
        return animal_id

    def get_info(input_path, folder_id, seg_type=False):
        if not seg_type:
            data_dir = input_path.joinpath(folder_id)
            data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.tif')])
        else:
            data_dir = input_path.joinpath(folder_id, seg_type)
            data_list = natsorted([f.parts[-1] for f in data_dir.glob('*.csv')])
        data_suffix = find_common_suffix(data_list, folder=folder_id)
        return data_dir, data_list, data_suffix



    def check_results_dir(input_path, seg_type):
        results_dir = input_path.joinpath('results', seg_type)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
            print("creating results folder under: " + str(results_dir))
        return results_dir

    @thread_worker
    def create_results_file(input_path, seg_type):

        if seg_type == 'cells':
            print("not implemented yet")
            return
        animal_id = get_animal_id(input_path)
        regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'registration')
        with open(regi_dir.joinpath('registration.json')) as fn:
            regi_data = json.load(fn)

        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        stats_dir, stats_list, stats_suffix = get_info(input_path, 'stats', seg_type=seg_type)
        results_dir = check_results_dir(input_path, seg_type)
        s = sliceHandle(regi_dir.joinpath('registration.json'))
        injection_data = pd.DataFrame()  # todo not only for injection data
        for im in stats_list:
            curr_im = im[:-len(stats_suffix)]
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

            stats = pd.read_csv(stats_dir.joinpath(im))
            y_pos = list(stats['Position Y'])
            x_pos = list(stats['Position X'])
            # append mix max values for rescaling
            y_pos.append(0)
            y_pos.append(y_im)
            x_pos.append(0)
            x_pos.append(x_im)

            y_scaled = np.ceil(minmax_scale(y_pos, feature_range=(0, y_low)))[:-2].astype(int)
            x_scaled = np.ceil(minmax_scale(x_pos, feature_range=(0, x_low)))[:-2].astype(int)
            # coords = np.stack([x_scaled, y_scaled], axis=1)

            poly_points = [(x_scaled[i], y_scaled[i]) for i in range(0, len(x_scaled))]
            polygon = path.Path(poly_points)

            # get all possible points on canvas
            # xx, yy = np.meshgrid(np.arange(img_regi.shape[1]), np.arange(img_regi.shape[0]))
            x_min, x_max = x_scaled.min(), x_scaled.max()
            y_min, y_max = y_scaled.min(), y_scaled.max()
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, (x_max - x_min) + 1),
                                 np.linspace(y_min, y_max, (y_max - y_min) + 1))
            # xx, yy = np.meshgrid(np.arange(img_regi.shape[1]), np.arange(img_regi.shape[0]))
            canvas_points = [(np.ndarray.flatten(xx)[i], np.ndarray.flatten(yy)[i]) for i in
                             range(0, len(np.ndarray.flatten(xx)))]
            idx_in_polygon = polygon.contains_points(canvas_points)
            points_in_polygon = [c for c, i in zip(canvas_points, idx_in_polygon) if i]
            x_poly = [p[0] for p in points_in_polygon]
            y_poly = [p[1] for p in points_in_polygon]
            coords = np.stack([x_poly, y_poly], axis=1)
            slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
            s.setImgFolder(regi_dir)
            # set which slice in there
            s.setSlice(slice_idx)
            # s.visualizeMapping(coords)
            section_data = s.getBrainArea(coords, (curr_im + regi_suffix))
            injection_data = pd.concat((injection_data, section_data))
        fn = results_dir.joinpath(animal_id + '_injection.csv')
        injection_data.to_csv(fn)
        print("done! data saved to: " + fn)


    # todo think about solution to check and load atlas data
    @magicgui(
        layout='vertical',
        input_path=dict(widget_type='FileEdit', label='input path: ', mode='d',
                        tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
        seg_type=dict(widget_type='ComboBox', label='segmentation type',
                      choices=['cells', 'injection_side'], value='cells',
                      tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'),
        results_button=dict(widget_type='PushButton', text='create results file',
                               tooltip='create one combined results datafile for segmentation results specified above'),
        quant_button=dict(widget_type='PushButton', text='quantify injection side',
                              tooltip='quick way to quantify injection side data'),  # todo: specify level, and give option for quantitfy cells
        call_button=False
    )

    def widget(
            input_path,
            seg_type,
            results_button,
            quant_button
    ) -> None:
        if not hasattr(widget, 'dummy'):  # todo, delete this or None exception?
            widget.dummy = []

    @widget.results_button.changed.connect
    def _create_results_file():
        # todo check input path
        input_path = widget.input_path.value
        seg_type = widget.seg_type.value
        worker_results_file = create_results_file(input_path, seg_type)
        worker_results_file.start()

    return widget

# def get_regi_info(input_path):
        #     regi_dir = input_path.joinpath('registration')
        #     regi_im_list = natsorted([f.parts[-1] for f in regi_dir.glob('*.tif')])  # todo check if this is necessary
        #     regi_suffix = find_common_suffix(regi_im_list, folder='registration')
        #
        #     return regi_dir, regi_im_list, regi_suffix, regi_data
        #
        # def get_stats_info(input_path, seg_type):
        #     stats_dir = input_path.joinpath('stats', seg_type)
        #     stats_list = natsorted([f.parts[-1] for f in stats_dir.glob('*.tif')])
        #     stats_suffix = find_common_suffix(stats_list, folder='stats')
        #     return stats_dir, stats_list, stats_suffix
        #
        # def get_seg_info(input_path):
        #     seg_im_dir = input_path.joinpath('rgb')
        #     seg_im_list = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])
        #     seg_im_suffix = find_common_suffix(seg_im_list, folder='segmented images')
        #     return seg_im_dir, seg_im_list, seg_im_suffix
