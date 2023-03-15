"""
todo: write documentation
- edit input params like contrast adjustment, channels to be loaded
- new layers via fucntion?
"""


from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
from natsort import natsorted
import cv2
import pandas as pd
from napari_dmc_brainmap.utils import get_animal_id, get_info

def segment_widget():



    def change_index(image_idx):
        widget.image_idx.value = image_idx

    def default_save_dict():
        save_dict = {
            "image_idx": False,
            "seg_type": False,
            "chan_list": False
        }
        return save_dict
    def update_save_dict(save_dict, image_idx, seg_type):
        # get image idx and segmentation type for saving segmentation data
        print('in update')
        save_dict['image_idx'] = image_idx
        save_dict['seg_type'] = seg_type
        return save_dict

    # todo add options for channel limits etc.
    def load_and_save(viewer, input_path, save_dict, seg_type, image_idx, load_dapi):
        stats_dir = get_info(input_path, 'stats', seg_type=seg_type, create_dir=True, only_dir=True)
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        if save_dict['image_idx']:
            save_data(viewer, input_path, save_dict)
        im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
            image_idx]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
        path_to_im = seg_im_dir.joinpath(im)
        print('before load')
        save_dict = load_next(viewer, path_to_im, seg_type, image_idx, load_dapi)
        return save_dict


    def load_next(viewer, path_to_im, save_dict, seg_type, image_idx, load_dapi):
        print('in load')
        save_dict = update_save_dict(save_dict, image_idx, seg_type)
        im_loaded = cv2.imread(str(path_to_im))  # loads RGB as BGR
        viewer.add_image(im_loaded[:, :, 2], name='cy3 channel', colormap='red', opacity=1.0)
        viewer.add_image(im_loaded[:, :, 1], name='green channel', colormap='green', opacity=0.5)
        if load_dapi:
            viewer.add_image(im_loaded[:, :, 0], name='dapi channel')
        viewer.layers['cy3 channel'].contrast_limits = [0, 100]
        viewer.layers['green channel'].contrast_limits = [0, 100]
        if seg_type == 'injection_side':
            viewer.add_shapes(name='injection', face_color='purple', opacity=0.4)
        elif seg_type == 'cells':  # todo presegment for cells
            viewer.add_points(size=5, name='green', face_color='magenta')
            viewer.add_points(size=5, name='cy3', face_color='cyan')
        print("loaded " + path_to_im.parts[-1] + " (cnt=" + str(image_idx) + ")")
        image_idx += 1
        change_index(image_idx)
        return viewer, image_idx, save_dict

    # @thread_worker
    def save_data(viewer, input_path, save_dict):
        # points data in [y, x] format
        # todo use save_dict with info for saving
        # todo edit channels
        # todo here come errors when going 'down' in index and layers are still open
        save_idx = save_dict['idx']
        seg_type_save = save_dict['seg_type']

        stats_dir = get_info(input_path, 'stats', seg_type=seg_type_save, only_dir=True)
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        path_to_im = seg_im_dir.joinpath(seg_im_list[save_idx])
        im_name_str = path_to_im.with_suffix('').parts[-1]
        if seg_type_save == 'injection_side':
            inj_side = pd.DataFrame(viewer.layers['injection'].data[0], columns=['Position Y', 'Position X'])
            save_name_inj = stats_dir.joinpath(im_name_str + '_injection_side.csv')
            if viewer.layers['injection'].data[0].shape[0] > 0:
                inj_side.to_csv(save_name_inj)
        elif seg_type_save == 'cells':
            green_cells = pd.DataFrame(viewer.layers['green'].data, columns=['Position Y', 'Position X'])
            cy3_cells = pd.DataFrame(viewer.layers['cy3'].data, columns=['Position Y', 'Position X'])
            save_name_green = stats_dir.joinpath(im_name_str + '_green.csv')
            save_name_cy3 = stats_dir.joinpath(im_name_str + '_cy3.csv')
            if viewer.layers['green'].data.shape[0] > 0:
                green_cells.to_csv(save_name_green)
            if viewer.layers['cy3'].data.shape[0] > 0:
                cy3_cells.to_csv(save_name_cy3)
        del (viewer.layers[:])



    @magicgui(
        # todo sate that only RGB images for now, think about different image formats
        layout='vertical',
        input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                        tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
        seg_type=dict(widget_type='ComboBox', label='segmentation type',
                      choices=['cells', 'injection_side'], value='cells',
                      tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'
                              'IMPORTANT: before switching between types, load next image, delete all image layers'
                              'and reload image of interest!'),
        image_idx=dict(widget_type='LineEdit', label='image to be loaded', value=0,
                       tooltip='index (int) of image to be loaded and segmented next'),
        load_dapi_box=dict(widget_type='CheckBox', text='load blue channel', value=False,
                           tooltip='option to load blue channel (0: red; 1: green; 2: dapi)'),
        # load_image_button=dict(widget_type='PushButton', text='load next images',
        #                        tooltip='load the next image (index specified above) for segmentation'),
        close_images_box=dict(widget_type='CheckBox', text='close images after saving', value=True,
                              tooltip='option to close all layers (images and segmentation data) after saving'),
        save_data_box=dict(widget_type='CheckBox', text='save segmentation data', value=True,
                              tooltip='option to save segmentation data before closing previous image'),
        # save_data_button=dict(widget_type='PushButton', text='save data',
        #                       tooltip='segmentation data is saved and optionally all open layers closed'),
        call_button="save data and load next image"
    )
    def widget(
        viewer: Viewer,
        input_path,  # posix path
        seg_type,
        image_idx,
        load_dapi_box,
        # load_image_button,
        close_images_box,
        save_data_box,
        save_dict=None,
        # save_data_button
    ) -> None:
        # if not hasattr(widget, 'segment_layers'):
        #     widget.segment_layers = []
        if save_dict is None:
            save_dict = default_save_dict()
        save_dict = load_and_save(viewer, input_path, save_dict, seg_type, int(image_idx), load_dapi_box)
    return widget
