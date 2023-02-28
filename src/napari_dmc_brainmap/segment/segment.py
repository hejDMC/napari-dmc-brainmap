from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
import sys

def segment_widget():
    from napari.qt.threading import thread_worker

    @thread_worker
    def 

    @magicgui(
        # todo sate that only RGB images for now, think about different image formats
        layout='vertical',
        input_path = dict(widget_type='FileEdit', label='input path: ', tooltip='directory of folder containing images to be segmented'),
        seg_type = dict(widget_type='ComboBox', label='segmentation type', choices=['cells (points)', 'injection side (areas)'], value='cells (points)', tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'),
        image_idx = dict(widget_type='LineEdit', label='image to be loaded', value=0, tooltip='index of image to be loaded and segmented next'),
        load_dapi_box = dict(widget_type='CheckBox', text='load blue channel', value=False, tooltip='option to load blue channel (0: red; 1: green; 2: dapi)'),
        load_image_button  = dict(widget_type='PushButton', text='load next images', tooltip='load the next image (index specified above) for segmentation'),
        close_images_box = dict(widget_type='CheckBox', text='close images after saving', value=True, tooltip='option to close all layers (images and segmentation data) after saving'),
        save_data_button  = dict(widget_type='PushButton', text='save data', tooltip='segmentation data is saved and optionally all open layers closed'),
        call_button = False
    )
    def widget(
        # viewer: Viewer,
        input_path,
        seg_type,
        image_idx,
        load_dapi_box,
        load_image_button,
        close_images_box,
        save_data_button
    ) -> None:
        if not hasattr(widget, 'something'):
            widget.something = []
    return widget
