from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui


def registration_widget():
    from napari.qt.threading import thread_worker


    # todo think about solution to check and load atlas data
    @magicgui(
        layout='vertical',
        input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                        tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
        call_button='start registration GUI'
    )

    def widget()-> None:
        # if not hasattr(widget, 'segment_layers'):
        #     widget.segment_layers = []
        load_next(viewer, input_path, seg_type, int(image_idx), load_dapi_box)
    return widget
