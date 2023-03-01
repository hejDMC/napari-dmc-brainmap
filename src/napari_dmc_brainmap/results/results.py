from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
from natsort import natsorted
from skimage import io
import pandas as pd

def results_widget():

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


