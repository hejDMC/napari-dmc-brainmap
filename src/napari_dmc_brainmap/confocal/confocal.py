from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui

import numpy as np
import cv2
from aicsimageio.readers import CziReader

from napari_dmc_brainmap.utils import get_animal_id


fn = r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dopamine\analysis\anatomy\data\478017_glp1r_syp_5488_555_1.czi'
fn2 = r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dopamine\analysis\anatomy\data\478017_glp1r_syp_5488_555_1-Stitching-13-Orthogonal Projection-26.czi'

chan_label_map = {
    'dapi': '',
    'green': 'AF488-T3',
    'n3': '',
    'cy3': 'AF555-T2',
    'cy5': '',
}

reader = CziReader(fn)
channels = reader.channel_names

"""shape: [1, 1, channel, z, x, y, 1]"""
num_z = reader.mosaic_data.shape[3]
img = np.squeeze(reader.mosaic_data)

for i, chan in enumerate(channels):
    curr_img = img[i, :, :, :]
    if num_z > 1:
        # do max intensity projection
        curr_img = np.max(curr_img, axis=0)
    save_name = r'C:\Users\felix-arbeit\Documents\Academia\DMC-lab\projects\dopamine\analysis\anatomy\data\478017_glp1r_syp_5488_555_1.tif'
    cv2.imwrite(curr_img, save_name)

@thread_worker
def create_tifs(input_path, chans_imaged, chan_label_map):
    animal_id = get_animal_id(input_path)
    data_dir = input_path.joinpath('confocal')
    im_list =
    stitched_dir = get_info(input_path, 'stitched', only_dir=True)
    filter_dir = [f for f in stitched_dir.glob('**/*') if f.is_dir()][0]  # just take the first folder
    image_list = natsorted([f.parts[-1] for f in data_dir.glob('*.czi')])


@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders. Folder with NOT STITCHED confocal images (czi)'
                            ' should be called >>confocal<< (without arrows). '),
    channels=dict(widget_type='Select', label='imaged channels', value=['green', 'cy3'],
                      choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                      tooltip='select the imaged channels, '
                              'to select multiple hold ctrl/shift'),
    dapi_name=dict(widget_type='LineEdit', label='name of dapi channel at confocal',
                       value='', tooltip='enter the name of the dapi channel at the confocal'),
    green_name=dict(widget_type='LineEdit', label='name of green channel at confocal',
                       value='AF488-T3', tooltip='enter the name of the green channel at the confocal'),
    n3_name=dict(widget_type='LineEdit', label='name of n3 channel at confocal',
                       value='', tooltip='enter the name of the n3 channel at the confocal'),
    cy3_name=dict(widget_type='LineEdit', label='name of cy3 channel at confocal',
                       value='AF555-T2', tooltip='enter the name of the cy3 channel at the confocal'),
    cy5_name=dict(widget_type='LineEdit', label='name of cy5 channel at confocal',
                       value='', tooltip='enter the name of the cy5 channel at the confocal'),

    call_button=False
)
def confocal_widget(
    viewer: Viewer,
    input_path,  # posix path
    channels,
    dapi_name,
    green_name,
    cy3_name,
    cy5_name

) -> None:

    return confocal_widget

class ConfocalWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        confocal = confocal_widget
        btn = QPushButton("create tif images from czi files")
        btn.clicked.connect(self._create_tifs)

        self.layout().addWidget(confocal.native)
        self.layout().addWidget(btn)

    def _get_channel_name_map(self):
        chan_label_map = {
            'dapi': confocal_widget.dapi_name.value,
            'green': confocal_widget.green_name.value,
            'n3': confocal_widget.n3_name.value,
            'cy3': confocal_widget.cy3_name.value,
            'cy5': confocal_widget.cy5_name.value
        }
        return chan_label_map

    def _create_tifs(self):
        input_path = confocal_widget.input_path.value
        chans_imaged = confocal_widget.chans_imaged.value
        chan_label_map = self._get_channel_name_map()

        preprocessing_worker = create_tifs(input_path, chans_imaged, chan_label_map)
        preprocessing_worker.start()