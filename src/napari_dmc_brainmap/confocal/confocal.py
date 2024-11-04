from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import numpy as np
import cv2
from aicsimageio.readers import CziReader

from napari_dmc_brainmap.utils import get_animal_id, get_info, get_image_list
from napari_dmc_brainmap.stitching.stitching_tools import padding_for_atlas




@thread_worker
def create_tifs(input_path, chan_label_map):
    #animal_id = get_animal_id(input_path)
    #data_dir = input_path.joinpath('confocal')
    image_list = get_image_list(input_path, folder_id='confocal', file_id='*.czi')
    for im in image_list:
        print('started with ' + im)
        im_fn = input_path.joinpath('confocal', im + '.czi')
        reader = CziReader(im_fn)
        channels = reader.channel_names
        num_chan = len(channels)
        num_z = reader.mosaic_data.shape[3]  # shape: [1, 1, channel, z, x, y, 1]
        img = np.squeeze(reader.mosaic_data)
        for i, chan in enumerate(channels):
            chan_new = [c for c in chan_label_map if chan_label_map[c] == chan][0]
            stitched_dir = get_info(input_path, 'stitched', channel=chan_new, create_dir=True, only_dir=True)
            if num_chan > 1:
                curr_img = img[i, :, :, :]
            else:
                curr_img = img
            if num_z > 1:
                # do max intensity projection
                curr_img = np.max(curr_img, axis=0)
            curr_img = padding_for_atlas(curr_img)  # padding
            save_fn = stitched_dir.joinpath(im + '_stitched.tif')
            cv2.imwrite(str(save_fn), curr_img)

    print('DONE!')


def initialize_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders. Folder with NOT STITCHED confocal images (czi)'
                                ' should be called >>confocal<< (without arrows). '),
              # channels=dict(widget_type='Select', label='imaged channels', value=['green', 'cy3'],
              #                   choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
              #                   tooltip='select the imaged channels, '
              #                           'to select multiple hold ctrl/shift'),
              dapi_name=dict(widget_type='LineEdit', 
                             label='name of dapi channel at confocal',
                             value='', 
                             tooltip='enter the name of the dapi channel at the confocal'),
              green_name=dict(widget_type='LineEdit', 
                              label='name of green channel at confocal',
                              value='AF488-T3', 
                              tooltip='enter the name of the green channel at the confocal'),
              n3_name=dict(widget_type='LineEdit', 
                           label='name of n3 channel at confocal',
                           value='', 
                           tooltip='enter the name of the n3 channel at the confocal'),
              cy3_name=dict(widget_type='LineEdit', 
                            label='name of cy3 channel at confocal',
                            value='AF555-T2', 
                            tooltip='enter the name of the cy3 channel at the confocal'),
              cy5_name=dict(widget_type='LineEdit', 
                            label='name of cy5 channel at confocal',
                            value='', 
                            tooltip='enter the name of the cy5 channel at the confocal'),
              call_button=False)
    def confocal_widget(
        viewer: Viewer,
        input_path,  # posix path
        # channels,
        dapi_name,
        green_name,
        n3_name,
        cy3_name,
        cy5_name):
        pass
    return confocal_widget


class ConfocalWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.confocal = initialize_widget()
        btn = QPushButton("create tif images from czi files")
        btn.clicked.connect(self._create_tifs)

        self.layout().addWidget(self.confocal.native)
        self.layout().addWidget(btn)

    def _get_channel_name_map(self):
        chan_label_map = {
            'dapi': self.confocal.dapi_name.value,
            'green': self.confocal.green_name.value,
            'n3': self.confocal.n3_name.value,
            'cy3': self.confocal.cy3_name.value,
            'cy5': self.confocal.cy5_name.value
        }
        return chan_label_map

    def _create_tifs(self):
        input_path = self.confocal.input_path.value
        # chans_imaged = confocal_widget.chans_imaged.value
        chan_label_map = self._get_channel_name_map()

        tif_worker = create_tifs(input_path, chan_label_map)
        tif_worker.start()

