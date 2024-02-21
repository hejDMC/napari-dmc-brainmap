
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import cv2

from napari_dmc_brainmap.stitching.stitching_tools import padding_for_atlas
from napari_dmc_brainmap.utils import load_params, get_info

@thread_worker
def do_padding(input_path, channels, pad_folder, resolution):
    print('doing padding of ...')
    for chan in channels:
        print('... channel ' + ' chan')
        pad_dir, pad_im_list, pad_suffix = get_info(input_path, pad_folder, channel=chan)
        for im in pad_im_list:
            print('... ' + im)
            im_fn = pad_dir.joinpath(im)
            im_array = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)  # 0 for grayscale mode
            im_padded = padding_for_atlas(im_array, resolution)
            cv2.imwrite(str(im_fn), im_padded)

    print('done!')


def initialize_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                    'folder containing segmentation results'),
              pad_folder=dict(widget_type='LineEdit', 
                              label='folder name images to be padded: ', 
                              value='stitched',
                              tooltip='name of folder containing the stitched images to be padded '
                                '(animal_id/>pad_folder</chan1'),
              channels=dict(widget_type='Select', 
                            label='imaged channels', 
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select the imaged channels, '
                                'to select multiple hold ctrl/shift'),
              call_button=False)
    
    def padding_widget(
        viewer: Viewer,
        input_path,  # posix path
        channels,
        pad_folder):
        pass
    return padding_widget


class PaddingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.padding = initialize_widget()
        btn = QPushButton("do the padding (WARNING - overriding existing files!)")
        btn.clicked.connect(self._do_padding)

        self.layout().addWidget(self.padding.native)
        self.layout().addWidget(btn)


    def _do_padding(self):
        input_path = self.padding.input_path.value
        channels = self.padding.channels.value
        pad_folder = self.padding.pad_folder.value
        params_dict = load_params(input_path)
        resolution = params_dict['atlas_info']['resolution']  # [x,y]
        padding_worker = do_padding(input_path, channels, pad_folder, resolution)
        padding_worker.start()