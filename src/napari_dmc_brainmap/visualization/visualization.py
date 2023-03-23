from napari import Viewer
from natsort import natsorted
import cv2
from napari_dmc_brainmap.utils import get_info
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
import pandas as pd

def change_index(image_idx):
    segment_widget.image_idx.value = image_idx


def cmap_cells():
    # return default colormap for channel and color of cells
    cmap = {
        'dapi': 'yellow',
        'green': 'magenta',
        'cy3': 'cyan',
        # 'cy5': 'lightblue'
    }
    return cmap


def default_save_dict():
    save_dict = {
        "image_idx": False,
        "seg_type": False,
        "chan_list": False
    }
    return save_dict

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
    channels=dict(widget_type='Select', label='selected channels', value=['green', 'cy3'],
                      choices=['dapi', 'green', 'cy3'],
                      tooltip='select channels to be selected for cell segmentation, '
                              'to select multiple hold ctrl/shift'),
    image_idx=dict(widget_type='LineEdit', label='image to be loaded', value=0,
                    tooltip='index (int) of image to be loaded and segmented next'),
    # load_dapi_bool=dict(widget_type='CheckBox', text='load blue channel', value=False,
    #                     tooltip='option to load blue channel (0: cy3; 1: green; 2: dapi)'),
    call_button=False
)
def segment_widget(
    viewer: Viewer,
    input_path,  # posix path
    seg_type,
    channels,
    image_idx,
    # load_dapi_bool
) -> None:

    return segment_widget



class VisualizationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        segment = segment_widget
        self.save_dict = default_save_dict()
        btn = QPushButton("save data and load next image")
        btn.clicked.connect(self._save_and_load)

        self.layout().addWidget(segment.native)
        self.layout().addWidget(btn)

    def _update_save_dict(self, image_idx, seg_type):
        # get image idx and segmentation type for saving segmentation data
        self.save_dict['image_idx'] = image_idx
        self.save_dict['seg_type'] = seg_type
        return self.save_dict

    def _save_and_load(self):
        # stats_dir = get_info(input_path, 'stats', seg_type=seg_type, create_dir=True, only_dir=True)

        input_path = segment_widget.input_path.value
        image_idx = int(segment_widget.image_idx.value)
        seg_type = segment_widget.seg_type.value
        channels = segment_widget.channels.value
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        if len(self.viewer.layers) == 0:  # no open images, set save_dict to defaults
            self.save_dict = default_save_dict()
        if type(self.save_dict['image_idx']) == int:  # todo there must be a better way :-D (for image_idx = 0)
            self._save_data(input_path)
        del (self.viewer.layers[:])  # remove open layers

        try:
            im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
                image_idx]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
            path_to_im = seg_im_dir.joinpath(im)
            self._load_next(path_to_im, seg_type, channels, image_idx)
        except IndexError:
            print("Index out of range, check that index matches image count in " + str(seg_im_dir))


    def _load_next(self, path_to_im, seg_type, channels, image_idx):
        self.save_dict = self._update_save_dict(image_idx, seg_type)
        im_loaded = cv2.imread(str(path_to_im))  # loads RGB as BGR
        if 'cy3' in channels:
            self.viewer.add_image(im_loaded[:, :, 2], name='cy3 channel', colormap='red', opacity=1.0)
            self.viewer.layers['cy3 channel'].contrast_limits = [0, 100]
        if 'green' in channels:
            self.viewer.add_image(im_loaded[:, :, 1], name='green channel', colormap='green', opacity=0.5)
            self.viewer.layers['green channel'].contrast_limits = [0, 100]
        if 'dapi' in channels:
            self.viewer.add_image(im_loaded[:, :, 0], name='dapi channel')
            self.viewer.layers['dapi channel'].contrast_limits = [0, 100]
        if seg_type == 'injection_side':
            self.viewer.add_shapes(name='injection', face_color='purple', opacity=0.4)
        elif seg_type == 'cells':  # todo presegment for cells
            cmap_dict = cmap_cells()
            for chan in channels:
                self.viewer.add_points(size=5, name=chan, face_color=cmap_dict[chan])
        print("loaded " + path_to_im.parts[-1] + " (cnt=" + str(image_idx) + ")")
        image_idx += 1
        change_index(image_idx)

    def _save_data(self, input_path):
        # points data in [y, x] format
        # todo edit channels etc. this is very stiff at the moment
        save_idx = self.save_dict['image_idx']
        seg_type_save = self.save_dict['seg_type']
        stats_dir = get_info(input_path, 'stats', seg_type=seg_type_save, create_dir=True, only_dir=True)
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        path_to_im = seg_im_dir.joinpath(seg_im_list[save_idx])
        im_name_str = path_to_im.with_suffix('').parts[-1]
        if seg_type_save == 'injection_side':
            if len(self.viewer.layers['injection'].data) > 0:
                inj_side = pd.DataFrame(self.viewer.layers['injection'].data[0], columns=['Position Y', 'Position X'])
                save_name_inj = stats_dir.joinpath(im_name_str + '_injection_side.csv')
                inj_side.to_csv(save_name_inj)
        elif seg_type_save == 'cells':
            if len(self.viewer.layers['green'].data) > 0:
                green_cells = pd.DataFrame(self.viewer.layers['green'].data, columns=['Position Y', 'Position X'])
                save_name_green = stats_dir.joinpath(im_name_str + '_green.csv')
                green_cells.to_csv(save_name_green)
            if len(self.viewer.layers['cy3'].data) > 0:
                cy3_cells = pd.DataFrame(self.viewer.layers['cy3'].data, columns=['Position Y', 'Position X'])
                save_name_cy3 = stats_dir.joinpath(im_name_str + '_cy3.csv')
                cy3_cells.to_csv(save_name_cy3)


