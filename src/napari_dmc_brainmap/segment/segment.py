from napari import Viewer
from natsort import natsorted
import cv2
from napari_dmc_brainmap.utils import get_info
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
import pandas as pd
import random
import matplotlib.colors as mcolors

def change_index(image_idx):
    segment_widget.image_idx.value = image_idx


def cmap_cells():
    # return default colormap for channel and color of cells
    cmap = {
        'dapi': 'yellow',
        'green': 'magenta',
        'n3': 'gray',
        'cy3': 'cyan',
        'cy5': 'lightblue'
    }
    return cmap

def cmap_display():
    cmap = {
        'dapi': 'blue',
        'green': 'green',
        'n3': 'orange',
        'cy3': 'red',
        'cy5': 'pink'
    }
    return cmap


def default_save_dict():
    save_dict = {
        "image_idx": False,
        "seg_type": False,
        "chan_list": False,
        "n_probes": False
    }
    return save_dict

def get_path_to_im(input_path, image_idx, single_channel=False, chan=False):
    if single_channel:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channel=chan)
    else:
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
    im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
        image_idx]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
    path_to_im = seg_im_dir.joinpath(im)
    return path_to_im

@magicgui(
    # todo sate that only RGB images for now, think about different image formats
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
    single_channel_bool=dict(widget_type='CheckBox', text='use single channel', value=False,
                             tooltip='tick to use single channel images (not RGB), one can still select '
                                     'multiple channels'),
    seg_type=dict(widget_type='ComboBox', label='segmentation type',
                    choices=['cells', 'injection_side', 'optic_fiber', 'neuropixels_probe'], value='cells',  # todo option for more than one probe
                    tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'
                            'IMPORTANT: before switching between types, load next image, delete all image layers'
                            'and reload image of interest!'),
    n_probes=dict(widget_type='LineEdit', label='number of fibers/probes', value=1,
                    tooltip='number (int) of optic fibres and or probes used to segment, ignore this value for '
                            'segmenting cells/areas/'),
    channels=dict(widget_type='Select', label='selected channels', value=['green', 'cy3'],
                      choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                      tooltip='select channels to be selected for cell segmentation, '
                              'to select multiple hold ctrl/shift'),
    image_idx=dict(widget_type='LineEdit', label='image to be loaded', value=0,
                    tooltip='index (int) of image to be loaded and segmented next'),
    call_button=False
)
def segment_widget(
    viewer: Viewer,
    input_path,  # posix path
    seg_type,
    n_probes,
    channels,
    image_idx,
    single_channel_bool
) -> None:

    return segment_widget



class SegmentWidget(QWidget):
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

    def _update_save_dict(self, image_idx, seg_type, n_probes):
        # get image idx and segmentation type for saving segmentation data
        self.save_dict['image_idx'] = image_idx
        self.save_dict['seg_type'] = seg_type
        self.save_dict['n_probes'] = n_probes
        return self.save_dict

    def _save_and_load(self):

        input_path = segment_widget.input_path.value
        image_idx = int(segment_widget.image_idx.value)
        seg_type = segment_widget.seg_type.value
        channels = segment_widget.channels.value
        n_probes = int(segment_widget.n_probes.value)
        single_channel = segment_widget.single_channel_bool.value

        if len(self.viewer.layers) == 0:  # no open images, set save_dict to defaults
            self.save_dict = default_save_dict()
        if type(self.save_dict['image_idx']) == int:  # todo there must be a better way :-D (for image_idx = 0)
            self._save_data(input_path, channels)
        del (self.viewer.layers[:])  # remove open layers

        try:
            # if not single_channel:
            #     seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
            #     im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
            #         image_idx]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
            #     path_to_im = seg_im_dir.joinpath(im)
            #     self._load_next_rgb(path_to_im, seg_type, channels, image_idx, n_probes)
            # else:
            #     for chan in channels:
            #         seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'single_channel', channel=chan)
            #         im = natsorted([f.parts[-1] for f in seg_im_dir.glob('*.tif')])[
            #             image_idx]  # this detour due to some weird bug, list of paths was only sorted, not natsorted
            #         path_to_im = seg_im_dir.joinpath(im)
            self._load_next(input_path, seg_type, channels, image_idx, n_probes, single_channel)

        except IndexError:
            print("Index out of range, check that index matches image count in target folder")

    def _load_next(self, input_path, seg_type, channels, image_idx, n_probes, single_channel):
        self.save_dict = self._update_save_dict(image_idx, seg_type, n_probes)
        if single_channel:
            for chan in channels:
                path_to_im = get_path_to_im(input_path, image_idx, single_channel=single_channel, chan=chan)
                self._load_single(path_to_im, chan)
        else:
            path_to_im = get_path_to_im(input_path, image_idx)
            self._load_rgb(path_to_im, channels)
        self._create_seg_objects(seg_type, channels, n_probes)
        print("loaded " + path_to_im.parts[-1] + " (cnt=" + str(image_idx) + ")")
        image_idx += 1
        change_index(image_idx)


    def _load_rgb(self, path_to_im, channels):
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

    def _load_single(self, path_to_im, chan):

        cmap_disp = cmap_display()
        im_loaded = cv2.imread(str(path_to_im), cv2.IMREAD_GRAYSCALE)
        self.viewer.add_image(im_loaded, name=chan + ' channel', colormap=cmap_disp[chan], opacity=0.5)
        self.viewer.layers[chan + ' channel'].contrast_limits = [0, 100]

    def _create_seg_objects(self, seg_type, channels, n_probes):
        if seg_type == 'injection_side':
            if 'injection' not in self.viewer.layers:
                self.viewer.add_shapes(name='injection', face_color='purple', opacity=0.4)
        elif seg_type == 'cells':  # todo presegment for cells
            cmap_dict = cmap_cells()
            for chan in channels:
                self.viewer.add_points(size=5, name=chan, face_color=cmap_dict[chan])
        else:
            # todo keep colors constant
            for i in range(n_probes):
                p_color = random.choice(list(mcolors.CSS4_COLORS.keys()))
                if n_probes > 1:
                    p_name = seg_type + '_' +str(i)
                else:
                    p_name = seg_type
                self.viewer.add_points(size=20, name=p_name, face_color=p_color)

    def _save_data(self, input_path, channels):
        # points data in [y, x] format
        save_idx = self.save_dict['image_idx']
        seg_type_save = self.save_dict['seg_type']
        seg_im_dir, seg_im_list, seg_im_suffix = get_info(input_path, 'rgb')
        path_to_im = seg_im_dir.joinpath(seg_im_list[save_idx])
        im_name_str = path_to_im.with_suffix('').parts[-1]
        if seg_type_save == 'injection_side':
            if len(self.viewer.layers['injection'].data) > 0:
                segment_dir = get_info(input_path, 'segmentation', seg_type=seg_type_save, create_dir=True, only_dir=True)
                inj_side = pd.DataFrame()
                for i in range(len(self.viewer.layers['injection'].data)):
                    inj_data_temp = pd.DataFrame(self.viewer.layers['injection'].data[i], columns=['Position Y', 'Position X'])
                    inj_data_temp['idx_shape'] = [i] * len(inj_data_temp)
                    inj_side = pd.concat((inj_side, inj_data_temp))
                save_name_inj = segment_dir.joinpath(im_name_str + '_injection_side.csv')
                inj_side.to_csv(save_name_inj)
        elif seg_type_save == 'cells':
            for chan in channels:
                if len(self.viewer.layers[chan].data) > 0:
                    segment_dir = get_info(input_path, 'segmentation', channel=chan, seg_type=seg_type_save, create_dir=True,
                                         only_dir=True)
                    coords = pd.DataFrame(self.viewer.layers[chan].data, columns=['Position Y', 'Position X'])
                    save_name = segment_dir.joinpath(im_name_str + '_cells.csv')
                    coords.to_csv(save_name)
        else:
            for i in range(self.save_dict['n_probes']):
                if self.save_dict['n_probes'] > 1:
                    p_name = seg_type_save + '_' +str(i)
                else:
                    p_name = seg_type_save
                segment_dir = get_info(input_path, 'segmentation', seg_type=seg_type_save, create_dir=True,
                                         only_dir=True)
                coords = pd.DataFrame(self.viewer.layers[p_name].data, columns=['Position Y', 'Position X'])
                save_name = segment_dir.joinpath(im_name_str + '_' + p_name + '.csv')
                coords.to_csv(save_name)



