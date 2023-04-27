from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui

from natsort import natsorted
import json
import tifffile as tiff
import numpy as np

from napari_dmc_brainmap.stitching.stitching_tools import stitch_stack
from napari_dmc_brainmap.utils import get_info, get_animal_id, update_params_dict, clean_params_dict


@thread_worker
def do_stitching(input_path, filter_list, params_dict):
    animal_id = get_animal_id(input_path)
    direct_sharpy_track = params_dict['operations']['sharpy_track']

    # get obj sub-dirs
    data_dir = input_path.joinpath('raw')
    objs = natsorted([o.parts[-1] for o in data_dir.iterdir() if o.is_dir()])
    if not objs:
        print('no object slides under raw-data folder!')
        return

    # iterate objs and chans
    for obj in objs:
        in_obj = data_dir.joinpath(obj)
        for f in filter_list:
            stitch_dir = get_info(input_path, 'stitching', channel=f, create_dir=True, only_dir=True)
            in_chan = in_obj.joinpath(obj + '_' + f + '_1')
            ## load tile stack name
            stack = []
            im_list = natsorted([im.parts[-1] for im in in_chan.glob('*.tif')])
            for fn in im_list:
                stack.append(fn)
            stack.sort()
            ## load stack data
            whole_stack = []
            for stk in stack:
                with tiff.TiffFile(in_chan.joinpath(stk)) as tif:  # read multipaged tif
                    for page in tif.pages:  # iterate over pages
                        image = page.asarray()  # convert to array
                        whole_stack.append(image)  # append to whole_stack container
            ## convert to numpy array
            whole_stack = np.array(whole_stack)

            # load tile location meta data from meta folder
            meta_json_where = in_obj.joinpath(obj + '_meta_1', 'regions_pos.json')

            with open(meta_json_where, 'r') as data:
                img_meta = json.load(data)
            # get number of regions on this objective slide
            region_n = len(img_meta)
            # iterate regions
            for rn in range(region_n):
                pos_list = img_meta['region_' + str(rn)]
                stitched_path = stitch_dir.joinpath(animal_id + '_' + obj + '_' + str(rn + 1) + '_stitched.tif')
                if direct_sharpy_track:
                    sharpy_chans = params_dict['sharpy_track_params']['channels']
                    if f in sharpy_chans:
                        sharpy_dir = get_info(input_path, 'stitching', channel=f, create_dir=True, only_dir=True)
                        sharpy_im_dir = sharpy_dir.joinpath(animal_id + '_' + obj + '_' +
                                                            str(rn + 1) + '_downsampled.tif')
                        pop_img = stitch_stack(pos_list, whole_stack, 205, stitched_path, params_dict, f, sharpy_im_dir)
                    else:
                        pop_img = stitch_stack(pos_list, whole_stack, 205, stitched_path, params_dict, f)
                else:
                    pop_img = stitch_stack(pos_list, whole_stack, 205, stitched_path, params_dict, f)
                # remove stitched tiles from whole_stack
                whole_stack = np.delete(whole_stack, [np.arange(pop_img)], axis=0)
    params_dict = clean_params_dict(params_dict, "operations")
    params_fn = input_path.joinpath('params.json')
    params_dict = update_params_dict(input_path, params_dict)
    with open(params_fn, 'w') as fn:
        json.dump(params_dict, fn, indent=4)
    print('all finished!')


@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
                    tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                'folder containing segmentation results'),
    channels=dict(widget_type='Select', label='imaged channels', value=['green', 'cy3'],
                      choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                      tooltip='select the imaged channels, '
                              'to select multiple hold ctrl/shift'),
    sharpy_bool=dict(widget_type='CheckBox', text='get images for sharpy-track', value=True,
                     tooltip='downsample image to resolution in sharpy track (1140 x 800)'),
    sharpy_chan=dict(widget_type='Select', label='selected channels', value='green',
                     choices=['all', 'dapi', 'green', 'n3', 'cy3', 'cy5'],
                     tooltip='select channels to be processed, to select multiple hold ctrl/shift'),
    contrast_bool=dict(widget_type='CheckBox', text='perform contrast adjustment on images for registration', value=True,
                       tooltip='option to adjust contrast on images, see option details below'),
    contrast_dapi=dict(widget_type='LineEdit', label='set contrast limits for the dapi channel',
                       value='50,1000', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_green=dict(widget_type='LineEdit', label='set contrast limits for the green channel',
                        value='50,300', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_n3=dict(widget_type='LineEdit', label='set contrast limits for the n3 channel',
                      value='50,500', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy3=dict(widget_type='LineEdit', label='set contrast limits for the cy3 channel',
                      value='50,500', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),
    contrast_cy5=dict(widget_type='LineEdit', label='set contrast limits for the cy5 channel',
                      value='50,500', tooltip='enter contrast limits: min,max (default values for 16-bit image)'),


    call_button=False
)
def stitching_widget(
    viewer: Viewer,
    input_path,  # posix path
    channels,
    sharpy_bool,
    sharpy_chan,
    contrast_bool,
    contrast_dapi,
    contrast_green,
    contrast_n3,
    contrast_cy3,
    contrast_cy5

) -> None:

    return stitching_widget



class StitchingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        stitching = stitching_widget
        btn = QPushButton("stitch images")
        btn.clicked.connect(self._do_stitching)

        self.layout().addWidget(stitching.native)
        self.layout().addWidget(btn)
    def _get_info(self, widget):

            return {
                "channels": widget.sharpy_chan.value,
                "contrast_adjustment": widget.contrast_bool.value,
                "dapi": [int(i) for i in widget.contrast_dapi.value.split(',')],
                "green": [int(i) for i in widget.contrast_green.value.split(',')],
                "n3": [int(i) for i in widget.contrast_n3.value.split(',')],
                "cy3": [int(i) for i in widget.contrast_cy3.value.split(',')],
                "cy5": [int(i) for i in widget.contrast_cy5.value.split(',')]
            }

    def _get_stitching_params(self):
        params_dict = {
            "general":
                {
                    "animal_id": get_animal_id(stitching_widget.input_path.value),
                    "chans_imaged": stitching_widget.chans_imaged.value
                },
            "operations":
                {
                    "sharpy_track": stitching_widget.sharpy_bool.value,
                },
            "sharpy_track_params": self._get_info(stitching_widget),

        }
        return params_dict
    def _do_stitching(self):
        input_path = stitching_widget.input_path.value
        params_dict = self._get_stitching_params()
        filter_list = params_dict['general']['chans_imaged']
        preprocessing_worker = do_stitching(input_path, filter_list, params_dict)
        preprocessing_worker.start()

