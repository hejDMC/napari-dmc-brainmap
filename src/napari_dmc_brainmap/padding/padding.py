
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from napari import Viewer
from napari.qt.threading import thread_worker
from magicgui import magicgui
from magicgui.widgets import FunctionGui
import tifffile
from natsort import natsorted
import pandas as pd
from napari_dmc_brainmap.stitching.stitching_tools import padding_for_atlas
from napari_dmc_brainmap.utils import load_params, get_info

@thread_worker
def do_padding(input_path, channels, pad_folder, resolution):
    # check if pad_folder name is "confocal"
    if pad_folder == "confocal":
        raise NotImplementedError("'confocal' is a keyword reserved for CZI file format,"
                                "\nPlease rename the folder something else (such as 'to_pad'), if these are tif files to pad."
                                "\nFor CZI files, please goto 'Stitch czi images' function."
                                "\nExiting padding function!")

    print('doing padding of ...')
    for chan in channels:
        print(f'... channel {chan}')
        # check if to pad images has _stitched.tif suffix
        # get first image name
        if not [tif.name for tif in input_path.joinpath(pad_folder, chan).glob("*.tif")][0].endswith("_stitched.tif"):
            # rename image files
            for im in [tif.name for tif in input_path.joinpath(pad_folder, chan).glob("*.tif")]:
                im_old = input_path.joinpath(pad_folder, chan, im)
                im_new = input_path.joinpath(pad_folder, chan, f"{im.split('.tif')[0]}_stitched.tif")
                print(f"renaming ===> {str(im_old)} \nto {str(im_new)} <===")
                im_old.rename(im_new)
            # save image_names.csv
            # check if image_names.csv already exists
            image_names_csv = input_path.joinpath("image_names.csv")
            if image_names_csv.exists():
                pass
            else:
                image_list = natsorted([tif.name for tif in input_path.joinpath(pad_folder, chan).glob("*.tif")])
                image_list = [tif.split("_stitched.tif")[0] for tif in image_list]
                # store data as .csv file
                image_list_store = pd.DataFrame(image_list)
                image_list_store.to_csv(image_names_csv)

        pad_dir, pad_im_list, pad_suffix = get_info(input_path, pad_folder, channel=chan)
        for im in pad_im_list:
            print(f'... {im}')
            im_fn = pad_dir.joinpath(im)
            # use tiffile to read instead of cv2
            im_array = tifffile.imread(str(im_fn))
            # im_array = cv2.imread(str(im_fn), cv2.IMREAD_ANYDEPTH)  # 0 for grayscale mode
            im_padded = padding_for_atlas(im_array, resolution)
            # use tifffile to write instead of cv2
            tifffile.imwrite(str(im_fn), im_padded)
            # cv2.imwrite(str(im_fn), im_padded)
            
    print('done!')


def initialize_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
                                    'folder containing images'),
              pad_folder=dict(widget_type='LineEdit', 
                              label='folder name images to be padded: ', 
                              value='stitched',
                              tooltip='name of folder containing the stitched images to be padded '
                                '(animal_id/>pad_folder</chan1)'),
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