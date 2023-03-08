from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout
from superqt import QCollapsible




def preprocessing_widget():

    def do_something(self,do_something: bool = True):
        print("Is true: ", do_something)
    class Demo(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setLayout(QVBoxLayout())

            # QCollapsible creates a collapse container for inner widgets
            self._collapse2 = QCollapsible('Widget2: expand for more', self)
            # magicgui doesn't need to be used as a decorator, you can call it
            # repeatedly to create new widgets:
            new_widget = magicgui(do_something)
            # if you're mixing Qt and magicgui, you need to use the "native"
            # attribute in magicgui to access the QWidget
            self._collapse2.addWidget(new_widget.native)

            self._collapse3 = QCollapsible('Widget3: expand for more', self)

            self.layout().addWidget(self._collapse2)
            self.layout().addWidget(self._collapse3)


    @magicgui(
    # #     # todo sate that only RGB images for now, think about different image formats
    # #     layout='vertical',
    #     input_path=dict(widget_type='FileEdit', label='input path (animal_id): ', mode='d',
    #                     tooltip='directory of folder containing subfolders with e.g. images, segmentation results, NOT '
    #                             'folder containing segmentation results'),
    # #     chans_imaged=dict(widget_type='Select', label='imaged channels', choices=['dapi', 'green', 'cy3', 'cy5'],
    # #                       tooltip='select all channels imaged, to select multiple hold shift'),
    # #     rgb_box=dict(widget_type='CheckBox', text='create RGB images', value=True,
    # #                           tooltip='option to create RGB images'),
    # #     rgb_channels=dict(widget_type='Select', label='rgb channels', choices=['all', 'dapi', 'green', 'cy3'],
    # #                       allow_multiple=True, tooltip='select all channels to be used to create RGB image, '
    # #                                                    'hold shift to select multiple'),
    # #     rgb_ds=dict(widget_type='CheckBox', text='perform downsampling on RGB images', value=True,
    # #                           tooltip='option to downsample images, see option details below'),
    # #     rgb_ds2=dict(widget_type='CheckBox', value=True,
    # #                 tooltip='option to downsample images, see option details below'),
    # #     rgb_contrast=dict(widget_type='CheckBox', text='perform contrast adjustment on RGB images', value=True,
    # #                           tooltip='option to adjust contrast on images, see option details below'),
    # #     ds_params=dict(widget_type='LineEdit', label='enter downsampling factor', value=0,
    # #                    tooltip='enter scale factor for downsampling'),
    # #     # seg_type=dict(widget_type='ComboBox', label='segmentation type',
    # #     #               choices=['cells', 'injection_side'], value='cells',
    # #     #               tooltip='select to either segment cells (points) or areas (e.g. for the injection side)'
    # #     #                       'IMPORTANT: before switching between types, load next image, delete all image layers'
    # #     #                       'and reload image of interest!'),
    # #     # image_idx=dict(widget_type='LineEdit', label='image to be loaded', value=0,
    # #     #                tooltip='index (int) of image to be loaded and segmented next'),
    # #     # load_dapi_box=dict(widget_type='CheckBox', text='load blue channel', value=False,
    # #     #                    tooltip='option to load blue channel (0: red; 1: green; 2: dapi)'),
    # #     # # load_image_button=dict(widget_type='PushButton', text='load next images',
    # #     # #                        tooltip='load the next image (index specified above) for segmentation'),
    # #     # close_images_box=dict(widget_type='CheckBox', text='close images after saving', value=True,
    # #     #                       tooltip='option to close all layers (images and segmentation data) after saving'),
    # #     # save_data_box=dict(widget_type='CheckBox', text='save segmentation data', value=True,
    # #     #                       tooltip='option to save segmentation data before closing previous image'),
    # #     # # save_data_button=dict(widget_type='PushButton', text='save data',
    # #     # #                       tooltip='segmentation data is saved and optionally all open layers closed'),
        call_button="load image"
    )
    def widget(
            viewer: Viewer,
            call_button
    #         input_path,
    #         dummy: Demo,
    #         # chans_imaged,
    #         call_button,
    #         # rgb_box,
    #         # rgb_channels,
    #         # rgb_ds,
    #         # rgb_ds2,
    #         # rgb_contrast,
    #         # ds_params
    ):
    #     print('yolo')
        viewer = Viewer
        dummy = Demo()
        viewer.window.add_dock_widget(dummy)
    return widget
#





# import datetime
# from enum import Enum
# from pathlib import Path
#
# from magicgui import magicgui
#
# def preprocessing_widget():
#     class Medium(Enum):
#         """Using Enums is a great way to make a dropdown menu."""
#         Glass = 1.520
#         Oil = 1.515
#         Water = 1.333
#         Air = 1.0003
#
#
#     @magicgui(
#         call_button="Calculate",
#         layout="vertical",
#         result_widget=True,
#         # numbers default to spinbox widgets, but we can make
#         # them sliders using the 'widget_type' option
#         slider_float={"widget_type": "FloatSlider", "max": 100},
#         slider_int={"widget_type": "Slider", "readout": False},
#         radio_option={
#             "widget_type": "RadioButtons",
#             "orientation": "horizontal",
#             "choices": [("first option", 1), ("second option", 2)],
#             "allow_multiple": True
#         },
#         filename={"label": "Pick a file:"},  # custom label
#     )
#     def widget_demo(
#         boolean=True,
#         integer=1,
#         spin_float=3.14159,
#         slider_float=43.5,
#         slider_int=550,
#         string="Text goes here",
#         dropdown=Medium,
#         radio_option=2,
#         date=datetime.date(1999, 12, 31),
#         time=datetime.time(1, 30, 20),
#         datetime=datetime.datetime.now(),
#         filename=Path.home(),  # path objects are provided a file picker
#     ):
#         """Run some computation."""
#         # return locals().values()
#
#     return widget_demo