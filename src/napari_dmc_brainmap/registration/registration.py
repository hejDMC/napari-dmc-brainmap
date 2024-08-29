import concurrent.futures
import json
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from napari_dmc_brainmap.utils import create_regi_dict, split_to_list, get_info, get_bregma
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer
from napari_dmc_brainmap.visualization.visualization_brain_section import get_orient_map
from napari_dmc_brainmap.visualization.visualization_tools import plot_brain_schematic
from bg_atlasapi import BrainGlobeAtlas


def get_schematic_plotting_params(schematic_widget, regi_dict):
    orient_dict = {
        'coronal': [['sagittal', 1], ['horizontal', 0]],
        'sagittal': [['coronal', 0], ['horizontal', 1]],
        'horizontal': [['coronal', 0], ['sagittal', 1]]
    }

    plotting_params = {
        'section_orient': regi_dict['orientation'],
        'orient_list': [o[0] for o in orient_dict[regi_dict['orientation']]],
        'orient_idx_list': [o[1] for o in orient_dict[regi_dict['orientation']]],
        'plot_outline': schematic_widget.plot_outline.value,
        'brain_areas': False,
        'unilateral': False,
        'coronal': float(schematic_widget.coronal.value),
        'sagittal': float(schematic_widget.sagittal.value),
        'horizontal': float(schematic_widget.horizontal.value),
        'highlight_section': split_to_list(schematic_widget.highlight_section.value),
        'highlight_color': schematic_widget.highlight_color.value,
        'save_fig': schematic_widget.save_fig.value,
        'save_name': schematic_widget.save_name.value
    }
    return plotting_params

def caculate_line(brain_sec_dim, point, angle):

    h, w = brain_sec_dim[0], brain_sec_dim[1]
    y1, x1 = point  # point as y, x
    angle_rad = np.radians(angle)
    slope = np.tan(angle_rad)
    # Calculate where the line intersects the boundaries of the matrix
    # Extend to the left edge (x = 0)
    x_l = 0
    y_l = y1 - (x1 - x_l) * slope

    # Extend to the right edge (x = width)
    x_r = w
    y_r = y1 + (x_r - x1) * slope

    # If the line is steep, it might intersect the top or bottom before the left/right edges
    if y_l < 0:
        y_l = 0
        x_l = x1 - (y1 - y_l) / slope
    elif y_l > h:
        y_l = h
        x_l = x1 - (y1 - y_l) / slope

    if y_r < 0:
        y_r = 0
        x_r = x1 + (y_r - y1) / slope
    elif y_r > h:
        y_r = h
        x_r = x1 + (y_r - y1) / slope
    point_l = [y_l, x_l]
    point_r = [y_r, x_r]
    return [point_l, point_r]  # points as y,x


def plot_section(atlas, plotting_params, regi_data, bregma, orient, i):
    dummy_params = {'section_orient': orient}
    orient_mapping = get_orient_map(atlas, plotting_params)
    plot_mapping = get_orient_map(atlas, dummy_params)
    slice_idx = int(-(plotting_params[orient] / plot_mapping['z_plot'][2] - bregma[plot_mapping['z_plot'][1]]))

    annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx, plot_mapping['z_plot'][1],
                                                                     plotting_params)

    section_lines = []
    orient_idx_list = plotting_params['orient_idx_list']
    for section in regi_data["atlasLocation"].keys():
        angle = regi_data["atlasLocation"][section][orient_idx_list[i]]
        regi_loc = int(-(regi_data["atlasLocation"][section][2] / orient_mapping['z_plot'][2] - bregma[
            orient_mapping['z_plot'][1]]))

        if plotting_params['section_orient'] == 'sagittal' or \
                (plotting_params["section_orient"] == 'horizontal' and orient == 'sagittal'):
            angle *= (-1)
            angle += 90
            section_point = [int(annot_section_plt.shape[0] / 2), regi_loc]
        else:
            section_point = [regi_loc, int(annot_section_plt.shape[1] / 2)]

        point_l, point_r = caculate_line(annot_section_plt.shape, section_point, angle)
        if section in plotting_params['highlight_section']:
            clr = plotting_params['highlight_color']
        else:
            clr = 'black'

        section_lines.append(([point_l[1], point_r[1]], [point_l[0], point_r[0]], clr))

    return (annot_section_plt, annot_section_contours, section_lines)


def do_schematic(atlas, plotting_params, regi_data, save_path):
    mpl_widget = FigureCanvas(Figure(figsize=(8, 6)))  # todo option to change figsize
    static_ax = mpl_widget.figure.subplots(1, 2)
    bregma = get_bregma(atlas.atlas_name)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for i, orient in enumerate(plotting_params['orient_list']):
            futures.append(executor.submit(plot_section, atlas, plotting_params, regi_data, bregma, orient, i))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Draw the results in the corresponding subplots
    for i, (annot_section_plt, annot_section_contours, section_lines) in enumerate(results):
        static_ax[i].imshow(annot_section_plt)
        if annot_section_contours:
            static_ax[i].imshow(annot_section_contours)

        for line in section_lines:
            static_ax[i].plot(line[0], line[1], color=line[2], lw=1)

        static_ax[i].axis('off')

    if plotting_params["save_fig"]:
        mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))

    return mpl_widget

# def do_schematic(atlas, plotting_params, regi_data, save_path):
#     mpl_widget = FigureCanvas(Figure(figsize=(8, 6)))  # todo option to change figsize
#     static_ax = mpl_widget.figure.subplots(1, 2)
#     bregma = get_bregma(atlas.atlas_name)
#
#     for i, orient in enumerate(plotting_params['orient_list']):
#         dummy_params = {'section_orient': orient}
#         orient_mapping = get_orient_map(atlas, plotting_params)
#         plot_mapping = get_orient_map(atlas, dummy_params)
#         slice_idx = int(-(plotting_params[orient] / plot_mapping['z_plot'][2] - bregma[plot_mapping['z_plot'][1]]))
#         annot_section_plt, annot_section_contours = plot_brain_schematic(atlas, slice_idx, plot_mapping['z_plot'][1],
#                                                                      plotting_params)
#         static_ax[i].imshow(annot_section_plt)
#         if annot_section_contours:
#             static_ax[i].imshow(annot_section_contours)
#         orient_idx_list = plotting_params['orient_idx_list']
#         for section in regi_data["atlasLocation"].keys():
#             angle = regi_data["atlasLocation"][section][orient_idx_list[i]]
#             regi_loc = int(-(regi_data["atlasLocation"][section][2]
#                              / orient_mapping['z_plot'][2] - bregma[orient_mapping['z_plot'][1]]))
#             if plotting_params['section_orient'] == 'sagittal' or \
#                     (plotting_params["section_orient"] == 'horizontal' and orient == 'sagittal'):
#                 angle *= (-1)
#                 angle += 90
#                 section_point = [int(annot_section_plt.shape[0]/2), regi_loc]
#             else:
#                 section_point = [regi_loc, int(annot_section_plt.shape[1]/2)]
#
#             point_l, point_r = caculate_line(annot_section_plt.shape, section_point, angle)
#             if section in plotting_params['highlight_section']:
#                 clr = plotting_params['highlight_color']
#             else:
#                 clr = 'black'
#             static_ax[i].plot([point_l[1], point_r[1]], [point_l[0], point_r[0]], color=clr, lw=1)
#         static_ax[i].axis('off')
#     if plotting_params["save_fig"]:
#         mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))
#     return mpl_widget


def initialize_widget() -> FunctionGui:
    @magicgui(input_path=dict(widget_type='FileEdit', 
                              label='input path (animal_id): ', 
                              mode='d',
                              tooltip='directory of folder containing subfolders with SHARPy-track images, '
                                'NOT folder containing SHARPy-track images itself'),
              regi_chan=dict(widget_type='ComboBox', 
                             label='registration channel',
                             choices=['dapi', 'green', 'n3', 'cy3', 'cy5'], 
                             value='green',
                             tooltip="select the registration channel (channel subfolder with images needs to be in sharpy_track folder)"),
              call_button=False)

    def header_widget(
            self,
            input_path,
            regi_chan):
        pass
    return header_widget

def initialize_schematic_widget() -> FunctionGui:
    @magicgui(save_fig=dict(widget_type='CheckBox',
                            label='save figure?',
                            value=False,
                            tooltip='tick to save figure under directory and name'),
              save_name=dict(widget_type='LineEdit',
                             label='enter name of figure to save',
                             value='test.svg',
                             tooltip='enter name of figure (incl. extension (.svg/.png etc.)'),
              save_path=dict(widget_type='FileEdit',
                             label='save path: ',
                             mode='d',
                             value='',
                             tooltip='select a folder for saving plots, if left empty, plot will be saved under animal '
                                     'directory'),
              plot_outline=dict(widget_type='CheckBox',
                            label='plot contours of brain areas',
                            value=False,
                            tooltip='tick to plot contours of brain areas'),
              coronal=dict(widget_type='LineEdit',
                                label='coronal coordinate (mm)',
                                value='0.0',
                                tooltip='enter the coordinate (mm relative to bregma) of the schematic section to be plotted,'
                                        'ignore the orientation you used for registration, e.g. if your'
                                        'section were registered in coronal orientation, change the values for'
                                        'sagittal/horizontal (on which the sections will be drawn)'),
              sagittal=dict(widget_type='LineEdit',
                           label='sagittal coordinate (mm)',
                           value='-1.0',
                           tooltip='enter the coordinate (mm relative to bregma) of the schematic section to be plotted,'
                                   'ignore the orientation you used for registration, e.g. if your'
                                   'section were registered in coronal orientation, change the values for'
                                   'sagittal/horizontal (on which the sections will be drawn)'),
              horizontal=dict(widget_type='LineEdit',
                            label='horizontal coordinate (mm)',
                            value='-3.0',
                            tooltip='enter the coordinate (mm relative to bregma) of the schematic section to be plotted,'
                                    'ignore the orientation you used for registration, e.g. if your'
                                    'section were registered in coronal orientation, change the values for'
                                    'sagittal/horizontal (on which the sections will be drawn)'),
              highlight_section=dict(widget_type='LineEdit',
                                label='highlight sections (#)',
                                value='0,5,10',
                                tooltip='enter a COMMA SEPERATED list of sections (by their number) you want to '
                                        'highlight (see first column of image_names.csv file for number of section)'),
              highlight_color=dict(widget_type='LineEdit',
                             label='highlight color',
                             value='tomato',
                             tooltip="enter the color you want to use for highlighting sections, "
                                     "all non highlighted section are schematized by black lines"),
              call_button=False)
    def schematic_widget(
            self,
            save_fig,
            save_name,
            save_path,
            plot_outline,
            coronal,
            sagittal,
            horizontal,
            highlight_section,
            highlight_color):
        pass
    return schematic_widget


class RegistrationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.header = initialize_widget()
        btn = QPushButton("start registration GUI")
        btn.clicked.connect(self._start_sharpy_track)

        self._collapse_schematic = QCollapsible("Plot schematic of section locations: expand for more")
        self.schematic = initialize_schematic_widget()
        self._collapse_schematic.addWidget(self.schematic.native)
        btn_schematic = QPushButton("Create plot")
        btn_schematic.clicked.connect(self._do_schematic)
        self._collapse_schematic.addWidget(btn_schematic)

        self.layout().addWidget(self.header.native)
        self.layout().addWidget(btn)
        self.layout().addWidget(self._collapse_schematic)


    def _start_sharpy_track(self):
        # todo think about solution to check and load atlas data
        input_path = self.header.input_path.value
        regi_chan = self.header.regi_chan.value

        regi_dict = create_regi_dict(input_path, regi_chan)

        self.reg_viewer = RegistrationViewer(self, regi_dict)
        self.reg_viewer.show()


    def del_regviewer_instance(self): # temporary fix for memory leak, maybe not complete, get back to this in the future
        self.reg_viewer.widget.viewerLeft.scene.changed.disconnect()
        self.reg_viewer.widget.viewerRight.scene.changed.disconnect()

        if self.reg_viewer.helperAct.isEnabled():
            pass
        else: # if registration helper is opened, close it too
            self.reg_viewer.helperPage.close()
    
        del self.reg_viewer.regViewerWidget
        del self.reg_viewer.app
        del self.reg_viewer.regi_dict
        del self.reg_viewer.widget
        del self.reg_viewer.status
        del self.reg_viewer.atlasModel
        del self.reg_viewer
        
    def _do_schematic(self):
        input_path = self.header.input_path.value
        regi_chan = self.header.regi_chan.value
        regi_dict = create_regi_dict(input_path, regi_chan)
        regi_dir, regi_im_list, regi_suffix = get_info(input_path, 'sharpy_track', channel=regi_chan)
        with open(regi_dir.joinpath('registration.json')) as fn:
            regi_data = json.load(fn)
        if str(self.schematic.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.schematic.save_path.value
        plotting_params = get_schematic_plotting_params(self.schematic, regi_dict)
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(regi_dict['atlas'])
        mpl_widget = do_schematic(atlas, plotting_params, regi_data, save_path)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)
