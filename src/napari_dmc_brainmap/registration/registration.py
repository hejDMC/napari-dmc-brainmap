import concurrent.futures
import json
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Union
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
from napari_dmc_brainmap.utils.gui_utils import check_input_path
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.general_utils import split_to_list, create_regi_dict
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, get_orient_map
from napari_dmc_brainmap.utils.color_manager import ColorManager
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.view.RegistrationViewer import RegistrationViewer
from napari_dmc_brainmap.visualization.vis_plots.brainsection_plotter import BrainsectionPlotter
from bg_atlasapi import BrainGlobeAtlas


def get_schematic_plotting_params(schematic_widget, regi_dict: Dict) -> Dict:
    """
    Extract plotting parameters for schematics from the widget and registration dictionary.

    Parameters:
        schematic_widget: The schematic widget instance.
        regi_dict (Dict): Registration dictionary containing atlas information.

    Returns:
        Dict: A dictionary of plotting parameters.
    """
    orient_dict = {
        'coronal': [['sagittal', 1], ['horizontal', 0]],
        'sagittal': [['coronal', 0], ['horizontal', 1]],
        'horizontal': [['coronal', 0], ['sagittal', 1]]
    }

    return {
        'section_orient': regi_dict['orientation'],
        'orient_list': [o[0] for o in orient_dict[regi_dict['orientation']]],
        'orient_idx_list': [o[1] for o in orient_dict[regi_dict['orientation']]],
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


def calculate_line(brain_sec_dim: Tuple[int, int], point: Tuple[int, int], angle: float) -> List[List[float]]:
    """
    Calculate the start and end points of a line given its angle and a point.

    Parameters:
        brain_sec_dim (Tuple[int, int]): Dimensions of the brain section (height, width).
        point (Tuple[int, int]): Point through which the line passes (y, x).
        angle (float): Angle of the line in degrees.

    Returns:
        List[List[float]]: Start and end points of the line [[y_start, x_start], [y_end, x_end]].
    """
    h, w = brain_sec_dim
    y1, x1 = point  # point as y, x
    angle_rad = np.radians(angle)
    slope = np.tan(angle_rad)

    # Calculate intersection with edges
    x_l, y_l = 0, y1 - (x1 - 0) * slope
    x_r, y_r = w, y1 + (w - x1) * slope

    # Adjust for top or bottom intersection
    if y_l < 0:
        y_l, x_l = 0, x1 - y1 / slope
    elif y_l > h:
        y_l, x_l = h, x1 - (y1 - h) / slope
    if y_r < 0:
        y_r, x_r = 0, x1 - y1 / slope
    elif y_r > h:
        y_r, x_r = h, x1 + (h - y1) / slope

    return [[y_l, x_l], [y_r, x_r]]


def plot_section(
    brainsection_plotter: BrainsectionPlotter,
    atlas: BrainGlobeAtlas,
    plotting_params: Dict,
    regi_data: Dict,
    bregma: Dict[str, float],
    orient: str,
    idx: int
) -> Tuple:
    """
    Plot a schematic section with atlas registration data.

    Parameters:
        brainsection_plotter (BrainsectionPlotter): Instance for plotting brain sections.
        atlas (BrainGlobeAtlas): Atlas instance for anatomical reference.
        plotting_params (Dict): Parameters for plotting.
        regi_data (Dict): Registration data.
        bregma (Dict[str, float]): Bregma reference coordinates.
        orient (str): Orientation for the section ('coronal', 'sagittal', etc.).
        idx (int): Index for orientation mapping.

    Returns:
        Tuple: Annotation data and lines for the section plot.
    """
    dummy_params = {'section_orient': orient}
    orient_mapping = get_orient_map(atlas, plotting_params)
    plot_mapping = get_orient_map(atlas, dummy_params)
    slice_idx = int(-(plotting_params[orient] / plot_mapping['z_plot'][2] - bregma[plot_mapping['z_plot'][1]]))

    annot_data = brainsection_plotter.plot_brain_schematic(slice_idx, plot_mapping['z_plot'][1])
    section_lines = []
    orient_idx_list = plotting_params['orient_idx_list']

    for section in regi_data["atlasLocation"].keys():
        angle = regi_data["atlasLocation"][section][orient_idx_list[idx]]
        regi_loc = int(-(regi_data["atlasLocation"][section][2] / orient_mapping['z_plot'][2]
                         - bregma[orient_mapping['z_plot'][1]]))

        if plotting_params['section_orient'] == 'sagittal' or \
                (plotting_params["section_orient"] == 'horizontal' and orient == 'sagittal'):
            angle *= (-1)
            angle += 90
            section_point = [int(annot_data[0].shape[0] / 2), regi_loc]
        else:
            section_point = [regi_loc, int(annot_data[0].shape[1] / 2)]

        point_l, point_r = calculate_line(annot_data[0].shape, section_point, angle)
        try:
            color = plotting_params['highlight_color'] if section in plotting_params['highlight_section'] else 'black'
        except TypeError:
            color = 'black'

        section_lines.append(([point_l[1], point_r[1]], [point_l[0], point_r[0]], color))

    return annot_data, section_lines

def do_schematic(
    brainsection_plotter: BrainsectionPlotter,
    atlas: BrainGlobeAtlas,
    plotting_params: Dict,
    regi_data: Dict,
    save_path: Path
) -> FigureCanvas:
    """
    Generate and save a schematic plot of section locations.

    Parameters:
        brainsection_plotter (BrainsectionPlotter): Instance for plotting brain sections.
        atlas (BrainGlobeAtlas): Atlas instance for anatomical reference.
        plotting_params (Dict): Parameters for plotting.
        regi_data (Dict): Registration data.
        save_path (Path): Path to save the schematic figure.

    Returns:
        FigureCanvas: The generated matplotlib figure canvas.
    """
    mpl_widget = FigureCanvas(Figure(figsize=(8, 6)))
    static_ax = mpl_widget.figure.subplots(1, 2)
    bregma = get_bregma(atlas.atlas_name)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                plot_section, brainsection_plotter, atlas, plotting_params, regi_data, bregma, orient, i
            )
            for i, orient in enumerate(plotting_params['orient_list'])
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Draw results in subplots
    for i, (annot_data, section_lines) in enumerate(results):
        annot_section, unique_ids, color_dict = annot_data
        for uid in unique_ids:
            mask = np.uint8(annot_section == uid)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                if len(points) > 4:
                    poly = Polygon(points)
                    x, y = poly.exterior.xy
                    vertices = np.column_stack((x, y))
                    static_ax[i].add_patch(
                        plt.Polygon(vertices, fc=color_dict[uid], ec='gainsboro', lw=0.5, alpha=1.0)
                    )
        for line in section_lines:
            static_ax[i].plot(line[0], line[1], color=line[2], lw=1)
        static_ax[i].axis('off')

    if plotting_params["save_fig"]:
        mpl_widget.figure.savefig(save_path.joinpath(plotting_params["save_name"]))

    return mpl_widget


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
            coronal,
            sagittal,
            horizontal,
            highlight_section,
            highlight_color):
        pass

    return schematic_widget


class RegistrationWidget(QWidget):
    """
    QWidget for configuring and performing atlas registration and schematic plotting.
    """
    def __init__(self, napari_viewer) -> None:
        """
        Initialize the RegistrationWidget.

        Parameters:
            napari_viewer: Napari viewer instance.
        """
        super().__init__()
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.header = initialize_widget()

        start_button = QPushButton("Start Registration GUI")
        start_button.clicked.connect(self._start_sharpy_track)

        self._collapse_schematic = QCollapsible("Plot schematic of section locations: expand for more")
        self.schematic = initialize_schematic_widget()
        self._collapse_schematic.addWidget(self.schematic.native)

        schematic_button = QPushButton("Create Plot")
        schematic_button.clicked.connect(self._do_schematic)
        self._collapse_schematic.addWidget(schematic_button)

        self.layout().addWidget(self.header.native)
        self.layout().addWidget(start_button)
        self.layout().addWidget(self._collapse_schematic)

    def _start_sharpy_track(self) -> None:
        """
        Launch the SHARPy-track registration GUI.
        """
        input_path = self.header.input_path.value
        if not check_input_path(input_path):
            return
        regi_chan = self.header.regi_chan.value
        regi_dir = get_info(input_path, 'sharpy_track', channel=regi_chan, only_dir=True)
        regi_dict = create_regi_dict(input_path, regi_dir)

        self.reg_viewer = RegistrationViewer(self, regi_dict)
        self.reg_viewer.show()

    def del_regviewer_instance(self) -> None:
        """
        Delete the RegistrationViewer instance to address memory leaks.
        """
        self.reg_viewer.widget.viewerLeft.scene.changed.disconnect()
        self.reg_viewer.widget.viewerRight.scene.changed.disconnect()

        if not self.reg_viewer.interpolatePositionAct.isEnabled():
            self.reg_viewer.interpolatePositionPage.close()
        
        if not self.reg_viewer.measurementAct.isEnabled():
            self.reg_viewer.measurementPage.close()

        if not self.reg_viewer.shortcutsAct.isEnabled():
            self.reg_viewer.shortcutsPage.close()

        del self.reg_viewer.regViewerWidget
        del self.reg_viewer.app
        del self.reg_viewer.regi_dict
        del self.reg_viewer.widget
        del self.reg_viewer.status
        del self.reg_viewer.atlasModel
        del self.reg_viewer

    def _do_schematic(self) -> None:
        """
        Generate and display a schematic plot of section locations.
        """
        input_path = self.header.input_path.value
        if not check_input_path(input_path):
            return

        regi_chan = self.header.regi_chan.value
        regi_dir, _, _ = get_info(input_path, 'sharpy_track', channel=regi_chan)
        regi_dict = create_regi_dict(input_path, regi_dir)
        try:
            with open(regi_dir.joinpath('registration.json')) as file:
                regi_data = json.load(file)
        except FileNotFoundError:
            print(" ['registration.json'] file missing for " + regi_dir.parts[-1] + " \n"
            "Check Data Integrity at folder: {} \n"
            "and try again!".format(regi_dir)
            )
            return

        save_path = Path(self.schematic.save_path.value) if str(self.schematic.save_path.value) != '.' else input_path
        plotting_params = get_schematic_plotting_params(self.schematic, regi_dict)
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(regi_dict['atlas'])
        color_manager = ColorManager()
        brainsection_plotter = BrainsectionPlotter(atlas, plotting_params, None, color_manager, None)

        mpl_widget = do_schematic(brainsection_plotter, atlas, plotting_params, regi_data, save_path)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

