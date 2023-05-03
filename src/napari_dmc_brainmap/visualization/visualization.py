from napari import Viewer
import json

from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

import pandas as pd
from napari_dmc_brainmap.utils import split_to_list
from napari_dmc_brainmap.visualization.visualization_tools import load_data
from napari_dmc_brainmap.visualization.visualization_bar_plot import get_bar_plot_params, do_bar_plot


@magicgui(
    layout='vertical',
    input_path=dict(widget_type='FileEdit', label='input path: ',
                    value='',
                    mode='d',
                    tooltip='directory of folder containing folders with animals'),  # todo here groups
    save_path=dict(widget_type='FileEdit', label='save path: ', mode='d',
                   value='',
                   tooltip='select a folder for saving plots'),
    animal_list=dict(widget_type='LineEdit', label='list of animals',
                        value='', tooltip='enter the COMMA SEPERATED list of animals (no white spaces: animal1,animal2)'),
    channels=dict(widget_type='Select', label='select channels to plot', value=['green', 'cy3'],
                      choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                      tooltip='select the channels with segmented cells to be plotted, '
                              'to select multiple hold ctrl/shift'),

    call_button=False
)
def header_widget(
    viewer: Viewer,
    input_path,  # posix path
    save_path,
    animal_list,
    channels,

) -> None:

    return header_widget

@magicgui(
    layout='vertical',
    save_fig=dict(widget_type='CheckBox', label='save figure?', value=False,
                   tooltip='tick to save figure under directory and name'),
    save_name=dict(widget_type='LineEdit', label='enter name of figure to save',
                        value='test.svg', tooltip='enter name of figure (incl. extension (.svg/.png etc.)'),
    hemisphere=dict(widget_type='ComboBox', label='injection side',
                  choices=['left', 'right', 'both'], value='both',
                  tooltip="select hemisphere to visualize (relative to injection side)"),
    tgt_list=dict(widget_type='LineEdit', label='list of brain areas (ABA)',
                        value='area1,area2', tooltip='enter the COMMA SEPERATED list of names of areas (ABA nomenclature)'
                                          ' to plot (no white spaces: area1,area2)'),
    tgt_colors=dict(widget_type='LineEdit', label='list of colors',
                            value='c:Blues', tooltip='enter the COMMA SEPERATED list of colors used for plotting '
                                              '(no white spaces: color1,color2); '
                                              'for using a colormap start with "c:NAMEOFCMAP"'),
    plot_size=dict(widget_type='LineEdit', label='enter plot size',
                        value='8,6', tooltip='enter the COMMA SEPERATED size of the plot'),
    orient=dict(widget_type='ComboBox', label='select orientation of plot', value='vertical',
                              choices=['horizontal', 'vertical'],
                              tooltip='select orientation of plot'),
    xlabel=dict(widget_type='LineEdit', label='enter the xlabel',
                        value='Brain regions', tooltip='enter the xlabel of the plot'),
    xlabel_size=dict(widget_type='SpinBox', label='size of xlabel', value=14, min=1,
                   tooltip='select the size of the xlabel'),
    rotate_xticks=dict(widget_type='SpinBox', label='rotation of xticklabels',
                        value='45', tooltip='enter rotation of xticklabels, set to 0 for no rotation'),
    ylabel=dict(widget_type='LineEdit', label='enter the ylabel',
                        value='Proportion of cells [%]', tooltip='enter the ylabel of the plot'),
    ylabel_size=dict(widget_type='SpinBox', label='size of ylabel', value=14, min=1,
                   tooltip='select the size of the ylabel'),
    title=dict(widget_type='LineEdit', label='enter the title',
                            value='', tooltip='enter the title of the plot'),
    title_size=dict(widget_type='SpinBox', label='size of title', value=18, min=1,
                   tooltip='select the size of the title'),
    tick_size=dict(widget_type='SpinBox', label='size of ticks', value=12, min=1,
                       tooltip='select the size of the ticks'),
    alphabetic=dict(widget_type='CheckBox', label='alphabetic order of brain areas', value=False,
                   tooltip='choose to order brain areas alphabetically or in order of list provided above'),
    style=dict(widget_type='ComboBox', label='background of plot', value='white',
                              choices=['white', 'black'],
                              tooltip='select background of plot'),
    color=dict(widget_type='ComboBox', label='color of plot', value='black',
                                  choices=['white', 'black'],
                                  tooltip='select main color of plot for axis etc.'),
    scatter_hue=dict(widget_type='CheckBox', label='plot individual data points', value=True,
                   tooltip='option to add individual data points'),
    scatter_palette=dict(widget_type='LineEdit', label='colors of data points',
                                value='c:Greys', tooltip='enter the COMMA SEPERATED list of colors used scatter plot'
                                                  '(no white spaces: color1,color2); '
                                                  'for using a colormap start with "c:NAMEOFCMAP"'),
    scatter_size=dict(widget_type='SpinBox', label='size of data points', value=5, min=1,
                       tooltip='select the size individual data points'),
    scatter_legend_hide=dict(widget_type='CheckBox', label='hide data points legend', value=True,
                   tooltip='option to hide legend for individual data points'),
    absolute_numbers=dict(widget_type='CheckBox', label='plot absolute numbers', value=False,
                       tooltip='option to plot absolute numbers, if not ticked, relative percentages of used'),
    call_button=False
)
def barplot_widget(
    viewer: Viewer,
    save_fig,
    save_name,
    hemisphere,
    tgt_list,
    tgt_colors,
    plot_size,
    orient,
    xlabel,
    xlabel_size,
    rotate_xticks,
    ylabel,
    ylabel_size,
    title,
    title_size,
    tick_size,
    alphabetic,
    style,
    color,
    scatter_hue,
    scatter_palette,
    scatter_size,
    scatter_legend_hide,
    absolute_numbers
) -> None:

    return barplot_widget

class VisualizationWidget(QWidget):
    def __init__(self, napari_viewer):  #parent=None):
        super().__init__()  # (parent)
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        header = header_widget
        self._collapse_bar = QCollapsible('Bar plot: expand for more', self)
        barplot = barplot_widget
        self._collapse_bar.addWidget(barplot.native)
        btn = QPushButton("Create the plot")
        btn.clicked.connect(self._do_plotting)
        self.layout().addWidget(header.native)
        self.layout().addWidget(self._collapse_bar)
        self.layout().addWidget(btn)





    def _do_plotting(self):
        input_path = header_widget.input_path.value
        save_path = header_widget.save_path.value
        animal_list = split_to_list(header_widget.animal_list.value)
        channels = header_widget.channels.value
        plotting_params = get_bar_plot_params(barplot_widget)

        df = load_data(input_path, animal_list, channels)
        tgt_list = split_to_list(barplot_widget.tgt_list.value)
        mpl_widget = do_bar_plot(df, plotting_params, animal_list, tgt_list, barplot_widget, save_path)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)


