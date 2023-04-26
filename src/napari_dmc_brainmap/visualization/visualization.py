from napari import Viewer
import json

from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import pandas as pd
from napari_dmc_brainmap.utils import get_animal_id, get_info, split_strings_layers, clean_results_df, split_to_list
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.find_structure import sliceHandle
from napari_dmc_brainmap.visualization.visualization_tools import *


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

    def _get_plotting_params(self):
        plotting_params = {

            "horizontal": barplot_widget.orient.value,
            "xlabel": [barplot_widget.xlabel.value, int(barplot_widget.xlabel_size.value)],  # 0: label, 1: fontsize
            "ylabel": [barplot_widget.ylabel.value, int(barplot_widget.ylabel_size.value)],
            "tick_size": int(barplot_widget.tick_size.value),  # for now only y and x same size
            "rotate_xticks": int(barplot_widget.rotate_xticks.value),  # set to False of no rotation
            "title": [barplot_widget.title.value, int(barplot_widget.title_size.value)],
            "alphabetic": barplot_widget.alphabetic.value,
            "style": barplot_widget.style.value,
            "color": barplot_widget.color.value,
            "bar_palette": split_to_list(barplot_widget.tgt_colors.value),
            "scatter_palette": split_to_list(barplot_widget.scatter_palette.value),
            "scatter_hue": barplot_widget.scatter_hue.value,
            "scatter_size": int(barplot_widget.scatter_size.value),
            "scatter_legend_hide": barplot_widget.scatter_legend_hide.value,
            "save_name": barplot_widget.save_name.value,
            "save_fig": barplot_widget.save_fig.value,
            "absolute_numbers": barplot_widget.absolute_numbers.value
        }
        return plotting_params

    def _load_data(self, input_path, animal_list, channels):
        s = sliceHandle()
        st = s.df_tree
        #  loop over animal_ids
        results_data_merged = pd.DataFrame()  # initialize merged dataframe
        for animal_id in animal_list:
            # for animal_idx, animal_id in enumerate(animal_list):
            for channel in channels:
                results_dir = get_info(input_path.joinpath(animal_id), 'results', seg_type='cells', channel=channel,
                                       only_dir=True)
                results_file = results_dir.joinpath(animal_id + '_cells.csv')

                if results_file.exists():
                    results_data = pd.read_csv(results_file)  # load the data
                    results_data['sphinx_id'] -= 1  # correct for indices starting at 1
                    results_data['animal_id'] = [animal_id] * len(
                        results_data)  # add the animal_id as a column for later identification
                    results_data['channel'] = [channel] * len(results_data)
                    # add the injection hemisphere stored in params.json file
                    params_file = input_path.joinpath(animal_id, 'params.json')  # directory of params.json file
                    with open(params_file) as fn:  # load the file
                        params_data = json.load(fn)
                    injection_side = params_data['general']['injection_side']  # add the injection_side as a column
                    results_data['injection_side'] = [injection_side] * len(results_data)
                    # add if the location of a cell is ipsi or contralateral to the injection side
                    results_data = get_ipsi_contra(results_data)
                    results_data_merged = pd.concat([results_data_merged, results_data])
            print("Done with " + animal_id)
            results_data_merged = clean_results_df(results_data_merged, st)
        return results_data_merged

    def _do_plotting(self):
        input_path = header_widget.input_path.value
        save_path = header_widget.save_path.value
        animal_list = split_to_list(header_widget.animal_list.value)
        channels = header_widget.channels.value
        plotting_params = self._get_plotting_params()

        df = self._load_data(input_path, animal_list, channels)
        tgt_list = split_to_list(barplot_widget.tgt_list.value)
        hemisphere = barplot_widget.hemisphere.value

        # if applicable only get the ipsi or contralateral cells
        if hemisphere == 'ipsi':
            df = df[df['ipsi_contra'] == 'ipsi']
        elif hemisphere == 'contra':
            df = df[df['ipsi_contra'] == 'contra']

        if plotting_params["horizontal"] == "horizontal":
            plot_orient = 'h'
            x_var = "percent_cells"
            y_var = "tgt_name"
        else:
            plot_orient = 'v'
            y_var = "percent_cells"
            x_var = "tgt_name"
        # get re-structured dataframe for plotting
        tgt_data_to_plot = calculate_percentage_bar_plot(df, animal_list, tgt_list, plotting_params)

        if not plotting_params[
            "alphabetic"]:  # re-structuring of df creates alphabetic order of brain areas, if tgt_list order should be kept do resort
            tgt_data_to_plot = resort_df(tgt_data_to_plot, tgt_list)

        mpl_widget = FigureCanvas(Figure(figsize=([int(i) for i in barplot_widget.plot_size.value.split(',')])))
        static_ax = mpl_widget.figure.subplots()

        sns.set(style=plotting_params["style"])  # set style todo: dark style not really implemented
        sns.barplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot, palette=plotting_params["bar_palette"],
                         capsize=.1, errorbar=None, orient=plot_orient)  # do the barplot
        if plotting_params["scatter_hue"]:  # color code dots by animals
            sns.swarmplot(ax=static_ax, x=x_var, y=y_var, hue='animal_id', data=tgt_data_to_plot,
                               palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
                               orient=plot_orient)
        else:
            sns.swarmplot(ax=static_ax, x=x_var, y=y_var, data=tgt_data_to_plot,
                               palette=plotting_params["scatter_palette"], size=plotting_params["scatter_size"],
                               orient=plot_orient)
        if plot_orient == 'v':
            static_ax.set_xlabel(plotting_params["xlabel"][0], fontsize=plotting_params["xlabel"][1])
            static_ax.set_ylabel(plotting_params["ylabel"][0], fontsize=plotting_params["ylabel"][1])
        else:
            static_ax.set_ylabel(plotting_params["xlabel"][0], fontsize=plotting_params["xlabel"][1])
            static_ax.set_xlabel(plotting_params["ylabel"][0], fontsize=plotting_params["ylabel"][1])

        static_ax.set_title(plotting_params["title"][0], fontsize=plotting_params["title"][1])
        static_ax.spines['top'].set_visible(False)
        static_ax.spines['right'].set_visible(False)
        if plotting_params["scatter_hue"]:  # adjust color for legend
            leg = static_ax.get_legend()
            leg.get_title().set_color(plotting_params["color"])
            frame = leg.get_frame()
            frame.set_alpha(None)
            frame.set_facecolor((0, 0, 1, 0))
            for text in leg.get_texts():
                text.set_color(plotting_params["color"])
        if plotting_params["scatter_legend_hide"]:  # remove legend from scatter plot
            static_ax.legend_.remove()
        static_ax.spines['bottom'].set_color(plotting_params["color"])
        static_ax.spines['left'].set_color(plotting_params["color"])
        static_ax.xaxis.label.set_color(plotting_params["color"])
        static_ax.yaxis.label.set_color(plotting_params["color"])
        static_ax.tick_params(colors=plotting_params["color"], labelsize=plotting_params["tick_size"])
        if plotting_params["rotate_xticks"]:  # rotate x-ticks
            static_ax.set_xticklabels(static_ax.get_xticklabels(), rotation=plotting_params["rotate_xticks"])
        if plotting_params["save_fig"]:
            static_ax.figure.savefig(save_path.joinpath(plotting_params["save_name"]))

        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)


