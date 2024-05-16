from napari import Viewer


from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout
from magicgui import magicgui
from magicgui.widgets import FunctionGui

from napari_dmc_brainmap.utils import split_to_list, load_params
from napari_dmc_brainmap.visualization.visualization_tools import load_data
from napari_dmc_brainmap.visualization.visualization_bar_plot import get_bar_plot_params, do_bar_plot
from napari_dmc_brainmap.visualization.visualization_heatmap import get_heatmap_params, do_heatmap
from napari_dmc_brainmap.visualization.visualization_brain_section import get_brain_section_params, \
    do_brain_section_plot
from bg_atlasapi import BrainGlobeAtlas


def initialize_header_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path: ',
                              value='',
                              mode='d',
                              tooltip='directory of folder containing folders with animals'),  # todo here groups
              save_path=dict(widget_type='FileEdit', 
                             label='save path: ', 
                             mode='d',
                             value='',
                             tooltip='select a folder for saving plots'),
              animal_list=dict(widget_type='LineEdit', 
                               label='list of animals',
                               value='animal1,animal2', 
                               tooltip='enter the COMMA SEPERATED list of animals (no white spaces: animal1,animal2)'),
              channels=dict(widget_type='Select', 
                            label='select channels to plot', 
                            value=['green', 'cy3'],
                            choices=['dapi', 'green', 'n3', 'cy3', 'cy5'],
                            tooltip='select the channels with segmented cells to be plotted, '
                                'to select multiple hold ctrl/shift'),
              call_button=False)

    def header_widget(
        viewer: Viewer,
        input_path,  # posix path
        save_path,
        animal_list,
        channels):
        pass
    return header_widget


def initialize_barplot_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox', 
                            label='save figure?', 
                            value=False,
                            tooltip='tick to save figure under directory and name'),
              save_name=dict(widget_type='LineEdit', 
                             label='enter name of figure to save',
                             value='test.svg', 
                             tooltip='enter name of figure (incl. extension (.svg/.png etc.)'),
              hemisphere=dict(widget_type='ComboBox', 
                              label='hemisphere',
                              choices=['ipsi', 'contra', 'both'], 
                              value='both',
                              tooltip="select hemisphere to visualize (relative to injection side)"),
              tgt_list=dict(widget_type='LineEdit', 
                            label='list of brain areas (ABA)',
                            value='area1,area2', 
                            tooltip='enter the COMMA SEPERATED list of names of areas (ABA nomenclature)'
                                            ' to plot (no white spaces: area1,area2)'),
              tgt_colors=dict(widget_type='LineEdit', 
                              label='list of colors',
                              value='c:Blues', 
                              tooltip='enter the COMMA SEPERATED list of colors used for plotting '
                                                '(no white spaces: color1,color2); '
                                                'for using a colormap start with "c:NAMEOFCMAP"'),
              plot_size=dict(widget_type='LineEdit', 
                             label='enter plot size',
                             value='8,6', 
                             tooltip='enter the COMMA SEPERATED size of the plot'),
              orient=dict(widget_type='ComboBox', 
                          label='select orientation of plot', 
                          value='vertical',
                          choices=['horizontal', 'vertical'],
                          tooltip='select orientation of plot'),
              xlabel=dict(widget_type='LineEdit', 
                          label='enter the xlabel',
                          value='Brain regions', 
                          tooltip='enter the xlabel of the plot'),
              xlabel_size=dict(widget_type='SpinBox', 
                               label='size of xlabel', 
                               value=14, 
                               min=1,
                               tooltip='select the size of the xlabel'),
              rotate_xticks=dict(widget_type='SpinBox', 
                                 label='rotation of xticklabels',
                                 value='45', 
                                 tooltip='enter rotation of xticklabels, set to 0 for no rotation'),
              ylabel=dict(widget_type='LineEdit', 
                          label='enter the ylabel',
                          value='Proportion of cells [%]', 
                          tooltip='enter the ylabel of the plot'),
              ylabel_size=dict(widget_type='SpinBox', 
                               label='size of ylabel', 
                               value=14, 
                               min=1,
                               tooltip='select the size of the ylabel'),
              title=dict(widget_type='LineEdit', 
                         label='enter the title',
                         value='', 
                         tooltip='enter the title of the plot'),
              title_size=dict(widget_type='SpinBox', 
                              label='size of title', 
                              value=18, 
                              min=1,
                              tooltip='select the size of the title'),
              tick_size=dict(widget_type='SpinBox', 
                             label='size of ticks', 
                             value=12, 
                             min=1,
                             tooltip='select the size of the ticks'),
              alphabetic=dict(widget_type='CheckBox', 
                              label='alphabetic order of brain areas', 
                              value=False,
                              tooltip='choose to order brain areas alphabetically or in order of list provided above'),
              style=dict(widget_type='ComboBox', 
                         label='background of plot', 
                         value='white',
                         choices=['white', 'black'],
                         tooltip='select background of plot'),
              color=dict(widget_type='ComboBox', 
                         label='color of plot', 
                         value='black',
                         choices=['white', 'black'],
                         tooltip='select main color of plot for axis etc.'),
              scatter_hue=dict(widget_type='CheckBox', 
                               label='plot individual data points', 
                               value=True,
                               tooltip='option to add individual data points'),
              scatter_palette=dict(widget_type='LineEdit', 
                                   label='colors of data points',
                                   value='c:Greys', 
                                   tooltip='enter the COMMA SEPERATED list of colors used scatter plot'
                                                    '(no white spaces: color1,color2); '
                                                    'for using a colormap start with "c:NAMEOFCMAP"'),
              scatter_size=dict(widget_type='SpinBox', 
                                label='size of data points', 
                                value=5, 
                                min=1,
                                tooltip='select the size individual data points'),
              scatter_legend_hide=dict(widget_type='CheckBox', 
                                       label='hide data points legend', 
                                       value=True,
                                       tooltip='option to hide legend for individual data points'),
              absolute_numbers=dict(widget_type='CheckBox', 
                                    label='plot absolute numbers', 
                                    value=False,
                                    tooltip='option to plot absolute numbers, if not ticked, relative percentages of used'),
              call_button=False,
              scrollable=True)

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
        absolute_numbers):
        pass
    return barplot_widget


def initialize_heatmap_widget() -> FunctionGui:
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox', 
                            label='save figure?', 
                            value=False,
                            tooltip='tick to save figure under directory and name'),
              save_name=dict(widget_type='LineEdit', 
                             label='enter name of figure to save',
                             value='test.svg', 
                             tooltip='enter name of figure (incl. extension (.svg/.png etc.)'),
              hemisphere=dict(widget_type='ComboBox', 
                              label='hemisphere',
                              choices=['ipsi', 'contra', 'both'], 
                              value='both',
                              tooltip="select hemisphere to visualize (relative to injection side)"),
              tgt_list=dict(widget_type='LineEdit', 
                            label='list of brain areas (ABA)',
                            value='area1,area2', 
                            tooltip='enter the COMMA SEPERATED list of names of areas (ABA nomenclature)'
                                            ' to plot (no white spaces: area1,area2)'),
              intervals=dict(widget_type='LineEdit', 
                             label='intervals',
                             value='-0.5,0.0,0.5,1.0,1.5', 
                             tooltip='enter a COMMA SEPERATED list of mm coordinates relative '
                                                            'to bregma defining the intervals to plot (increasing in value)'),
              include_layers=dict(widget_type='CheckBox', 
                                  label='include layers?', 
                                  value=True,
                                  tooltip='option to include layers of brain areas defined above, e.g. PL1, PL2/3 etc. '
                                'for PL (NOT IMPLEMENTED YET)'),
              cmap=dict(widget_type='LineEdit', 
                        label='colormap',
                        value='c:Blues', 
                        tooltip='enter colormap to use for heatmap, start with a c: ; '
                                                        'e.g. "c:NAMEOFCMAP"'),
              cmap_min_max=dict(widget_type='LineEdit', 
                                label='colormap range',
                                value='-1,0.75', 
                                tooltip="enter COMMA SEPERATED [0] minimum value for colormap and "
                                                "[1] factor to multiply max range with"),
              cbar_label=dict(widget_type='LineEdit', 
                              label='colormap',
                              value='Proportion of cells [%]', 
                              tooltip='enter a label for the colorbar'),
              plot_size=dict(widget_type='LineEdit', 
                             label='enter plot size',
                             value='8,6', 
                             tooltip='enter the COMMA SEPERATED size of the plot'),
              xlabel=dict(widget_type='LineEdit', 
                          label='enter the xlabel',
                          value='', 
                          tooltip='enter the xlabel of the plot'),
              xlabel_size=dict(widget_type='SpinBox', 
                               label='size of xlabel', 
                               value=16, 
                               min=1,
                               tooltip='select the size of the xlabel'),
        #rotate_xticks=dict(widget_type='SpinBox', label='rotation of xticklabels',
        #                    value='45', tooltip='enter rotation of xticklabels, set to 0 for no rotation'),
              ylabel=dict(widget_type='LineEdit', 
                          label='enter the ylabel',
                          value='Distance relative to bregma', 
                          tooltip='enter the ylabel of the plot'),
              ylabel_size=dict(widget_type='SpinBox', 
                               label='size of ylabel', 
                               value=16, 
                               min=1,
                               tooltip='select the size of the ylabel'),
              title=dict(widget_type='LineEdit', 
                         label='enter the title',
                         value='', 
                         tooltip='enter the title of the plot'),
              title_size=dict(widget_type='SpinBox', 
                              label='size of title', 
                              value=18, 
                              min=1,
                              tooltip='select the size of the title'),
        #tick_size=dict(widget_type='SpinBox', label='size of ticks', value=12, min=1,
        #                   tooltip='select the size of the ticks'),
              style=dict(widget_type='ComboBox', 
                         label='background of plot', 
                         value='white',
                         choices=['white', 'black'],
                         tooltip='select background of plot'),
              color=dict(widget_type='ComboBox', 
                         label='color of plot', 
                         value='black',
                         choices=['white', 'black'],
                         tooltip='select main color of plot for axis etc.'),
              transpose=dict(widget_type='CheckBox', 
                             label='transpose data?', 
                             value=True,
                             tooltip='option to transpose data'),  # todo explain what this does
              absolute_numbers=dict(widget_type='CheckBox', 
                                    label='plot absolute numbers', 
                                    value=False,
                                    tooltip='option to plot absolute numbers, if not ticked, relative percentages of used'),
              call_button=False,
              scrollable=True)

    def heatmap_widget(
        viewer: Viewer,
        save_fig,
        save_name,
        hemisphere,
        tgt_list,
        intervals,
        include_layers,
        cmap,
        cmap_min_max,
        cbar_label,
        plot_size,
        xlabel,
        xlabel_size,
        ylabel,
        ylabel_size,
        title,
        title_size,
        style,
        color,
        transpose,
        absolute_numbers):
        pass
    return heatmap_widget



def initialize_brainsection_widget() -> FunctionGui:
    # todo option for only one hemisphere
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox', 
                            label='save figure?', 
                            value=False,
                            tooltip='tick to save figure under directory and name'),
              save_name=dict(widget_type='LineEdit', 
                             label='enter name of figure to save',
                             value='test.svg', 
                             tooltip='enter name of figure (incl. extension (.svg/.png etc.)'),
              plot_outline=dict(widget_type='CheckBox',
                            label='plot contours of brain areas',
                            value=False,
                            tooltip='tick to plot contours of brain areas'),
              plot_item=dict(widget_type='Select', 
                             label='item to plot',
                             choices=['cells', 'injection_side', 'projections', 'optic_fiber', 'neuropixels_probe'],
                             tooltip='select items to plot cells/injection side/projection density, hold ctrl/shift to select multiple'),
              section_orient=dict(widget_type='ComboBox', 
                                  label='orientation of sectioning',
                                  choices=['coronal', 'sagittal', 'horizontal'], 
                                  value='coronal',
                                  tooltip="select the how you sliced the brain"),
              brain_areas=dict(widget_type='LineEdit', 
                               label='list of brain areas',
                               tooltip='enter the COMMA SEPERATED list of names of brain areas (acronym)'
                                ' to plot (no white spaces: area1,area2)'),
              brain_areas_color=dict(widget_type='LineEdit', 
                                     label='brain area colors',
                                     tooltip='enter the COMMA SEPERATED list of colors for brain areas '
                                    '(no white spaces: red,blue,yellow)'),
              brain_areas_transparency=dict(widget_type='LineEdit', 
                                            label='brain area transparency',
                                            tooltip='enter the COMMA SEPERATED transparency values for colors for brain areas '
                                    ' in 8-bit range (max 255, min 0; no white spaces: 100,42,255)'),
              plot_size=dict(widget_type='LineEdit', 
                             label='enter plot size',
                             value='8,6', 
                             tooltip='enter the COMMA SEPERATED size of the plot'),
              dot_size=dict(widget_type='LineEdit',
                             label='enter dot size',
                             value='10',
                             tooltip='enter the size of the dots (only for visualizing cells, neuropixels and optic fibres'),
              section_list=dict(widget_type='LineEdit', 
                                label='list of sections',
                                value='-0.5,0.0,0.5,1.0,1.5', 
                                tooltip='enter a COMMA SEPERATED list of mm coordinates '
                                                            '(relative to bregma)indicating '
                                                            'the brain sections you want to plot'),
              section_range=dict(widget_type='LineEdit', 
                                 label='range around section', 
                                 value='0.05',
                                 tooltip='enter the range around the section to include data from, set to zero if only include '
                                'data from that particular coordinate, otherwise this value will be taken plus/minus to '
                                'include more data'),
              groups=dict(widget_type='ComboBox', 
                          label='channel/group/genotype/animals separately?',
                          choices=['', 'channel', 'group', 'genotype', 'animal_id'], 
                          value='',
                          tooltip="if you want to plot channel/group/genotype or individual animals in different colors, "
                            "select accordingly, otherwise leave empty"),
              color_cells_atlas=dict(widget_type='CheckBox',
                            label='color cells according to atlas?',
                            value=False,
                            tooltip='tick to color cells according to atlas (if ticked this override manual colors '
                                    'entered below (e.g. colors of different groups).'),
              color_cells=dict(widget_type='LineEdit', 
                               label='colors (cell plot)',
                               value='',
                               tooltip='enter COMMA SEPERATED list of colors (or c:map), should have the same length as '
                                'the groups/genotypes you want to plot'),
              cmap_projection=dict(widget_type='LineEdit', 
                                   label='colormap (projection density)',
                                   value='Blues', 
                                   tooltip='enter a colormap for visualizing projections (e.g. Reds, Blues etc.)'),
              bin_width=dict(widget_type='SpinBox', 
                             label='bin_width (projection density)', 
                             value=5, 
                             min=1, 
                             max=800,
                             tooltip='bin width for visualization of axonal density'),
              vmax=dict(widget_type='LineEdit', 
                        label='vmax (projection density)', 
                        value='2000',
                        tooltip='max value for colorbar for visualizing projection densities '
                        '(depends on actual density and bin_width)'),
              color_inj=dict(widget_type='LineEdit', 
                             label='colors (injection side)',
                             value='Blue,Yellow', 
                             tooltip='enter a COMMA SEPERATED list for colors to use for the injection side'),
              color_optic=dict(widget_type='LineEdit', 
                               label='colors (optic fiber)',
                               value='Green,Pink', 
                               tooltip='enter a COMMA SEPERATED list for colors to use for the optic fiber(s)'),
              color_npx=dict(widget_type='LineEdit', 
                             label='colors (neuropixels)',
                             value='Red,Brown', 
                             tooltip='enter a COMMA SEPERATED list for colors to use for the neuropixels probes(s)'),
              call_button=False,
              scrollable=True)

    def brain_section_widget(
        viewer: Viewer,
        save_fig,
        save_name,
        plot_outline,
        plot_item,
        section_orient,
        brain_areas,
        brain_areas_color,
        brain_areas_transparency,
        plot_size,
        dot_size,
        section_list,
        section_range,
        groups,
        color_cells_atlas,
        color_cells,
        cmap_projection,
        bin_width,
        vmax,
        color_inj,
        color_optic,
        color_npx):
        pass
    return brain_section_widget


class VisualizationWidget(QWidget):
    def __init__(self, napari_viewer):  #parent=None):
        super().__init__()  # (parent)
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.header = initialize_header_widget()
        self.header.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        self._collapse_bar = QCollapsible('Bar plot: expand for more', self)
        self.barplot = initialize_barplot_widget()
        self.barplot.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_bar.addWidget(self.barplot.root_native_widget)
        btn_bar = QPushButton("Create bar plot")
        btn_bar.clicked.connect(self._do_bar_plot)
        self._collapse_bar.addWidget(btn_bar)

        self._collapse_heat = QCollapsible('Heatmap: expand for more', self)
        self.heatmap = initialize_heatmap_widget()
        self.heatmap.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_heat.addWidget(self.heatmap.root_native_widget)
        btn_heat = QPushButton("Create heatmap")
        btn_heat.clicked.connect(self._do_heatmap)
        self._collapse_heat.addWidget(btn_heat)

        self._collapse_section = QCollapsible('Brain section plot: expand for more', self)
        self.brain_section = initialize_brainsection_widget()
        self.brain_section.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_section.addWidget(self.brain_section.root_native_widget)
        btn_section = QPushButton("Create brain section plot")
        btn_section.clicked.connect(self._do_brain_section_plot)
        self._collapse_section.addWidget(btn_section)

        self.layout().addWidget(self.header.native)
        self.layout().addWidget(self._collapse_bar)
        self.layout().addWidget(self._collapse_heat)
        self.layout().addWidget(self._collapse_section)

    # todo: for now all animals need to be registered to the same atlas
    def _do_bar_plot(self):
        input_path = self.header.input_path.value
        if str(self.header.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.header.save_path.value
        animal_list = split_to_list(self.header.animal_list.value)
        channels = self.header.channels.value
        plotting_params = get_bar_plot_params(self.barplot)
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
        df = load_data(input_path, atlas, animal_list, channels)
        tgt_list = split_to_list(self.barplot.tgt_list.value)
        mpl_widget = do_bar_plot(df, atlas, plotting_params, animal_list, tgt_list, self.barplot, save_path)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)


    def _do_heatmap(self):
        input_path = self.header.input_path.value
        if str(self.header.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.header.save_path.value
        animal_list = split_to_list(self.header.animal_list.value)
        channels = self.header.channels.value
        plotting_params = get_heatmap_params(self.heatmap)
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
        df = load_data(input_path, atlas, animal_list, channels)
        tgt_list = split_to_list(self.heatmap.tgt_list.value)
        mpl_widget = do_heatmap(df, atlas, animal_list, tgt_list, plotting_params, self.heatmap, save_path)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

    def _do_brain_section_plot(self):
        input_path = self.header.input_path.value
        if str(self.header.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.header.save_path.value
        animal_list = split_to_list(self.header.animal_list.value)
        channels = self.header.channels.value
        plotting_params = get_brain_section_params(self.brain_section)
        plot_item = self.brain_section.plot_item.value
        data_dict = {}
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        print("loading reference atlas...")
        atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])

        for item in plot_item:
            data_dict[item] = load_data(input_path, atlas, animal_list, channels, data_type=item)

        mpl_widget = do_brain_section_plot(input_path, atlas, data_dict, animal_list, plotting_params, self.brain_section,
                                                     save_path)
        self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

