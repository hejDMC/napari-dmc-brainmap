import pandas as pd
from napari import Viewer
import time
from qtpy.QtCore import Signal
from superqt import QCollapsible
from qtpy.QtWidgets import QPushButton, QWidget, QVBoxLayout, QMessageBox
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from typing import List, Optional

from napari_dmc_brainmap.utils.general_utils import split_to_list
from napari_dmc_brainmap.utils.params_utils import load_params
from bg_atlasapi import BrainGlobeAtlas
from napari_dmc_brainmap.utils.gui_utils import ProgressBar, check_input_path
from napari_dmc_brainmap.utils.data_loader import DataLoader
from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import get_descendants
from napari_dmc_brainmap.visualization.vis_utils.gene_info_dialog import GeneInfoDialog
from napari_dmc_brainmap.visualization.vis_plots.barplot_visualization import BarplotVisualization
from napari_dmc_brainmap.visualization.vis_plots.heatmap_visualization import HeatmapVisualization
from napari_dmc_brainmap.visualization.vis_plots.brainsection_visualization import BrainsectionVisualization



@thread_worker
def calculate_barplot(barplot_vis: BarplotVisualization) -> pd.DataFrame:
    """
    Calculate data for barplot visualization in a background thread.

    Parameters:
        barplot_vis (BarplotVisualization): Barplot visualization object.

    Returns:
        pd.DataFrame: Dataframe containing the barplot data.
    """
    return barplot_vis.calculate_plot()

@thread_worker
def calculate_heatmap(heatmap_vis: HeatmapVisualization, progress_heatmap: Signal) -> pd.DataFrame:
    """
    Calculate data for heatmap visualization in a background thread.

    Parameters:
        heatmap_vis (HeatmapVisualization): Heatmap visualization object.
        progress_heatmap (Signal): Signal to update progress.

    Returns:
        pd.DataFrame: Dataframe containing the heatmap data.
    """
    def progress_callback(progress):
        progress_heatmap.emit(progress)
    return heatmap_vis.calculate_plot(progress_callback=progress_callback)

@thread_worker
def calculate_brainsection(brainsection_vis: BrainsectionVisualization, progress_brainsection: Signal) -> List:
    """
    Calculate data for brain section visualization in a background thread.

    Parameters:
        brainsection_vis (BrainsectionVisualization): Brainsection visualization object.
        progress_brainsection (Signal): Signal to update progress.

    Returns:
        List: Calculated data for brain section plots.
    """
    def progress_callback(progress):
        progress_brainsection.emit(progress)
    return brainsection_vis.calculate_plot(progress_callback=progress_callback)

def initialize_header_widget() -> FunctionGui:
    """
    Initialize the header widget with input and save path options.

    Returns:
        FunctionGui: Header widget.
    """
    @magicgui(layout='vertical',
              input_path=dict(widget_type='FileEdit', 
                              label='input path: ',
                              value='',
                              mode='d',
                              tooltip='directory of folder containing folders with animals'),
              save_path=dict(widget_type='FileEdit', 
                             label='save path: ', 
                             mode='d',
                             value='',
                             tooltip='select a folder for saving plots'),
              animal_list=dict(widget_type='LineEdit', 
                               label='list of animals',
                               value='animal1,animal2', 
                               tooltip="enter the COMMA SEPERATED list of animals (no white spaces: animal1,animal2) "
                                       "\n to select all animal in input path enter ':' (without ' ')"),
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
    """
    Initialize the barplot widget with configuration options.

    Returns:
        FunctionGui: Barplot widget.
    """
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox',
                            label='save figure/data?',
                            value=False,
                            tooltip='tick to save figure (and data)'),
              save_name=dict(widget_type='LineEdit',
                             label='enter folder name to save figure/data in',
                             value='test',
                             tooltip='enter the name of the folder in which figure/data will be saved under same name'),
              plot_item=dict(widget_type='ComboBox',
                              label='item to plot',
                              choices=['cells', 'injection_site', 'projections', 'genes', 'hcr'],
                              value='cells',
                              tooltip="select item to plot"),
              hemisphere=dict(widget_type='ComboBox', 
                              label='hemisphere',
                              choices=['ipsi', 'contra', 'both'], 
                              value='both',
                              tooltip="select hemisphere to visualize (relative to injection site)"),
              groups=dict(widget_type='ComboBox',
                          label='channel/group/genotype/animals separately?',
                          choices=['', 'channel', 'group', 'genotype', 'animal_id', 'ipsi_contra'],
                          value='',
                          tooltip="if you want to plot channel/group/genotype/individual animals or ipsi/contralateral to injection site in different colors (no colormaps), "
                                  "select accordingly, otherwise leave empty"),
              tgt_list=dict(widget_type='LineEdit', 
                            label='list of brain areas (ABA)',
                            value='area1,area2', 
                            tooltip='enter the COMMA SEPERATED list of names of areas (acronyms)'
                                            ' to plot (no white spaces: area1,area2)'),
              tgt_colors=dict(widget_type='LineEdit', 
                              label='list of colors',
                              value='c:Blues', 
                              tooltip='enter the COMMA SEPERATED list of colors used for plotting '
                                      '(no white spaces: color1,color2); '
                                      'for using a colormap start with "c:NAMEOFCMAP. '
                                      'colormaps only work when no grouping parametrs is selected.'),
              plot_size=dict(widget_type='LineEdit', 
                             label='enter plot size',
                             value='16,12',
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
                          value='Number of cells',  #'Proportion of cells [%]'
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
              absolute_numbers=dict(widget_type='ComboBox',
                          label='plotting absolute numbers or percentage',
                          value='absolute',
                          choices=['absolute', 'percentage_dataset', 'percentage_selection'],
                          tooltip='select to plot absolute cell numbers, percentage relative to all cells in dataset or relative to cells in selected regions'),
              call_button=False,
              scrollable=True)

    def barplot_widget(
        viewer: Viewer,
        save_fig,
        save_name,
        plot_item,
        hemisphere,
        groups,
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
        absolute_numbers):
        pass
    return barplot_widget


def initialize_heatmap_widget() -> FunctionGui:
    """
    Initialize the heatmap widget with configuration options.

    Returns:
        FunctionGui: Heatmap widget.
    """
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox',
                            label='save figure/data?',
                            value=False,
                            tooltip='tick to save figure (and data)'),
              save_name=dict(widget_type='LineEdit',
                             label='enter folder name to save figure/data in',
                             value='test',
                             tooltip='enter the name of the folder in which figure/data will be saved under same name'),
              plot_item=dict(widget_type='ComboBox',
                             label='item to plot',
                             choices=['cells', 'injection_site', 'projections', 'genes'],
                             value='cells',
                             tooltip="select item to plot"),
              group_diff=dict(widget_type='ComboBox',
                          label='difference of channel/group/genotype/animals?',
                          choices=['', 'channel', 'group', 'genotype', 'animal_id', 'ipsi_contra',
                                   'channel (index)', 'group (index)', 'genotype (index)', 'animal_id (index)',
                                   'ipsi_contra (index)'],
                          value='',
                          tooltip="if you want to plot a difference heatmap between channel/group/genotype/individual animals or ipsi/contralateral to injection site, "
                                  "select accordingly, otherwise leave empty.\nif there are >2 item per channel, enter below which one to select."
                                  "\n If choosing the option with (index), a difference index = (group1-group2)/(group1+group2) is calculated, resulting in an index from +1 to -1."),
              group_diff_items=dict(widget_type='LineEdit',
                             label='difference heatmap for items:',
                             value='item1-item2',
                             tooltip='enter the two items you want to subtract from each other, item1-item2, e.g. experiment-contral'),
              hemisphere=dict(widget_type='ComboBox', 
                              label='hemisphere',
                              choices=['ipsi', 'contra', 'both'], 
                              value='both',
                              tooltip="select hemisphere to visualize (relative to injection site)"),
              tgt_list=dict(widget_type='LineEdit', 
                            label='list of brain areas (ABA)',
                            value='area1,area2', 
                            tooltip='enter the COMMA SEPERATED list of names of areas (acronyms)'
                                            ' to plot (no white spaces: area1,area2)'),
              intervals=dict(widget_type='LineEdit', 
                             label='intervals',
                             value='-0.5,0.0,0.5,1.0,1.5', 
                             tooltip='enter a COMMA SEPERATED list of mm coordinates relative '
                                                            'to bregma defining the intervals to plot (increasing in value)'),
              descendants=dict(widget_type='CheckBox',
                                  label='include descendants?',
                                  value=True,
                                  tooltip='option to include descendants of brain areas defined above, e.g. PL1, PL2/3 etc. '
                                'for PL'),
              cmap=dict(widget_type='LineEdit', 
                        label='colormap',
                        value='c:Blues', 
                        tooltip='enter colormap to use for heatmap, start with a c: ; '
                                                        'e.g. "c:NAMEOFCMAP"'),
              cmap_min_max=dict(widget_type='LineEdit', 
                                label='colormap range',
                                value='0,100',
                                tooltip="enter COMMA SEPERATED list of vmin, vmax values for the colormap, type 'auto'"
                                        "if values should be estimated based on data"),
              cbar_label=dict(widget_type='LineEdit', 
                              label='colormap label',
                              value='Number of cells',
                              tooltip='enter a label for the colorbar'),
              plot_size=dict(widget_type='LineEdit', 
                             label='enter plot size',
                             value='16,12',
                             tooltip='enter the COMMA SEPERATED size of the plot'),
              # xlabel=dict(widget_type='LineEdit',
              #             label='enter the xlabel',
              #             value='',
              #             tooltip='enter the xlabel of the plot'),
              # xlabel_size=dict(widget_type='SpinBox',
              #                  label='size of xlabel',
              #                  value=14,
              #                  min=1,
              #                  tooltip='select the size of the xlabel'),
              xticklabel_size=dict(widget_type='SpinBox',
                                   label='size of xticklabel',
                                   value=10,
                                   min=1,
                                   tooltip='select the size of the xticklabel'),
        #rotate_xticks=dict(widget_type='SpinBox', label='rotation of xticklabels',
        #                    value='45', tooltip='enter rotation of xticklabels, set to 0 for no rotation'),
              ylabel=dict(widget_type='LineEdit', 
                          label='enter the ylabel',
                          value='Distance relative to bregma', 
                          tooltip='enter the ylabel of the plot'),
              ylabel_size=dict(widget_type='SpinBox', 
                               label='size of ylabel', 
                               value=14,
                               min=1,
                               tooltip='select the size of the ylabel'),
              yticklabel_size=dict(widget_type='SpinBox',
                                   label='size of yticklabel',
                                   value=10,
                                   min=1,
                                   tooltip='select the size of the yticklabel'),
              # title=dict(widget_type='LineEdit',
              #            label='enter the title',
              #            value='',
              #            tooltip='enter the title of the plot'),
              # title_size=dict(widget_type='SpinBox',
              #                 label='size of title',
              #                 value=18,
              #                 min=1,
              #                 tooltip='select the size of the title'),
              subtitle_size=dict(widget_type='SpinBox',
                              label='size of subtitle (brain areas)',
                              value=12,
                              min=1,
                              tooltip='select the size of the subtitle (brain areas)'),
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
              absolute_numbers=dict(widget_type='ComboBox',
                          label='plotting absolute numbers or percentage',
                          value='absolute',
                          choices=['absolute', 'percentage_dataset', 'percentage_selection'],
                          tooltip='select to plot absolute cell numbers, percentage relative to all cells in dataset or relative to cells in selected regions'),
              call_button=False,
              scrollable=True)

    def heatmap_widget(
        viewer: Viewer,
        save_fig,
        save_name,
        plot_item,
        group_diff,
        group_diff_items,
        hemisphere,
        tgt_list,
        intervals,
        descendants,
        cmap,
        cmap_min_max,
        cbar_label,
        plot_size,
        # xlabel,
        # xlabel_size,
        xticklabel_size,
        ylabel,
        ylabel_size,
        yticklabel_size,
        # title,
        # title_size,
        subtitle_size,
        style,
        color,
        absolute_numbers):
        pass
    return heatmap_widget



def initialize_brainsection_widget() -> FunctionGui:
    """
    Initialize the brain section widget with configuration options.

    Returns:
        FunctionGui: Brain section widget.
    """
    @magicgui(layout='vertical',
              save_fig=dict(widget_type='CheckBox',
                            label='save figure/data?',
                            value=False,
                            tooltip='tick to save figure (and data)'),
              save_name=dict(widget_type='LineEdit',
                             label='enter folder name to save figure/data in',
                             value='test',
                             tooltip='enter the name of the folder in which figure/data will be saved under same name'),
              plot_item=dict(widget_type='Select', 
                             label='item to plot',
                             choices=['cells', 'cells_density', 'injection_site', 'projections', 'optic_fiber',
                                      'neuropixels_probe', 'genes', 'hcr', 'swc'],
                             tooltip='select items to plot cells/injection site/projection density, '
                                     'hold ctrl/shift to select multiple.'),
              section_orient=dict(widget_type='ComboBox', 
                                  label='section orientation',
                                  choices=['coronal', 'sagittal', 'horizontal'], 
                                  value='coronal',
                                  tooltip="select the orientation for plotting"),
              hemisphere=dict(widget_type='ComboBox',
                              label='data from hemisphere',
                              choices=['ipsi', 'contra', 'both'],
                              value='both',
                              tooltip='visualize data from both hemispheres, or only one hemisphere '
                                      '(relative to injection site specified in params.json file)'),
              unilateral=dict(widget_type='ComboBox',
                              label='plotting hemisphere',
                              choices=['both', 'left', 'right'],
                              value='both',
                              tooltip="choose to either plot both hemisphere, or left (ML<0 mm)/right (ML > 0 mm) "
                                      "hemisphere (only for coronal/horizontal orientations)"),
              brain_areas=dict(widget_type='LineEdit', 
                               label='list of brain areas',
                               tooltip='enter the COMMA SEPERATED list of names of brain areas (acronym)'
                                ' to plot (no white spaces: area1,area2).\n'
                                       'If you want to select all brain areas, typ in ALL.'),
              brain_areas_color=dict(widget_type='LineEdit', 
                                     label='brain area colors',
                                     tooltip='enter the COMMA SEPERATED list of colors for brain areas '
                                    '(no white spaces: red,blue,yellow)\n'
                                             'If you want to color all selected areas in the same color, mark the'
                                             'color by an asteriks (e.g. yellow*).\n'
                                             'If you want to color the brain sections according to the '
                                             'reference atlas type ATLAS.'),
              color_brain_density=dict(widget_type='CheckBox',
                            label='color brain areas according to cell/projection density?',
                            value=False,
                            tooltip='Tick option to color brain areas according to density of cells/projections'
                                    '(relative to complete dataset). If both projections and cells are plotted, the '
                                    'first item ticked is choosen. It is currently not possible to use this option when'
                                    ' using the option for seperate plotting by channel/group/genotype/animal. '
                                    '\nThe colormap is chosen based on color used for plotting cells/projections.'),
              # plot_size=dict(widget_type='LineEdit',
              #                label='enter plot size',
              #                value='16,12',
              #                tooltip='enter the COMMA SEPERATED size of the plot'),
              dot_size=dict(widget_type='LineEdit',
                             label='enter dot size',
                             value='10',
                             tooltip='enter the size of the dots (only for visualizing cells'),
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
                            label='color dots according to atlas?',
                            value=False,
                            tooltip='tick to color cells/spots (genes) according to atlas (if ticked this override manual colors '
                                    'entered below (e.g. colors of different groups).'),
              color_cells=dict(widget_type='LineEdit', 
                               label='colors (cell plot)',
                               value='Blue',
                               tooltip='enter COMMA SEPERATED list of colors (or c:map), should have the same length as '
                                'the groups/genotypes you want to plot'),
              show_cbar=dict(widget_type='CheckBox',
                                     label='show colorbar?',
                                     value=False,
                                     tooltip='tick to show colorbar. global parameter for all plots with colorbars.'),
              cmap_cells=dict(widget_type='LineEdit',
                                   label='colormap (cell density)',
                                   value='c:Greens',
                                   tooltip='enter a colormap for visualizing cell densities (e.g. Reds, Blues etc.)'),
              bin_size_cells=dict(widget_type='SpinBox',
                             label='bin_width (cell density)',
                             value=5,
                             min=1,
                             max=800,
                             tooltip='bin width for visualization of cell density'),
              vmin_cells=dict(widget_type='LineEdit',
                        label='vmin (cell density)',
                        value='0',
                        tooltip='min value for colorbar for visualizing cell densities '
                                '(depends on actual density and bin_width)'),
              vmax_cells=dict(widget_type='LineEdit',
                        label='vmax (cell density)',
                        value='20',
                        tooltip='max value for colorbar for visualizing cell densities '
                                '(depends on actual density and bin_width)'),
              group_diff_cells=dict(widget_type='ComboBox',
                              label='difference of channel/group/genotype/animals (cell density)?',
                              choices=['', 'channel', 'group', 'genotype', 'animal_id', 'ipsi_contra',
                                       'channel (index)', 'group (index)', 'genotype (index)', 'animal_id (index)',
                                       'ipsi_contra (index)'],
                              value='',
                              tooltip="if you want to plot a difference heatmap between channel/group/genotype/individual animals or ipsi/contralateral to injection site, "
                                      "select accordingly, otherwise leave empty.\nif there are >2 item per channel, enter below which one to select."
                                      "\n If choosing the option with (index), a difference index = (group1-group2)/(group1+group2) is calculated, resulting in an index from +1 to -1."),
              group_diff_items_cells=dict(widget_type='LineEdit',
                                    label='difference for items (cell density):',
                                    value='item1-item2',
                                    tooltip='enter the two items you want to subtract from each other, item1-item2, e.g. experiment-contral'),
              cmap_projection=dict(widget_type='LineEdit',
                                   label='colormap (projection density)',
                                   value='c:Blues',
                                   tooltip='enter a colormap for visualizing projections (e.g. Reds, Blues etc.)'),
              bin_size_proj=dict(widget_type='SpinBox',
                             label='bin_width (projection density)', 
                             value=5, 
                             min=1, 
                             max=800,
                             tooltip='bin width for visualization of axonal density'),
              vmin_proj=dict(widget_type='LineEdit',
                        label='vmin (projection density)',
                        value='0',
                        tooltip='min value for colorbar for visualizing projection densities '
                                '(depends on actual density and bin_width)'),
              vmax_proj=dict(widget_type='LineEdit',
                        label='vmax (projection density)', 
                        value='2000',
                        tooltip='max value for colorbar for visualizing projection densities '
                        '(depends on actual density and bin_width)'),
              group_diff_proj=dict(widget_type='ComboBox',
                                    label='difference of channel/group/genotype/animals (projection density)?',
                                    choices=['', 'channel', 'group', 'genotype', 'animal_id', 'ipsi_contra',
                                             'channel (index)', 'group (index)', 'genotype (index)',
                                             'animal_id (index)', 'ipsi_contra (index)'],
                                    value='',
                                    tooltip="if you want to plot a difference heatmap between channel/group/genotype/individual animals or ipsi/contralateral to injection site, "
                                            "select accordingly, otherwise leave empty.\nif there are >2 item per channel, enter below which one to select. "
                                            "\n If choosing the option with (index), a difference index = (group1-group2)/(group1+group2) is calculated, resulting in an index from +1 to -1."),
              group_diff_items_proj=dict(widget_type='LineEdit',
                                          label='difference for items (projection density):',
                                          value='item1-item2',
                                          tooltip='enter the two items you want to subtract from each other, item1-item2, e.g. experiment-contral'),
              color_inj=dict(widget_type='LineEdit', 
                             label='colors (injection site)',
                             value='Blue,Yellow', 
                             tooltip='enter a COMMA SEPERATED list for colors to use for the injection site'),
              color_optic=dict(widget_type='LineEdit', 
                               label='colors (optic fiber)',
                               value='Green,Pink', 
                               tooltip='enter a COMMA SEPERATED list for colors to use for the optic fiber(s)'),
              color_npx=dict(widget_type='LineEdit', 
                             label='colors (neuropixels)',
                             value='Red,Brown', 
                             tooltip='enter a COMMA SEPERATED list for colors to use for the neuropixels probes(s)'),
              plot_gene=dict(widget_type='ComboBox',
                             label='gene clusters or expression levels?',
                             choices=['clusters', 'expression'],
                             value='clusters',
                             tooltip="Choose to either visualize location of gene clusters by spots/brain area or to"
                                     "visualize the expression levels of one target gene. The second option requires "
                                     "the presence of a .csv file holding gene expression data, rows are gene "
                                     "expression, columns genes plus one column named 'spot_id' containing the spot ID"),
              color_genes=dict(widget_type='LineEdit',
                               label='colors (genes)',
                               value='Purple,Blue',
                               tooltip="enter a COMMA SEPERATED list for colors to use for the gene clusters. "
                                       "NOTE: if you have >148 colors to set, use hex keys for setting colors."
                                       "Use colormap for gene expression data, i.e. add 'c:' prior to colormap "
                                       "(e.g. c:Reds)"),
              color_brain_genes=dict(widget_type='ComboBox',
                            label='color brain areas according to gene cluster/expression?',
                            choices=['no_color', 'brain_areas', 'voronoi'],
                            value='no_color',
                            tooltip='Choose to color brain areas on section according to gene cluster (majority of '
                                    'cluster in brain area defines color) or gene expression (filepath needs to given '
                                    'above) or to perform Voronoi tessellation (area around spot receives color '
                                    'according to gene cluster or expression level of gene).'),
              color_hcr=dict(widget_type='LineEdit',
                               label='colors (hcr)',
                               value='Blue,Orange',
                               tooltip="enter a COMMA SEPERATED list for colors to use for the individual hcr genes. "
                                       "NOTE: if you have >148 colors to set, use hex keys for setting colors."),
              color_swc=dict(widget_type='LineEdit',
                               label='colors (swc)',
                               value='Black,Gray',
                               tooltip="enter a COMMA SEPERATED list for colors to use for the .swc neuron morphologies. "
                                       "NOTE: if you have >148 colors to set, use hex keys for setting colors.\n"
                                       "If you want to color all neurons differently, type 'random' here. Otherwise all "
                                       "are in the same color. For using the same color to all neurons add '*' to "
                                       "color name, e.g. 'black*'."
                                       "Consult the Github Wiki page for details on data formatting."),
              group_swc=dict(widget_type='CheckBox',
                            label='Use grouping variable for swc data?',
                            value=True,
                            tooltip='Tick to select grouping variable for swc data. '
                                    'Consult the Github Wiki page for details on data formatting.'),
              call_button=False,
              scrollable=True)

    def brain_section_widget(
        viewer: Viewer,
        save_fig,
        save_name,
        plot_item,
        hemisphere,
        unilateral,
        section_orient,
        brain_areas,
        brain_areas_color,
        color_brain_density,
        # plot_size,
        dot_size,
        section_list,
        section_range,
        groups,
        color_cells_atlas,
        color_cells,
        show_cbar,
        cmap_cells,
        bin_size_cells,
        vmin_cells,
        vmax_cells,
        group_diff_cells,
        group_diff_items_cells,
        cmap_projection,
        bin_size_proj,
        vmin_proj,
        vmax_proj,
        group_diff_proj,
        group_diff_items_proj,
        color_inj,
        color_optic,
        color_npx,
        plot_gene,
        color_genes,
        color_brain_genes,
        color_hcr,
        color_swc,
        group_swc):
        pass
    return brain_section_widget


class VisualizationWidget(QWidget):
    """
    Main widget for handling visualization tasks, including barplot, heatmap, and brain section plots.
    """
    progress_heatmap = Signal(int)
    """
    """
    progress_brainsection = Signal(int)
    """
    """
    
    def __init__(self, napari_viewer: Viewer) -> None:
        """
        Initialize the VisualizationWidget.

        Parameters:
            napari_viewer (Viewer): Napari viewer instance.
        """
        super().__init__()  # (parent)
        self.viewer = napari_viewer
        self.setLayout(QVBoxLayout())
        self.header = initialize_header_widget()
        self.header.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)

        self._collapse_bar = QCollapsible('Bar plot: expand for more', self)
        self.barplot = initialize_barplot_widget()
        self.barplot.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_bar.addWidget(self.barplot.root_native_widget)
        self.btn_bar = QPushButton("Create bar plot")
        self.btn_bar.clicked.connect(self._do_bar_plot)
        self._collapse_bar.addWidget(self.btn_bar)

        self._collapse_heat = QCollapsible('Heatmap: expand for more', self)
        self.heatmap = initialize_heatmap_widget()
        self.heatmap.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_heat.addWidget(self.heatmap.root_native_widget)
        self.btn_heat = QPushButton("Create heatmap")
        self.btn_heat.clicked.connect(self._do_heatmap)
        self._collapse_heat.addWidget(self.btn_heat)
        self.progress_bar_heatmap = ProgressBar(self)
        self._collapse_heat.addWidget(self.progress_bar_heatmap)
        self.progress_heatmap.connect(self.progress_bar_heatmap.set_value)

        self._collapse_section = QCollapsible('Brain section plot: expand for more', self)
        self.brainsection = initialize_brainsection_widget()
        self.brainsection.native.layout().setSizeConstraint(QVBoxLayout.SetFixedSize)
        self._collapse_section.addWidget(self.brainsection.root_native_widget)
        self.btn_brainsection = QPushButton("Create brainsection plot")
        self.btn_brainsection.clicked.connect(self._do_brainsection)
        self._collapse_section.addWidget(self.btn_brainsection)
        self.progress_bar_section = ProgressBar(self)
        self._collapse_section.addWidget(self.progress_bar_section)
        self.progress_brainsection.connect(self.progress_bar_section.set_value)

        self.layout().addWidget(self.header.native)
        self.layout().addWidget(self._collapse_bar)
        self.layout().addWidget(self._collapse_heat)
        self.layout().addWidget(self._collapse_section)

        self.barplot_vis = None
        self.heatmap_vis = None
        self.brainsection_vis = None
        self.atlas = None
        self.gene_expression_df = pd.DataFrame()
        self.gene_list, self.gene, self.gene_expression_fn, self.round_expression = None, None, None, None

    def _do_bar_plot(self) -> None:
        """
        Trigger the calculation and display of a bar plot.
        """

        self.barplot_vis = self._create_barplot_visualization()
        if not self.barplot_vis:
            return
        worker_barplot = calculate_barplot(self.barplot_vis)
        worker_barplot.started.connect(
            lambda: self.btn_bar.setText("Analyzing data..."))
        # worker_barplot.started.connect(lambda: self.barplot.setEnabled(False))  # Disable UI during processing
        # worker_barplot.finished.connect(lambda: self.barplot.setEnabled(True))  # Re-enable UI when done
        worker_barplot.returned.connect(self._show_barplot)
        worker_barplot.returned.connect(
            lambda: self.btn_bar.setText("Create bar plot"))
        worker_barplot.start()

    def _show_barplot(self, df_to_plot: pd.DataFrame) -> None:
        """
        Display the generated barplot in the viewer.

        Parameters:
            df_to_plot (pd.DataFrame): Dataframe containing the barplot data.
        """
        if not self.barplot_vis:
            show_info("Error: BarplotVisualization object does not exist.")
            return
        mpl_widget = self.barplot_vis.do_bar_plot(df_to_plot)

        # Display the mpl_widget
        if mpl_widget:
            self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

        # mpl_widget = do_bar_plot(df, atlas, plotting_params, animal_list, tgt_list, self.barplot, save_path)
        # self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)

    def _create_barplot_visualization(self) -> Optional[BarplotVisualization]:
        """
        Create a BarplotVisualization instance based on the input parameters.

        Returns:
            Optional[BarplotVisualization]: Barplot visualization object, or None if creation fails.
        """
        input_path = self.header.input_path.value
        if not check_input_path(input_path):
            return
        if str(self.header.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.header.save_path.value
            if not check_input_path(save_path):
                return
        if self.header.animal_list.value == ':':
            animal_list = [f.parts[-1] for f in input_path.iterdir() if f.is_dir()]
        else:
            animal_list = split_to_list(self.header.animal_list.value)
        channels = self.header.channels.value
        plot_item = self.barplot.plot_item.value
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        if not self.atlas:
            show_info("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
            show_info("...done!")
        data_loader = DataLoader(input_path, self.atlas, animal_list, channels, data_type=plot_item,
                                 hemisphere=self.barplot.hemisphere.value)
        df_all = data_loader.load_data()

        if plot_item == 'genes':
            # Open the pop-up dialog to get gene expression info
            if self.gene_expression_df.empty:
                gene_dialog = GeneInfoDialog()
                if gene_dialog.exec_():
                    self.gene_expression_fn, self.gene_list, _ = gene_dialog.get_gene_info()
                    if not self.gene_expression_fn or not self.gene_list:
                        show_info("Gene expression file or gene list was not provided. Aborting the bar plot for genes.")
                        return  # Stop if no input is provided
                columns_to_load = ['spot_id']
                columns_to_load += self.gene_list
                try:
                    show_info("loading gene expression data...")
                    self.gene_expression_df = pd.read_csv(self.gene_expression_fn, usecols=columns_to_load)

                except ValueError:
                    show_info(f"{','.join(self.gene_list)} does not match!")
                    return


            df_all = pd.merge(df_all, self.gene_expression_df, on='spot_id', how='left').fillna(0)
        else:
            self.gene_expression_df = pd.DataFrame()
            self.gene_expression_fn, self.gene_list = None, None
        tgt_list = split_to_list(self.barplot.tgt_list.value)
        use_na = 'NA' in tgt_list
        df = data_loader.get_tgt_data_only(df_all, tgt_list, use_na=use_na)

        return BarplotVisualization(df_all, df, self.atlas, animal_list, tgt_list, save_path, self.barplot, self.gene_list)



    def _do_heatmap(self) -> None:
        """
        Trigger the calculation and display of a heatmap.
        """
        self.heatmap_vis = self._create_heatmap_visualization()
        if not self.heatmap_vis:
            return
        worker_heatmap = calculate_heatmap(self.heatmap_vis, self.progress_heatmap)
        worker_heatmap.started.connect(
            lambda: self.btn_heat.setText("Analyzing data..."))
        worker_heatmap.returned.connect(self._show_heatmap)
        worker_heatmap.returned.connect(
            lambda: self.btn_heat.setText("Create heatmap"))
        worker_heatmap.returned.connect(lambda: self.progress_heatmap.emit(0))
        worker_heatmap.start()

    def _show_heatmap(self, df_to_plot: pd.DataFrame) -> None:
        """
        Display the generated heatmap in the viewer.

        Parameters:
            df_to_plot (pd.DataFrame): Dataframe containing the heatmap data.
        """
        if not self.heatmap_vis:
            show_info("Error: HeatmapVisualization object does not exist.")
            return
        mpl_widget = self.heatmap_vis.do_plot(df_to_plot)

        # Display the mpl_widget
        if mpl_widget:
            self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)


    def _create_heatmap_visualization(self) -> Optional[HeatmapVisualization]:
        """
        Create a HeatmapVisualization instance based on the input parameters.

        Returns:
            Optional[HeatmapVisualization]: Heatmap visualization object, or None if creation fails.
        """
        input_path = self.header.input_path.value
        if not check_input_path(input_path):
            return
        if str(self.header.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.header.save_path.value
            if not check_input_path(save_path):
                return
        if self.header.animal_list.value == ':':
            animal_list = [f.parts[-1] for f in input_path.iterdir() if f.is_dir()]
        else:
            animal_list = split_to_list(self.header.animal_list.value)
        channels = self.header.channels.value
        plot_item = self.heatmap.plot_item.value
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        if not self.atlas:
            show_info("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
            show_info("...done!")
        data_loader = DataLoader(input_path, self.atlas, animal_list, channels, data_type=plot_item,
                                 hemisphere=self.heatmap.hemisphere.value)
        df_all = data_loader.load_data()

        if plot_item == 'genes':
            if self.gene_expression_df.empty:
                # Open the pop-up dialog to get gene expression info
                gene_dialog = GeneInfoDialog(only_gene=True)
                if gene_dialog.exec_():
                    self.gene_expression_fn, self.gene_list, _ = gene_dialog.get_gene_info()
                    if not self.gene_expression_fn or not self.gene_list:
                        show_info("Gene expression file or gene list was not provided. Aborting the heatmap plot for genes.")
                        return  # Stop if no input is provided
                columns_to_load = ['spot_id']
                self.gene = self.gene_list[0]
                columns_to_load.append(self.gene)
                try:
                    show_info("loading gene expression data...")
                    self.gene_expression_df = pd.read_csv(self.gene_expression_fn, usecols=columns_to_load)

                except ValueError:
                    show_info(f'{self.gene} does not match!')
                    return
            df_all = pd.merge(df_all, self.gene_expression_df, on='spot_id', how='left').fillna(0)
        else:
            self.gene_expression_df = pd.DataFrame()
            self.gene_expression_fn, self.gene_list, self.gene = None, None, None
        tgt_list = split_to_list(self.heatmap.tgt_list.value)
        try:
            if self.heatmap.descendants.value:
                tgt_list = get_descendants(tgt_list, self.atlas)
            df = data_loader.get_tgt_data_only(df_all, tgt_list)
        except KeyError:
            show_info(f"Brain areas {tgt_list} not found in the data. Aborting the bar plot.")
            return

        return HeatmapVisualization(df_all, df, self.atlas, animal_list, tgt_list, self.gene, self.heatmap,
                                    save_path)


    def _do_brainsection(self) -> None:
        """
        Trigger the calculation and display of a brain section plot.
        """
        self.brainsection_vis = self._create_brainsection_visualization()
        if not self.brainsection_vis:
            return
        worker_brainsection = calculate_brainsection(self.brainsection_vis, self.progress_brainsection)
        # worker_brainsection.started.connect(
        #     lambda: self.btn_brainsection.setText("Analyzing data..."))
        worker_brainsection.started.connect(
            lambda: self.btn_brainsection.setText("Creating plots..."))
        worker_brainsection.returned.connect(self._show_brainsection)
        worker_brainsection.returned.connect(
            lambda: self.btn_brainsection.setText("Create brainsection plot"))
        worker_brainsection.returned.connect(lambda: self.progress_brainsection.emit(0))
        worker_brainsection.start()

    def _show_brainsection(self, results: List) -> None:
        """
        Display the generated brain section plot in the viewer.

        Parameters:
            results (List): Data and annotations for the brain section plots.
        """
        if not self.brainsection_vis:
            show_info("Error: BrainsectionVisualization object does not exist.")
            return
        # self.btn_brainsection.setText("Creating plots...")
        mpl_widget = self.brainsection_vis.do_plot(results)

        # Display the mpl_widget
        if mpl_widget:
            self.viewer.window.add_dock_widget(mpl_widget, area='left').setFloating(True)


    def _create_brainsection_visualization(self) -> Optional[BrainsectionVisualization]:
        """
        Create a BrainsectionVisualization instance based on the input parameters.

        Returns:
            Optional[BrainsectionVisualization]: Brainsection visualization object, or None if creation fails.
        """
        input_path = self.header.input_path.value
        if not check_input_path(input_path):
            return
        if str(self.header.save_path.value) == '.':
            save_path = input_path
        else:
            save_path = self.header.save_path.value
            if not check_input_path(save_path):
                return
        if self.header.animal_list.value == ':':
            animal_list = [f.parts[-1] for f in input_path.iterdir() if f.is_dir()]
        else:
            animal_list = split_to_list(self.header.animal_list.value)
        channels = self.header.channels.value
        plot_item = self.brainsection.plot_item.value
        params_dict = load_params(input_path.joinpath(animal_list[0]))
        if not self.atlas:
            show_info("loading reference atlas...")
            self.atlas = BrainGlobeAtlas(params_dict['atlas_info']['atlas'])
            show_info("...done!")

        data_dict = {
            item: DataLoader(
                input_path,
                self.atlas,
                animal_list,
                channels,
                data_type='cells' if item == 'cells_density' else item,
                hemisphere=self.brainsection.hemisphere.value
            ).load_data()
            for item in plot_item
        }
        if self.brainsection.plot_gene.value == 'expression':
            if self.gene_expression_df.empty:
                self.btn_brainsection.setText("Loading gene expression data...")
                # Open the pop-up dialog to get gene expression info
                gene_dialog = GeneInfoDialog(only_gene=True, round_expression=True)
                if gene_dialog.exec_():
                    self.gene_expression_fn, self.gene_list, self.round_expression = gene_dialog.get_gene_info()
                    if not self.gene_expression_fn or not self.gene_list:
                        show_info("Gene expression file or gene list was not provided. Aborting the brainsection plot for genes.")
                        return  # Stop if no input is provided
                columns_to_load = ['spot_id']
                self.gene = self.gene_list[0]
                columns_to_load.append(self.gene)
                try:
                    show_info("loading gene expression data...")
                    self.gene_expression_df = pd.read_csv(self.gene_expression_fn, usecols=columns_to_load)
                    self.gene_expression_df.rename(columns={self.gene: 'gene_expression'}, inplace=True)
                except ValueError:
                    show_info(f'{self.gene} does not match!')
                    return
            data_dict['genes'] = pd.merge(data_dict['genes'], self.gene_expression_df, on='spot_id', how='left').fillna(0)
            data_dict['genes']['gene_expression_norm'] = (
                    data_dict['genes']['gene_expression'] / data_dict['genes']['gene_expression'].max()
            ).clip(lower=0)
            if self.round_expression > 0:
                data_dict['genes']['gene_expression_norm'] = data_dict['genes']['gene_expression_norm'].round(
                    self.round_expression)
        else:
            self.gene_expression_df = pd.DataFrame()
            self.gene_expression_fn, self.gene_list, self.gene, self.round_expression= None, None, None, None

        return BrainsectionVisualization(input_path, self.atlas, data_dict, animal_list, self.brainsection, save_path,
                                         self.gene)




