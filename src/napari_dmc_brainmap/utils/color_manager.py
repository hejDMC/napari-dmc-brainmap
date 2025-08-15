import json
import pathlib
import random
from pathlib import Path
from typing import List, Dict, Union

from natsort import natsorted
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from napari.utils.notifications import show_info

class ColorManager:
    def __init__(self):
        """
        Initialize the ColorManager with default and XKCD colors.
        """
        self.default_colors = list(mcolors.CSS4_COLORS.keys())
        self.xkcd_colors = self._load_xkcd_colors()

    def _load_xkcd_colors(self) -> List[str]:
        """
        Load XKCD colors from a JSON file or provide a fallback list.

        Returns:
            List[str]: A list of hex color codes.
        """
        try:
            with open(Path(__file__).resolve().parent.joinpath('xkcd.json')) as fn:
                xcol_data = json.load(fn)
                return [f"{i['hex']}" for i in xcol_data['colors']]
        except FileNotFoundError:
            show_info("Warning: xkcd.json not found, using fallback color list.")
            return self.default_colors

    def check_color_name(self, clr_name: str) -> str:
        """
        Validate the color name and replace with a random default color if invalid.

        Parameters:
            clr_name (str): The name of the color to check.

        Returns:
            str: A valid color name.
        """
        if not mcolors.is_color_like(clr_name):
            show_info(f'{clr_name} does not exist, selecting random color instead.')
            clr_name = random.choice(self.default_colors)
        return clr_name

    def create_color_palette(self,
                             animal_dict: Dict[str, str],
                             plotting_params: Dict[str, Union[str, List[str]]],
                             clr_id: str,
                             df: pd.DataFrame = pd.DataFrame(),
                             hue_id: str = 'channel') -> Dict[Union[str, int], Union[tuple, List[float], str]]:
        """
        Create a color palette based on group IDs or plotting parameters.

        Parameters:
            animal_dict (Dict[str, str]): Dictionary of animal information.
            plotting_params (Dict[str, Union[str, List[str]]]): Plotting parameters.
            clr_id (str): The color ID to use for the palette.
            df (pd.DataFrame, optional): Dataframe containing grouping information. Defaults to an empty DataFrame.
            hue_id (str, optional): Column in the DataFrame for grouping. Defaults to 'channel'.

        Returns:
            Dict[Union[str, int], Union[tuple, List[float]]]: A mapping of group IDs to RGBA colors.
        """
        cmap = {}
        if not df.empty:
            group_ids = list(df[hue_id].unique())
            if hue_id == 'cluster_id':
                group_ids = natsorted(group_ids)
        else:
            group_ids = list(animal_dict.keys())
        cmap_groups = plotting_params.get(clr_id, [])
        if not isinstance(cmap_groups, list):
            cmap_groups = []
        num_groups = len(group_ids)
        num_colors = len(cmap_groups) if isinstance(cmap_groups, list) else 0
        if num_groups > num_colors:
            diff = num_groups - num_colors
            show_info(f"Warning: {num_groups} groups but only {num_colors} colors provided. Adding random colors.")
            if clr_id == 'colors_projections':
                colormaps = [cc for cc in cm.datad]
                for _ in range(diff):
                    cmap_groups.append(random.choice(colormaps))
            else:
                cmap_groups += random.sample(self.default_colors, diff) if num_groups <= len(self.default_colors) \
                    else random.sample(self.xkcd_colors, diff)
        elif num_groups < num_colors:
            show_info(f"Warning: {num_groups} groups, but {len(cmap_groups)} colors provided. Truncating colors.")
            cmap_groups = cmap_groups[:num_groups]
        cmap_groups = [mcolors.to_rgba(self.check_color_name(c)) for c in cmap_groups]
        for g, c in zip(group_ids, cmap_groups):
            cmap[g] = c
        return cmap

    def create_color_dict(self,
                          input_path: Path,
                          animal_list: List[str],
                          data_dict: Dict[str, pd.DataFrame],
                          plotting_params: Dict[str, Union[str, List[str]]]) -> Dict[str, Dict[str, Union[bool, Dict]]]:
        """
        Generate a dictionary of color mappings for various plot items.

        Parameters:
            input_path (Path): The base directory containing input data.
            animal_list (List[str]): List of animal identifiers.
            data_dict (Dict[str, pd.DataFrame]): Dictionary of dataframes for plot items.
            plotting_params (Dict[str, Union[str, List[str]]]): Parameters for plotting.

        Returns:
            Dict[str, Dict[str, Union[bool, Dict]]]: The color mapping dictionary.
        """
        color_dict = {}
        for item in plotting_params['plot_item']:
            color_dict[item] = {}
            clr_id = f'color_{item}'
            if item in ['cells', 'injection_site']:
                if plotting_params['groups'] in ['genotype', 'group']:
                    animal_dict = self._load_group_dict(input_path, animal_list, group_id=plotting_params['groups'])
                    cmap = self.create_color_palette(animal_dict, plotting_params, clr_id)
                    single_color = False
                elif plotting_params['groups'] in ['channel', 'animal_id']:
                    cmap = self.create_color_palette({}, plotting_params, clr_id, df=data_dict[item],
                                                     hue_id=plotting_params['groups'])
                    single_color = False
                else:
                    try:
                        cmap = plotting_params.get(clr_id, [random.choice(self.default_colors)])[0]
                    except TypeError:
                        cmap = random.choice(self.default_colors)
                    cmap = self.check_color_name(cmap)
                    single_color = True
            elif item == 'genes':
                if plotting_params['plot_gene'] == 'clusters':
                    cmap = self.create_color_palette({}, plotting_params, clr_id, df=data_dict[item],
                                                     hue_id='cluster_id')
                    single_color = False
                else:
                    cmap = self.create_custom_colormap(plotting_params.get(clr_id, False))
                    single_color = True
            elif item in ['projections', 'cells_density']:
                cmap = self.create_custom_colormap(plotting_params.get(clr_id, False))
                single_color = True
            elif item == 'hcr':
                cmap = self.create_color_palette({}, plotting_params, clr_id, df=data_dict[item],
                                                 hue_id='hcr')
                single_color = False
            elif item == 'swc':
                if plotting_params["group_swc"]:
                    cmap = self.create_color_palette({}, plotting_params, clr_id, df=data_dict[item],
                                                     hue_id='group_id')
                    single_color = False
                else:
                    if plotting_params.get("color_swc", 'black') == 'random':
                        cmap = self.create_color_palette({}, plotting_params, [], df=data_dict[item],
                                                         hue_id='neuron_id')
                        single_color = False
                    elif '*' in plotting_params['color_swc'][0]:
                        try:
                            cmap = plotting_params.get(clr_id, [random.choice(self.default_colors)])[0].strip('*')
                        except TypeError:
                            cmap = random.choice(self.default_colors)
                        cmap = self.check_color_name(cmap)
                        single_color = True
                    else:
                        cmap = self.create_color_palette({}, plotting_params, clr_id, df=data_dict[item],
                                                         hue_id='neuron_id')
                        single_color = False
            else:
                num_probe = len(data_dict[item]['channel'].unique())
                try:
                    cmap = plotting_params.get(clr_id, [random.choice(self.default_colors)])[
                        0] if num_probe == 1 else self.create_color_palette({}, plotting_params, clr_id, df=data_dict[item])
                except TypeError:
                    cmap = random.choice(self.default_colors)
                cmap = self.check_color_name(cmap) if num_probe == 1 else cmap
                single_color = (num_probe == 1)
            color_dict[item]['cmap'] = cmap
            color_dict[item]['single_color'] = single_color
        return color_dict

    def create_custom_colormap(self, cmap: Union[str, List[str]]) -> mcolors.Colormap:
        """
        Create a custom colormap.

        Parameters:
            cmap (Union[str, List[str]]): The name or definition of the colormap.

        Returns:
            mcolors.Colormap: The resulting colormap.
        """
        try:
            cmap = plt.get_cmap(cmap)
        except ValueError:
            if isinstance(cmap, list):
                cmap = cmap[0]
            if not isinstance(cmap, str):
                show_info(f"Invalid input for colormap '{cmap}', selecting random colormap instead.")
                return random.choice(plt.colormaps())
            try:
                if '-' in cmap:
                    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap',
                                                                     [cmap.split('-')[0], cmap.split('-')[1]])
                elif ':' in cmap:
                    if len(cmap.split(':')) == 2:
                        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [cmap.split(':')[0], 'white',
                                                                                     cmap.split(':')[1]])
                    elif len(cmap.split(':')) == 3:
                        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [cmap.split(':')[0], cmap.split(':')[1],
                                                                                     cmap.split(':')[2]])
                    else:
                        show_info('Invalid colormap specified, selecting random colormap instead.')
                        cmap = random.choice(plt.colormaps())


                else:
                    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['white', cmap])
            except ValueError:
                show_info('Invalid colormap specified, selecting random colormap instead.')
                cmap = random.choice(plt.colormaps())
        return cmap

    def _load_group_dict(self, input_path: Path, animal_list: List[str], group_id: str = 'genotype') -> Dict[str, List[str]]:
        """
        Collect group information from the specified input path.

        Parameters:
            input_path (Path): The base path to data files.
            animal_list (List[str]): List of animal IDs.
            group_id (str): The group identifier to extract.

        Returns:
            Dict[str, List[str]]: A dictionary mapping group IDs to lists of animals.
        """
        group_dict = {}
        for animal_id in animal_list:
            data_dir = input_path.joinpath(animal_id)
            params_fn = data_dir.joinpath('params.json')
            if params_fn.exists():
                with open(params_fn) as fn:
                    params_dict = json.load(fn)
                try:
                    g_id = params_dict['general'][group_id]
                    if g_id in group_dict.keys():
                        group_dict[g_id].append(animal_id)
                    else:
                        group_dict[g_id] = [animal_id]
                except KeyError:
                    show_info(f"No group_id value (* {group_id} *) specified for {animal_id}")
                    show_info(f"    --> skipping {animal_id}")
                    pass
            else:
                show_info(f"No params.json file under {str(params_fn)}")
                show_info(f"    --> skipping {animal_id}")
        return group_dict
