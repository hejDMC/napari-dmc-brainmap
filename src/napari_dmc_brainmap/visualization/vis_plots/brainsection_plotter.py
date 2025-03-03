import random
from natsort import natsorted
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, get_xyz
from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import get_descendants, match_lists
from bg_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info

class BrainsectionPlotter:
    """
    Class for plotting brain sections and generating visualizations such as schematics, Voronoi diagrams,
    density maps, and heatmaps.
    """
    def __init__(self, atlas: BrainGlobeAtlas, plotting_params: Dict, data_dict: Dict, color_manager,
                 color_dict: Dict) -> None:
        """
        Initialize the BrainsectionPlotter.

        Parameters:
            atlas: BrainGlobeAtlas instance.
            plotting_params (Dict): Parameters for visualization.
            data_dict (Dict): Dictionary containing data to be visualized.
            color_manager: Color manager for handling custom color maps.
            color_dict (Dict): Dictionary containing predefined color schemes.
        """
        self.atlas = atlas
        self.plotting_params = plotting_params
        self.data_dict = data_dict
        self.color_manager = color_manager
        self.color_dict = color_dict



    def plot_brain_schematic(self, slice_idx: int, orient_idx: int) -> List:
        """
        Generate a schematic plot for a specific brain slice.

        Parameters:
            slice_idx (int): Index of the brain slice.
            orient_idx (int): Orientation index (0: coronal, 1: sagittal, 2: horizontal).

        Returns:
            List: Annotated section, unique IDs, and color dictionary.
        """

        annot_section = self._get_slice(slice_idx, orient_idx)
        annot_section = self._check_unilateral_plotting(annot_section, orient_idx)

        unique_ids = np.unique(annot_section)
        unique_ids = unique_ids[unique_ids != 0]

        color_dict = self._generate_color_dict(unique_ids)


        return [annot_section, unique_ids, color_dict]

    def plot_brain_schematic_voronoi(self, df: pd.DataFrame, slice_idx: int, orient_mapping: Dict) -> List:
        """
        Generate a Voronoi plot for a specific brain slice.

        Parameters:
            df (pd.DataFrame): Dataframe containing plot points.
            slice_idx (int): Index of the brain slice.
            orient_mapping (Dict): Mapping for orientation coordinates.

        Returns:
            List: Voronoi mask, unique IDs, and color dictionary.
        """

        orient_idx = orient_mapping['z_plot'][1]
        annot_section = self._get_slice(slice_idx, orient_idx)
        annot_section = self._check_unilateral_plotting(annot_section, orient_idx)
        voronoi_mask, color_dict = self._get_voronoi_mask(df, orient_mapping)
        if voronoi_mask is None:
            return self.plot_brain_schematic(slice_idx, orient_idx)
        voronoi_mask = self._check_unilateral_plotting(voronoi_mask, orient_idx)
        voronoi_mask[annot_section == 0] = -1
        unique_ids = np.unique(voronoi_mask)
        unique_ids = unique_ids[unique_ids != 0]
        #
        # color_dict = self._generate_color_dict(unique_ids)


        return [voronoi_mask, unique_ids, color_dict]

    def _get_slice(self, slice_idx: int, orient_idx: int) -> np.ndarray:
        """
        Retrieve a slice of the atlas based on orientation and index.

        Parameters:
            slice_idx (int): Index of the brain slice.
            orient_idx (int): Orientation index (0: coronal, 1: sagittal, 2: horizontal).

        Returns:
            np.ndarray: Extracted slice from the atlas.
        """
        if orient_idx == 0:
            return self.atlas.annotation[slice_idx, :, :].copy()
        elif orient_idx == 1:
            return self.atlas.annotation[:, slice_idx, :].copy()
        else:
            return self.atlas.annotation[:, :, slice_idx].copy()

    def _check_unilateral_plotting(self, annot_section: np.ndarray, orient_idx: int) -> np.ndarray:
        """
        Adjust plotting for unilateral visualizations.

        Parameters:
            annot_section (np.ndarray): Annotated section.
            orient_idx (int): Orientation index (0: coronal, 1: sagittal, 2: horizontal).

        Returns:
            np.ndarray: Adjusted annotated section.
        """
        unilateral = self.plotting_params.get('unilateral', False)
        if unilateral in ['left', 'right'] and orient_idx < 2:
            bregma = get_bregma(self.atlas.atlas_name)
            midline_idx = bregma[self.atlas.space.axes_description.index('rl')]
            if unilateral == 'left':
                annot_section = annot_section[:, midline_idx:]
            else:
                annot_section = annot_section[:, :midline_idx]
        return annot_section

    def _generate_color_dict(self, unique_ids: np.ndarray, default_color: str = "linen") -> Dict:
        """
        Generate a color dictionary for unique region IDs.

        Parameters:
            unique_ids (np.ndarray): Array of unique IDs.
            default_color (str, optional): Default color for regions. Defaults to "linen".

        Returns:
            Dict: Dictionary mapping region IDs to colors.
        """
        area_list, color_list, default_color = self._get_custom_colormap(unique_ids)
        color_dict = {uid: default_color for uid in unique_ids}

        if area_list and color_list:
            for a, c in zip(area_list, color_list):
                try:
                    color_dict.update({
                        self.atlas.structures[s]['id']: c
                        for s in get_descendants([a], self.atlas)
                        if self.atlas.structures[s]['id'] in unique_ids
                    })
                except KeyError:
                    continue
        return color_dict

    def _get_custom_colormap(self, unique_ids: np.ndarray) -> Tuple[List[str], List[str], str]:
        """
        Retrieve a custom colormap for specified regions.

        Parameters:
            unique_ids (np.ndarray): Array of unique IDs.

        Returns:
            Tuple[List[str], List[str], str]: Lists of areas and corresponding colors and default color.
        """

        area_list = ['fiber tracts', 'VS']
        color_list = ['lightgray', 'lightcyan']
        default_color = "linen"
        if (self.plotting_params.get('brain_areas') or self.plotting_params.get('brain_areas_color')) \
                and not self.plotting_params.get('color_brain_density'):
            brain_areas, brain_areas_color = self._brain_region_custom_color(unique_ids)
        elif self.plotting_params.get('color_brain_density'):
            default_color = "whitesmoke"
            area_density_key = next((key for key in ['cells', 'projections'] if key in self.data_dict), None)
            self.plotting_params['area_density'] = area_density_key
            show_info(f"color brain areas according to {self.plotting_params['area_density']}")
            brain_areas, brain_areas_color = self._calculate_density()
            if self.plotting_params.get('brain_areas'):
                tgt_brain_areas = get_descendants(self.plotting_params['brain_areas'], self.atlas)
                brain_areas, brain_areas_color = zip(*[(a, c) for a, c in zip(brain_areas, brain_areas_color) if a in tgt_brain_areas])


        elif self.plotting_params.get('color_brain_genes') == 'brain_areas':
            color_list = ['lightgray', 'white']
            default_color = "white"
            brain_areas, brain_areas_color = self._brain_region_gene_color()
        else:
            brain_areas, brain_areas_color = None, None

        if brain_areas and brain_areas_color:
            area_list.extend(brain_areas)
            color_list.extend(brain_areas_color)

        return area_list, color_list, default_color

    def _brain_region_custom_color(self, unique_ids: np.ndarray) -> Tuple[
        List[str], List[Union[str, Tuple[float, float, float]]]]:
        """
        Assign custom colors to brain regions based on user preferences or atlas data.

        Parameters:
            unique_ids (np.ndarray): Array of unique region IDs.

        Returns:
            Tuple[List[str], List[Union[str, Tuple[float, float, float]]]]: Matched lists of brain region names and their colors.
        """
        brain_areas = self.plotting_params.get('brain_areas')
        if not brain_areas:
            brain_areas = []
        elif 'ALL' in brain_areas:
            brain_areas = [self.atlas.structures[b]['acronym'] for b in unique_ids]
            try:
                del brain_areas[brain_areas.index('root')]
            except ValueError:
                pass
        brain_areas_color = self.plotting_params['brain_areas_color']
        if not brain_areas_color:
            brain_areas_color = []
        elif any('*' in c for c in brain_areas_color):
            brain_areas_color = [next((c for c in brain_areas_color if '*' in c)).replace('*','')]*len(brain_areas)

        if brain_areas_color and 'ATLAS' in brain_areas_color:
            try:
                brain_areas_color = [
                    tuple(c / 255 for c in self.atlas.structures[self.atlas.structures.acronym_to_id_map[b]]['rgb_triplet'])
                    for b in brain_areas
                ]
            except KeyError:
                brain_areas_color = []
                show_info(f"Could not find acronyms in {','.join(brain_areas_color)} - using random colors")

        elif not brain_areas_color:
            brain_areas_color = [random.choice(list(mcolors.CSS4_COLORS.keys())) for _ in brain_areas]
        brain_areas, brain_areas_color = match_lists(brain_areas, brain_areas_color)
        # brain_areas = [self.atlas.structures[b]['id'] for b in brain_areas]
        return brain_areas, brain_areas_color

    def _calculate_density(self) -> Tuple[List[str], List[str]]:
        """
        Calculate the density of data points across brain regions.

        Returns:
            Tuple[List[str], List[str]]: Sorted brain areas and corresponding colors.
        """
        df = self.data_dict[self.plotting_params['area_density']].assign(
            left_right=np.where(self.data_dict[self.plotting_params['area_density']]['ml_mm'] < 0, 'right', 'left'))
        animal_list = df['animal_id'].unique()
        df_density = self._calculate_density_pivot(df, animal_list)
        df_density = self._normalize_density(df_density)
        # if self.plotting_params.get('brain_areas'):
        #     df_density = df_density[df_density['acronym'].isin(get_descendants(self.plotting_params['brain_areas'],
        #                                                                        self.atlas))]
        #     df_density = df_density.loc[df_density.index.isin(self.plotting_params['brain_areas'])]
        # df_density = self._filter_acronyms(df_density, self.atlas)  # todo check if included, only for ABA
        clr = self.color_dict[self.plotting_params['area_density']]['cmap'] \
            if self.color_dict[self.plotting_params['area_density']]['single_color'] \
            else self.color_dict[self.plotting_params['area_density']]['cmap'][0]
        return self._brain_region_density_color(df_density, clr)

    def _calculate_density_pivot(self, df: pd.DataFrame, animal_list: List[str]) -> pd.DataFrame:
        """
        Create a pivot table summarizing density data for each brain region.

        Parameters:
            df (pd.DataFrame): Dataframe containing density data.
            animal_list (List[str]): List of animal IDs.

        Returns:
            pd.DataFrame: Normalized density data.
        """
        df_pivot = df.pivot_table(index='acronym', columns=['animal_id', 'left_right'], aggfunc='count').fillna(0)[
            'ap_coords']
        new_columns = pd.MultiIndex.from_product([animal_list, ['left', 'right']], names=['animal_id', 'left_right'])
        df_pivot = df_pivot.reindex(columns=new_columns, fill_value=0)
        df_density = pd.DataFrame(np.zeros((len(df_pivot.index), len(df.left_right.unique()))),
                                  index=df_pivot.index, columns=df.left_right.unique())
        for animal_id in animal_list:
            data = (df_pivot[animal_id] / len(df[df['animal_id'] == animal_id]))
            data = data.fillna(0)
            df_density += data
        df_density /= len(animal_list)
        df_density['acronym'] = df_density.index.to_list()
        df_density = pd.melt(df_density, id_vars=['acronym'], var_name='left_right', value_name='density')
        df_density['left_right'] = df_density['left_right'].map({'left': 1, 'right': 0})
        return df_density

    def _normalize_density(self, df_density: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize density values to a range between 0.1 and 1.0.

        Parameters:
            df_density (pd.DataFrame): Dataframe containing density values.

        Returns:
            pd.DataFrame: Normalized density dataframe.
        """
        scaler = MinMaxScaler(feature_range=(0.1, 1.0))
        dens = np.append(df_density['density'].to_numpy(), 0)  # Add zero for scaling
        dens_norm = scaler.fit_transform(dens.reshape(-1, 1))
        df_density['density'] = dens_norm[:-1]  # Remove the added zero
        return df_density

    def _sort_and_extract_brain_areas(self, df: pd.DataFrame, plot_type: Optional[str] = None) -> List[List[str]]:
        """
        Sort brain areas by hierarchical structure and extract relevant data.

        Parameters:
            df (pd.DataFrame): Dataframe containing brain region data.
            plot_type (Optional[str]): Type of plot ('density' for density plot, None for others).

        Returns:
            List[List[str]]: Lists of brain areas and their corresponding colors.
        """
        df['len'] = df['structure_id'].apply(
            lambda a: len(self.atlas.structures.data[a]['structure_id_path'])
        )
        df.sort_values(by='len', inplace=True)
        brain_areas = df['acronym'].tolist()
        brain_areas_colors = df['brain_areas_color'].tolist()
        # todo how to best integrate hemispheres for density plot
        # if plot_type == 'density':
        #     brain_areas_hemisphere = df['left_right'].tolist()
        #     return [brain_areas, brain_areas_colors, brain_areas_hemisphere]
        # else:
        #     return [brain_areas, brain_areas_colors]
        return [brain_areas, brain_areas_colors]

    def _brain_region_density_color(self, df: pd.DataFrame, cmap: str) -> List[List[str]]:
        """
        Assign colors to brain regions based on density values.

        Parameters:
            df (pd.DataFrame): Dataframe containing density data.
            cmap (str): Colormap for assigning colors.

        Returns:
            List[List[str]]: Sorted brain areas and their corresponding colors.
        """

        df['structure_id'] = df['acronym'].map(self.atlas.structures.acronym_to_id_map)
        curr_cmap = self.color_manager.create_custom_colormap(cmap)
        df['brain_areas_color'] = df['density'].map(curr_cmap)

        return self._sort_and_extract_brain_areas(df, plot_type='density')

    def _brain_region_gene_color(self) -> List[List[str]]:
        """
        Assign colors to brain regions based on gene expression data.

        Returns:
            List[List[str]]: Sorted brain areas and their corresponding colors.
        """

        if self.plotting_params['plot_gene'] == 'clusters':
            count_clusters = self.data_dict['genes'].groupby(['acronym', 'structure_id', 'cluster_id']).size().reset_index(name='count')
            brain_region_colors = count_clusters.loc[count_clusters.groupby('acronym')['count'].idxmax()]
            brain_region_colors['brain_areas_color'] = brain_region_colors['cluster_id'].map(self.color_dict['genes']['cmap'])
        else:
            brain_region_colors = self.data_dict['genes'].groupby('acronym')['gene_expression_norm'].mean().reset_index()
            brain_region_colors['structure_id'] = brain_region_colors['acronym'].map(
                self.atlas.structures.acronym_to_id_map)
            brain_region_colors['brain_areas_color'] = brain_region_colors['gene_expression_norm'].apply(self.color_dict['genes']['cmap'])
        return self._sort_and_extract_brain_areas(brain_region_colors)

    def _get_voronoi_mask(self, df: pd.DataFrame, orient_mapping: Dict) -> Tuple[
        Optional[np.ndarray], Optional[Dict[int, Tuple[float, float, float]]]]:
        """
        Generate a Voronoi mask for visualizing spatial distribution of data.

        Parameters:
            df (pd.DataFrame): Dataframe containing points for the Voronoi diagram.
            orient_mapping (Dict): Mapping of orientation coordinates.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Dict[int, Tuple[float, float, float]]]]: Voronoi mask and color dictionary.
        """

        df = df.reset_index(drop=True)
        xyz_dict = get_xyz(self.atlas, self.plotting_params['section_orient'])
        ydim, xdim = xyz_dict['y'][1], xyz_dict['x'][1]
        # unilateral = self.plotting_params.get('unilateral', False)
        # if unilateral in ['left', 'right'] and orient_idx < 2:
        #     xdim //= 2

        matrix_shape = (ydim, xdim)
        matrix = np.zeros(matrix_shape)

        points = df[[orient_mapping['x_plot'], orient_mapping['y_plot']]].to_numpy()
        if len(points) < 4:
            show_info('too few spots in section, consider increasing range around section to include more')
            return None, None

        if self.plotting_params['plot_gene'] == 'clusters':
            df.loc[:, 'voronoi_colors'] = df['cluster_id'].map(self.color_dict['genes']['cmap'])
            df = self._create_color_ids(df)
        else:
            df.loc[:, 'voronoi_colors'] = df['gene_expression_norm'].apply(self.color_dict['genes']['cmap'])
            df.loc[:, 'clr_id'] = np.arange(len(df))

        xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        nearest_point_index = np.argmin(cdist(grid_points, points), axis=1)


        matrix.ravel()[:] = nearest_point_index
        matrix = matrix.reshape(matrix_shape).astype('int')
        matrix = np.vectorize(lambda x: int(df.loc[x, 'clr_id']))(matrix)

        color_dict = {
            clr_id: tuple(df.loc[df['clr_id'] == clr_id, 'voronoi_colors'].iloc[0])
            for clr_id in natsorted(df['clr_id'].unique())
        }
        return [matrix, color_dict]

    def _create_color_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign unique integer IDs to clusters for consistent coloring.

        Parameters:
            df (pd.DataFrame): Dataframe containing cluster IDs.

        Returns:
            pd.DataFrame: Updated dataframe with integer color IDs.
        """
        unique_clusters = natsorted(df['cluster_id'].unique())
        map_dict = {cluster: idx for idx, cluster in enumerate(unique_clusters)}

        # Map cluster IDs to the new integer values
        df['clr_id'] = df['cluster_id'].map(map_dict)

        return df


    def calculate_heatmap(
        self, annot_section_plt: np.ndarray, df: pd.DataFrame, orient_mapping: Dict, y_bins: np.ndarray, x_bins: np.ndarray, bin_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate a heatmap based on data distribution in the specified slice.

        Parameters:
            annot_section_plt (np.ndarray): Annotated section to use as a base.
            df (pd.DataFrame): Dataframe containing plotting data.
            orient_mapping (Dict): Mapping for orientation coordinates.
            y_bins (np.ndarray): Bin edges for the y-axis.
            x_bins (np.ndarray): Bin edges for the x-axis.
            bin_size (float): Size of each bin in the heatmap.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Heatmap data and mask.
        """
        animal_data = []
        for animal_id, group_data in df.groupby('animal_id'):
            # Calculate 2D histogram for the current animal
            h_data, _, _ = np.histogram2d(
                group_data[orient_mapping['y_plot']],
                group_data[orient_mapping['x_plot']],
                bins=[y_bins, x_bins]
            )
            num_sections = len(df[df['animal_id'] == animal_id]['section_name'].unique())
            h_data /= num_sections
            animal_data.append(h_data)
        heatmap_data = np.mean(np.stack(animal_data), axis=0)
        heatmap_data, mask = self._resize_heatmap(heatmap_data, annot_section_plt)
        return heatmap_data, mask

    def calculate_heatmap_difference(
        self, annot_section_plt: np.ndarray, df: pd.DataFrame, plotting_params: Dict, orient_mapping: Dict,
        y_bins: np.ndarray, x_bins: np.ndarray, bin_size: float, diff_type: str, diff_items: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the difference between heatmaps for two groups.

        Parameters:
            annot_section_plt (np.ndarray): Annotated section to use as a base.
            df (pd.DataFrame): Dataframe containing plotting data.
            plotting_params (Dict): Parameters for plotting.
            orient_mapping (Dict): Mapping for orientation coordinates.
            y_bins (np.ndarray): Bin edges for the y-axis.
            x_bins (np.ndarray): Bin edges for the x-axis.
            bin_size (float): Size of each bin in the heatmap.
            diff_type (str): Column name used for grouping.
            diff_items (List[str]): Group names to compare.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Heatmap difference data and mask.
        """
        group_list = df[plotting_params[diff_type]].unique()
        show_info(f'group list: {group_list}')
        required_items = plotting_params[diff_items]
        show_info(f'group list: {required_items}')
        if all([item in group_list for item in required_items]):
            diff_data = []
            for group_item in required_items:
                animal_sub_list = df[df[plotting_params[diff_type]] == group_item]['animal_id'].unique()
                group_df = df[df['animal_id'].isin(animal_sub_list)]
                heatmap_sub_data, _ = self.calculate_heatmap(annot_section_plt, group_df,
                                                        orient_mapping, y_bins, x_bins, bin_size)
                diff_data.append(heatmap_sub_data)
            if self.plotting_params[f'{diff_type}_idx']:
                heatmap_data = (diff_data[0] - diff_data[1])/(diff_data[0] + diff_data[1])
            else:
                heatmap_data = diff_data[0] - diff_data[1]
            mask = self._get_heatmap_mask(heatmap_data, annot_section_plt)
        else:
            missing_items = [item for item in required_items if item not in group_list]
            show_info(
                f"selected items to calculate difference not found: {missing_items}  \n"
                f"check if items exists, also check params file if items are stated \n"
                f"--> plotting regular density map")
            heatmap_data, mask = self.calculate_heatmap(annot_section_plt, df, orient_mapping, y_bins, x_bins, bin_size)
        return heatmap_data, mask

    def _resize_heatmap(self, heatmap_data: np.ndarray, annot_section_plt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize a heatmap to match the size of an annotated section.

        Parameters:
            heatmap_data (np.ndarray): Heatmap data to resize.
            annot_section_plt (np.ndarray): Annotated section to match the size.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Resized heatmap and mask.
        """

        resized_heatmap_data = resize(heatmap_data, (annot_section_plt.shape[0], annot_section_plt.shape[1]), order=1,
                                      mode='reflect',
                                      anti_aliasing=True)
        resized_heatmap_data = gaussian_filter(resized_heatmap_data, sigma=1)
        mask = self._get_heatmap_mask(resized_heatmap_data, annot_section_plt)

        return resized_heatmap_data, mask

    def _get_heatmap_mask(self, heatmap_data: np.ndarray, annot_section_plt: np.ndarray) -> np.ndarray:
        """
        Generate a mask for a heatmap based on annotated section.

        Parameters:
            heatmap_data (np.ndarray): Heatmap data.
            annot_section_plt (np.ndarray): Annotated section.

        Returns:
            np.ndarray: Mask indicating valid regions.
        """
        mask1 = heatmap_data == 0
        mask2 = annot_section_plt == 0
        mask = mask1 | mask2

        return mask










