from pathlib import Path
import json
from typing import List, Union
import pandas as pd
from natsort import natsorted
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.atlas_utils import get_bregma, coord_mm_transform
from bg_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_info

class DataLoader:
    def __init__(self,
                 input_path: Path,
                 atlas: BrainGlobeAtlas,
                 animal_list: List[str],
                 channels: List[str],
                 data_type: str = 'cells',
                 hemisphere: str = 'both'):
        """
        Initialize the DataLoader.

        Parameters:
            input_path (Path): Path to the input data directory.
            atlas: The atlas object containing structure and hierarchy information.
            animal_list (List[str]): List of animal identifiers.
            channels (List[str]): List of channel identifiers.
            data_type (str): Type of data to load (e.g., 'cells', 'optic_fiber'). Default is 'cells'.
            hemisphere (str): Hemisphere to analyze ('both', 'ipsi', 'contra'). Default is 'both'.
        """
        self.input_path = input_path
        self.atlas = atlas
        self.animal_list = animal_list
        self.channels = channels
        self.data_type = data_type
        self.hemisphere = hemisphere
        self.bregma = get_bregma(atlas.atlas_name)

    def load_data(self) -> pd.DataFrame:
        """
        Load and merge data for all animals.

        Returns:
            pd.DataFrame: Merged DataFrame containing data for all animals and channels.
        """
        results_data_merged = pd.DataFrame()

        for animal_id in self.animal_list:
            if self.data_type in ["optic_fiber", "neuropixels_probe"]:
                seg_super_dir = get_info(self.input_path.joinpath(animal_id), 'results', seg_type=self.data_type, only_dir=True)
                self.channels = natsorted([f.parts[-1] for f in seg_super_dir.iterdir() if f.is_dir()])

            params = self._load_params(self.input_path.joinpath(animal_id, 'params.json'))

            for channel in self.channels:
                if self.data_type == 'swc':
                    results_data = self._load_swc_data(animal_id, channel)
                else:
                    results_data = self._load_channel_data(animal_id, channel)
                if results_data is None:
                    continue
                if self.data_type in ["optic_fiber", "neuropixels_probe"]:
                    results_data = results_data[results_data['inside_brain']]
                results_data['ml_mm'] *= -1  # Convert negative values for the left hemisphere
                results_data['animal_id'] = animal_id
                results_data['channel'] = f"{animal_id}_{channel}" if (
                        self.data_type in ["optic_fiber", "neuropixels_probe"] and len(self.animal_list) > 1) else channel
                results_data['injection_site'] = params.get('injection_site', 'right')
                results_data['genotype'] = params.get('genotype', 0)
                results_data['group'] = params.get('group', 0)

                results_data = self._get_ipsi_contra(results_data)
                results_data_merged = pd.concat([results_data_merged, results_data], ignore_index=True)

            show_info(f"Loaded data from {animal_id}")

            if self.atlas.metadata['name'] == 'allen_mouse':
                results_data_merged = self._clean_results_df(results_data_merged)
            if self.hemisphere in ['ipsi', 'contra']:
                results_data_merged = results_data_merged[results_data_merged['ipsi_contra'] == self.hemisphere]

        return results_data_merged.reset_index(drop=True)

    def get_tgt_data_only(self,
                          df: pd.DataFrame,
                          tgt_list: List[str],
                          negative: bool = False,
                          use_na: bool = False) -> pd.DataFrame:
        """
        Filter data to include only target regions.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing structure data.
            tgt_list (List[str]): List of target region names.
            negative (bool): Whether to exclude target regions. Default is False.
            use_na (bool): Whether to include NA regions. Default is False.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only target regions.
        """
        ids = []
        for reg in tgt_list:
            if not reg == 'NA':
                try:
                    ids.append(self.atlas.structures[reg]['id'])
                except KeyError:
                    show_info(f'No region called >> {reg} <<, skipping that region.')
                    pass

        ids_child = [child_id for tgt_id in ids for child_id in self.atlas.hierarchy.is_branch(tgt_id)]
        ids.extend(ids_child)

        if 'NA' in tgt_list and use_na:
            ids.append(-42)
            df.loc[~df.structure_id.isin(ids), 'structure_id'] = -42

        condition = df['structure_id'].isin(ids)
        tgt_data = df[~condition] if negative else df[condition]
        tgt_data.loc[:, 'tgt_name'] = tgt_data['structure_id'].map(lambda s: self._get_tgt_name(s, tgt_list))
        return tgt_data.reset_index(drop=True)

    def _get_tgt_name(self, structure_id: int, tgt_list: List[str]) -> str:
        """
        Map a structure ID to its name or closest ancestor in the target list.

        Parameters:
            structure_id (int): The ID of the structure.
            tgt_list (List[str]): List of target region names.

        Returns:
            str: The name of the target region.
        """
        try:
            acronym = self.atlas.structures[structure_id]['acronym']
            if acronym in tgt_list:
                return acronym
            ancestors = list(set(self.atlas.get_structure_ancestors(structure_id)) & set(tgt_list))
            return ancestors[0] if ancestors else 'NA'
        except KeyError:
            return 'NA'

    def _load_params(self, params_file: Path) -> dict:
        """
        Load parameters from a JSON file.

        Parameters:
            params_file (Path): Path to the params.json file.

        Returns:
            dict: Loaded parameters.
        """
        try:
            with open(params_file) as fn:
                return json.load(fn).get('general', {})
        except FileNotFoundError:
            show_info(f"WARNING: Params file not found for {params_file}, using defaults.")
            return {}

    def _load_channel_data(self, animal_id: str, channel: str) -> Union[pd.DataFrame, None]:
        """
        Load data for a specific channel of an animal.

        Parameters:
            animal_id (str): Identifier of the animal.
            channel (str): Identifier of the channel.

        Returns:
            pd.DataFrame or None: Loaded channel data or None if file is not found.
        """
        results_dir = get_info(self.input_path.joinpath(animal_id), 'results', seg_type=self.data_type, channel=channel, only_dir=True)
        results_file = results_dir.joinpath(f"{animal_id}_{self.data_type}.csv")

        if not results_file.exists():
            show_info(f"WARNING: Data file {results_file} does not exist. Skipping...")
            return None

        return pd.read_csv(results_file)

    def _load_swc_data(self, animal_id: str, channel: str) -> Union[pd.DataFrame, None]:
        """
        Load swc data for a specific channel of an animal.

        Parameters:
            animal_id (str): Identifier of the animal.
            channel (str): Identifier of the channel.

        Returns:
            pd.DataFrame or None: Loaded channel data or None if file is not found.
        """
        swc_merge = pd.DataFrame()
        results_dir = get_info(self.input_path.joinpath(animal_id), 'results', seg_type=self.data_type, channel=channel,
                               only_dir=True)
        results_file = results_dir.joinpath(f"{animal_id}_{self.data_type}.csv")
        if not results_file.exists():
            show_info(f"WARNING: Data file {results_file} does not exist. Skipping...")
            return None

        results_descriptor = pd.read_csv(results_file)
        for index, row in results_descriptor.iterrows():
            swc_fn = results_dir.joinpath(row['swc_file'])
            cols = ["id", "type", "ml_coords", "ap_coords", "dv_coords", "radius", "parent"]
            swc_df = pd.read_csv(swc_fn, comment="#", delim_whitespace=True, names=cols)
            swc_df['neuron_id'] = row['swc_file'].split('.')[0]
            swc_df['group_id'] = row['group']
            swc_df['animal_id'] = animal_id
            swc_merge = pd.concat((swc_merge, swc_df), ignore_index=True)
        coords = swc_merge[["ap_coords", "dv_coords", "ml_coords"]].to_numpy()
        mm_coords = [coord_mm_transform(row, self.bregma, self.atlas.space.resolution) for row in coords]
        swc_merge[["ap_mm", "dv_mm", "ml_mm"]] = pd.DataFrame(mm_coords, index=swc_merge.index)
        return swc_merge

    def _get_ipsi_contra(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Determine if cells are ipsilateral ('ipsi') or contralateral ('contra') to the injection site.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing 'ml_mm' and 'injection_site' columns.

        Returns:
            pd.DataFrame: DataFrame with an additional 'ipsi_contra' column.
        """
        if 'injection_site' not in df.columns or df.empty:
            raise ValueError("The dataframe must contain a non-empty 'injection_site' column.")

        injection_site = df['injection_site'].iloc[0]
        df['ipsi_contra'] = 'ipsi'

        if injection_site == 'left':
            df.loc[df['ml_mm'] < 0, 'ipsi_contra'] = 'contra'
        elif injection_site == 'right':
            df.loc[df['ml_mm'] > 0, 'ipsi_contra'] = 'contra'
        else:
            raise ValueError(f"Unexpected value in 'injection_site': {injection_site}. Expected 'left' or 'right'.")

        return df

    def _clean_results_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the results DataFrame by removing unwanted structures.

        Parameters:
            df (pd.DataFrame): DataFrame containing atlas data.

        Returns:
            pd.DataFrame: Cleaned DataFrame with unwanted structures removed.
        """
        try:
            list_delete = ['root']
            for item in ['fiber tracts', 'VS']:
                list_delete.extend(self.atlas.get_structure_descendants(item))
            df = df.drop(df[df['acronym'].isin(list_delete)].index)
            df = df.reset_index(drop=True)
        except KeyError:
            pass
        return df
