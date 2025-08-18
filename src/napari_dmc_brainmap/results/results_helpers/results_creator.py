import pandas as pd
import numpy as np
from natsort import natsorted
import cv2
from sklearn.preprocessing import minmax_scale
from matplotlib import path
from typing import List, Union, Generator
from pathlib import Path
from napari.utils.notifications import show_info
from napari_dmc_brainmap.results.results_helpers.tract_calculator import TractCalculator
from napari_dmc_brainmap.results.results_helpers.slice_handle import SliceHandle
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.general_utils import get_animal_id, create_regi_dict


class ResultsCreator:
    """
    Class for creating segmentation result files, including registration data,
    transformed segmentation data, and optional export for brainrender visualization.
    """
    def __init__(
        self,
        input_path: Path,
        seg_type: str,
        channels: List[str],
        seg_folder: str,
        regi_chan: str,
        include_all: bool,
        export: bool,
        probe_insert: List[int]
    ) -> None:
        """
        Initialize the ResultsCreator.

        Parameters:
            input_path (Path): Path to the input directory.
            seg_type (str): Type of segmentation (e.g., 'cells', 'injection_site').
            channels (List[str]): List of segmentation channels.
            seg_folder (str): Folder containing segmentation results.
            regi_chan (str): Registration channel name.
            include_all (bool): Whether to include unregistered data.
            export (bool): Whether to export data for brainrender visualization.
            probe_insert (List[int]): Probe insertion coordinates (optional).
        """
        self.input_path = input_path
        self.seg_type = seg_type
        self.channels = channels
        self.seg_folder = seg_folder
        self.regi_chan = regi_chan
        self.include_all = include_all
        self.export = export
        self.probe_insert = probe_insert
        self.s = None
        self.atlas = None
        self.results_dir = None
        self.tract_calculator = None


    def initialize(self) -> None:
        """
        Initialize the registration and segmentation channels based on segmentation type.
        """
        self.s = self._initialize_registration()
        if self.seg_type in ["optic_fiber", "neuropixels_probe"]:
            self.channels = self._get_segmentation_channels()
        elif self.seg_type == "single_cell":
            self.channels = self._get_segmentation_channels()



    def create_results(self, progress_callback: Union[None, callable] = None) -> None:
        """
        Create results for the specified segmentation type.

        Parameters:
            progress_callback (Union[None, callable], optional): Function to report progress. Defaults to None.
        """
        try:
            self.initialize()
            completed_steps = 0
            total_steps = sum(len(self._get_segment_list(chan)) for chan in self.channels)
            if self.seg_type in ["optic_fiber", "neuropixels_probe"]:
                total_steps += 1

            for chan in self.channels:
                for im_idx in self._process_channel(chan):
                    completed_steps += 1
                    if progress_callback:
                        progress = int((completed_steps / total_steps) * 100)
                        progress_callback(progress)
            if self.seg_type in ["optic_fiber", "neuropixels_probe"]:
                self._calculate_probe_tract()
                completed_steps += 1
                if progress_callback:
                    progress = int((completed_steps / total_steps) * 100)
                    progress_callback(progress)

        except Exception as e:
            show_info(f"An error occurred during results creation: {e}")

    def _initialize_registration(self) -> SliceHandle:
        """
        Initialize registration data for the atlas and segmentation.

        Returns:
            SliceHandle: Object containing registration data and methods.
        """
        regi_dir = get_info(self.input_path, 'sharpy_track', channel=self.regi_chan, only_dir=True)
        regi_dict = create_regi_dict(self.input_path, regi_dir)
        return SliceHandle(regi_dict)

    def _get_segmentation_channels(self) -> List[str]:
        """
        Retrieve the segmentation channels from the segmentation folder.

        Returns:
            List[str]: List of segmentation channels.
        """
        seg_super_dir = get_info(self.input_path, 'segmentation', seg_type=self.seg_type, only_dir=True)
        return natsorted([f.parts[-1] for f in seg_super_dir.iterdir() if f.is_dir()])

    def _get_segment_list(self, chan: str) -> List[str]:
        """
        Get the list of segmentation images for the specified channel.

        Parameters:
            chan (str): Channel name.

        Returns:
            List[str]: List of segmentation image filenames.
        """
        # Retrieve the list of segmentation images for the given channel
        segment_dir, segment_list, _ = get_info(self.input_path, 'segmentation', channel=chan, seg_type=self.seg_type)
        return segment_list

    def _process_channel(self, chan: str) -> Generator[int, None, None]:
        """
        Process segmentation data for a specific channel.

        Parameters:
            chan (str): Channel name.

        Yields:
            int: Index of the processed image.
        """

        regi_dir, regi_im_list, regi_suffix = get_info(self.input_path, 'sharpy_track', channel=self.regi_chan)
        if self.seg_folder == 'rgb':
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(self.input_path, self.seg_folder)
        else:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(self.input_path, self.seg_folder, channel=chan)
        segment_dir, segment_list, segment_suffix = get_info(self.input_path, 'segmentation', channel=chan, seg_type=self.seg_type)
        data = pd.DataFrame()
        if len(segment_list) > 0:
            self.results_dir = get_info(self.input_path, 'results', channel=chan, seg_type=self.seg_type,
                                        create_dir=True, only_dir=True)
            for im_idx, im in enumerate(segment_list):
                try:
                    section_data = self._transform_points_to_regi(im, segment_dir, segment_suffix, seg_im_dir,
                                                                seg_im_suffix, regi_dir, regi_suffix)
                    if not self.include_all:
                        if section_data is None:
                            print(f"Skipping empty section data for image: {im}")
                        else:
                            print(f"Processing section data for image: {im}")
                            section_data = section_data[section_data['structure_id'] != 0].reset_index(drop=True)
                    if section_data is not None:
                        data = pd.concat((data, section_data))
                except KeyError:
                    show_info(f"Registration data for channel {chan} is incomplete, skipping.")
                yield im_idx
            self._save_results(data, chan)
        else:
            show_info(f"No segmentation images found in {str(segment_dir)}")



    def _transform_points_to_regi(
        self,
        im: str,
        segment_dir: Path,
        segment_suffix: str,
        seg_im_dir: Path,
        seg_im_suffix: str,
        regi_dir: Path,
        regi_suffix: str
    ) -> pd.DataFrame:
        """
        Transform segmentation points to registration coordinates.

        Parameters:
            im (str): Image name.
            segment_dir (Path): Path to segmentation directory.
            segment_suffix (str): Suffix of segmentation files.
            seg_im_dir (Path): Path to segmented image directory.
            seg_im_suffix (str): Suffix of segmented image files.
            regi_dir (Path): Path to registration directory.
            regi_suffix (str): Suffix of registration files.

        Returns:
            pd.DataFrame: Transformed segmentation data.
        """
        curr_im = im[:-len(segment_suffix)]
        img = cv2.imread(str(seg_im_dir.joinpath(curr_im + seg_im_suffix)))
        y_im, x_im, z_im = img.shape  # original resolution of image
        # correct for 0 indices
        y_im -= 1
        x_im -= 1
        img_regi = cv2.imread(str(regi_dir.joinpath(curr_im + regi_suffix)))
        y_low, x_low, z_low = img_regi.shape  # original resolution of image
        # correct for 0 indices
        y_low -= 1
        x_low -= 1

        segment_data = pd.read_csv(segment_dir.joinpath(im))
        y_pos = list(segment_data['Position Y'])
        x_pos = list(segment_data['Position X'])
        # append mix max values for rescaling
        y_pos.append(0)
        y_pos.append(y_im)
        x_pos.append(0)
        x_pos.append(x_im)
        y_scaled = np.ceil(minmax_scale(y_pos, feature_range=(0, y_low)))[:-2].astype(int)
        x_scaled = np.ceil(minmax_scale(x_pos, feature_range=(0, x_low)))[:-2].astype(int)
        if self.seg_type == 'injection_site':
            for n in segment_data['idx_shape'].unique():
                n_idx = segment_data.index[segment_data['idx_shape'] == n].tolist()
                curr_x = np.array([x_scaled[i] for i in n_idx])
                curr_y = np.array([y_scaled[i] for i in n_idx])
                curr_coords = self._regi_points_polygon(curr_x, curr_y)
                if n == 0:
                    coords = curr_coords
                else:
                    coords = np.concatenate((coords, curr_coords), axis=0)

        else:
            coords = np.stack([x_scaled, y_scaled], axis=1)

        # slice_idx = list(regi_data['imgName'].values()).index(curr_im + regi_suffix)
        self.s.setImgFolder(regi_dir)
        # set which slice in there
        self.s.setSlice(curr_im + regi_suffix)
        section_data = self.s.getBrainArea(coords, (curr_im + regi_suffix))
        if self.seg_type == "genes":
            assert section_data is not None
            section_data['cluster_id'] = segment_data['cluster_id']
            section_data['spot_id'] = segment_data['spot_id']
        elif self.seg_type == 'hcr':
            section_data['hcr'] = segment_data['hcr']
        return section_data

    def _regi_points_polygon(self, x_scaled: np.ndarray, y_scaled: np.ndarray) -> np.ndarray:
        """
        Create a polygon from scaled coordinates and return the points inside.

        Parameters:
            x_scaled (np.ndarray): Scaled x-coordinates.
            y_scaled (np.ndarray): Scaled y-coordinates.

        Returns:
            np.ndarray: Points within the polygon.
        """

        poly_points = [(x_scaled[i], y_scaled[i]) for i in range(0, len(x_scaled))]
        polygon = path.Path(poly_points)
        x_min, x_max = x_scaled.min(), x_scaled.max()
        y_min, y_max = y_scaled.min(), y_scaled.max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, (x_max - x_min) + 1),
                             np.linspace(y_min, y_max, (y_max - y_min) + 1))
        canvas_points = [(np.ndarray.flatten(xx)[i], np.ndarray.flatten(yy)[i]) for i in
                         range(0, len(np.ndarray.flatten(xx)))]
        idx_in_polygon = polygon.contains_points(canvas_points)
        points_in_polygon = [c for c, i in zip(canvas_points, idx_in_polygon) if i]
        x_poly = [p[0] for p in points_in_polygon]
        y_poly = [p[1] for p in points_in_polygon]
        coords = np.stack([x_poly, y_poly], axis=1)
        return coords


    def _save_results(self, data: pd.DataFrame, chan: str) -> None:
        """
        Save segmentation results to a CSV file.

        Parameters:
            data (pd.DataFrame): Segmentation data.
            chan (str): Channel name.
        """
        fn = self.results_dir.joinpath(f'{get_animal_id(self.input_path)}_{self.seg_type}.csv')
        data.to_csv(fn)

        if self.export and self.seg_type == 'cells':
            self._export_data_to_brainrender(data, chan)

    def _export_data_to_brainrender(self, data: pd.DataFrame, chan: str) -> None:
        """
        Export segmentation data to brainrender-compatible format.

        Parameters:
            data (pd.DataFrame): Segmentation data.
            chan (str): Channel name.
        """
        orient_dict = {
            'ap': 'ap_coords',
            'rl': 'ml_coords',
            'si': 'dv_coords'
        }
        atlas_ax = self.s.atlas.space.axes_description
        X, Y, Z = data[orient_dict[atlas_ax[0]]].to_numpy(), data[orient_dict[atlas_ax[1]]].to_numpy(), data[
            orient_dict[atlas_ax[2]]].to_numpy()
        pts = [[x * self.s.atlas.resolution[0], y * self.s.atlas.resolution[1], z * self.s.atlas.resolution[2]] for x, y, z in zip(X, Y, Z)]
        bg_data = np.vstack(pts)
        bg_fn = self.results_dir.joinpath(f'{get_animal_id(self.input_path)}_{self.seg_type}_{chan}.npy')
        np.save(bg_fn, bg_data)
        show_info(f"Exported data to brainrender format in {str(bg_fn)}")

    def _calculate_probe_tract(self) -> None:
        """
        Calculate the probe tract for neuropixels or optic fiber segmentation.
        """
        tract_calculator = TractCalculator(self.s, self.input_path, self.seg_type, self.probe_insert)
        tract_calculator.calculate_probe_tract()
