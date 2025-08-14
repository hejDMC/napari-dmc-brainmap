import numpy as np
import pandas as pd
from pathlib import Path
import json
import cv2
from typing import List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from aicsimageio import AICSImage
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from aicssegmentation.core.seg_dot import dot_3d
from aicssegmentation.core.utils import hole_filling
from aicsimageio.writers import OmeTiffWriter
from bg_atlasapi import BrainGlobeAtlas
from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import fitGeoTrans, mapPointTransform
from napari_dmc_brainmap.segment.processing.atlas_utils import loadAnnotBool, angleSlice
from napari_dmc_brainmap.utils.path_utils import get_info
from napari_dmc_brainmap.utils.atlas_utils import get_bregma

from napari.utils.notifications import show_info

class PreSegmenter:
    """
    Base class for pre-segmentation tasks including loading data, preparing directories,
    and excluding objects based on registration data.
    """
    def __init__(self, input_path: Path, general_params: dict) -> None:
        """
        Initialize the PreSegmenter with input path and general parameters.

        Parameters:
            input_path (Path): Path to the input directory.
            general_params (dict): General configuration parameters.
        """

        self.input_path = input_path
        self.general_params = general_params

        # self.registration_data = None


    def load_registration_data(self) -> None:
        """
        Load registration data including transformation matrices and atlas information.
        """

        if self.general_params["regi_bool"]:
            try:
                regi_dir = get_info(self.input_path, 'sharpy_track', channel=self.general_params["regi_chan"], only_dir=True)
                regi_fn = regi_dir.joinpath("registration.json")
                with open(regi_fn, 'r') as f:
                    regi_data = json.load(f)
                annot_bool = loadAnnotBool(self.general_params["atlas_id"])
                show_info(f"check existence of local version of {self.general_params['atlas_id']} atlas ...")
                show_info(f"loading reference atlas {self.general_params['atlas_id']} ...")
                atlas = BrainGlobeAtlas(self.general_params["atlas_id"])
                z_idx = atlas.space.axes_description.index(self.general_params["xyz_dict"]['z'][0])
                z_res = self.general_params["xyz_dict"]['z'][2]
                bregma = get_bregma(self.general_params["atlas_id"])
                self.registration_data = {
                    "regi_data": regi_data,
                    "annot_bool": annot_bool,
                    "z_idx": z_idx,
                    "z_res": z_res,
                    "bregma": bregma
                }
            except FileNotFoundError:
                show_info('NO REGISTRATION DATA FOUND')
                self.registration_data = None

    def get_regi_index(self, im: str, seg_im_suffix: str) -> int:
        """
        Get the registration index corresponding to the given image.

        Parameters:
            im (str): Image name.
            seg_im_suffix (str): Suffix of the segmented image.

        Returns:
            int: Registration index.
        """
        for k, v in self.registration_data["regi_data"]['imgName'].items():
            if v.startswith(im[:-(len(seg_im_suffix) - 1)]):
                regi_index = k
        return regi_index

    def exclude_segment_objects(self, im: str, segmented_image: np.ndarray, seg_idx: np.ndarray, seg_im_suffix: str) -> np.ndarray:
        """
        Exclude objects from the segmentation based on atlas registration.

        Parameters:
            im (str): Image name.
            segmented_image (np.ndarray): Binary segmentation mask.
            seg_idx (np.ndarray): Array of segment indices.
            seg_im_suffix (str): Suffix of the segmented image.

        Returns:
            np.ndarray: Updated segment indices after exclusion.
        """

        dim_image = segmented_image.shape
        x_res = self.general_params["xyz_dict"]['x'][1]
        y_res = self.general_params["xyz_dict"]['y'][1]
        x_im = seg_idx[:, 1] / dim_image[1] * x_res
        y_im = seg_idx[:, 0] / dim_image[0] * y_res
        regi_index = self.get_regi_index(im, seg_im_suffix)
        drop_mask = self.get_drop_mask(regi_index, x_im, y_im)
        if len(drop_mask) > 0:
            seg_idx = seg_idx[drop_mask]
        return seg_idx

    def get_drop_mask(self, regi_index: int, x_im: np.ndarray, y_im: np.ndarray) -> np.ndarray:
        """
        Get a mask for excluding objects based on atlas registration.

        Parameters:
            regi_index (int): Registration index.
            x_im (np.ndarray): X-coordinates of image objects.
            y_im (np.ndarray): Y-coordinates of image objects.

        Returns:
            np.ndarray: Boolean mask for excluding objects.
        """

        try:
            # get transformation
            tform = fitGeoTrans(self.registration_data["regi_data"]['sampleDots'][regi_index],
                                self.registration_data["regi_data"]['atlasDots'][regi_index])
            # slice annotation volume
            x_angle, y_angle, z = self.registration_data["regi_data"]['atlasLocation'][regi_index]

            annot_slice = angleSlice(x_angle, y_angle, z, self.registration_data["annot_bool"],
                                     self.registration_data["z_idx"], self.registration_data["z_res"],
                                     self.registration_data["bregma"], self.general_params["xyz_dict"])
            # mark invalid coordinates
            drop_mask = []
            for x, y in zip(x_im, y_im):
                x_atlas, y_atlas = mapPointTransform(x, y, tform)
                x_atlas, y_atlas = int(x_atlas), int(y_atlas)
                if (x_atlas < 0) | (y_atlas < 0) | (x_atlas >= self.general_params["xyz_dict"]['x'][1]) | (
                        y_atlas >= self.general_params["xyz_dict"]['y'][1]):
                    drop_mask.append(0)
                else:
                    if annot_slice[y_atlas, x_atlas] == 0:
                        drop_mask.append(0)
                    else:
                        drop_mask.append(1)
            drop_mask = np.array(drop_mask, dtype=bool)
        except Exception:
            show_info(f"No registration data for {self.registration_data['regi_data']['imgName'][regi_index]}")
            drop_mask = []
        return drop_mask


    def save_to_csv(self, data: pd.DataFrame, file_path: Path) -> None:
        """
        Save data to a CSV file.

        Parameters:
            data (pd.DataFrame): Data to save.
            file_path (Path): Path to save the CSV file.
        """
        data.to_csv(file_path, index=False)


    def load_image_list(self, chan: str, im_class: str) -> Tuple[Path, List[str], str]:
        """
        Load the list of images to be segmented.

        Parameters:
            chan (str): Channel identifier.
            im_class (str): Class of the image (e.g., 'rgb', 'single_channel').

        Returns:
            Tuple[Path, List[str], str]: Directory path, list of images, and image suffix.
        """
        if im_class == 'rgb':
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(self.input_path, im_class)
        else:
            seg_im_dir, seg_im_list, seg_im_suffix = get_info(self.input_path, im_class, channel=chan)

        if self.general_params["start_end_im"]:
            if len(self.general_params["start_end_im"]) == 2:
                seg_im_list = seg_im_list[self.general_params["start_end_im"][0]:self.general_params["start_end_im"][1] + 1]
        return seg_im_dir, seg_im_list, seg_im_suffix


    def prepare_segmentation_folders(self, seg_folder: str, chan: str) -> Union[Tuple[Path, Path], Path]:
        """
        Prepare directories for segmentation tasks.

        Parameters:
            seg_folder (str): Folder name for segmentation masks.
            chan (str): Channel identifier.

        Returns:
            Union[Tuple[Path, Path], Path]: Paths to prepared directories.
        """
        output_dir = get_info(self.input_path, self.general_params["output_folder"], channel=chan,
                              seg_type=self.general_params["seg_type"], create_dir=True, only_dir=True)
        if self.general_params["seg_type"] == 'projections':
            return output_dir
        else:
            mask_dir = get_info(self.input_path, seg_folder, channel=chan, seg_type=self.general_params["seg_type"],
                                create_dir=True, only_dir=True)
            return mask_dir, output_dir
#%%

class CellsSegmenter(PreSegmenter):
    """
    Class for performing cell segmentation tasks including preprocessing, segmentation,
    and centroid detection.
    """
    def __init__(self, input_path: Path, general_params: dict, cells_params: dict, preseg_params: dict) -> None:
        """
        Initialize the CellsSegmenter.

        Parameters:
            input_path (Path): Path to the input directory.
            general_params (dict): General configuration parameters.
            cells_params (dict): Parameters specific to cell segmentation.
            preseg_params (dict): Pre-segmentation parameters.
        """
        super().__init__(input_path, general_params)
        self.cells_params = cells_params
        self.preseg_params = preseg_params


    def load_image(self, image_path: Path, chan: str) -> np.ndarray:
        """
        Load an image for segmentation.

        Parameters:
            image_path (Path): Path to the image file.
            chan (str): Channel identifier.

        Returns:
            np.ndarray: Loaded image as a NumPy array.
        """

        reader = AICSImage(str(image_path))
        img = reader.data.astype(np.float32)
        if self.cells_params["single_channel"]:
            img_struct = img[0, 0, 0, :, :].copy()
        else: # for RGB images
            chan_dict = {
                'cy3': 0,
                'green': 1,
                'dapi': 2
            }
            img_struct = img[0, 0, 0, :, :, chan_dict[chan]].copy()
        return np.array([img_struct, img_struct])  # Duplicate layer stack

    def save_mask_image(self, segmentation: np.ndarray, mask_save_fn: Path) -> None:
        """
        Save the segmentation mask as a TIFF image.

        Parameters:
            segmentation (np.ndarray): Segmentation mask.
            mask_save_fn (Path): File path to save the mask.
        """
        writer = OmeTiffWriter()
        writer.save(segmentation[0], str(mask_save_fn))

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image for segmentation.

        Parameters:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        image = intensity_normalization(image, scaling_param=self.preseg_params["intensity_norm"])
        return image_smoothing_gaussian_slice_by_slice(image, sigma=self.preseg_params["gaussian_smoothing_sigma"])

    def segment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Perform segmentation on the preprocessed image.

        Parameters:
            image (np.ndarray): Preprocessed image.

        Returns:
            np.ndarray: Segmentation mask.
        """
        response = dot_3d(image, log_sigma=self.preseg_params["dot_3d_sigma"])
        bw = response > self.preseg_params["dot_3d_cutoff"]
        bw_filled = hole_filling(bw, self.preseg_params["hole_min_max"][0], self.preseg_params["hole_min_max"][1], True)
        seg = remove_small_objects(bw_filled, min_size=self.preseg_params["minArea"], connectivity=1)

        seg = seg > 0
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255
        return seg[0]

    def find_centroids(self, segmented_image: np.ndarray) -> np.ndarray:
        """
        Find centroids in the segmented image.

        Parameters:
            segmented_image (np.ndarray): Segmentation mask.

        Returns:
            np.ndarray: Centroid coordinates.
        """
        # Label the image to find individual segmented regions
        label_img = label(segmented_image)

        # Get properties of segmented regions, particularly the centroids
        regions = regionprops(label_img)

        # Extract centroids
        centroids = np.zeros((len(regions), 2))
        for idx, props in enumerate(regions):
            centroids[idx, 0] = props.centroid[0]  # Y-coordinate
            centroids[idx, 1] = props.centroid[1]  # X-coordinate

        return centroids

    def segment_cells(self, image_path: Path, save_path: Path, mask_save_fn: Path, chan: str, seg_im_suffix: str) -> None:
        """
        Perform segmentation on a single image and save the results.

        Parameters:
            image_path (Path): Path to the image file.
            save_path (Path): Path to save segmentation results.
            mask_save_fn (Path): Path to save segmentation mask.
            chan (str): Channel identifier.
            seg_im_suffix (str): Image suffix.
        """
        image = self.load_image(image_path, chan)
        preprocessed_image = self.preprocess_image(image)
        segmented_image = self.segment_image(preprocessed_image)
        seg_cells = self.find_centroids(segmented_image)
        if self.general_params['regi_bool']:
            seg_cells = self.exclude_segment_objects(image_path.stem, segmented_image, seg_cells, seg_im_suffix)
        # csv_to_save = pd.DataFrame({'Position Y': idx[0], 'Position X': idx[1]})
        csv_to_save = pd.DataFrame(seg_cells, columns=["Position Y", "Position X"])
        self.save_to_csv(csv_to_save, save_path)

        # Create a binary image with only centroid points for further visualization
        # centroid_binary = np.zeros(segmented_image.shape, dtype='uint8')
        # centroids_int = np.round(seg_cells).astype(int)
        # for val in centroids_int:
        #     centroid_binary[val[0], val[1]] = 255
        # Save centroid binary image
        # centroid_image_save_path = mask_image_path.with_name(f"{mask_image_path.stem}_centroids.tif")
        # print(mask_save_fn)
        # cv2.imwrite(str(mask_save_fn), centroid_binary)


    def process_images(self, progress_callback: Union[None, callable] = None) -> None:
        """
        Process all images for segmentation.

        Parameters:
            progress_callback (Union[None, callable]): Callback function for progress updates.
        """
        self.load_registration_data()
        if self.cells_params['single_channel']:
            im_class = 'single_channel'
        else:
            im_class = 'rgb'
        total_images = 0
        for chan in self.general_params["channels"]:
            _, seg_im_list, _ = self.load_image_list(chan, im_class)
            total_images += len(seg_im_list)

        processed_images = 0
        for chan in self.general_params["channels"]:
            mask_dir, output_dir = self.prepare_segmentation_folders(self.cells_params["mask_folder"], chan)
            seg_im_dir, seg_im_list, seg_im_suffix = self.load_image_list(chan, im_class)

            for im_name in seg_im_list:
                path_to_im = seg_im_dir.joinpath(im_name)
                save_path = output_dir.joinpath(f"{im_name.split('.')[0]}_{self.general_params['seg_type']}.csv")
                mask_save_fn = mask_dir.joinpath(im_name[:-len(seg_im_suffix)] + '_masks.tiff')
                self.segment_cells(path_to_im, save_path, mask_save_fn, chan, seg_im_suffix)
                processed_images += 1
                if progress_callback:
                    progress = int((processed_images / total_images) * 100)
                    progress_callback(progress)

    def process_images_parallel(self, max_workers: int = 4, progress_callback: Union[None, callable] = None) -> None:
        """
        Process images in parallel.

        Parameters:
            max_workers (int): Number of parallel workers.
            progress_callback (Union[None, callable]): Callback function for progress updates.
        """
        tasks = []
        self.load_registration_data()
        im_class = 'single_channel' if self.cells_params['single_channel'] else 'rgb'
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Determine if we're using single-channel or RGB images

            # Load the image lists for all channels
            for chan in self.general_params["channels"]:
                mask_dir, output_dir = self.prepare_segmentation_folders(self.cells_params["mask_folder"], chan)
                seg_im_dir, seg_im_list, seg_im_suffix = self.load_image_list(chan, im_class)

                # Submit each image processing task to the executor
                for im_name in seg_im_list:
                    image_path = seg_im_dir.joinpath(im_name)
                    save_path = output_dir.joinpath(f"{im_name.split('.')[0]}_{self.general_params['seg_type']}.csv")
                    mask_save_fn = mask_dir.joinpath(im_name[:-len(seg_im_suffix)] + '_masks.tiff')
                    tasks.append(executor.submit(self.segment_cells, image_path, save_path, mask_save_fn, chan, seg_im_suffix))

            # Monitor the progress as tasks complete
            for idx, future in enumerate(as_completed(tasks)):
                try:
                    # Retrieve the result or handle any exception raised during processing
                    future.result()
                    if progress_callback:
                        progress = int((idx + 1) / len(tasks) * 100)
                        progress_callback(progress)
                except Exception as e:
                    print(f"Error processing image: {e}")

#%%
class ProjectionSegmenter(PreSegmenter):
    """
    Class for performing projection segmentation tasks.
    """
    def __init__(self, input_path: Path, general_params: dict, projection_params: Union[dict, None] = None) -> None:
        """
        Initialize the ProjectionSegmenter.

        Parameters:
            input_path (Path): Path to the input directory.
            general_params (dict): General configuration parameters.
            projection_params (Union[dict, None]): Parameters specific to projection segmentation.
        """
        super().__init__(input_path, general_params)
        self.projection_params = projection_params


    def segment_projection(self, image_path: Path, save_path: Path, binary_suffix: str) -> None:
        """
        Perform segmentation on a single projection image and save the results.

        Parameters:
            image_path (Path): Path to the image file.
            save_path (Path): Path to save segmentation results.
            binary_suffix (str): Suffix of the binary image.
        """
        # Load the image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) # self.load_image(image_path, single_channel=True, structure_channel=0)

        # Find positions of segmented regions (non-zero pixels)
        idx = np.where(image == 255)
        seg_proj = np.zeros((len(idx[0]), 2))
        for i, (y, x) in enumerate(zip(idx[0], idx[1])):
            seg_proj[i, 0] = y  # Y-coordinate
            seg_proj[i, 1] = x  # X-coordinate
        if self.general_params["regi_bool"]:
            seg_proj = self.exclude_segment_objects(image_path.stem, image, seg_proj, binary_suffix)

        # Save results
        csv_to_save = pd.DataFrame(seg_proj, columns=["Position Y", "Position X"])
        self.save_to_csv(csv_to_save, save_path)


    def process_images(self, progress_callback: Union[None, callable] = None) -> None:
        """
        Process all projection images for segmentation.

        Parameters:
               progress_callback (Union[None, callable]): Callback function for progress updates.

        """
        self.load_registration_data()
        im_class = self.projection_params['binary_folder']
        total_images = 0
        for chan in self.general_params["channels"]:
            _, seg_im_list, _ = self.load_image_list(chan, self.projection_params['binary_folder'])
            total_images += len(seg_im_list)

        processed_images = 0
        for chan in self.general_params["channels"]:
            output_dir = self.prepare_segmentation_folders(self.projection_params["binary_folder"], chan)
            binary_dir, binary_images, binary_suffix = self.load_image_list(chan, im_class)
            for im_name in binary_images:
                path_to_im = binary_dir.joinpath(im_name)
                save_path = output_dir.joinpath(f"{im_name.split('.')[0]}_{self.general_params['seg_type']}.csv")
                self.segment_projection(path_to_im, save_path, binary_suffix)
                processed_images += 1
                if progress_callback:
                    progress = int((processed_images / total_images) * 100)
                    progress_callback(progress)

    def process_images_parallel(self, max_workers=4, progress_callback: Union[None, callable] =None) -> None:
        """
        Process projection images in parallel using multiple CPU cores.

        Parameters:
            max_workers (int): Number of worker processes to run in parallel.
            progress_callback (Union[None, callable]): Callback function to report progress.
        """
        tasks = []

        im_class = self.projection_params['binary_folder']
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            # Load the image lists for all channels
            for chan in self.general_params["channels"]:
                output_dir = self.prepare_segmentation_folders(self.projection_params["binary_folder"], chan)
                binary_dir, binary_images, binary_suffix = self.load_image_list(chan, im_class)

                # Submit each image processing task to the executor
                for im_name in binary_images:
                    self.load_registration_data()
                    image_path = binary_dir.joinpath(im_name)
                    save_path = output_dir.joinpath(f"{im_name.split('.')[0]}_{self.general_params['seg_type']}.csv")
                    tasks.append(executor.submit(self.segment_projection, image_path, save_path, binary_suffix))

            # Monitor the progress as tasks complete
            for idx, future in enumerate(as_completed(tasks)):
                try:
                    # Retrieve the result or handle any exception raised during processing
                    future.result()
                    if progress_callback:
                        progress = int((idx + 1) / len(tasks) * 100)
                        progress_callback(progress)
                except Exception as e:
                    print(f"Error processing image: {e}")