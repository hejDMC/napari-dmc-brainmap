import cv2
import numpy as np
import pandas as pd
from typing import List, Union
from pathlib import Path
from skimage.measure import label, regionprops
from natsort import natsorted
from napari_dmc_brainmap.utils.path_utils import get_info
from concurrent.futures import ProcessPoolExecutor, as_completed

class CentroidFinder:
    """
    Class for finding centroids in segmentation mask images and saving them as CSV files.
    """
    def __init__(
        self, input_path: Path, mask_folder: str, output_folder: str, channels: List[str], mask_type: str = "cells"
    ) -> None:
        """
        Initialize the CentroidFinder.

        Parameters:
            input_path (Path): Path to the input directory containing images.
            mask_folder (str): Name of the folder containing mask images.
            output_folder (str): Name of the output folder for saving centroid CSVs.
            channels (List[str]): List of channels to process.
            mask_type (str): Type of segmentation mask ('cells' by default).
        """
        self.input_path = input_path
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.channels = channels
        self.mask_type = mask_type

    def find_centroids_for_image(self, mask_image_path: Path, save_path: Path) -> None:
        """
        Find centroids for a given mask image and save results as a CSV.

        Parameters:
            mask_image_path (Path): Path to the mask image file.
            save_path (Path): Path to save the centroid CSV file.
        """
        # Load the mask image
        image = cv2.imread(str(mask_image_path), cv2.IMREAD_GRAYSCALE)

        # Label the image to find individual segmented regions
        label_img = label(image)

        # Get properties of segmented regions, particularly the centroids
        regions = regionprops(label_img)

        # Extract centroids
        centroids = np.zeros((len(regions), 2))
        for idx, props in enumerate(regions):
            centroids[idx, 0] = props.centroid[0]  # Y-coordinate
            centroids[idx, 1] = props.centroid[1]  # X-coordinate

        # Save the centroids as a CSV
        csv_to_save = pd.DataFrame(centroids, columns=["Position Y", "Position X"])
        csv_to_save.to_csv(save_path, index=False)
        # Create a binary image with only centroid points for further visualization
        centroid_binary = np.zeros(image.shape, dtype='uint8')
        centroids_int = np.round(centroids).astype(int)
        for val in centroids_int:
            centroid_binary[val[0], val[1]] = 255

        # Save centroid binary image
        centroid_image_save_path = mask_image_path.with_name(f"{mask_image_path.stem}_centroids.tif")
        cv2.imwrite(str(centroid_image_save_path), centroid_binary)

    def process_all_masks(self, progress_callback: Union[None, callable] = None) -> None:
        """
        Iterate over all mask images in the specified folder and process them to find centroids.

        Parameters:
            progress_callback (Union[None, callable], optional): Callback function for updating progress. Defaults to None.
        """
        total_images = 0
        for chan in self.channels:
            mask_dir = get_info(self.input_path, self.mask_folder, seg_type=self.mask_type, channel=chan, only_dir=True)
            total_images += len([im for im in mask_dir.glob('*.tiff')])

        processed_images = 0
        for chan in self.channels:
            # Set up directories
            mask_dir = get_info(self.input_path, self.mask_folder, seg_type=self.mask_type, channel=chan, only_dir=True)
            output_dir = get_info(self.input_path, self.output_folder, seg_type=self.mask_type, channel=chan, create_dir=True,
                                  only_dir=True)

            mask_images = natsorted([im for im in mask_dir.glob('*.tiff')])
            for mask_image_path in mask_images:
                print(f"Processing mask image: {mask_image_path}")
                save_path = output_dir.joinpath(f"{mask_image_path.stem}_{self.mask_type}.csv")
                self.find_centroids_for_image(mask_image_path, save_path)

                processed_images += 1
                if progress_callback:
                    progress = int((processed_images / total_images) * 100)
                    progress_callback(progress)

    def process_all_masks_parallel(
        self, max_workers: int = 4, progress_callback: Union[None, callable] = None
    ) -> None:
        """
        Parallel processing of mask images using multiple CPU cores.

        Parameters:
            max_workers (int, optional): Maximum number of parallel workers. Defaults to 4.
            progress_callback (Union[None, callable], optional): Callback function for updating progress. Defaults to None.
        """
        tasks = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chan in self.channels:
                mask_dir = get_info(self.input_path, self.mask_folder, seg_type=self.mask_type, channel=chan, only_dir=True)
                output_dir = get_info(self.input_path, self.output_folder, seg_type=self.mask_type, channel=chan, create_dir=True, only_dir=True)

                mask_images = natsorted([im for im in mask_dir.glob('*.tiff')])
                for mask_image_path in mask_images:
                    save_path = output_dir.joinpath(f"{mask_image_path.stem}_{self.mask_type}.csv")
                    # Submit each image processing task to the executor
                    tasks.append(executor.submit(self.find_centroids_for_image, mask_image_path, save_path))
            for idx, future in enumerate(as_completed(tasks)):
                try:
                    future.result()  # this will raise any exception that happened during processing
                    if progress_callback:
                        progress = int((idx + 1) / len(tasks) * 100)
                        progress_callback(progress)
                except Exception as e:
                    print(f"Error processing image: {e}")