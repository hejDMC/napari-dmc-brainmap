"""Presegmentation primitives implemented without AICS dependencies."""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects


RGB_SAMPLE_INDEX = {"cy3": 0, "green": 1, "dapi": 2}


def load_presegmentation_image(
    image_path: Path,
    channel: str,
    single_channel: bool,
) -> np.ndarray:
    """Load a grayscale or RGB TIFF as the duplicated ZYX stack used today."""
    image = tifffile.imread(image_path)
    if single_channel:
        if image.ndim != 2:
            raise ValueError(
                f"Expected a 2D grayscale TIFF, got shape {image.shape}"
            )
        selected = image
    else:
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(
                "Expected an RGB/RGBA TIFF with shape (Y, X, 3/4), "
                f"got {image.shape}"
            )
        try:
            selected = image[..., RGB_SAMPLE_INDEX[channel]]
        except KeyError as error:
            supported = ", ".join(RGB_SAMPLE_INDEX)
            raise ValueError(
                f"Unsupported RGB channel {channel!r}; expected one of "
                f"{supported}"
            ) from error

    duplicated = np.empty((2, *selected.shape), dtype=np.float32)
    duplicated[0] = selected
    duplicated[1] = selected
    return duplicated


def normalize_intensity(
    image: np.ndarray,
    scaling_param: Sequence[float],
) -> np.ndarray:
    """Normalize using the current min/max and standard-deviation rules."""
    if len(scaling_param) == 1:
        upper_bound = scaling_param[0]
        if upper_bound >= 1:
            image = image.copy()
            image[image > upper_bound] = image.min()
        stretch_min = image.min()
        stretch_max = image.max()
    elif len(scaling_param) == 2:
        mean = image.mean()
        standard_deviation = image.std()
        stretch_min = max(
            mean - scaling_param[0] * standard_deviation,
            image.min(),
        )
        stretch_max = min(
            mean + scaling_param[1] * standard_deviation,
            image.max(),
        )
    elif len(scaling_param) == 4:
        valid = image[
            (image > scaling_param[2]) & (image < scaling_param[3])
        ]
        if valid.size == 0:
            raise ValueError(
                "The intensity normalization bounds do not include any pixels"
            )
        standard_deviation = valid.std()
        stretch_min = max(
            scaling_param[2] - scaling_param[0] * standard_deviation,
            image.min(),
        )
        stretch_max = min(
            scaling_param[3] + scaling_param[1] * standard_deviation,
            image.max(),
        )
    else:
        raise ValueError(
            "Intensity normalization expects 1, 2, or 4 values"
        )

    if stretch_min > stretch_max:
        raise ValueError(
            "Invalid intensity normalization range: "
            f"{stretch_min} > {stretch_max}"
        )
    clipped = np.clip(image, stretch_min, stretch_max)
    return (clipped - stretch_min + 1e-8) / (
        stretch_max - stretch_min + 1e-8
    )


def smooth_slices(
    image: np.ndarray,
    sigma: float,
    truncate: float = 3.0,
) -> np.ndarray:
    """Apply 2D Gaussian smoothing independently to each Z slice."""
    smoothed = np.empty_like(image)
    for z_index, image_slice in enumerate(image):
        smoothed[z_index] = gaussian_filter(
            image_slice,
            sigma=sigma,
            mode="nearest",
            truncate=truncate,
        )
    return smoothed


def _fill_holes_by_size(
    binary: np.ndarray,
    minimum_size: int,
    maximum_size: int,
) -> np.ndarray:
    filled = binary.copy()
    for z_index, image_slice in enumerate(binary):
        background_labels = label(~image_slice, connectivity=1)
        component_sizes = np.bincount(background_labels.ravel())
        fill_labels = (
            (component_sizes >= minimum_size)
            & (component_sizes <= maximum_size)
        )
        fill_labels[0] = False
        filled[z_index] |= fill_labels[background_labels]
    return filled


def segment_preprocessed_image(
    image: np.ndarray,
    sigma: float,
    cutoff: float,
    hole_sizes: Sequence[int],
    minimum_area: int,
) -> np.ndarray:
    """Detect dots and return the first duplicated slice as a uint8 mask."""
    response = -(sigma**2) * gaussian_laplace(image, sigma)
    binary = response > cutoff
    filled = _fill_holes_by_size(
        binary,
        minimum_size=hole_sizes[0],
        maximum_size=hole_sizes[1],
    )
    segmented = remove_small_objects(
        filled,
        min_size=minimum_area,
        connectivity=1,
    )
    return segmented[0].astype(np.uint8) * 255


def find_mask_centroids(mask: np.ndarray) -> np.ndarray:
    """Return connected-component centroids in YX order."""
    regions = regionprops(label(mask))
    return np.asarray(
        [region.centroid for region in regions],
        dtype=np.float64,
    ).reshape((-1, 2))


def run_native_presegmentation(
    image_path: Path,
    channel: str,
    single_channel: bool,
    params: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Run the complete non-AICS presegmentation candidate."""
    loaded = load_presegmentation_image(
        image_path,
        channel=channel,
        single_channel=single_channel,
    )
    normalized = normalize_intensity(
        loaded,
        scaling_param=params["intensity_norm"],
    )
    smoothed = smooth_slices(
        normalized,
        sigma=params["gaussian_smoothing_sigma"],
    )
    mask = segment_preprocessed_image(
        smoothed,
        sigma=params["dot_3d_sigma"],
        cutoff=params["dot_3d_cutoff"],
        hole_sizes=params["hole_min_max"],
        minimum_area=params["minArea"],
    )
    return mask, find_mask_centroids(mask)
