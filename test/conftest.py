from pathlib import Path

import numpy as np
import pytest
import tifffile


def _spot_image_uint8() -> np.ndarray:
    """Return a deterministic synthetic image with three bright spots."""
    y, x = np.indices((32, 32))
    image = (12 + ((3 * y + 5 * x) % 16)).astype(np.uint8)
    image[(y - 9) ** 2 + (x - 10) ** 2 <= 3**2] = 220
    image[(y - 23) ** 2 + (x - 22) ** 2 <= 4**2] = 180
    image[4, 27] = np.iinfo(np.uint8).max
    return image


@pytest.fixture
def grayscale_uint16_tiff(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A small 16-bit grayscale TIFF with values spanning the full range."""
    image = _spot_image_uint8().astype(np.uint16) * 257
    image_path = tmp_path / "grayscale_uint16.tif"
    tifffile.imwrite(image_path, image, photometric="minisblack")
    return image_path, image


@pytest.fixture
def grayscale_uint8_tiff(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A small 8-bit grayscale TIFF with values spanning the full range."""
    image = _spot_image_uint8()
    image_path = tmp_path / "grayscale_uint8.tif"
    tifffile.imwrite(image_path, image, photometric="minisblack")
    return image_path, image


@pytest.fixture
def rgb_uint8_tiff(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A small RGB TIFF whose three sample channels are easy to distinguish."""
    red = _spot_image_uint8()
    green = np.roll(red, 3, axis=1)
    blue = 255 - red
    image = np.stack((red, green, blue), axis=-1)
    image_path = tmp_path / "rgb_uint8.tif"
    tifffile.imwrite(image_path, image, photometric="rgb")
    return image_path, image


@pytest.fixture
def presegmentation_params() -> dict[str, object]:
    """The current presegmentation defaults whose behavior is under test."""
    return {
        "intensity_norm": [0.5, 17.5],
        "gaussian_smoothing_sigma": 1,
        "dot_3d_sigma": 1,
        "dot_3d_cutoff": 0.03,
        "hole_min_max": [0, 81],
        "minArea": 3,
    }


@pytest.fixture
def presegmentation_golden_output() -> tuple[np.ndarray, np.ndarray]:
    """Expected mask and YX centroids from the current AICS pipeline."""
    mask_rows = (
        "................................",
        "................................",
        "................................",
        "...........................#....",
        "..........................###...",
        "...........................#....",
        "..........#.....................",
        "........#####...................",
        "........#####...................",
        ".......#######..................",
        "........#####...................",
        "........#####...................",
        "..........#.....................",
        "................................",
        "................................",
        "................................",
        "................................",
        "................................",
        "................................",
        "................................",
        "....................#####.......",
        "...................#######......",
        "...................#######......",
        "..................#########.....",
        "...................#######......",
        "...................#######......",
        "....................#####.......",
        "................................",
        "................................",
        "................................",
        "................................",
        "................................",
    )
    mask = np.array(
        [[255 if pixel == "#" else 0 for pixel in row] for row in mask_rows],
        dtype=np.uint8,
    )
    centroids = np.array(
        [[4.0, 27.0], [9.0, 10.0], [23.0, 22.0]],
        dtype=np.float64,
    )
    return mask, centroids
