from pathlib import Path

import numpy as np
import pytest

from napari_dmc_brainmap.segment.processing.presegmentation_tools import (
    CellsSegmenter,
)


@pytest.mark.parametrize(
    "fixture_name",
    ["grayscale_uint16_tiff", "grayscale_uint8_tiff"],
)
def test_load_single_channel_tiff_preserves_intensities(
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    image_path, source_image = request.getfixturevalue(fixture_name)
    segmenter = CellsSegmenter(
        input_path=Path(),
        general_params={},
        cells_params={"single_channel": True},
        preseg_params={},
    )

    loaded = segmenter.load_image(image_path, chan="cy3")

    expected = source_image[np.newaxis, ...].astype(np.float32)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, expected)


@pytest.mark.parametrize(
    ("channel", "sample_index"),
    [("cy3", 0), ("green", 1), ("dapi", 2)],
)
def test_load_rgb_uint8_tiff_selects_requested_sample_channel(
    rgb_uint8_tiff: tuple[Path, np.ndarray],
    channel: str,
    sample_index: int,
) -> None:
    image_path, source_image = rgb_uint8_tiff
    segmenter = CellsSegmenter(
        input_path=Path(),
        general_params={},
        cells_params={"single_channel": False},
        preseg_params={},
    )

    loaded = segmenter.load_image(image_path, chan=channel)

    selected_channel = source_image[..., sample_index]
    expected = selected_channel[np.newaxis, ...].astype(np.float32)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, expected)
