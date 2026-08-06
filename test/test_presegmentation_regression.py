from pathlib import Path

import numpy as np
import pytest

from napari_dmc_brainmap.segment.processing.presegmentation_tools import (
    CellsSegmenter,
)


@pytest.mark.parametrize(
    ("fixture_name", "single_channel"),
    [
        ("grayscale_uint16_tiff", True),
        ("grayscale_uint8_tiff", True),
        ("rgb_uint8_tiff", False),
    ],
)
def test_presegmentation_mask_and_centroids_match_golden_output(
    fixture_name: str,
    single_channel: bool,
    request: pytest.FixtureRequest,
    presegmentation_params: dict[str, object],
    presegmentation_golden_output: tuple[np.ndarray, np.ndarray],
) -> None:
    image_path, _ = request.getfixturevalue(fixture_name)
    expected_mask, expected_centroids = presegmentation_golden_output
    segmenter = CellsSegmenter(
        input_path=Path(),
        general_params={},
        cells_params={"single_channel": single_channel},
        preseg_params=presegmentation_params,
    )

    loaded = segmenter.load_image(image_path, chan="cy3")
    preprocessed = segmenter.preprocess_image(loaded)
    mask = segmenter.segment_image(preprocessed)
    centroids = segmenter.find_centroids(mask)

    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(centroids, expected_centroids)
