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


@pytest.mark.parametrize(
    ("fixture_name", "single_channel"),
    [
        ("grayscale_uint16_tiff", True),
        ("grayscale_uint8_tiff", True),
        ("rgb_uint8_tiff", False),
    ],
)
def test_min_area_3_matches_legacy_duplicate_stack_min_area_5(
    fixture_name: str,
    single_channel: bool,
    request: pytest.FixtureRequest,
    presegmentation_params: dict[str, object],
) -> None:
    image_path, _ = request.getfixturevalue(fixture_name)
    current_segmenter = CellsSegmenter(
        input_path=Path(),
        general_params={},
        cells_params={"single_channel": single_channel},
        preseg_params=presegmentation_params,
    )
    legacy_params = {**presegmentation_params, "minArea": 5}
    legacy_segmenter = CellsSegmenter(
        input_path=Path(),
        general_params={},
        cells_params={"single_channel": single_channel},
        preseg_params=legacy_params,
    )

    singleton = current_segmenter.load_image(image_path, chan="cy3")
    duplicated = np.repeat(singleton, 2, axis=0)
    current_mask = current_segmenter.segment_image(
        current_segmenter.preprocess_image(singleton)
    )
    legacy_mask = legacy_segmenter.segment_image(
        legacy_segmenter.preprocess_image(duplicated)
    )

    np.testing.assert_array_equal(current_mask, legacy_mask)
    np.testing.assert_array_equal(
        current_segmenter.find_centroids(current_mask),
        legacy_segmenter.find_centroids(legacy_mask),
    )
