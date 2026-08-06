from pathlib import Path

import numpy as np
import pytest

from napari_dmc_brainmap.segment.processing.native_presegmentation import (
    _maximum_removed_area,
    run_native_presegmentation,
    segment_preprocessed_image,
)


@pytest.mark.parametrize(
    ("fixture_name", "single_channel"),
    [
        ("grayscale_uint16_tiff", True),
        ("grayscale_uint8_tiff", True),
        ("rgb_uint8_tiff", False),
    ],
)
def test_native_presegmentation_matches_golden_output(
    fixture_name: str,
    single_channel: bool,
    request: pytest.FixtureRequest,
    presegmentation_params: dict[str, object],
    presegmentation_golden_output: tuple[np.ndarray, np.ndarray],
) -> None:
    image_path, _ = request.getfixturevalue(fixture_name)
    expected_mask, expected_centroids = presegmentation_golden_output

    mask, centroids = run_native_presegmentation(
        Path(image_path),
        channel="cy3",
        single_channel=single_channel,
        params=presegmentation_params,
    )

    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_equal(centroids, expected_centroids)


@pytest.mark.parametrize(
    ("minimum_area", "maximum_removed_area"),
    [(0, 0), (1, 0), (3, 2), (5, 4)],
)
def test_minimum_area_is_translated_to_inclusive_removal_limit(
    minimum_area: int,
    maximum_removed_area: int,
) -> None:
    assert _maximum_removed_area(minimum_area) == maximum_removed_area


def test_minimum_area_keeps_component_at_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detected = np.zeros((1, 4, 9), dtype=np.float32)
    detected[0, 1, 1:3] = 1
    detected[0, 1, 5:8] = 1
    monkeypatch.setattr(
        "napari_dmc_brainmap.segment.processing.native_presegmentation."
        "gaussian_laplace",
        lambda image, sigma: -detected,
    )

    mask = segment_preprocessed_image(
        detected,
        sigma=1,
        cutoff=0.5,
        hole_sizes=(0, 0),
        minimum_area=3,
    )

    expected = np.zeros((4, 9), dtype=np.uint8)
    expected[1, 5:8] = 255
    np.testing.assert_array_equal(mask, expected)
