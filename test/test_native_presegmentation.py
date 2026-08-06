from pathlib import Path

import numpy as np
import pytest

from napari_dmc_brainmap.segment.processing.native_presegmentation import (
    run_native_presegmentation,
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
