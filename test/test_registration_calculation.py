import numpy as np
import pytest

from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.calculation import (
    fitGeoTrans,
    mapPointTransform,
    predictPointSample,
)


SAMPLE_POINTS = np.array(
    [[0, 0], [10, 0], [10, 10], [0, 10]],
    dtype=np.float32,
)
ATLAS_POINTS = np.array(
    [[2, 3], [14, 2], [13, 15], [1, 12]],
    dtype=np.float32,
)


def test_fit_projective_transform_maps_control_points() -> None:
    transform = fitGeoTrans(SAMPLE_POINTS, ATLAS_POINTS)
    mapped = np.array(
        [mapPointTransform(x, y, transform) for x, y in SAMPLE_POINTS]
    )

    np.testing.assert_allclose(mapped, ATLAS_POINTS, atol=1e-5)


def test_reverse_projective_transform_maps_atlas_to_sample() -> None:
    reverse_transform = fitGeoTrans(ATLAS_POINTS, SAMPLE_POINTS)
    mapped = np.array(
        [
            predictPointSample(x, y, reverse_transform)
            for x, y in ATLAS_POINTS
        ]
    )

    np.testing.assert_allclose(mapped, SAMPLE_POINTS, atol=1e-5)


def test_separately_fitted_transforms_round_trip_non_control_points() -> None:
    forward_transform = fitGeoTrans(SAMPLE_POINTS, ATLAS_POINTS)
    reverse_transform = fitGeoTrans(ATLAS_POINTS, SAMPLE_POINTS)
    sample_points = np.array(
        [[2.5, 3.25], [5.0, 5.0], [8.25, 6.75]],
        dtype=np.float64,
    )

    atlas_points = np.array(
        [
            mapPointTransform(x, y, forward_transform)
            for x, y in sample_points
        ]
    )
    restored_sample_points = np.array(
        [
            predictPointSample(x, y, reverse_transform)
            for x, y in atlas_points
        ]
    )

    np.testing.assert_allclose(
        restored_sample_points,
        sample_points,
        atol=1e-5,
    )


def test_fit_projective_transform_rejects_degenerate_points() -> None:
    source = np.ones((4, 2), dtype=np.float32)
    destination = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="Could not estimate"):
        fitGeoTrans(source, destination)
