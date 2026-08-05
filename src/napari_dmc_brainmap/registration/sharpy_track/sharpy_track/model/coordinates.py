"""Coordinate conversion between native atlas pixels and display pixels."""

from collections.abc import Sequence

import numpy as np


def scale_coordinate(
    value: float,
    source_extent: int,
    target_extent: int,
) -> float:
    """Map one coordinate between endpoint-aligned pixel coordinate systems."""
    if source_extent < 1 or target_extent < 1:
        raise ValueError("Coordinate extents must be positive integers.")
    if source_extent == 1 or target_extent == 1:
        return 0.0

    scaled = float(value) * (target_extent - 1) / (source_extent - 1)
    return min(max(scaled, 0.0), float(target_extent - 1))


def native_to_display_point(
    point: Sequence[float],
    native_size: Sequence[int],
    display_size: Sequence[int],
) -> list[float]:
    """Convert an ``[x, y]`` native point to a floating display point."""
    return [
        scale_coordinate(point[0], native_size[0], display_size[0]),
        scale_coordinate(point[1], native_size[1], display_size[1]),
    ]


def display_to_native_point(
    point: Sequence[float],
    native_size: Sequence[int],
    display_size: Sequence[int],
) -> list[int]:
    """Convert an ``[x, y]`` display point to integer native coordinates."""
    return [
        int(round(scale_coordinate(point[0], display_size[0], native_size[0]))),
        int(round(scale_coordinate(point[1], display_size[1], native_size[1]))),
    ]


def coordinate_scale_matrix(
    source_size: Sequence[int],
    target_size: Sequence[int],
) -> np.ndarray:
    """Return a 3x3 endpoint-aligned native/display scaling matrix."""
    if any(extent < 2 for extent in [*source_size, *target_size]):
        raise ValueError(
            "Homography coordinate extents must each contain at least 2 pixels."
        )
    scale_x = scale_coordinate(1.0, source_size[0], target_size[0])
    scale_y = scale_coordinate(1.0, source_size[1], target_size[1])
    return np.array(
        [
            [scale_x, 0.0, 0.0],
            [0.0, scale_y, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def homography_to_display_space(
    native_transform: np.ndarray,
    sample_native_size: Sequence[int],
    sample_display_size: Sequence[int],
    reference_native_size: Sequence[int],
    reference_display_size: Sequence[int],
) -> np.ndarray:
    """Convert a sample-to-reference homography from native to display space."""
    transform = np.asarray(native_transform, dtype=np.float64).reshape(3, 3)
    sample_scale = coordinate_scale_matrix(
        sample_native_size,
        sample_display_size,
    )
    reference_scale = coordinate_scale_matrix(
        reference_native_size,
        reference_display_size,
    )
    display_transform = reference_scale @ transform @ np.linalg.inv(sample_scale)
    if not np.isclose(display_transform[2, 2], 0.0):
        display_transform /= display_transform[2, 2]
    return display_transform
