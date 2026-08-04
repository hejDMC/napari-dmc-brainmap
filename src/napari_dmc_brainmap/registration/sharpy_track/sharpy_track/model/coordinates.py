"""Coordinate conversion between native atlas pixels and display pixels."""

from collections.abc import Sequence


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
