import json

import numpy as np
import pytest

from napari_dmc_brainmap.registration.sharpy_track.sharpy_track.model.coordinates import (
    display_to_native_point,
    native_to_display_point,
    scale_coordinate,
)


NATIVE_SIZE = [1140, 800]
DISPLAY_SIZE = [763, 536]


def test_native_to_display_maps_both_axes_end_to_end():
    assert native_to_display_point([0, 0], NATIVE_SIZE, DISPLAY_SIZE) == [
        0.0,
        0.0,
    ]
    assert native_to_display_point(
        [1139, 799], NATIVE_SIZE, DISPLAY_SIZE
    ) == [762.0, 535.0]


def test_display_to_native_maps_both_axes_end_to_end_as_ints():
    point = display_to_native_point(
        [762.0, 535.0], NATIVE_SIZE, DISPLAY_SIZE
    )

    assert point == [1139, 799]
    assert all(type(value) is int for value in point)


def test_subpixel_display_position_is_rounded_only_in_native_space():
    display_point = [123.75, 271.125]
    native_point = display_to_native_point(
        display_point,
        NATIVE_SIZE,
        DISPLAY_SIZE,
    )
    expected = [
        round(display_point[0] * 1139 / 762),
        round(display_point[1] * 799 / 535),
    ]

    assert native_point == expected
    assert all(type(value) is int for value in native_point)


def test_integer_native_points_remain_json_integers():
    point = display_to_native_point(
        [123.75, 271.125],
        NATIVE_SIZE,
        DISPLAY_SIZE,
    )
    decoded = json.loads(json.dumps({"atlasDots": {"0": [point]}}))

    saved_point = decoded["atlasDots"]["0"][0]
    assert saved_point == point
    assert all(type(value) is int for value in saved_point)


@pytest.mark.parametrize(
    "display_size",
    [[912, 640], [763, 536], [570, 400], [376, 264]],
)
def test_native_points_survive_display_round_trip_exactly(display_size):
    for x in range(NATIVE_SIZE[0]):
        display_point = native_to_display_point(
            [x, 0],
            NATIVE_SIZE,
            display_size,
        )
        assert display_to_native_point(
            display_point,
            NATIVE_SIZE,
            display_size,
        ) == [x, 0]

    for y in range(NATIVE_SIZE[1]):
        display_point = native_to_display_point(
            [0, y],
            NATIVE_SIZE,
            display_size,
        )
        assert display_to_native_point(
            display_point,
            NATIVE_SIZE,
            display_size,
        ) == [0, y]


@pytest.mark.parametrize(
    "display_size",
    [[912, 640], [763, 536], [570, 400], [376, 264]],
)
def test_downscaled_round_trip_stays_within_half_a_display_pixel(display_size):
    for display_point in np.linspace(
        [0.0, 0.0],
        [display_size[0] - 1, display_size[1] - 1],
        num=101,
    ):
        native_point = display_to_native_point(
            display_point,
            NATIVE_SIZE,
            display_size,
        )
        restored_display = native_to_display_point(
            native_point,
            NATIVE_SIZE,
            display_size,
        )
        np.testing.assert_allclose(
            restored_display,
            display_point,
            atol=0.5,
        )


def test_scale_coordinate_clamps_to_valid_pixel_endpoints():
    assert scale_coordinate(-10, 1140, 763) == 0.0
    assert scale_coordinate(2000, 1140, 763) == 762.0


@pytest.mark.parametrize(
    ("source_extent", "target_extent"),
    [(0, 10), (10, 0), (-1, 10), (10, -1)],
)
def test_scale_coordinate_rejects_invalid_extents(source_extent, target_extent):
    with pytest.raises(ValueError, match="positive"):
        scale_coordinate(0, source_extent, target_extent)
