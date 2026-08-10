import numpy as np
import pytest

from napari_dmc_brainmap.stitching.layout import (
    AcquisitionOrder,
    TILE_OVERLAP_PX,
    TILE_SIZE_PX,
    TILE_STRIDE_PX,
    acquisition_cells,
    classify_acquisition_order,
    describe_layout,
    layout_from_grid_metadata,
    layout_from_stage_positions,
    legacy_serpentine_index_grid,
)


def test_rigid_stitching_geometry_is_fixed():
    assert TILE_SIZE_PX == 2048
    assert TILE_OVERLAP_PX == 205
    assert TILE_STRIDE_PX == 1843


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1, 1, [[0]]),
        (3, 2, [[3, 4, 5], [2, 1, 0]]),
        (3, 3, [[8, 7, 6], [3, 4, 5], [2, 1, 0]]),
        (2, 4, [[6, 7], [5, 4], [2, 3], [1, 0]]),
    ],
)
def test_legacy_serpentine_grid_is_preserved(width, height, expected):
    np.testing.assert_array_equal(
        legacy_serpentine_index_grid(width, height),
        np.asarray(expected),
    )


@pytest.mark.parametrize("order", list(AcquisitionOrder))
def test_all_imagej_orders_round_trip_through_classifier(order):
    cells = acquisition_cells(4, 3, order)

    detected, match = classify_acquisition_order(cells, 4, 3)

    assert detected is order
    assert match == 1.0


def test_incomplete_layout_is_reported_as_unknown():
    cells = acquisition_cells(
        4,
        3,
        AcquisitionOrder.SNAKE_ROWS_RIGHT_DOWN,
    )[:-1]

    detected, match = classify_acquisition_order(cells, 4, 3)

    assert detected is None
    assert 0.0 < match < 1.0


def test_indistinguishable_single_tile_order_is_reported_as_unknown():
    detected, match = classify_acquisition_order([(0, 0)], 1, 1)

    assert detected is None
    assert match == 1.0


def test_stage_coordinates_are_clustered_and_legacy_orientation_is_applied():
    raw_cells = acquisition_cells(
        3,
        2,
        AcquisitionOrder.SNAKE_ROWS_RIGHT_DOWN,
    )
    positions = [
        (
            column * 1179.648 + row * 0.01,
            row * 1179.648 + column * 0.01,
        )
        for row, column in raw_cells
    ]

    layout = layout_from_stage_positions(positions)

    assert layout.width == 3
    assert layout.height == 2
    assert layout.acquisition_order is AcquisitionOrder.SNAKE_ROWS_RIGHT_DOWN
    assert layout.output_order is AcquisitionOrder.SNAKE_ROWS_LEFT_UP
    actual_grid = np.empty((layout.height, layout.width), dtype=int)
    for placement in layout.placements:
        actual_grid[placement.row, placement.column] = placement.source_index
    np.testing.assert_array_equal(
        actual_grid,
        legacy_serpentine_index_grid(3, 2),
    )


def test_micro_manager_grid_coordinates_preserve_legacy_orientation():
    stage_positions = [
        {"Label": "Pos0", "GridRow": 0, "GridCol": 0},
        {"Label": "Pos1", "GridRow": 0, "GridCol": 1},
        {"Label": "Pos2", "GridRow": 1, "GridCol": 1},
        {"Label": "Pos3", "GridRow": 1, "GridCol": 0},
    ]

    layout = layout_from_grid_metadata(stage_positions)

    assert layout.acquisition_order is AcquisitionOrder.SNAKE_ROWS_RIGHT_DOWN
    assert layout.output_order is AcquisitionOrder.SNAKE_ROWS_LEFT_UP
    assert {
        placement.source_index: (placement.row, placement.column)
        for placement in layout.placements
    } == {0: (1, 1), 1: (1, 0), 2: (0, 0), 3: (0, 1)}


def test_micro_manager_placement_does_not_depend_on_metadata_list_order():
    stage_positions = [
        {"Label": "Pos2", "GridRow": 1, "GridCol": 1},
        {"Label": "Pos0", "GridRow": 0, "GridCol": 0},
        {"Label": "Pos3", "GridRow": 1, "GridCol": 0},
        {"Label": "Pos1", "GridRow": 0, "GridCol": 1},
    ]

    layout = layout_from_grid_metadata(stage_positions)

    assert layout.acquisition_order is None
    assert {
        placement.source_index: (placement.row, placement.column)
        for placement in layout.placements
    } == {0: (0, 0), 1: (1, 1), 2: (0, 1), 3: (1, 0)}


def test_duplicate_grid_positions_are_rejected():
    stage_positions = [
        {"GridRow": 0, "GridCol": 0},
        {"GridRow": 0, "GridCol": 0},
    ]

    with pytest.raises(ValueError, match="duplicate grid positions"):
        layout_from_grid_metadata(stage_positions)


def test_layout_description_reports_acquisition_and_fixed_geometry():
    positions = [
        (0.0, 0.0),
        (100.0, 0.0),
        (100.0, 100.0),
        (0.0, 100.0),
    ]
    layout = layout_from_stage_positions(positions)

    description = describe_layout(layout, "section 12")

    assert "section 12: 2 × 2 tiles" in description
    assert "acquired: Grid: snake by rows — Right & Down" in description
    assert "placed: Grid: snake by rows — Left & Up" in description
    assert "overlap 205 px" in description
    assert "stride 1843 px" in description
