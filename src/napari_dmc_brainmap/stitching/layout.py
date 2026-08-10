"""Tile-grid layout helpers for rigid-overlap stitching."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np


TILE_SIZE_PX = 2048
# This is intentionally a fixed compatibility value, not recalculated at run time.
TILE_OVERLAP_PX = 205
TILE_STRIDE_PX = TILE_SIZE_PX - TILE_OVERLAP_PX


class AcquisitionOrder(str, Enum):
    """ImageJ Grid/Collection acquisition-order names."""

    ROW_RIGHT_DOWN = "Grid: row-by-row — Right & Down"
    ROW_LEFT_DOWN = "Grid: row-by-row — Left & Down"
    ROW_RIGHT_UP = "Grid: row-by-row — Right & Up"
    ROW_LEFT_UP = "Grid: row-by-row — Left & Up"

    COLUMN_DOWN_RIGHT = "Grid: column-by-column — Down & Right"
    COLUMN_DOWN_LEFT = "Grid: column-by-column — Down & Left"
    COLUMN_UP_RIGHT = "Grid: column-by-column — Up & Right"
    COLUMN_UP_LEFT = "Grid: column-by-column — Up & Left"

    SNAKE_ROWS_RIGHT_DOWN = "Grid: snake by rows — Right & Down"
    SNAKE_ROWS_LEFT_DOWN = "Grid: snake by rows — Left & Down"
    SNAKE_ROWS_RIGHT_UP = "Grid: snake by rows — Right & Up"
    SNAKE_ROWS_LEFT_UP = "Grid: snake by rows — Left & Up"

    SNAKE_COLUMNS_DOWN_RIGHT = "Grid: snake by columns — Down & Right"
    SNAKE_COLUMNS_DOWN_LEFT = "Grid: snake by columns — Down & Left"
    SNAKE_COLUMNS_UP_RIGHT = "Grid: snake by columns — Up & Right"
    SNAKE_COLUMNS_UP_LEFT = "Grid: snake by columns — Up & Left"


@dataclass(frozen=True)
class _OrderSpec:
    by_rows: bool
    snake: bool
    first_axis_forward: bool
    second_axis_forward: bool


_ORDER_SPECS = {
    AcquisitionOrder.ROW_RIGHT_DOWN: _OrderSpec(True, False, True, True),
    AcquisitionOrder.ROW_LEFT_DOWN: _OrderSpec(True, False, False, True),
    AcquisitionOrder.ROW_RIGHT_UP: _OrderSpec(True, False, True, False),
    AcquisitionOrder.ROW_LEFT_UP: _OrderSpec(True, False, False, False),
    AcquisitionOrder.COLUMN_DOWN_RIGHT: _OrderSpec(False, False, True, True),
    AcquisitionOrder.COLUMN_DOWN_LEFT: _OrderSpec(False, False, True, False),
    AcquisitionOrder.COLUMN_UP_RIGHT: _OrderSpec(False, False, False, True),
    AcquisitionOrder.COLUMN_UP_LEFT: _OrderSpec(False, False, False, False),
    AcquisitionOrder.SNAKE_ROWS_RIGHT_DOWN: _OrderSpec(True, True, True, True),
    AcquisitionOrder.SNAKE_ROWS_LEFT_DOWN: _OrderSpec(True, True, False, True),
    AcquisitionOrder.SNAKE_ROWS_RIGHT_UP: _OrderSpec(True, True, True, False),
    AcquisitionOrder.SNAKE_ROWS_LEFT_UP: _OrderSpec(True, True, False, False),
    AcquisitionOrder.SNAKE_COLUMNS_DOWN_RIGHT: _OrderSpec(False, True, True, True),
    AcquisitionOrder.SNAKE_COLUMNS_DOWN_LEFT: _OrderSpec(False, True, True, False),
    AcquisitionOrder.SNAKE_COLUMNS_UP_RIGHT: _OrderSpec(False, True, False, True),
    AcquisitionOrder.SNAKE_COLUMNS_UP_LEFT: _OrderSpec(False, True, False, False),
}


@dataclass(frozen=True)
class TilePlacement:
    """Location of one acquisition tile in the output grid."""

    source_index: int
    row: int
    column: int


@dataclass(frozen=True)
class StitchLayout:
    """Detected acquisition geometry and output placement."""

    width: int
    height: int
    placements: tuple[TilePlacement, ...]
    acquisition_order: AcquisitionOrder | None
    output_order: AcquisitionOrder | None
    acquisition_match: float
    output_match: float

    @property
    def tile_count(self) -> int:
        return len(self.placements)


def acquisition_cells(
    width: int,
    height: int,
    order: AcquisitionOrder,
) -> tuple[tuple[int, int], ...]:
    """Return ``(row, column)`` cells in acquisition order."""
    if width < 1 or height < 1:
        raise ValueError("Grid width and height must both be positive.")

    spec = _ORDER_SPECS[order]
    cells: list[tuple[int, int]] = []

    if spec.by_rows:
        rows = list(range(height))
        columns = list(range(width))
        if not spec.second_axis_forward:
            rows.reverse()
        if not spec.first_axis_forward:
            columns.reverse()

        for line_index, row in enumerate(rows):
            line_columns = columns
            if spec.snake and line_index % 2:
                line_columns = list(reversed(columns))
            cells.extend((row, column) for column in line_columns)
    else:
        rows = list(range(height))
        columns = list(range(width))
        if not spec.first_axis_forward:
            rows.reverse()
        if not spec.second_axis_forward:
            columns.reverse()

        for line_index, column in enumerate(columns):
            line_rows = rows
            if spec.snake and line_index % 2:
                line_rows = list(reversed(rows))
            cells.extend((row, column) for row in line_rows)

    return tuple(cells)


def classify_acquisition_order(
    cells: Sequence[tuple[int, int]],
    width: int,
    height: int,
) -> tuple[AcquisitionOrder | None, float]:
    """Classify cells using ImageJ order names and return the best match."""
    observed = tuple(cells)
    expected_count = width * height
    denominator = max(len(observed), expected_count, 1)
    best_orders: list[AcquisitionOrder] = []
    best_match = 0.0

    for order in AcquisitionOrder:
        expected = acquisition_cells(width, height, order)
        matches = sum(
            actual == candidate
            for actual, candidate in zip(observed, expected, strict=False)
        )
        match = matches / denominator
        if match > best_match:
            best_match = match
            best_orders = [order]
        elif match == best_match:
            best_orders.append(order)

    if (
        len(observed) != expected_count
        or best_match < 1.0
        or len(best_orders) != 1
    ):
        return None, best_match
    return best_orders[0], best_match


def legacy_serpentine_index_grid(width: int, height: int) -> np.ndarray:
    """Return the source-index grid produced by the legacy stitcher."""
    grid = np.empty((height, width), dtype=np.intp)
    cells = acquisition_cells(
        width,
        height,
        AcquisitionOrder.SNAKE_ROWS_LEFT_UP,
    )
    for source_index, (row, column) in enumerate(cells):
        grid[row, column] = source_index
    return grid


def _cluster_axis(values: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    """Assign slightly jittered stage coordinates to ordered grid cells."""
    order = np.argsort(values, kind="stable")
    labels = np.empty(len(values), dtype=np.intp)
    clusters: list[list[float]] = []

    for source_index in order:
        value = float(values[source_index])
        if not clusters or value - float(np.mean(clusters[-1])) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
        labels[source_index] = len(clusters) - 1

    return labels, len(clusters)


def _stage_grid_cells(
    positions: Sequence[Sequence[float]],
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        raise ValueError("Stage positions must be a non-empty sequence of XY pairs.")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("Stage positions must contain finite XY coordinates.")

    movements = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    movements = movements[movements > np.finfo(float).eps]
    spacing = float(np.median(movements)) if len(movements) else 1.0
    tolerance = max(spacing * 0.25, np.finfo(float).eps)

    columns, width = _cluster_axis(points[:, 0], tolerance)
    rows, height = _cluster_axis(points[:, 1], tolerance)
    cells = tuple(
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
    )
    return cells, width, height


def _make_layout(
    raw_cells: Sequence[tuple[int, int]],
    width: int,
    height: int,
    *,
    invert_x: bool,
    invert_y: bool,
) -> StitchLayout:
    if len(set(raw_cells)) != len(raw_cells):
        raise ValueError("Tile metadata contains duplicate grid positions.")

    acquisition_order, acquisition_match = classify_acquisition_order(
        raw_cells,
        width,
        height,
    )
    output_cells = tuple(
        (
            height - 1 - row if invert_y else row,
            width - 1 - column if invert_x else column,
        )
        for row, column in raw_cells
    )
    output_order, output_match = classify_acquisition_order(
        output_cells,
        width,
        height,
    )
    placements = tuple(
        TilePlacement(source_index, row, column)
        for source_index, (row, column) in enumerate(output_cells)
    )
    return StitchLayout(
        width=width,
        height=height,
        placements=placements,
        acquisition_order=acquisition_order,
        output_order=output_order,
        acquisition_match=acquisition_match,
        output_match=output_match,
    )


def layout_from_stage_positions(
    positions: Sequence[Sequence[float]],
    *,
    invert_x: bool = True,
    invert_y: bool = True,
) -> StitchLayout:
    """Create a rigid grid layout from approximate XY stage coordinates."""
    raw_cells, width, height = _stage_grid_cells(positions)
    return _make_layout(
        raw_cells,
        width,
        height,
        invert_x=invert_x,
        invert_y=invert_y,
    )


def layout_from_grid_metadata(
    stage_positions: Sequence[Mapping[str, object]],
    *,
    invert_x: bool = True,
    invert_y: bool = True,
) -> StitchLayout:
    """Create a rigid grid layout from Micro-Manager grid metadata."""
    if not stage_positions:
        raise ValueError("StagePositions metadata is empty.")

    metadata_cells = [
        (int(position["GridRow"]), int(position["GridCol"]))
        for position in stage_positions
    ]
    minimum_row = min(row for row, _ in metadata_cells)
    minimum_column = min(column for _, column in metadata_cells)
    raw_cells = tuple(
        (row - minimum_row, column - minimum_column)
        for row, column in metadata_cells
    )
    height = max(row for row, _ in raw_cells) + 1
    width = max(column for _, column in raw_cells) + 1
    return _make_layout(
        raw_cells,
        width,
        height,
        invert_x=invert_x,
        invert_y=invert_y,
    )


def describe_layout(layout: StitchLayout, section: str) -> str:
    """Format a concise, user-facing stitching status message."""
    acquired = (
        layout.acquisition_order.value
        if layout.acquisition_order is not None
        else f"Custom or unknown ({layout.acquisition_match:.0%} match)"
    )
    placed = (
        layout.output_order.value
        if layout.output_order is not None
        else f"Custom or unknown ({layout.output_match:.0%} match)"
    )
    return (
        f"{section}: {layout.width} × {layout.height} tiles | "
        f"acquired: {acquired} | placed: {placed} | "
        f"tile {TILE_SIZE_PX} px | overlap {TILE_OVERLAP_PX} px | "
        f"stride {TILE_STRIDE_PX} px"
    )
