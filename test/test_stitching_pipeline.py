import json

import numpy as np
import pytest
import tifffile

from napari_dmc_brainmap.stitching.layout import (
    layout_from_stage_positions,
    legacy_serpentine_index_grid,
)
from napari_dmc_brainmap.stitching.stitching import (
    iter_tile_pages,
    load_embedded_stage_positions,
    load_tile_section,
    split_stage_position_regions,
)
from napari_dmc_brainmap.stitching import stitching
from napari_dmc_brainmap.stitching import tiff_output
from napari_dmc_brainmap.stitching.stitching_tools import (
    downsample_image,
    padding_for_atlas,
    stitch_folder,
    stitch_stack,
)


def _tiles(count: int, size: int = 3) -> np.ndarray:
    return np.stack(
        [np.full((size, size), index + 1, dtype=np.uint16) for index in range(count)]
    )


@pytest.fixture(autouse=True)
def _isolated_stitching_scratch(monkeypatch, tmp_path):
    scratch_root = tmp_path / "app-cache" / "stitching" / "scratch"
    scratch_root.mkdir(parents=True)
    monkeypatch.setattr(
        tiff_output,
        "stitching_scratch_root",
        lambda: scratch_root,
    )


def test_stack_workflow_rejects_output_inside_raw_directory(tmp_path):
    input_path = tmp_path / "animal"
    raw_directory = input_path / "raw"
    raw_directory.mkdir(parents=True)
    sentinel = raw_directory / "source-data"
    sentinel.write_text("unchanged", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Refusing to write stitching output outside its expected",
    ):
        stitching.process_stitch_stack(
            input_path,
            raw_directory / "obj3",
            "cy3",
            raw_directory,
            "animal",
            "obj3",
            {},
            (1140, 800),
            False,
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged"


def _legacy_canvas(
    tiles: np.ndarray,
    width: int,
    height: int,
    overlap: int,
) -> np.ndarray:
    tile_size = tiles.shape[-1]
    stride = tile_size - overlap
    canvas = np.zeros(
        (
            tile_size + stride * (height - 1),
            tile_size + stride * (width - 1),
        ),
        dtype=np.uint16,
    )
    source_grid = legacy_serpentine_index_grid(width, height)
    for row, column in np.ndindex(height, width):
        top = row * stride
        left = column * stride
        canvas[top:top + tile_size, left:left + tile_size] = tiles[
            source_grid[row, column]
        ]
    return canvas


def _write_embedded_stack(path, tiles, positions):
    with tifffile.TiffWriter(path) as writer:
        for tile, (x_position, y_position) in zip(
            tiles,
            positions,
            strict=True,
        ):
            metadata = json.dumps(
                {
                    "XPosition_um_Intended": x_position,
                    "YPosition_um_Intended": y_position,
                    "ZPosition_um_Intended": 100.0,
                }
            )
            writer.write(
                tile,
                photometric="minisblack",
                extratags=[(51123, "s", 0, metadata, False)],
            )


def test_embedded_tiff_positions_replace_regions_json(tmp_path):
    first_region = [(0, 0), (100, 0), (100, 100), (0, 100)]
    second_region = [
        (1000, 1000),
        (1100, 1000),
        (1100, 1100),
        (1000, 1100),
    ]
    positions = first_region + second_region
    stack_path = tmp_path / "stack.tif"
    _write_embedded_stack(stack_path, _tiles(len(positions)), positions)

    loaded = load_embedded_stage_positions(tmp_path, [stack_path.name])
    regions = split_stage_position_regions(loaded)

    assert loaded == [(float(x), float(y)) for x, y in positions]
    assert regions == [
        [(float(x), float(y)) for x, y in first_region],
        [(float(x), float(y)) for x, y in second_region],
    ]


def test_ndtiff_index_is_used_only_when_tiff_positions_are_missing(
    tmp_path,
    monkeypatch,
):
    stack_path = tmp_path / "stack.tif"
    tifffile.imwrite(stack_path, _tiles(1)[0], photometric="minisblack")
    (tmp_path / "NDTiff.index").write_bytes(b"fixture")
    monkeypatch.setattr(
        stitching.tiff,
        "read_ndtiff_index",
        lambda _: iter(
            [
                (
                    {"pos_x": 12, "pos_y": 34},
                    stack_path.name,
                    0,
                    3,
                    3,
                    1,
                    0,
                    0,
                    0,
                    0,
                )
            ]
        ),
    )

    positions = load_embedded_stage_positions(tmp_path, [stack_path.name])

    assert positions == [(12.0, 34.0)]


def test_layout_report_is_printed_to_terminal(capsys):
    layout = layout_from_stage_positions(
        [(0, 0), (100, 0), (100, 100), (0, 100)]
    )

    stitching._report_layout(layout, "section 1")

    report = capsys.readouterr().out
    assert "section 1: 2 × 2 tiles" in report
    assert "acquired: Grid: snake by rows — Right & Down" in report
    assert "placed: Grid: snake by rows — Left & Up" in report
    assert "overlap 205 px" in report


def test_tile_pages_are_loaded_one_region_at_a_time(tmp_path):
    tiles = _tiles(6)
    positions = [(index * 100, 0) for index in range(len(tiles))]
    first_stack = tmp_path / "stack.tif"
    second_stack = tmp_path / "stack_1.tif"
    _write_embedded_stack(first_stack, tiles[:4], positions[:4])
    _write_embedded_stack(second_stack, tiles[4:], positions[4:])
    pages = iter_tile_pages(
        tmp_path,
        [first_stack.name, second_stack.name],
    )

    first_section = load_tile_section(pages, 4, c_size=3)
    second_section = load_tile_section(pages, 2, c_size=3)

    np.testing.assert_array_equal(first_section, tiles[:4])
    np.testing.assert_array_equal(second_section, tiles[4:])


def test_stack_stitching_preserves_legacy_pixels_dtype_and_input(tmp_path):
    positions = [(0, 0), (100, 0), (100, 100), (0, 100)]
    tiles = _tiles(4)
    original = tiles.copy()
    output_path = tmp_path / "stitched.tif"

    layout = stitch_stack(
        positions,
        tiles,
        1,
        output_path,
        {},
        "cy3",
        resolution=False,
        c_size=3,
    )
    actual = tifffile.imread(output_path)
    expected = _legacy_canvas(tiles, 2, 2, 1)

    assert layout.width == 2
    assert layout.height == 2
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(tiles, original)
    with tifffile.TiffFile(output_path) as tif:
        assert tif.pages[0].tags["Compression"].value in (8, 32946)
        assert tif.pages[0].tags["Predictor"].value == 2


def test_stack_stitching_builds_padding_into_output_file(tmp_path):
    positions = [(0, 0), (100, 0)]
    tiles = _tiles(2)
    output_path = tmp_path / "padded.tif"
    expected = padding_for_atlas(
        _legacy_canvas(tiles, 2, 1, 1),
        (4, 3),
    )

    stitch_stack(
        positions,
        tiles,
        1,
        output_path,
        {},
        "cy3",
        resolution=(4, 3),
        c_size=3,
    )

    np.testing.assert_array_equal(tifffile.imread(output_path), expected)


def test_stack_stitching_preserves_downsampled_registration_output(tmp_path):
    positions = [(0, 0), (100, 0), (100, 100), (0, 100)]
    tiles = _tiles(4)
    output_path = tmp_path / "stitched.tif"
    downsampled_path = tmp_path / "downsampled.tif"
    resolution = (4, 3)
    contrast = (0, 4)
    expected_stitched = padding_for_atlas(
        _legacy_canvas(tiles, 2, 2, 1),
        resolution,
    )
    expected_downsampled = downsample_image(
        expected_stitched,
        resolution,
        contrast,
    )

    stitch_stack(
        positions,
        tiles,
        1,
        output_path,
        {"sharpy_track_params": {"cy3": list(contrast)}},
        "cy3",
        downsampled_path=downsampled_path,
        resolution=resolution,
        c_size=3,
    )

    np.testing.assert_array_equal(
        tifffile.imread(downsampled_path),
        expected_downsampled,
    )


def test_folder_stitching_uses_grid_metadata_and_matches_legacy(tmp_path):
    positions = [
        {"Label": "Pos0", "GridRow": 0, "GridCol": 0},
        {"Label": "Pos1", "GridRow": 0, "GridCol": 1},
        {"Label": "Pos2", "GridRow": 1, "GridCol": 1},
        {"Label": "Pos3", "GridRow": 1, "GridCol": 0},
    ]
    metadata = {"Prefix": "section", "StagePositions": positions}
    tiles = _tiles(4)
    for tile, position in zip(tiles, positions, strict=True):
        tifffile.imwrite(
            tmp_path / f"section_MMStack_{position['Label']}.ome.tif",
            tile,
            imagej=True,
            metadata={"Info": json.dumps(metadata)},
        )
    output_path = tmp_path / "stitched.tif"

    stitch_folder(
        tmp_path,
        1,
        output_path,
        {},
        "cy3",
        resolution=False,
        c_size=3,
    )

    np.testing.assert_array_equal(
        tifffile.imread(output_path),
        _legacy_canvas(tiles, 2, 2, 1),
    )
