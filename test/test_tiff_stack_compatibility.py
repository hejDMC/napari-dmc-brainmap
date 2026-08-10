import json

import numpy as np
import pytest
import tifffile

from napari_dmc_brainmap.preprocessing import preprocessing_tools
from napari_dmc_brainmap.stitching import stitching_tools


@pytest.mark.parametrize("compression", ["deflate", "lzw"])
@pytest.mark.parametrize(
    ("image", "photometric"),
    [
        (np.arange(30, dtype=np.uint8).reshape(5, 6), "minisblack"),
        (np.arange(30, dtype=np.uint16).reshape(5, 6) * 1000, "minisblack"),
        (
            np.arange(90, dtype=np.uint8).reshape(5, 6, 3),
            "rgb",
        ),
    ],
)
def test_compressed_tiff_round_trip_preserves_array(
    tmp_path, image, photometric, compression
):
    image_path = tmp_path / f"{photometric}_{image.dtype}_{compression}.tif"
    expected = image.copy()

    tifffile.imwrite(
        image_path,
        image,
        photometric=photometric,
        compression=compression,
    )
    loaded = tifffile.imread(image_path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == expected.dtype
    np.testing.assert_array_equal(loaded, expected)
    np.testing.assert_array_equal(image, expected)


def test_save_zstack_preserves_page_order_dtype_and_values(tmp_path):
    stack_path = tmp_path / "channels_stack.tif"
    stack = {
        "dapi": np.full((4, 5), 100, dtype=np.uint16),
        "green": np.full((4, 5), 200, dtype=np.uint16),
        "cy3": np.full((4, 5), 300, dtype=np.uint16),
    }
    expected = [image.copy() for image in stack.values()]

    preprocessing_tools.save_zstack(stack_path, stack)

    with tifffile.TiffFile(stack_path) as tif:
        assert len(tif.pages) == 3
        loaded = [page.asarray() for page in tif.pages]

    for actual, expected_page in zip(loaded, expected, strict=True):
        assert actual.dtype == np.uint16
        np.testing.assert_array_equal(actual, expected_page)

    for actual, expected_page in zip(stack.values(), expected, strict=True):
        np.testing.assert_array_equal(actual, expected_page)


def test_load_meta_preserves_imagej_info_json(tmp_path):
    metadata = {
        "Prefix": "animal_section",
        "StagePositions": [
            {"Label": "Pos0", "GridCol": 0, "GridRow": 0},
            {"Label": "Pos1", "GridCol": 1, "GridRow": 0},
        ],
    }
    image_path = tmp_path / "metadata.tif"
    tifffile.imwrite(
        image_path,
        np.zeros((4, 5), dtype=np.uint16),
        imagej=True,
        metadata={"Info": json.dumps(metadata)},
    )

    assert stitching_tools.load_meta(tmp_path) == metadata


def test_tifffile_fallback_loads_compressed_uint16_without_mutation(
    tmp_path, monkeypatch
):
    image_dir = tmp_path / "stitched" / "green"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "section_1_stitched.tif"
    expected = np.arange(42, dtype=np.uint16).reshape(6, 7) * 1000
    tifffile.imwrite(
        image_path,
        expected,
        photometric="minisblack",
        compression="lzw",
    )
    monkeypatch.setattr(preprocessing_tools.cv2, "imread", lambda *_: None)

    loaded = preprocessing_tools.load_stitched_images(
        tmp_path, "green", "section_1"
    )

    assert loaded.shape == expected.shape
    assert loaded.dtype == expected.dtype
    np.testing.assert_array_equal(loaded, expected)
