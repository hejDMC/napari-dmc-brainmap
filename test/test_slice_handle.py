from types import SimpleNamespace

import numpy as np
import pytest

from napari_dmc_brainmap.results.results_helpers.slice_handle import SliceHandle


def _make_slice_handle(location_mm):
    handle = SliceHandle.__new__(SliceHandle)
    handle.regData = {"atlasLocation": {"0": [0, 0, location_mm]}}
    handle.regi_dict = {
        "xyz_dict": {
            "x": ("rl", 4),
            "y": ("si", 3),
            "z": ("ap", 10),
        }
    }
    handle.bregma = [5]
    handle.z_idx = 0
    handle.atlas = SimpleNamespace(
        space=SimpleNamespace(resolution=(10,)),
    )
    handle.calculateImageGrid()
    return handle


def test_get_z_plane_returns_signed_indices_for_valid_flat_slice():
    handle = _make_slice_handle(location_mm=0.0)

    z_plane = handle.get_z_plane("0")

    assert z_plane.dtype == np.intp
    np.testing.assert_array_equal(z_plane, np.full((3, 4), 5))


def test_get_z_plane_rejects_slice_outside_atlas_volume():
    handle = _make_slice_handle(location_mm=0.1)

    with pytest.raises(ValueError, match="outside volume bounds"):
        handle.get_z_plane("0")


def test_result_coordinates_report_left_positive_ml():
    handle = SliceHandle.__new__(SliceHandle)
    handle.currentSlice = 0
    handle.regi_dict = {
        "xyz_dict": {
            "x": ("ap", 1),
            "y": ("si", 1),
            "z": ("rl", 1),
        }
    }
    handle.bregma = [540, 0, 570]
    handle.atlas = SimpleNamespace(
        space=SimpleNamespace(
            axes_description=("ap", "si", "rl"),
            resolution=(10.0, 10.0, 10.0),
        ),
        structure_from_coords=lambda _coords: 1,
    )
    handle.sTree = SimpleNamespace(
        data={1: {"name": "region", "acronym": "REG"}}
    )
    handle.getVolumeIndex = lambda _slice, _coords: [[540, 0, 730]]

    result = handle.getBrainArea([(0, 0)], "section")

    assert result.loc[0, "ml_coords"] == 730
    assert result.loc[0, "ml_mm"] == 1.6


@pytest.mark.parametrize(
    ("xyz_dict", "slice_idx", "expected"),
    [
        (
            {
                "x": ("rl", 4),
                "y": ("si", 3),
                "z": ("ap", 2),
            },
            1,
            lambda annotation: annotation[1, :, :],
        ),
        (
            {
                "x": ("rl", 4),
                "y": ("ap", 2),
                "z": ("si", 3),
            },
            1,
            lambda annotation: annotation[:, 1, :],
        ),
        (
            {
                "x": ("si", 3),
                "y": ("ap", 2),
                "z": ("rl", 4),
            },
            1,
            lambda annotation: annotation[:, :, 1],
        ),
    ],
    ids=("coronal", "horizontal", "sagittal"),
)
def test_get_atlas_plane_respects_section_orientation(
    xyz_dict,
    slice_idx,
    expected,
):
    handle = SliceHandle.__new__(SliceHandle)
    handle.currentSlice = 0
    handle.regi_dict = {"xyz_dict": xyz_dict}
    handle.atlas = SimpleNamespace(
        space=SimpleNamespace(axes_description=("ap", "si", "rl"))
    )
    handle.annot = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    plane_shape = (xyz_dict["y"][1], xyz_dict["x"][1])
    handle.get_z_plane = lambda _slice: np.full(
        plane_shape,
        slice_idx,
        dtype=np.intp,
    )

    plane = handle.getAtlasPlane()

    np.testing.assert_array_equal(plane, expected(handle.annot))
