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
