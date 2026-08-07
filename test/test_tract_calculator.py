from types import SimpleNamespace

import numpy as np
import pandas as pd
from skspatial.objects import Line

from napari_dmc_brainmap.results.results_helpers.tract_calculator import (
    TractCalculator,
)


def test_get_linefit3d_preserves_fitted_line_and_voxelization():
    calculator = TractCalculator.__new__(TractCalculator)
    calculator.atlas = SimpleNamespace(
        shape=(10, 10, 10),
        space=SimpleNamespace(
            index_pairs=((1, 2), (0, 2), (0, 1)),
        ),
    )
    probe_df = pd.DataFrame(
        [
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (2.0, 4.0, 6.0),
            (3.0, 6.0, 9.2),
        ],
        columns=calculator.ABC_LIST,
    )

    linefit, linevox, primary_axis_idx = calculator._get_linefit3d(probe_df)

    np.testing.assert_allclose(
        linefit["point"].to_numpy(),
        [1.5, 3.0, 4.55],
    )
    np.testing.assert_allclose(
        np.abs(linefit["direction"].to_numpy()),
        [0.263828095, 0.52765619, 0.80744887],
        rtol=1e-7,
    )
    assert primary_axis_idx == 2
    assert linevox.columns.tolist() == calculator.ABC_LIST
    assert linevox["c_coord"].tolist() == list(range(10))
    assert linevox.iloc[[0, -1]].to_dict("records") == [
        {"a_coord": 0, "b_coord": 0, "c_coord": 0},
        {"a_coord": 2, "b_coord": 5, "c_coord": 9},
    ]


def test_get_voxelized_coord_clips_both_atlas_boundaries():
    calculator = TractCalculator.__new__(TractCalculator)
    calculator.atlas = SimpleNamespace(
        shape=(10, 10, 10),
        space=SimpleNamespace(
            index_pairs=((1, 2), (0, 2), (0, 1)),
        ),
    )
    line = Line(point=[5, 5, 5], direction=[1, 3, 3])

    coordinates = calculator._get_voxelized_coord(0, line)

    for coordinate, axis_size in zip(coordinates, calculator.atlas.shape):
        assert np.all(coordinate >= 0)
        assert np.all(coordinate < axis_size)
    np.testing.assert_array_equal(
        np.column_stack(coordinates)[0],
        [0, 0, 0],
    )
    np.testing.assert_array_equal(
        np.column_stack(coordinates)[-1],
        [9, 9, 9],
    )


def test_estimate_confidence_preserves_50_um_distances():
    calculator = TractCalculator.__new__(TractCalculator)
    calculator.annot = np.ones((21, 21, 21), dtype=np.int32)
    coordinates = pd.DataFrame(
        [[10, 10, 10]],
        columns=calculator.ABC_LIST,
    )

    confidence = calculator._estimate_confidence(
        coordinates,
        atlas_resolution_um=50,
    )

    assert confidence.dtype == np.float32
    np.testing.assert_array_equal(confidence, [500.0])
