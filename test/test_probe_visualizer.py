from types import SimpleNamespace

import numpy as np

from napari_dmc_brainmap.results.probe_vis.probe_vis.view.ProbeVisualizer import (
    ProbeVisualizer,
)


def test_organize_probe_keeps_coordinates_signed():
    visualizer = SimpleNamespace(
        probeDict={
            "probe": {
                "Voxel": [[0, -1, 2]],
                "axis": "AP",
            }
        },
        widget=SimpleNamespace(loadSlice=lambda _visualizer: None),
    )

    ProbeVisualizer.organizeProbe(visualizer)

    assert visualizer.probe_list[0].dtype == np.intp
    np.testing.assert_array_equal(visualizer.probe_list[0], [[0, -1, 2]])
    assert visualizer.probe_axis == [0]


def test_probe_visualizer_reports_left_positive_ml():
    visualizer = SimpleNamespace(
        bregma=[540, 0, 570],
        decimal=2,
        atlas=SimpleNamespace(
            resolution=(10.0, 10.0, 10.0),
            space=SimpleNamespace(axes_description=("ap", "si", "rl")),
        ),
    )

    coordinates = ProbeVisualizer.getCoordMM(visualizer, [540, 0, 730])

    assert coordinates == [0.0, 0.0, 1.6]
