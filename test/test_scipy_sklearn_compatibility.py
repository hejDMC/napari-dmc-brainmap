from types import SimpleNamespace

import numpy as np
import pandas as pd

from napari_dmc_brainmap.results.results_helpers.results_creator import (
    _scale_positions,
)
from napari_dmc_brainmap.segment.processing.native_presegmentation import (
    smooth_slices,
)
from napari_dmc_brainmap.visualization.vis_plots.brainsection_plotter import (
    BrainsectionPlotter,
)


def test_scale_positions_preserves_shape_dtype_and_input() -> None:
    positions = np.array([0.0, 2.5, 5.0, 10.0], dtype=np.float32)
    original = positions.copy()

    scaled = _scale_positions(positions, source_max=10, target_max=4)

    np.testing.assert_array_equal(scaled, [0, 1, 2, 4])
    np.testing.assert_array_equal(positions, original)
    assert scaled.shape == positions.shape
    assert scaled.dtype == np.dtype(np.intp)


def test_scale_positions_accepts_empty_input() -> None:
    scaled = _scale_positions([], source_max=10, target_max=4)

    assert scaled.shape == (0,)
    assert scaled.dtype == np.dtype(np.intp)


def test_scale_positions_accepts_constant_coordinates() -> None:
    scaled = _scale_positions([5, 5, 5], source_max=10, target_max=4)

    np.testing.assert_array_equal(scaled, [2, 2, 2])


def test_density_normalization_does_not_mutate_source() -> None:
    plotter = object.__new__(BrainsectionPlotter)
    source = pd.DataFrame(
        {
            "acronym": ["A", "B", "C"],
            "density": [0.0, 2.0, 4.0],
        }
    )
    original = source.copy(deep=True)

    normalized = plotter._normalize_density(source)

    pd.testing.assert_frame_equal(source, original)
    np.testing.assert_allclose(normalized["density"], [0.1, 0.55, 1.0])
    assert normalized.shape == source.shape
    assert normalized["density"].dtype == np.dtype(np.float64)


def test_density_normalization_accepts_constant_and_empty_inputs() -> None:
    plotter = object.__new__(BrainsectionPlotter)
    constant = pd.DataFrame({"density": [0.0, 0.0]})
    empty = pd.DataFrame({"density": pd.Series(dtype=np.float64)})

    constant_result = plotter._normalize_density(constant)
    empty_result = plotter._normalize_density(empty)

    np.testing.assert_allclose(constant_result["density"], [0.1, 0.1])
    assert empty_result.empty
    assert empty_result["density"].dtype == np.dtype(np.float64)


def test_slice_smoothing_preserves_shape_dtype_and_input() -> None:
    image = np.zeros((2, 5, 5), dtype=np.float32)
    image[0, 2, 2] = 1.0
    image[1, 1, 3] = 2.0
    original = image.copy()

    smoothed = smooth_slices(image, sigma=1.0)

    np.testing.assert_array_equal(image, original)
    assert smoothed.shape == image.shape
    assert smoothed.dtype == image.dtype
    assert np.all(smoothed >= 0)
    assert smoothed[0, 2, 2] < image[0, 2, 2]
    assert smoothed[1, 1, 3] < image[1, 1, 3]


def test_voronoi_assignment_preserves_input_and_integer_output(
    monkeypatch,
) -> None:
    plotter = object.__new__(BrainsectionPlotter)
    plotter.atlas = SimpleNamespace()
    plotter.plotting_params = {
        "section_orient": "coronal",
        "plot_gene": "expression",
    }
    plotter.color_dict = {
        "genes": {"cmap": lambda value: (value, 0.0, 1.0 - value)}
    }
    source = pd.DataFrame(
        {
            "x": [0, 2, 0, 2],
            "y": [0, 0, 1, 1],
            "gene_expression_norm": [0.0, 0.25, 0.75, 1.0],
        }
    )
    original = source.copy(deep=True)
    monkeypatch.setattr(
        "napari_dmc_brainmap.visualization.vis_plots."
        "brainsection_plotter.get_xyz",
        lambda atlas, orientation: {"x": (None, 3), "y": (None, 2)},
    )

    matrix, color_dict = plotter._get_voronoi_mask(
        source,
        {"x_plot": "x", "y_plot": "y"},
    )

    pd.testing.assert_frame_equal(source, original)
    np.testing.assert_array_equal(matrix, [[0, 0, 1], [2, 2, 3]])
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.dtype(np.intp)
    assert color_dict[3] == (1.0, 0.0, 0.0)


def test_heatmap_smoothing_preserves_inputs_shape_and_dtype() -> None:
    plotter = object.__new__(BrainsectionPlotter)
    heatmap = np.zeros((3, 3), dtype=np.float32)
    heatmap[1, 1] = 1.0
    annotation = np.ones((3, 3), dtype=np.int32)
    heatmap_original = heatmap.copy()
    annotation_original = annotation.copy()

    smoothed, mask = plotter._resize_heatmap(heatmap, annotation)

    np.testing.assert_array_equal(heatmap, heatmap_original)
    np.testing.assert_array_equal(annotation, annotation_original)
    assert smoothed.shape == annotation.shape
    assert smoothed.dtype == np.dtype(np.float32)
    assert mask.shape == annotation.shape
    assert mask.dtype == np.dtype(bool)
    assert smoothed[1, 1] < 1.0
    assert smoothed[1, 1] == smoothed.max()
    assert not mask.any()
