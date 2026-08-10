from __future__ import annotations

import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from napari_dmc_brainmap.utils.color_manager import ColorManager
from napari_dmc_brainmap.visualization.vis_plots.barplot_visualization import (
    BarplotVisualization,
)
from napari_dmc_brainmap.visualization.vis_plots.brainsection_visualization import (
    BrainsectionVisualization,
)


def _barplot(groups: str = "") -> BarplotVisualization:
    plotter = object.__new__(BarplotVisualization)
    plotter.plotting_params = {
        "bar_palette": ["#ff0000", "#0000ff"],
        "scatter_palette": ["#008000", "#800080"],
        "scatter_hue": False,
        "scatter_size": 4,
        "groups": groups,
        "gene_list": [],
    }
    plotter.color_manager = ColorManager()
    return plotter


def _facecolors(axis: plt.Axes) -> list[str]:
    return [mcolors.to_hex(patch.get_facecolor()) for patch in axis.patches]


def test_ungrouped_barplot_preserves_category_order_and_palette():
    plotter = _barplot()
    data = pd.DataFrame(
        {
            "tgt_name": ["VTA", "VTA", "MB", "MB"],
            "percent": [20.0, 40.0, 10.0, 30.0],
            "animal_id": ["a1", "a2", "a1", "a2"],
        }
    )
    source = data.copy(deep=True)
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plotter._plot_data(axis, data, "tgt_name", "percent", "v")

    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["VTA", "MB"]
    assert [patch.get_height() for patch in axis.patches] == [30.0, 20.0]
    assert _facecolors(axis) == ["#df2020", "#2020df"]
    assert not axis.lines
    pd.testing.assert_frame_equal(data, source)
    plt.close(figure)


def test_horizontal_ungrouped_barplot_uses_target_as_redundant_hue():
    plotter = _barplot()
    data = pd.DataFrame(
        {
            "tgt_name": ["VTA", "MB"],
            "percent": [30.0, 20.0],
        }
    )
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plotter._plot_data(axis, data, "percent", "tgt_name", "h")

    assert [tick.get_text() for tick in axis.get_yticklabels()] == ["VTA", "MB"]
    assert [patch.get_width() for patch in axis.patches] == [30.0, 20.0]
    assert _facecolors(axis) == ["#df2020", "#2020df"]
    assert axis.get_legend() is None
    plt.close(figure)


def test_grouped_barplot_preserves_hue_order_palette_and_values():
    plotter = _barplot(groups="group")
    data = pd.DataFrame(
        {
            "tgt_name": ["VTA", "VTA", "MB", "MB"],
            "percent": [20.0, 40.0, 10.0, 30.0],
            "groups": ["control", "treated", "control", "treated"],
        }
    )
    source = data.copy(deep=True)
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plotter._plot_group_data(axis, data, "tgt_name", "percent", "v")

    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["VTA", "MB"]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "control",
        "treated",
    ]
    assert [patch.get_height() for patch in axis.patches[:4]] == [20.0, 10.0, 40.0, 30.0]
    assert _facecolors(axis)[:4] == ["#df2020", "#df2020", "#2020df", "#2020df"]
    assert not axis.lines
    pd.testing.assert_frame_equal(data, source)
    plt.close(figure)


def test_single_gene_barplot_accepts_widget_palette_without_legend():
    plotter = _barplot()
    plotter.plotting_params["gene_list"] = ["gene_a"]
    data = pd.DataFrame(
        {
            "tgt_name": ["VTA", "MB"],
            "percent": [0.25, 0.75],
            "genes": ["gene_a", "gene_a"],
        }
    )
    source = data.copy(deep=True)
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plotter._plot_expression(axis, data, "tgt_name", "percent", "v")

    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["VTA", "MB"]
    assert [patch.get_height() for patch in axis.patches] == [0.25, 0.75]
    assert _facecolors(axis) == ["#df2020", "#2020df"]
    assert axis.get_legend() is None
    pd.testing.assert_frame_equal(data, source)
    plt.close(figure)


def test_multiple_gene_barplot_preserves_gene_hue_and_legend_order():
    plotter = _barplot()
    plotter.plotting_params["gene_list"] = ["gene_a", "gene_b"]
    data = pd.DataFrame(
        {
            "tgt_name": ["VTA", "VTA", "MB", "MB"],
            "percent": [0.25, 0.5, 0.75, 1.0],
            "genes": ["gene_a", "gene_b", "gene_a", "gene_b"],
        }
    )
    source = data.copy(deep=True)
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plotter._plot_expression(axis, data, "tgt_name", "percent", "v")

    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["VTA", "MB"]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "gene_a",
        "gene_b",
    ]
    assert [patch.get_height() for patch in axis.patches[:4]] == [
        0.25,
        0.75,
        0.5,
        1.0,
    ]
    assert _facecolors(axis)[:4] == ["#df2020", "#df2020", "#2020df", "#2020df"]
    assert not axis.lines
    pd.testing.assert_frame_equal(data, source)
    plt.close(figure)


def test_injection_kde_preserves_input_and_draws_filled_density():
    plotter = object.__new__(BrainsectionVisualization)
    plotter.orient_mapping = {"x_plot": "ml_coords", "y_plot": "dv_coords"}
    plotter.plotting_params = {"groups": "group"}
    plotter.color_dict = {
        "injection_site": {
            "single_color": True,
            "cmap": "#336699",
        }
    }
    data = pd.DataFrame(
        {
            "ml_coords": [0.0, 0.7, 1.1, 1.8, 2.2, 2.9],
            "dv_coords": [0.1, 0.4, 1.3, 1.0, 2.1, 2.7],
            "group": ["control"] * 6,
        }
    )
    source = data.copy(deep=True)
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.filterwarnings(
            "ignore",
            message="use_inf_as_na option is deprecated",
            category=FutureWarning,
        )
        plotter._plot_injection_site(axis, data, "injection_site")

    assert axis.collections
    pd.testing.assert_frame_equal(data, source)
    plt.close(figure)


def test_probe_regression_is_line_only_without_confidence_band():
    plotter = object.__new__(BrainsectionVisualization)
    plotter.orient_mapping = {"x_plot": "ml_coords", "y_plot": "dv_coords"}
    plotter.color_dict = {
        "optic_fiber": {
            "single_color": True,
            "cmap": "#336699",
        }
    }
    data = pd.DataFrame(
        {
            "ml_coords": [0.0, 1.0, 2.0, 3.0],
            "dv_coords": [1.0, 3.0, 5.0, 7.0],
        }
    )
    source = data.copy(deep=True)
    figure, axis = plt.subplots()

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        plotter._plot_optic_or_probe(axis, data, "optic_fiber")

    assert len(axis.lines) == 1
    assert axis.lines[0].get_xdata()[[0, -1]].tolist() == [0.0, 3.0]
    assert axis.lines[0].get_ydata()[[0, -1]].tolist() == pytest.approx([1.0, 7.0])
    assert not axis.collections
    pd.testing.assert_frame_equal(data, source)
    plt.close(figure)
