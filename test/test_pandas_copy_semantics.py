from contextlib import contextmanager
import warnings
from types import SimpleNamespace

import pandas as pd

from napari_dmc_brainmap.utils.data_loader import DataLoader
from napari_dmc_brainmap.utils.dataframe_utils import clip_negative_first_row
from napari_dmc_brainmap.visualization.vis_plots.brainsection_visualization import (
    BrainsectionVisualization,
)
from napari_dmc_brainmap.visualization.vis_plots.heatmap_visualization import (
    HeatmapVisualization,
)
from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import (
    resort_df,
)


class _HierarchyStub:
    def is_branch(self, structure_id):
        return [structure_id]


class _AtlasStub:
    hierarchy = _HierarchyStub()
    structures = {
        "VTA": {"id": 1},
        1: {"acronym": "VTA"},
    }

    def get_structure_ancestors(self, structure_id):
        return []


@contextmanager
def _setting_with_copy_is_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.SettingWithCopyWarning)
        yield


def test_target_filter_does_not_warn_or_mutate_source():
    loader = object.__new__(DataLoader)
    loader.atlas = _AtlasStub()
    source = pd.DataFrame({"structure_id": [1, 999], "value": [10, 20]})

    with _setting_with_copy_is_error():
        result = loader.get_tgt_data_only(
            source,
            ["VTA", "NA"],
            use_na=True,
        )

    assert source.to_dict("list") == {
        "structure_id": [1, 999],
        "value": [10, 20],
    }
    assert result["structure_id"].tolist() == [1, -42]
    assert result["tgt_name"].tolist() == ["VTA", "NA"]


def test_inside_brain_filter_does_not_warn_or_mutate_loaded_data(
    monkeypatch,
    tmp_path,
):
    result_dir = tmp_path / "results"
    (result_dir / "probe").mkdir(parents=True)
    source = pd.DataFrame(
        {
            "inside_brain": [True, False],
            "ml_mm": [1.0, 2.0],
        }
    )

    loader = object.__new__(DataLoader)
    loader.input_path = tmp_path
    loader.atlas = SimpleNamespace(metadata={"name": "custom_atlas"})
    loader.animal_list = ["animal-1"]
    loader.channels = ["probe"]
    loader.data_type = "neuropixels_probe"
    loader.hemisphere = "both"
    loader._load_params = lambda _: {"injection_site": "right"}
    loader._load_channel_data = lambda animal_id, channel: source
    monkeypatch.setattr(
        "napari_dmc_brainmap.utils.data_loader.get_info",
        lambda *args, **kwargs: result_dir,
    )

    with _setting_with_copy_is_error():
        result = loader.load_data()

    assert source.to_dict("list") == {
        "inside_brain": [True, False],
        "ml_mm": [1.0, 2.0],
    }
    assert result["ml_mm"].tolist() == [-1.0]
    assert result["ipsi_contra"].tolist() == ["ipsi"]


def test_heatmap_binning_accepts_a_filtered_frame_without_warning():
    heatmap = object.__new__(HeatmapVisualization)
    heatmap.plotting_params = {
        "interval_labels": ["0.0 to 0.5", "0.5 to 1.0"],
        "intervals": [0.0, 0.5, 1.0],
    }
    source = pd.DataFrame(
        {
            "group": ["a", "b"],
            "ap_mm": [0.25, 0.75],
        }
    )
    filtered = source.loc[source["group"] == "a"]

    with _setting_with_copy_is_error():
        result = heatmap._extract_target_data(filtered)

    assert "bin" not in source.columns
    assert "bin" not in filtered.columns
    assert str(result["bin"].iloc[0]) == "0.0 to 0.5"


def test_section_filter_returns_independent_coordinate_data():
    brainsection = object.__new__(BrainsectionVisualization)
    source = pd.DataFrame(
        {
            "ap_coords": [1, 1, 3],
            "ml_coords": [1.0, -1.0, 1.5],
        }
    )
    brainsection.data_dict = {"cells": source}
    brainsection.plotting_params = {
        "color_brain_genes": "no_color",
        "unilateral": "left",
    }
    brainsection.orient_mapping = {"z_plot": ("ap_coords", 0, 1)}
    brainsection.atlas = SimpleNamespace(
        space=SimpleNamespace(axes_description=["ap", "si", "rl"])
    )
    brainsection.bregma = [0, 0, 0.5]
    brainsection._generate_brain_schematic = lambda slice_idx: None

    with _setting_with_copy_is_error():
        plot_dict, _ = brainsection._get_section_filter_data(0, [0, 2])

    assert source["ml_coords"].tolist() == [1.0, -1.0, 1.5]
    assert plot_dict["cells"]["ml_coords"].tolist() == [0.5]


def test_resort_df_accepts_a_slice_without_mutating_it():
    source = pd.DataFrame(
        {
            "keep": [True, True, False],
            "tgt_name": ["VTA", "MB", "VTA"],
            "animal_id": ["animal-1", "animal-1", "animal-1"],
        }
    )
    filtered = source.loc[source["keep"]]

    with _setting_with_copy_is_error():
        result = resort_df(filtered, ["MB", "VTA"])

    assert "tgt_name_sort" not in source.columns
    assert "tgt_name_sort" not in filtered.columns
    assert result["tgt_name"].tolist() == ["MB", "VTA"]


def test_clip_negative_first_row_is_copy_on_write_safe():
    source = pd.DataFrame(
        [[-2.0, 3.0], [-4.0, 5.0]],
        columns=["a", "b"],
    )

    with pd.option_context("mode.copy_on_write", True):
        result = clip_negative_first_row(source)

    assert source.to_numpy().tolist() == [[-2.0, 3.0], [-4.0, 5.0]]
    assert result.to_numpy().tolist() == [[0.0, 3.0], [-4.0, 5.0]]
