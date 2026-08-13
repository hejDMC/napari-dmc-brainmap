import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from magicgui.widgets import FunctionGui
from qtpy.QtWidgets import QApplication

from napari_dmc_brainmap.padding import padding
from napari_dmc_brainmap.params import params
from napari_dmc_brainmap.preprocessing import preprocessing
from napari_dmc_brainmap.registration import registration
from napari_dmc_brainmap.results import results
from napari_dmc_brainmap.segment import segment
from napari_dmc_brainmap.stitching import stitching
from napari_dmc_brainmap.visualization import visualization


INITIALIZERS = [
    ("params", params.initialize_widget),
    ("padding", padding.initialize_widget),
    ("stitching", stitching.initialize_widget),
    ("preprocessing", preprocessing.initialize_header_widget),
    ("segment", segment.initialize_segment_widget),
    ("load presegmentation", segment.initialize_loadpreseg_widget),
    ("presegment cells", segment.initialize_presegcells_widget),
    ("presegment projections", segment.initialize_presegproj_widget),
    ("find centroids", segment.initialize_findcentroids_widget),
    ("registration", registration.initialize_widget),
    ("registration schematic", registration.initialize_schematic_widget),
    ("results", results.initialize_results_widget),
    ("quantification", results.initialize_quant_widget),
    ("merge", results.initialize_merge_widget),
    ("probe visualizer", results.initialize_probevis_widget),
    ("visualization header", visualization.initialize_header_widget),
    ("bar plot", visualization.initialize_barplot_widget),
    ("heatmap", visualization.initialize_heatmap_widget),
    ("brain section", visualization.initialize_brainsection_widget),
]


@pytest.fixture(scope="module", autouse=True)
def application():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize(("name", "initializer"), INITIALIZERS)
def test_magicgui_initializers_construct(name, initializer, monkeypatch):
    if initializer is params.initialize_widget:
        monkeypatch.setattr(
            params, "get_atlas_dropdown", lambda: ["allen_mouse_10um"]
        )

    widget = initializer()

    assert isinstance(widget, FunctionGui), name
    if name != "probe visualizer":
        assert len(widget) > 0, name
    widget.native.close()


def test_representative_widget_defaults_are_preserved(monkeypatch):
    monkeypatch.setattr(
        params, "get_atlas_dropdown", lambda: ["allen_mouse_10um"]
    )

    params_widget = params.initialize_widget()
    segment_widget = segment.initialize_presegcells_widget()
    visualization_widget = visualization.initialize_brainsection_widget()

    assert params_widget.inj_side.value == ""
    assert params_widget.chans_imaged.value == ["green", "cy3"]
    assert params_widget.section_orient.value == "coronal"
    assert params_widget.atlas.value == "allen_mouse_10um"

    assert segment_widget.single_channel_bool.value is False
    assert segment_widget.regi_chan.value == "green"
    assert segment_widget.minArea.value == "3"

    assert visualization_widget.save_fig.value is False
    assert visualization_widget.section_orient.value == "coronal"
    assert visualization_widget.hemisphere.value == "both"
    assert "left (ML > 0 mm)/right (ML < 0 mm)" in (
        visualization_widget.unilateral.tooltip
    )

    params_widget.native.close()
    segment_widget.native.close()
    visualization_widget.native.close()


def test_sharpy_cy3_default_matches_direct_stitching():
    preprocessing_widget = preprocessing.PreprocessingWidget()
    stitching_widget = stitching.initialize_widget()
    cy3_contrast = next(
        widget
        for widget in preprocessing_widget.sharpy_widget
        if widget.label == "Set contrast limits for cy3"
    )

    assert cy3_contrast.value == "50,500"
    assert stitching_widget.contrast_cy3.value == "50,500"

    preprocessing_widget.close()
    stitching_widget.native.close()


@pytest.mark.parametrize(
    (
        "seg_type",
        "show_probe_count",
        "show_point_size",
        "show_cells",
        "show_centroids",
        "show_projections",
    ),
    [
        ("cells", False, True, True, True, False),
        ("injection_site", False, False, False, False, False),
        ("optic_fiber", True, True, False, False, False),
        ("neuropixels_probe", True, True, False, False, False),
        ("projections", False, True, False, False, True),
    ],
)
def test_segmentation_type_shows_only_relevant_controls(
    seg_type,
    show_probe_count,
    show_point_size,
    show_cells,
    show_centroids,
    show_projections,
):
    widget = segment.SegmentWidget(SimpleNamespace())

    widget.segment.seg_type.value = seg_type

    assert widget.segment.n_probes.native.isHidden() is not show_probe_count
    assert widget.segment.point_size.native.isHidden() is not show_point_size
    assert widget._collapse_cells.isHidden() is not show_cells
    assert widget._collapse_centroid.isHidden() is not show_centroids
    assert widget._collapse_projections.isHidden() is not show_projections
    assert widget._collapse_load_preseg.isHidden() is False

    widget.close()


def test_segmentation_annotation_channels_follow_image_mode():
    widget = segment.SegmentWidget(SimpleNamespace())
    channels = widget.segment.channels

    assert [child.name for child in widget.segment][1:5] == [
        "input_path",
        "seg_type",
        "single_channel_bool",
        "channels",
    ]
    assert tuple(channels.choices) == ("dapi", "green", "cy3")
    assert channels.value == ["green", "cy3"]
    assert widget.segment.contrast_dapi.native.isHidden() is True
    assert widget.segment.contrast_green.native.isHidden() is False
    assert widget.segment.contrast_cy3.native.isHidden() is False
    assert widget.segment.contrast_n3.native.isHidden() is True
    assert widget.segment.contrast_cy5.native.isHidden() is True

    widget.segment.single_channel_bool.value = True
    assert tuple(channels.choices) == (
        "dapi",
        "green",
        "n3",
        "cy3",
        "cy5",
    )

    channels.value = ["green", "n3"]
    assert widget.segment.contrast_green.native.isHidden() is False
    assert widget.segment.contrast_n3.native.isHidden() is False
    assert widget.segment.contrast_cy3.native.isHidden() is True

    widget.segment.single_channel_bool.value = False
    assert tuple(channels.choices) == ("dapi", "green", "cy3")
    assert channels.value == ["green"]
    assert widget.segment.contrast_n3.native.isHidden() is True

    widget.close()


def test_hidden_contrast_fields_are_not_parsed():
    widget = segment.SegmentWidget(SimpleNamespace())
    widget.segment.contrast_n3.value = ""
    widget.segment.contrast_cy5.value = "not,a,range"

    contrast = widget._get_contrast_dict(
        widget.segment,
        widget.segment.channels.value,
    )

    assert contrast == {"green": [0, 100], "cy3": [0, 100]}
    widget.close()


def test_disabled_rgb_contrast_does_not_parse_limit_fields():
    preprocessing_widget = preprocessing.PreprocessingWidget()
    rgb_widget = preprocessing_widget.rgb_widget
    rgb_widget["contrast_mode"].value = "none"

    for channel in ("cy3", "green"):
        rgb_widget[f"{channel}_row"][
            f"{channel}_manual_range"
        ].value = ""

    rgb_params = preprocessing_widget._get_widget_info(rgb_widget, "rgb")

    assert rgb_params["contrast_mode"] == "none"
    assert rgb_params["contrast_adjustment"] is False
    assert rgb_params["green"] == []
    assert rgb_params["cy3"] == []

    preprocessing_widget.close()


def test_rgb_manual_mode_preserves_existing_defaults():
    preprocessing_widget = preprocessing.PreprocessingWidget()

    rgb_params = preprocessing_widget._get_rgb_widget_info()

    assert rgb_params == {
        "channels": ["cy3", "green"],
        "downsampling": 3,
        "contrast_mode": "manual",
        "contrast_adjustment": True,
        "cy3": [50, 2000],
        "green": [50, 1000],
    }

    preprocessing_widget.close()


def test_unselected_rgb_channel_disables_automatic_profile():
    preprocessing_widget = preprocessing.PreprocessingWidget()
    rgb_widget = preprocessing_widget.rgb_widget
    rgb_widget["contrast_mode"].value = "automatic"
    green_row = rgb_widget["green_row"]
    green_row["green_profile"].value = "preserve_strong_signal_probe"
    green_row["green_enabled"].value = False

    rgb_params = preprocessing_widget._get_rgb_widget_info()

    assert green_row["green_profile"].enabled is False
    assert green_row["green_estimated_range"].value == "Not selected"
    assert rgb_params["channels"] == ["cy3"]
    assert rgb_params["automatic_profiles"] == {
        "cy3": "preserve_strong_signal_probe"
    }

    green_row["green_enabled"].value = True
    assert green_row["green_profile"].enabled is True
    assert (
        green_row["green_profile"].value
        == "preserve_strong_signal_probe"
    )

    preprocessing_widget.close()


def test_rgb_estimation_progress_prints_section_cutoffs(capsys):
    preprocessing_widget = preprocessing.PreprocessingWidget()
    rgb_widget = preprocessing_widget.rgb_widget
    rgb_widget["contrast_mode"].value = "automatic"
    calibration_key = preprocessing_widget._current_rgb_calibration_key()

    preprocessing_widget._update_rgb_estimation_progress(
        (
            calibration_key,
            {
                "event": "section",
                "section_index": 2,
                "total_sections": 5,
                "image_name": "section_002",
                "status": "processed",
                "padding_fraction": 0.125,
                "channel_cutoffs": {
                    "cy3": [50, 60000],
                    "green": [100, 900],
                },
            },
        )
    )

    assert rgb_widget["estimation_progress"].value == 2
    assert rgb_widget["estimate_ranges"].text == "Estimating 2/5..."
    output = capsys.readouterr().out
    assert "[Automatic RGB contrast] [2/5] section_002" in output
    assert "cy3=50–60000" in output
    assert "green=100–900" in output
    assert "padding excluded=12.50%" in output
    assert all(
        widget.name != "estimation_report" for widget in rgb_widget
    )

    preprocessing_widget.close()
