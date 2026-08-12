import json

import numpy as np
import pytest
import tifffile

from napari_dmc_brainmap.preprocessing import preprocessing_tools


def _write_section(root, channel, name, image):
    channel_dir = root / "stitched" / channel
    channel_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        channel_dir / f"{name}_stitched.tif",
        image,
        photometric="minisblack",
    )


def test_shared_bounds_use_all_channels_and_keep_internal_zeros():
    sparse_signal = np.zeros((8, 10), dtype=np.uint16)
    sparse_signal[4, 5] = 5000
    structural = np.zeros_like(sparse_signal)
    structural[2:7, 3:9] = 100

    bounds = preprocessing_tools.find_shared_content_bounds(
        [sparse_signal, structural]
    )
    histogram = preprocessing_tools.image_histogram(sparse_signal, bounds)

    assert bounds == (slice(2, 7), slice(3, 9))
    assert histogram.sum() == 30
    assert histogram[0] == 29
    assert histogram[5000] == 1


def test_section_histograms_are_combined_with_equal_weight():
    small_section = np.zeros(65536, dtype=np.uint64)
    large_section = np.zeros(65536, dtype=np.uint64)
    small_section[100] = 10
    large_section[1000] = 1000

    combined = preprocessing_tools.average_normalized_histograms(
        [small_section, large_section]
    )

    assert combined[100] == pytest.approx(0.5)
    assert combined[1000] == pytest.approx(0.5)
    assert combined.sum() == pytest.approx(1.0)


def test_automatic_profiles_estimate_dataset_wide_ranges(tmp_path):
    image_names = ["section_1", "section_2", "section_3"]
    progress_reports = []
    for section_index, image_name in enumerate(image_names):
        green = np.zeros((12, 12), dtype=np.uint16)
        green[1:11, 1:11] = np.arange(100, 200, dtype=np.uint16).reshape(
            10, 10
        )
        cy3 = np.zeros_like(green)
        cy3[1:11, 1:11] = 50
        if section_index == 0:
            cy3[5, 5] = 60000
        _write_section(tmp_path, "green", image_name, green)
        _write_section(tmp_path, "cy3", image_name, cy3)

    calibration = preprocessing_tools.estimate_automatic_rgb_ranges(
        tmp_path,
        image_names,
        source_channels=["green", "cy3"],
        target_profiles={
            "green": "balanced_cells",
            "cy3": "preserve_strong_signal_probe",
        },
        progress_callback=progress_reports.append,
    )

    assert calibration["automatic_ranges"]["green"] == [100, 199]
    assert calibration["automatic_ranges"]["cy3"] == [50, 60000]
    assert calibration["automatic_calibration"]["sections_used"] == {
        "green": 3,
        "cy3": 3,
    }
    assert calibration["automatic_calibration"][
        "mean_padding_fraction"
    ] == pytest.approx(1 - 100 / 144)
    assert len(progress_reports) == 4
    assert progress_reports[0] == {
        "event": "section",
        "section_index": 1,
        "total_sections": 3,
        "image_name": "section_1",
        "status": "processed",
        "padding_fraction": pytest.approx(1 - 100 / 144),
        "channel_cutoffs": {
            "green": [100, 199],
            "cy3": [50, 60000],
        },
    }
    assert progress_reports[-1] == {
        "event": "complete",
        "section_index": 3,
        "total_sections": 3,
        "automatic_ranges": {
            "green": [100, 199],
            "cy3": [50, 60000],
        },
    }


def test_resolved_ranges_are_json_serializable_and_reused(tmp_path, monkeypatch):
    image = np.zeros((6, 6), dtype=np.uint16)
    image[1:5, 1:5] = np.arange(16, dtype=np.uint16).reshape(4, 4) + 100
    _write_section(tmp_path, "green", "section_1", image)
    params = {
        "general": {"chans_imaged": ["green"]},
        "rgb_params": {
            "channels": ["green"],
            "contrast_mode": "automatic",
            "contrast_adjustment": True,
            "automatic_profiles": {"green": "balanced_cells"},
            "green": [],
        },
    }

    preprocessing_tools.resolve_automatic_rgb_params(
        tmp_path, ["section_1"], params
    )
    resolved = params["rgb_params"]["green"]
    json.dumps(params)

    monkeypatch.setattr(
        preprocessing_tools,
        "estimate_automatic_rgb_ranges",
        lambda *_args, **_kwargs: pytest.fail("cached range was not reused"),
    )
    preprocessing_tools.resolve_automatic_rgb_params(
        tmp_path, ["section_1"], params
    )

    assert resolved == [100, 115]
    assert params["rgb_params"]["automatic_ranges"]["green"] == resolved


def test_automatic_contrast_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="Unknown automatic profile"):
        preprocessing_tools.estimate_automatic_rgb_ranges(
            tmp_path,
            ["section_1"],
            source_channels=["green"],
            target_profiles={"green": "unknown"},
        )
