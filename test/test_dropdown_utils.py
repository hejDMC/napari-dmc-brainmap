from napari_dmc_brainmap.utils import dropdown_utils


def test_available_atlases_uses_v3_listing_api(monkeypatch) -> None:
    monkeypatch.setattr(
        dropdown_utils,
        "get_all_atlases_lastversions",
        lambda: {
            "admba_3d_e11_5_mouse_16um": "3.0",
            "allen_mouse_25um": "1.2",
            "example_mouse_100um": "1.0",
            "allen_mouse_10um": "3.0",
            "whs_sd_rat_39um": "1.1",
        },
    )

    available = dropdown_utils.get_available_atlases()

    assert available == {
        "allen_mouse_10um": "3.0",
        "admba_3d_e11_5_mouse_16um": "3.0",
        "allen_mouse_25um": "1.2",
        "whs_sd_rat_39um": "1.1",
        "example_mouse_100um": "1.0",
    }


def test_atlas_dropdown_contains_v3_atlas_names(monkeypatch) -> None:
    monkeypatch.setattr(
        dropdown_utils,
        "get_available_atlases",
        lambda: {
            "allen_mouse_25um": "1.2",
            "example_mouse_100um": "1.0",
        },
    )

    atlas_dropdown = dropdown_utils.get_atlas_dropdown()

    assert [member.value for member in atlas_dropdown] == [
        "allen_mouse_25um",
        "example_mouse_100um",
    ]
