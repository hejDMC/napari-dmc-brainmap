import numpy as np
import pytest

from napari_dmc_brainmap.results.results_helpers.results_quantifier import (
    ResultsQuantifier,
)


def test_majority_hemisphere_uses_left_positive_atlas_direction():
    assert ResultsQuantifier._majority_hemisphere(
        np.array([571, 600, 500]),
        570,
    ) == 'left'
    assert ResultsQuantifier._majority_hemisphere(
        np.array([500, 520, 600]),
        570,
    ) == 'right'


@pytest.mark.parametrize(
    ("rl_axis", "hemisphere", "expected"),
    [
        ("x", "left", np.arange(24).reshape(4, 6)[:, 3:]),
        ("x", "right", np.arange(24).reshape(4, 6)[:, :3]),
        ("y", "left", np.arange(24).reshape(4, 6)[2:, :]),
        ("y", "right", np.arange(24).reshape(4, 6)[:2, :]),
    ],
)
def test_crop_plane_to_hemisphere_uses_rl_axis(
    rl_axis,
    hemisphere,
    expected,
):
    plane = np.arange(24).reshape(4, 6)
    bregma_rl = 3 if rl_axis == "x" else 2

    result = ResultsQuantifier._crop_plane_to_hemisphere(
        plane,
        rl_axis,
        bregma_rl,
        hemisphere,
    )

    np.testing.assert_array_equal(result, expected)


def test_crop_plane_to_hemisphere_rejects_slice_axis():
    with pytest.raises(ValueError, match="in-plane RL axis"):
        ResultsQuantifier._crop_plane_to_hemisphere(
            np.zeros((2, 2)),
            "z",
            1,
            "left",
        )
