import pytest

from napari_dmc_brainmap.visualization.vis_utils.visualization_utils import (
    deduplicate_preserve_order,
    get_descendants,
    normalize_seaborn_style,
)


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("white", "white"),
        ("dark", "dark"),
        ("whitegrid", "whitegrid"),
        ("darkgrid", "darkgrid"),
        ("ticks", "ticks"),
    ],
)
def test_normalize_seaborn_style(style, expected):
    assert normalize_seaborn_style(style) == expected


@pytest.mark.parametrize("style", ["black", "unknown"])
def test_normalize_seaborn_style_rejects_unknown_style(style):
    with pytest.raises(ValueError, match="style must be one of"):
        normalize_seaborn_style(style)


def test_deduplicate_preserve_order():
    assert deduplicate_preserve_order(["VTA", "MBsen", "VTA", "SCs"]) == [
        "VTA",
        "MBsen",
        "SCs",
    ]


def test_get_descendants_deduplicates_overlapping_targets():
    class AtlasStub:
        descendants = {
            "MB": ["MBsen", "VTA"],
            "VTA": [],
        }

        def get_structure_descendants(self, target):
            return self.descendants[target]

    assert get_descendants(["MB", "VTA"], AtlasStub()) == ["MBsen", "VTA"]
